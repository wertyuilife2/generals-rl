"""Real two-rank PPO lifecycle without needing two GPUs for regression tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rsl_rl.runners import OnPolicyRunnerCTS

from generals_rl.self_play import HistoricalPolicyPool
from generals_rl.self_play_training import prepare_self_play
from generals_rl.train import train


def small_config():
    config = json.loads((Path(__file__).parents[1] / "configs/v32-opt.json").read_text())
    config["task"].update(num_envs=4, backend="cpu", device="cpu", map_pool_size=0,
                          max_episode_steps=4)
    config["runner"]["num_steps_per_env"] = 4
    config["runner"]["save_interval"] = 1
    config["runner"]["algorithm"].update(num_learning_epochs=1, num_mini_batches=2,
                                          physical_minibatch_size=2, ppo_probe_samples=4,
                                          ppo_probe_batch_size=2)
    config["self_play"]["snapshot_interval"] = 1
    config["evaluation"] = {"enabled": False}
    config["training_control"] = {}
    return config


def train_worker(rank, directory, opponent="current_self_play"):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{root}/group", rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    environment = {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}
    observed, results, writes = {}, {}, []

    original_manifest = HistoricalPolicyPool._write_manifest

    def write_manifest(pool):
        assert rank == 0, "only rank zero may write the shared historical pool"
        writes.append({"run": pool.directory.parent.name, "ids": list(pool.ids)})
        return original_manifest(pool)

    def capture_session(task, runner, *args, **kwargs):
        session = prepare_self_play(task, runner, *args, **kwargs)
        observed["session"] = session
        if opponent == "self_play":
            assert session.pool.read_only == (rank != 0)
            assert kwargs.get("read_only", False) == (rank != 0)
            observed["initial_weights"] = torch.cat([
                parameter.detach().flatten().clone() for parameter in runner.alg.policy.parameters()])
            newest = session.pool.get(session.pool.ids[-1]).model
            newest_weights = torch.cat([parameter.detach().flatten() for parameter in newest.parameters()])
            torch.testing.assert_close(newest_weights, observed["initial_weights"], rtol=0, atol=0)
            observed["initial_pool"] = {"ids": list(session.pool.ids),
                "iterations": [session.pool.get(value).metadata["iteration"] for value in session.pool.ids]}
            observed["pool_updates"] = []
            observed["opponent_seed"] = task.opponent.generator.initial_seed()
        return session

    def verify_weights():
        weights = torch.cat([parameter.detach().flatten()
                             for parameter in observed["runner"].alg.policy.parameters()])
        gathered = [torch.empty_like(weights), torch.empty_like(weights)]
        dist.all_gather(gathered, weights)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        if opponent == "self_play":
            dist.all_gather(gathered, observed["initial_weights"])
            torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)

    def record_history(name):
        if opponent != "self_play":
            return
        task = observed["runner"].env.task
        results[name + "_history"] = {
            "initial_pool": observed["initial_pool"], "updates": observed["pool_updates"],
            "final_pool": list(task.opponent.pool.ids), "assignments": task.opponent.episode_assignments,
            "counts": {str(key): value for key, value in task.opponent.snapshot_counts().items()},
            "opponent_seed": observed["opponent_seed"], "task_seed": task.config.seed,
        }

    def runner_factory(*args, **kwargs):
        runner = OnPolicyRunnerCTS(*args, **kwargs)
        observed["runner"] = runner
        original_log = runner.logger.log

        def log(**values):
            if opponent == "self_play":
                observed["pool_updates"].append(list(observed["session"].pool.ids))
            result = original_log(**values)
            if rank == 0 and observed.get("request_stop"):
                (root / "stop/STOP").touch()
            return result

        runner.logger.log = log
        return runner

    try:
        with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()), \
                patch("generals_rl.train.OnPolicyRunnerCTS", runner_factory), \
                patch("generals_rl.train.prepare_self_play", capture_session), \
                patch.object(HistoricalPolicyPool, "_write_manifest", write_manifest):
            config = small_config()
            config["task"]["opponent"] = opponent
            if opponent == "self_play":
                config["self_play"].update(snapshot_interval=2, pool_size=3)
            original = deepcopy(config)
            results["fresh"] = train(config, root / "fresh", steps_per_env=6)
            assert config == original
            assert dist.is_initialized()  # The outer test still owns this group.
            if opponent == "current_self_play":
                assert observed["runner"].env.task.opponent.policy is observed["runner"].alg.policy
            assert observed["runner"].logger.disable_logs == (rank != 0)
            verify_weights()
            record_history("fresh")
            results["same_weights"] = True
            dist.barrier()  # Checkpoint is a rank-zero artifact, not an all-reduce output.
            source_pool = {path.name: path.read_bytes() for path in (root / "fresh/history_pool").iterdir()}
            selected = root / "fresh/model_0.pt" if opponent == "self_play" else root / "fresh/checkpoint.pt"
            selected_iteration = (torch.load(selected, map_location="cpu", weights_only=True))["iter"]
            results["resume"] = train(config, root / "resume", iterations=1,
                                       resume=str(selected))
            assert observed["runner"].current_learning_iteration == selected_iteration + 1
            assert {path.name: path.read_bytes() for path in (root / "fresh/history_pool").iterdir()} == source_pool
            verify_weights()
            record_history("resume")
            dist.barrier()
            observed["request_stop"] = True
            results["stop"] = train(config, root / "stop", iterations=5)
            assert results["stop"]["iterations"] == 1
            assert results["stop"]["stop_reason"] == "requested_stop"
            verify_weights()
            record_history("stop")
            assert dist.is_initialized()
            dist.barrier()
            try:
                train(config, root / "fresh", iterations=1)
                raise AssertionError("existing output accepted")
            except (FileExistsError, RuntimeError) as error:
                results["output_error"] = str(error)
            assert not torch.cuda.is_initialized()
            results["pool_writes"] = writes
            (root / f"rank_{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def historical_failure_worker(rank, directory):
    """A failed reader must release its peers from the same collective phase."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{root}/failure_group", rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    environment = {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}
    config = small_config()
    config["task"]["opponent"] = "self_play"
    original_refresh = HistoricalPolicyPool.refresh
    results = {}
    try:
        for phase in ("prepare", "refresh"):
            def refresh(pool):
                if rank == 1 and (phase == "prepare" or pool.ids):
                    raise OSError(f"injected {phase} failure")
                return original_refresh(pool)

            with patch.dict(os.environ, environment), redirect_stdout(io.StringIO()), \
                    patch.object(HistoricalPolicyPool, "refresh", refresh):
                try:
                    train(config, root / phase, iterations=1)
                    raise AssertionError("reader failure was accepted")
                except (OSError, RuntimeError) as error:
                    results[phase] = {"type": type(error).__name__, "error": str(error)}
            assert dist.is_initialized()
            dist.barrier()
        assert not torch.cuda.is_initialized()
        (root / f"failure_rank_{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


class DistributedTrainEntrypointTests(unittest.TestCase):
    def test_full_training_resume_partial_budget_rank_zero_stop_and_file_ownership(self):
        self._assert_lifecycle("current_self_play")

    def test_historical_training_shared_pool_refresh_resume_cutoff_and_global_counts(self):
        self._assert_lifecycle("self_play")

    def test_failed_historical_reader_prepare_or_refresh_stops_every_rank(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            mp.spawn(historical_failure_worker, args=(directory,), nprocs=2, join=True)
            ranks = [json.loads((Path(directory) / f"failure_rank_{rank}.json").read_text())
                     for rank in range(2)]
            for phase, operation in (("prepare", "preparation"), ("refresh", "refresh")):
                self.assertEqual(ranks[0][phase]["type"], "RuntimeError")
                self.assertIn(f"{operation} failed on another rank", ranks[0][phase]["error"])
                self.assertEqual(ranks[1][phase]["type"], "OSError")
                self.assertEqual(ranks[1][phase]["error"], f"injected {phase} failure")

    def _assert_lifecycle(self, opponent):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            mp.spawn(train_worker, args=(directory, opponent), nprocs=2, join=True)
            root = Path(directory)
            ranks = [json.loads((root / f"rank_{rank}.json").read_text()) for rank in range(2)]
            for rank in ranks:
                self.assertTrue(rank["same_weights"])
                self.assertEqual(rank["fresh"]["environment_steps"], 4 * 6 * 2)
                self.assertEqual(rank["fresh"]["student_envs"], 8)
                self.assertEqual(rank["fresh"]["iterations"], 2)
                self.assertEqual(rank["resume"]["environment_steps"], 4 * 4 * 2)
                self.assertEqual(rank["stop"]["environment_steps"], 4 * 4 * 2)
                self.assertEqual(rank["stop"]["stop_reason"], "requested_stop")
                self.assertIn("training output already exists", rank["output_error"])
            for name in ("fresh", "resume", "stop"):
                self.assertEqual(len(list((root / name).glob("events.out.*"))), 1)
                summary = json.loads((root / name / "summary.json").read_text())
                self.assertEqual(summary, ranks[0][name])
                self.assertEqual(summary["distributed"]["world_size"], 2)
                if opponent == "current_self_play":
                    self.assertFalse(summary["self_play"]["snapshots_used_for_training"])
                else:
                    self.assertEqual(summary["self_play"]["counts_scope"], "global")
                    self.assertEqual(ranks[0][name]["self_play"], ranks[1][name]["self_play"])
                    self.assertEqual(summary["self_play"]["new_publications"], 2)
                    history = [rank[name + "_history"] for rank in ranks]
                    self.assertEqual(history[0]["updates"], history[1]["updates"])
                    self.assertEqual(history[0]["final_pool"], history[1]["final_pool"])
                    self.assertNotEqual(history[0]["task_seed"], history[1]["task_seed"])
                    self.assertNotEqual(history[0]["opponent_seed"], history[1]["opponent_seed"])
                    self.assertEqual(summary["self_play"]["episode_assignments"],
                                     sum(rank["assignments"] for rank in history))
                    counts = {}
                    for rank in history:
                        for key, value in rank["counts"].items():
                            counts[key] = counts.get(key, 0) + value
                    self.assertEqual(summary["self_play"]["active_snapshot_counts"], counts)
                    self.assertEqual(sum(counts.values()), 8)
            self.assertTrue(ranks[0]["pool_writes"])
            self.assertEqual(ranks[1]["pool_writes"], [])
            if opponent == "self_play":
                manifest = json.loads((root / "resume/history_pool/manifest.json").read_text())
                self.assertEqual(manifest["resume_source"]["iteration_cutoff"], 0)
                for rank in ranks:
                    self.assertEqual(rank["resume_history"]["initial_pool"]["iterations"], [-1, 0])
                    self.assertEqual(rank["fresh_history"]["updates"], [[0], [0, 1]])
                    self.assertEqual(rank["stop_history"]["updates"], [[0]])
                for name in ("fresh", "resume", "stop"):
                    manifest = json.loads((root / name / "history_pool/manifest.json").read_text())
                    self.assertEqual({path.name for path in (root / name / "history_pool").iterdir()},
                                     {"manifest.json", *(entry["filename"] for entry in manifest["snapshots"])})


if __name__ == "__main__":
    unittest.main()
