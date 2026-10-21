"""Supported StudentPPO training, historical opponents and checkpoint continuation."""
import contextlib
from copy import deepcopy
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from generals_rl.self_play_training import prepare_self_play
from generals_rl.train import train
from rl_test_utils import assert_nested_equal, small_training_config


def self_play_config():
    cfg = small_training_config()
    cfg["self_play"] = {"pool_size": 3, "snapshot_interval": 1,
                        "stochastic": True, "seed": 10019, "sampling": "uniform"}
    cfg["runner"]["save_interval"] = 1
    return cfg


def directory_hashes(path):
    path = Path(path)
    return {str(value.relative_to(path)): sha256(value.read_bytes()).hexdigest()
            for value in path.rglob("*") if value.is_file()}


class SelfPlayTrainingTests(unittest.TestCase):
    def run_training(self, config, output, **budget):
        observed = {}
        records = []

        def capture_session(task, runner, *args, **kwargs):
            observed["runner"] = runner
            observed["starting_iteration"] = runner.current_learning_iteration
            observed["starting_learning_rate"] = runner.alg.learning_rate
            observed["starting_model"] = deepcopy(runner.alg.policy.state_dict())
            observed["starting_episode_lengths"] = task.episode_length_buf.clone()
            observed["starting_optimizers"] = {
                name: deepcopy(getattr(runner.alg, name).state_dict())
                for name in ("optimizer",)}
            session = prepare_self_play(task, runner, *args, **kwargs)
            observed["session"] = session
            original_log = runner.logger.log

            def log(**values):
                records.append({"iteration": values["it"], "losses": values["loss_dict"],
                                "rollout_steps": runner.cfg["num_steps_per_env"],
                                "self_play_pool_ids": list(session.pool.ids),
                                "self_play_active_opponents": task.opponent.snapshot_counts()})
                return original_log(**values)

            runner.logger.log = log
            return session

        with patch("generals_rl.train.prepare_self_play", capture_session), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = train(config, output, **budget)
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        return summary, checkpoint, records, observed

    def test_real_self_play_updates_frozen_history_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = self_play_config()
            original = deepcopy(config)
            summary, checkpoint, records, observed = self.run_training(
                config, output, iterations=4)
            self.assertEqual(config, original)
            self.assertEqual(json.loads((output / "run_config.json").read_text()), original)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            policy = observed["runner"].alg.policy
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in policy.parameters()))
            for prefix in ("actor.", "critic."):
                self.assertTrue(any(not torch.equal(value, observed["starting_model"][name])
                                    for name, value in policy.named_parameters() if name.startswith(prefix)))
            observed["runner"].eval_mode()
            with torch.inference_mode():
                local = observed["runner"].env.get_observations().exclude("critic")
                actions = policy.act_inference(local)
                self.assertTrue(local["action_mask"].gather(1, actions).all())
            self.assertEqual(summary["steps_per_environment"], 16)
            self.assertEqual(summary["environment_steps"], 64)
            self.assertGreater(summary["completed_episodes"], 0)
            self.assertEqual([r["self_play_pool_ids"] for r in records],
                             [[0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 4]])
            self.assertTrue(all(math.isfinite(v) for r in records for v in r["losses"].values()))
            evidence = summary["self_play"]
            self.assertEqual(evidence["new_publications"], 5)
            self.assertGreater(evidence["episode_assignments"], 4)
            self.assertEqual(sum(evidence["active_snapshot_counts"].values()), 4)
            observed_opponents = {key for record in records for key in record["self_play_active_opponents"]}
            self.assertGreater(len(observed_opponents), 1)
            self.assertEqual(checkpoint["iter"], 3)
            self.assertEqual({path.name for path in output.glob("*.pt")},
                             {"model_0.pt", "model_1.pt", "model_2.pt", "checkpoint.pt"})

            session = observed["session"]
            self.assertEqual(session.pool.ids, (0, 3, 4))
            self.assertEqual(len(list((output / "history_pool").glob("snapshot_*.pt"))), 3)
            learner_ids = {id(parameter) for parameter in session.policy.parameters()}
            optimizer_ids = {id(parameter) for name in ("optimizer",)
                             for group in getattr(observed["runner"].alg, name).param_groups
                             for parameter in group["params"]}
            self.assertLessEqual(optimizer_ids, learner_ids)
            for snapshot_id in session.pool.ids:
                model = session.pool.get(snapshot_id).model
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                self.assertTrue({id(p) for p in model.parameters()}.isdisjoint(optimizer_ids))
                self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
            assert_nested_equal(self, session.pool.get(4).model.state_dict(), checkpoint["model_state_dict"])
            seed_state = session.pool.get(0).model.state_dict()
            self.assertTrue(any(not torch.equal(value, checkpoint["model_state_dict"][name])
                                for name, value in seed_state.items()))
            for name in ("optimizer_state_dict",):
                self.assertEqual({int(state["step"]) for state in checkpoint[name]["state"].values()}, {16})

    def test_resume_copies_pool_restores_optimizer_and_respects_checkpoint_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            config = self_play_config()
            # Distinct checkpoint LRs expose accidental reuse of configuration
            # defaults; a fixed schedule keeps the resumed update unambiguous.
            config["runner"]["algorithm"]["schedule"] = "fixed"
            main_lr = 1e-5
            self.run_training(config, source, iterations=4)
            for filename in ("checkpoint.pt", "model_0.pt"):
                path = source / filename
                selected = torch.load(path, map_location="cpu", weights_only=False)
                for key, lr in (("optimizer_state_dict", main_lr),):
                    for group in selected[key]["param_groups"]:
                        group["lr"] = lr
                torch.save(selected, path)
            source_hashes = directory_hashes(source)
            for filename, expected_iteration in (("checkpoint.pt", 3), ("model_0.pt", 0)):
                with self.subTest(checkpoint=filename):
                    selected = torch.load(source / filename, map_location="cpu", weights_only=False)
                    target = Path(directory) / filename.removesuffix(".pt")
                    _, checkpoint, records, observed = self.run_training(
                        config, target, iterations=1, resume=str(source / filename))
                    self.assertEqual(observed["starting_iteration"], expected_iteration + 1)
                    self.assertEqual(checkpoint["iter"], expected_iteration + 1)
                    self.assertEqual(records[0]["iteration"], expected_iteration + 1)
                    assert_nested_equal(self, observed["starting_model"], selected["model_state_dict"])
                    for live, stored, lr in (("optimizer", "optimizer_state_dict", main_lr),):
                        assert_nested_equal(self, observed["starting_optimizers"][live], selected[stored])
                        before_steps = {int(value["step"]) for value in selected[stored]["state"].values()}
                        after_steps = {int(value["step"]) for value in checkpoint[stored]["state"].values()}
                        self.assertEqual(after_steps, {value + 4 for value in before_steps})
                        self.assertTrue(all(group["lr"] == lr for group in checkpoint[stored]["param_groups"]))
                    self.assertEqual(observed["starting_learning_rate"], main_lr)
                    logged_lr = EventAccumulator(str(target)).Reload().Scalars("Loss/learning_rate")
                    self.assertEqual([point.step for point in logged_lr], [expected_iteration + 1])
                    self.assertAlmostEqual(logged_lr[0].value, main_lr)
                    pool = json.loads((target / "history_pool/manifest.json").read_text())
                    self.assertEqual(pool["resume_source"]["iteration_cutoff"], expected_iteration)
                    self.assertEqual(pool["resume_source"]["directory"], str(source / "history_pool"))
                    self.assertEqual([s["iteration"] for s in pool["snapshots"]],
                                     [-1, expected_iteration, expected_iteration + 1])
                    self.assertTrue(pool["snapshots"][0]["seed"])
                    self.assertEqual(pool["snapshots"][1]["source"]["kind"], "learner_checkpoint")
                    self.assertEqual(pool["snapshots"][1]["source"]["sha256"],
                                     sha256((source / filename).read_bytes()).hexdigest())
                    self.assertTrue((observed["starting_episode_lengths"] == 0).all())
                    self.assertEqual(directory_hashes(source), source_hashes)

    def test_partial_final_rollout_publishes_final_policy_without_duplicate_snapshot(self):
        config = self_play_config()
        config["self_play"]["snapshot_interval"] = 2
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary, checkpoint, records, observed = self.run_training(config, output, steps_per_env=10)
            self.assertEqual([record["rollout_steps"] for record in records], [4, 4, 2])
            self.assertEqual([record["self_play_pool_ids"] for record in records], [[0], [0, 1], [0, 1]])
            self.assertEqual(summary["steps_per_environment"], 10)
            self.assertEqual(summary["self_play"]["new_publications"], 3)
            pool = json.loads((output / "history_pool/manifest.json").read_text())
            self.assertEqual([s["iteration"] for s in pool["snapshots"]], [-1, 1, 2])
            assert_nested_equal(self, observed["session"].pool.get(2).model.state_dict(), checkpoint["model_state_dict"])

    def test_invalid_self_play_settings_fail_before_creating_run_output(self):
        cases = ({"pool_size": 1}, {"pool_size": True}, {"snapshot_interval": 0},
                 {"seed": -1}, {"stochastic": "yes"}, {"sampling": "pfsp"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for settings in cases:
                config = self_play_config()
                config["self_play"].update(settings)
                with self.subTest(settings=settings), self.assertRaises((ValueError, TypeError)):
                    train(config, output, iterations=1)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
