"""Real complete CPU PPO work and synthetic distributed critical-path accounting."""
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import timedelta
import importlib.util
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict


spec = importlib.util.spec_from_file_location(
    "benchmark_training_tool", Path(__file__).resolve().parents[1] / "benchmark_training.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_counts_fresh_samples_once_and_uses_slowest_rank_per_complete_update():
    ranks = [{"iterations": [dict(collection_seconds=.03, gae_seconds=.02, update_seconds=.05, total_seconds=.10),
                              dict(collection_seconds=.10, gae_seconds=.02, update_seconds=.08, total_seconds=.20)]},
             {"iterations": [dict(collection_seconds=.01, gae_seconds=.02, update_seconds=.17, total_seconds=.20),
                              dict(collection_seconds=.03, gae_seconds=.02, update_seconds=.05, total_seconds=.10)]}]
    result = bench.aggregate_ranks(ranks, num_envs=8, steps=2, epochs=4, minibatches=2)
    assert result["global_num_envs"] == 16
    assert result["fresh_transitions_measured_global"] == 64
    assert result["critical_path_seconds"] == pytest.approx(.4)
    assert result["transitions_per_second"] == pytest.approx(160)
    assert result["max_cumulative_rank_seconds"] == pytest.approx(.3)
    assert result["ppo_sample_visits_per_update_global"] == 128
    assert result["optimizer_steps_per_update_per_rank"] == 8
    # Summing independently slow collection/update ranks is not the FPS denominator.
    assert sum(result["stage_seconds_sum_of_iteration_rank_maxima"][stage] for stage in bench.STAGES[:-1]) > .4


def test_storage_estimate_preserves_boolean_mask_and_matches_real_allocation():
    from rsl_rl.storage import RolloutStorageCTS
    obs = TensorDict({"policy": torch.zeros(8, 4), "critic": torch.zeros(8, 3),
                      "action_mask": torch.ones(8, 5, dtype=torch.bool)}, batch_size=[8])
    estimate = bench.estimate_storage(obs, 8, 2)
    assert estimate["observation_bytes_per_transition"] == 33
    assert estimate["observation_fields"]["action_mask"]["bytes_per_transition"] == 5
    storage = RolloutStorageCTS("rl", 8, 0, 2, obs, [1], action_dtype=torch.long,
                                store_action_mean=False, store_action_std=False)
    assert estimate["total_storage_bytes"] == bench.actual_storage_bytes(storage)["total_storage_bytes"]


@pytest.mark.parametrize("profile", (False, True))
def test_real_cpu_rollout_gae_four_epoch_updates_exclude_warmup_and_refuse_overwrite(tmp_path, monkeypatch, profile):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/v32.json").read_text())
    config["task"].update(backend="cpu", device="cpu", num_envs=8, max_episode_steps=4,
                          map_pool_size=0, opponent="current_self_play")
    config["runner"]["num_steps_per_env"] = 2
    config["runner"]["algorithm"].update(num_mini_batches=2, ppo_monitor=False,
                                          schedule="fixed", desired_kl=None)
    config["self_play"].update(pool_size=2, snapshot_interval=50)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "benchmark"
    command = ["--config", str(path), "--output", str(output), "--updates", "2", "--warmup", "1"]
    if profile:
        command.append("--profile")
    assert bench.main(command) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["performance"]["fresh_transitions_measured_global"] == 32
    assert report["performance"]["ppo_epochs"] == 4
    assert report["performance"]["optimizer_steps_per_update_per_rank"] == 8
    assert len(report["ranks"][0]["iterations"]) == 2
    assert len(report["ranks"][0]["warmup_iterations"]) == 1
    assert report["ranks"][0]["estimated_storage"]["total_storage_bytes"] == report["ranks"][0]["actual_storage"]["total_storage_bytes"]
    assert report["protocol"]["learner_and_opponent_action_nn_included"]
    estimate = json.loads((output / "storage_estimate.json").read_text())
    assert estimate["global_num_envs"] == 8
    assert estimate["total_storage_bytes"] == report["ranks"][0]["actual_storage"]["total_storage_bytes"]
    assert report["protocol"]["profile_updates_excluded"] == int(profile)
    if profile:
        assert report["profile"]["rollout_ticks_profiled"] == 1
        assert report["profile"]["logical_minibatches_profiled"] == 1
        assert set(report["profile"]["phases"]) == {"collection", "gae", "update"}
        for phase in report["profile"]["phases"].values():
            assert phase["top_operators"]
            assert json.loads(Path(phase["trace"]).read_text())["traceEvents"]
        # The profiler's extra full update contributes no measured transitions.
        assert len(report["ranks"][0]["iterations"]) == 2
    else:
        assert report["profile"] is None
    assert report["ranks"][0]["measured_memory"]["peak_allocated_bytes"] is None
    assert not list(output.glob("events*"))
    assert not (output / "checkpoint.pt").exists()
    original = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        bench.main(command)
    assert (output / "report.json").read_bytes() == original


def test_invalid_discount_is_rejected_before_output_or_environment(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/v32.json").read_text())
    config["task"]["reward"]["potential_gamma"] = .9
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="gamma"):
        bench.main(["--config", str(path), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("opponent", ("current_self_play", "self_play"))
def test_topology_and_explicit_global_environment_count(tmp_path, opponent):
    assert bench.topology({"WORLD_SIZE": "8", "RANK": "3", "LOCAL_RANK": "3"}) == {
        "world_size": 8, "rank": 3, "local_rank": 3}
    with pytest.raises(ValueError):
        bench.topology({"WORLD_SIZE": "8", "RANK": "8"})
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/v32.json").read_text())
    config["task"]["opponent"] = opponent
    path = tmp_path / "global.json"
    path.write_text(json.dumps(config))
    # Capture resolved settings without instantiating any GPU model or process group.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("WORLD_SIZE", "8")
        patch.setenv("RANK", "3")
        patch.setenv("LOCAL_RANK", "3")
        captured = []
        def capture(args):
            captured.append(bench.resolve_config(args, bench.topology()))
            return None
        patch.setattr(bench, "benchmark", capture)
        assert bench.main(["--config", str(path), "--output", str(tmp_path / "unused"),
                           "--global-num-envs", "16384", "--logical-batch-size", "4096"]) == 0
    assert captured[0]["task"]["num_envs"] == 2048
    assert captured[0]["task"]["device"] == "cuda:3"
    assert captured[0]["runner"]["algorithm"]["num_mini_batches"] == 16


def test_runner_borrows_existing_group_without_initializing_it_again(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    def forbidden(*args, **kwargs):
        pytest.fail("borrowed distributed group must not be initialized twice")
    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden)
    runner = bench.BenchmarkRunner.__new__(bench.BenchmarkRunner)
    runner.device = "cuda:3"
    runner._configure_multi_gpu()
    assert runner.is_distributed
    assert runner.multi_gpu_cfg == {"world_size": 8, "global_rank": 3, "local_rank": 3}
    runner.device = "cuda:2"
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        runner._configure_multi_gpu()
    runner.device = "cuda:3"
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 4)
    with pytest.raises(ValueError, match="disagrees"):
        runner._configure_multi_gpu()


def historical_config():
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/v32.json").read_text())
    config["task"].update(backend="cpu", device="cpu", num_envs=64, max_episode_steps=4,
                          map_pool_size=0, opponent="self_play")
    config["runner"]["num_steps_per_env"] = 2
    config["runner"]["algorithm"].update(num_mini_batches=2, ppo_monitor=False,
                                          schedule="fixed", desired_kl=None, physical_minibatch_size=16)
    config["self_play"].update(pool_size=8, snapshot_interval=50)
    return config


def frozen_actor_observer(model_ids):
    from generals_rl.information_ppo import InformationSpatialPPOPolicy
    original = InformationSpatialPPOPolicy.action_probabilities

    def probabilities(model, *args, **kwargs):
        if not any(parameter.requires_grad for parameter in model.parameters()):
            model_ids.add(id(model))
        return original(model, *args, **kwargs)

    return patch.object(InformationSpatialPPOPolicy, "action_probabilities", probabilities)


@pytest.mark.parametrize("fill", (False, True))
def test_historical_default_and_synthetic_fill_time_real_model_groups(tmp_path, monkeypatch, fill):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    config = historical_config()
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "historical"
    command = ["--config", str(path), "--output", str(output), "--updates", "2", "--warmup", "1"]
    if fill:
        command.append("--historical-pool-fill")
    actors = set()
    with frozen_actor_observer(actors):
        assert bench.main(command) == 0
    report = json.loads((output / "report.json").read_text())
    count = config["self_play"]["pool_size"] if fill else 1
    assert len(actors) == count  # Count actual frozen actor inference groups, not just pool metadata.
    assert report["ranks"][0]["historical_pool"]["snapshot_ids"] == list(range(count))
    assert report["protocol"]["historical_pool"]["synthetic_pool_fill"] == fill
    assert not report["protocol"]["historical_pool"]["trained_snapshots"]
    assert report["protocol"]["historical_pool"]["initialization_and_pool_fill_excluded_from_timing"]
    assert report["performance"]["fresh_transitions_measured_global"] == 64 * 2 * 2
    assert report["performance"]["optimizer_steps_per_update_per_rank"] == 8
    manifest = json.loads((output / "history_pool/manifest.json").read_text())
    assert len(manifest["snapshots"]) == count
    weights = []
    for metadata in manifest["snapshots"]:
        payload = torch.load(output / "history_pool" / metadata["filename"], weights_only=True)
        weights.append(payload["model_state_dict"])
        assert metadata["iteration"] == -1
        if metadata["snapshot_id"]:
            assert metadata["source"]["kind"] == "benchmark_synthetic_initial_policy_copy"
            assert not metadata["source"]["trained_snapshot"]
    for state in weights[1:]:
        for key, value in state.items():
            torch.testing.assert_close(value, weights[0][key], rtol=0, atol=0)


def historical_benchmark_worker(rank, directory, borrowed, fill):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    root = Path(directory)
    original_init = dist.init_process_group
    init_calls, writes, actors, observed = [], [], set(), {}

    def initialize(*args, **kwargs):
        init_calls.append(1)
        kwargs.update(init_method=f"file://{root}/group", timeout=timedelta(seconds=30))
        return original_init(*args, **kwargs)

    if borrowed:
        original_init("gloo", init_method=f"file://{root}/group", rank=rank, world_size=2,
                      timeout=timedelta(seconds=30))
    from generals_rl.self_play import HistoricalPolicyPool
    original_manifest = HistoricalPolicyPool._write_manifest
    original_runner = bench.BenchmarkRunner

    def write_manifest(pool):
        assert rank == 0, "reader wrote a benchmark historical manifest"
        writes.append(list(pool.ids))
        return original_manifest(pool)

    def runner(*args, **kwargs):
        observed["runner"] = original_runner(*args, **kwargs)
        return observed["runner"]

    try:
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": str(rank), "LOCAL_RANK": str(rank)}), \
                patch.object(dist, "init_process_group", initialize), \
                patch.object(HistoricalPolicyPool, "_write_manifest", write_manifest), \
                patch.object(bench, "BenchmarkRunner", runner), frozen_actor_observer(actors), \
                redirect_stdout(io.StringIO()):
            command = ["--config", str(root / "config.json"), "--output", str(root / "benchmark"),
                       "--updates", "2", "--warmup", "1"]
            if fill:
                command.append("--historical-pool-fill")
            assert bench.main(command) == 0
        assert dist.is_initialized() == borrowed
        assert len(init_calls) == (0 if borrowed else 1)
        assert not torch.cuda.is_initialized()
        policy = observed["runner"].alg.policy
        weights = torch.cat([parameter.detach().flatten().clone() for parameter in policy.parameters()])
        torch.save(weights, root / f"weights_{rank}.pt")
        (root / f"rank_{rank}.json").write_text(json.dumps({"actor_models": len(actors), "writes": writes}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("borrowed,fill", ((False, True), (True, True), (True, False)))
def test_real_distributed_historical_complete_updates_share_pool_and_preserve_group_ownership(tmp_path, borrowed, fill):
    (tmp_path / "config.json").write_text(json.dumps(historical_config()))
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        mp.spawn(historical_benchmark_worker, args=(str(tmp_path), borrowed, fill), nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["writes"]
    assert ranks[1]["writes"] == []
    count = 8 if fill else 1
    assert [rank["actor_models"] for rank in ranks] == [count, count]
    weights = [torch.load(tmp_path / f"weights_{rank}.pt", weights_only=True) for rank in range(2)]
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    report = json.loads((tmp_path / "benchmark/report.json").read_text())
    assert report["performance"]["fresh_transitions_measured_global"] == 2 * 64 * 2 * 2
    assert report["performance"]["ppo_epochs"] == 4
    assert report["protocol"]["historical_pool"]["synthetic_pool_fill"] == fill
    assert not report["protocol"]["historical_pool"]["trained_snapshots"]
    assert len(list((tmp_path / "benchmark/history_pool").glob("snapshot_*.pt"))) == count
    assert [rank["historical_pool"]["read_only"] for rank in report["ranks"]] == [False, True]
    assert all(rank["historical_pool"]["snapshot_ids"] == list(range(count)) for rank in report["ranks"])
    assert report["ranks"][0]["seed"] != report["ranks"][1]["seed"]
    assert report["ranks"][0]["historical_pool"]["opponent_seed"] != report["ranks"][1]["historical_pool"]["opponent_seed"]


def test_historical_fill_rejects_current_mode_before_creating_output(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(historical_config()))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="historical_pool_fill"):
        bench.main(["--config", str(path), "--output", str(output), "--opponent", "current_self_play",
                    "--historical-pool-fill"])
    assert not output.exists()
