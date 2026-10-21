"""Actual evaluated joint actions replay independently of policy numerics."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from generals_baselines import AI_KINDS
from generals.core.recording import replay, state_digest
from generals.core.state import Action, Direction, GameState, MapLayout
from generals_rl.config import task_config
from generals_rl.evaluate_builtin import (GameSpec, SelectedGameTraces, battle_batch,
                                         evaluate_builtin)
from rsl_rl.utils import resolve_callable
from generals_rl.task import GeneralsTask, TaskConfig
from rl_test_utils import small_training_config


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("trace_demo_tool", TOOLS / "analyze_iteration.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)
from recordings import replay_evaluation_trace


def corridor_state(*, gap=False):
    terrain = np.ones((3, 3), dtype=np.uint8)
    terrain.flat[:3 if gap else 2] = 0
    positions = (0, 2 if gap else 1)
    structure = np.zeros((3, 3), dtype=np.uint8)
    owner = np.full((3, 3), -1, dtype=np.int16)
    army = np.zeros((3, 3), dtype=np.int64)
    for pid, cell in enumerate(positions):
        structure.flat[cell], owner.flat[cell], army.flat[cell] = 2, pid, 40 if pid == 0 else 1
    return GameState(MapLayout(terrain), structure, owner, army,
                     np.ones(2, dtype=bool), np.array(positions, dtype=np.int32))


class CorridorStudent:
    def action_probabilities(self, obs):
        assert "critic" not in obs.keys()
        actions = torch.zeros(len(obs), 1, dtype=torch.long)
        for source in (1, 0):
            action = Action.move(source, Direction.RIGHT).encode()
            actions[obs["action_mask"][:, action]] = action
        return torch.zeros_like(obs["action_mask"], dtype=torch.float32).scatter_(1, actions, 1)


def recorded_games(tmp_path, *, kinds=AI_KINDS, gap=False):
    checkpoint = tmp_path / "snapshot.pt"
    checkpoint.write_bytes(b"fixture policy identity; replay must not load this file")
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    specs = [GameSpec(index * 2 + seat, kind, 0, 5, 6, seat)
             for index, kind in enumerate(kinds) for seat in (0, 1)]
    states = [corridor_state(gap=gap) for _ in specs]
    collector = SelectedGameTraces(tmp_path / "selected", max_ticks=32, opponents=kinds,
        metadata={"checkpoint_sha256": digest, "max_ticks": 32, "device": "cpu",
                  "evaluation_seed": 1000042, "sampling_seed": 1000042,
                  "sampling_protocol": "per_game_numpy_uniform_cdf_v1", "stochastic": True})
    config = TaskConfig(width=3, height=3, history_length=2, productive_actions=True)
    rng = torch.get_rng_state().clone()
    games, _ = battle_batch(specs, states, CorridorStudent(), config, max_ticks=32, recorder=collector)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    collector.finish_batch(games)
    without_recording, _ = battle_batch(specs, states, CorridorStudent(), config, max_ticks=32)
    assert without_recording == games
    report = {"checkpoint_sha256": digest, "device": "cpu", "stochastic": True,
              "sampling_protocol": "per_game_numpy_uniform_cdf_v1", "games": games,
              "selected_game_traces": collector.describe()}
    return checkpoint, specs, games, collector, report


def test_eight_real_baseline_traces_replay_without_policy_or_controller_and_reject_corruption(tmp_path):
    checkpoint, specs, games, collector, report = recorded_games(tmp_path)
    entries = report["selected_game_traces"]["cases"]
    assert len(entries) == 8 and report["selected_game_traces"]["missing"] == []
    assert not collector.candidates
    assert len(list((tmp_path / "selected").iterdir())) == 8
    assert not list(tmp_path.rglob("*.frames.json.gz"))
    assert not collector.needs_batch(specs)
    collector.finish_batch(games)
    assert len(list((tmp_path / "selected").iterdir())) == 8
    for entry in entries:
        final = replay(entry["path"], verify=True)
        original = next(game for game in games if game["game_id"] == entry["game_id"])
        assert final.tick == original["ticks"] and final.winner_id == original["winner_seat"]
        assert sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    rng = torch.get_rng_state().clone()
    with patch("generals_rl.evaluate_builtin._load_student", side_effect=AssertionError("no reinference")), \
            patch("generals_rl.evaluate_builtin.battle_batch", side_effect=AssertionError("no new battles")), \
            patch("generals_rl.evaluate_builtin.BatchBaselineControllers", side_effect=AssertionError("no new controllers")):
        demos = analyze.demos(checkpoint, report, tmp_path / "demos", render=False)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert len(demos["cases"]) == 8 and demos["replay_mode"] == "recorded_joint_actions"
    for case in demos["cases"]:
        assert case["recorded_action_replay_verified"] and case["native_core_replay_verified"]
        assert case["match"] == next(game for game in games if game["game_id"] == case["game_id"])
    # Even when a corrupt file's hash is updated, the original native digest
    # must fail; a hash-only check is insufficient for precise replay.
    corrupt = deepcopy(entries[0])
    rows = [json.loads(line) for line in Path(corrupt["path"]).read_text().splitlines()]
    rows[-1]["digest"] = "0" * 64
    path = tmp_path / "corrupt.trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    corrupt.update(path=str(path), sha256=sha256(path.read_bytes()).hexdigest())
    match = next(game for game in games if game["game_id"] == corrupt["game_id"])
    with pytest.raises(ValueError, match="native state mismatch"):
        replay_evaluation_trace(corrupt, match, tmp_path / "corrupt_demo",
                                metadata={"checkpoint_sha256": report["checkpoint_sha256"]})


def test_multitick_native_and_core_digests_and_first_game_id_selection(tmp_path):
    checkpoint, specs, games, collector, report = recorded_games(tmp_path, kinds=("aggressive",), gap=True)
    assert all(game["ticks"] > 1 for game in games)
    for entry in report["selected_game_traces"]["cases"]:
        rows = [json.loads(line) for line in Path(entry["path"]).read_text().splitlines()]
        assert [row["tick"] for row in rows[1:]] == list(range(1, entry["ticks"] + 1))
        final = replay(entry["path"], verify=True)
        assert state_digest(final) == rows[-1]["digest"]
    demos = analyze.demos(checkpoint, report, tmp_path / "demos", render=False)
    assert len(demos["cases"]) == 2
    assert all(case["recorded_action_replay_verified"] for case in demos["cases"])
    # Out-of-order completion cannot displace the first scheduled game_id.
    collector2 = SelectedGameTraces(tmp_path / "sorted", max_ticks=32, opponents=("aggressive",),
        metadata=collector.metadata)
    expanded = [GameSpec(10, "aggressive", 0, 5, 6, 0), GameSpec(2, "aggressive", 0, 5, 6, 0)]
    completed, _ = battle_batch(expanded, [corridor_state() for _ in expanded], CorridorStudent(),
        TaskConfig(width=3, height=3, history_length=2, productive_actions=True), max_ticks=32,
        recorder=collector2)
    collector2.finish_batch(completed)
    assert collector2.describe()["cases"][0]["game_id"] == 2
    assert collector2.describe()["missing"] == [{"opponent": "aggressive", "outcome": "loss"}]


def test_real_checkpoint_trace_option_preserves_default_cpu_schedule_rng_and_missing_outcomes(tmp_path):
    config = small_training_config(task_overrides={"opponent": "wait"})
    options = deepcopy(config["runner"]["policy"])
    config["runner"]["policy"] = deepcopy(options)
    policy_type = resolve_callable(options.pop("class_name"))
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    with redirect_stdout(io.StringIO()), GeneralsTask(task_config(config["task"])) as task:
        example = TensorDict(task.get_observations(), batch_size=[task.num_envs])
        policy = policy_type(example, config["runner"]["obs_groups"], task.action_space.size, **options)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    with redirect_stdout(io.StringIO()):
        default = evaluate_builtin(checkpoint, maps_per_ai=1, seed=73, batch_size=3, max_ticks=1)
        rng = torch.get_rng_state().clone()
        recorded = evaluate_builtin(checkpoint, maps_per_ai=1, seed=73, batch_size=3, max_ticks=1,
                                    trace_directory=tmp_path / "selected")
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert "selected_game_traces" not in default
    for key in ("games", "by_ai", "by_seat", "overall", "sampling_seed", "sampling_protocol"):
        assert default[key] == recorded[key]
    traces = recorded["selected_game_traces"]
    assert traces["cases"] == [] and len(traces["missing"]) == 8
    assert list((tmp_path / "selected").iterdir()) == []
    demos = analyze.demos(checkpoint, recorded, tmp_path / "demos", render=False)
    assert demos["cases"] == [] and len(demos["missing"]) == 8
    # A GPU report without actual recorded actions must not fall back to CPU.
    legacy_gpu = dict(default, device="cuda:0")
    with pytest.raises(ValueError, match="require recorded actions"):
        analyze.demos(checkpoint, legacy_gpu, tmp_path / "legacy_gpu", render=False)
