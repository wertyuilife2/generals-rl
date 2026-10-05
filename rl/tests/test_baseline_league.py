"""Disabled-path isolation and real mixed-task Core/feedback/source contracts."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from generals import MapConfig, VecEnv
from generals.controllers import make_controllers
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import Action
from generals_rl.baseline_league import BaselineLeagueConfig, make_league_opponent
from generals_rl.evaluate_builtin import decode_move
from generals_rl.opponents import SelfPlayOpponent
from generals_rl.task import GeneralsTask, TaskConfig
from generals_rl.train import train
from test_self_play_task import TestPool


def assert_controller(test, actual, expected):
    test.assertEqual(actual.rng.bit_generator.state, expected.rng.bit_generator.state)
    for name in ("terrain", "structure", "owner", "army", "last_seen"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("path", "path_tick", "path_strategy", "_pending"):
        test.assertEqual(getattr(actual, name), getattr(expected, name))


class LeagueContracts(unittest.TestCase):
    def test_default_is_exact_original_opponent_and_trajectory(self):
        config = TaskConfig(num_envs=50, opponent="self_play", max_episode_steps=8, seed=872)
        rng_before = torch.get_rng_state().clone()
        self.assertIsNone(make_league_opponent(config, BaselineLeagueConfig()))
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        with GeneralsTask(config) as original, GeneralsTask(config,
                opponent=make_league_opponent(config, BaselineLeagueConfig(baseline_envs=0))) as disabled:
            for task in (original, disabled):
                self.assertIs(type(task.opponent), SelfPlayOpponent)
                task.opponent.attach(TestPool(), stochastic=True, seed=778)
            for _ in range(16):
                left = original.step(torch.zeros(50, dtype=torch.long))
                right = disabled.step(torch.zeros(50, dtype=torch.long))
                for field in ("reward", "terminated", "truncated"):
                    torch.testing.assert_close(getattr(left, field), getattr(right, field), rtol=0, atol=0)
                for name in left.observations:
                    torch.testing.assert_close(left.observations[name], right.observations[name], rtol=0, atol=0)
                for name in original.backend.state:
                    np.testing.assert_array_equal(original.backend.state[name], disabled.backend.state[name])
                torch.testing.assert_close(original.opponent.generator.get_state(),
                                           disabled.opponent.generator.get_state(), rtol=0, atol=0)

    def test_real_mixed_task_python_native_core_and_episode_source_counts(self):
        from importlib.util import find_spec
        modes = (False, True) if find_spec("generals_rl.league_native_candidates") else (False,)
        for native in modes:
            with self.subTest(native=native):
                config = TaskConfig(num_envs=50, opponent="self_play", max_episode_steps=8,
                                    productive_actions=True, seed=881)
                league_config = BaselineLeagueConfig(baseline_envs=1, seed=349, native_candidates=native)
                league = make_league_opponent(config, league_config)
                maps = [generate_map(MapConfig(9, 9, 2), 1500 + lane) for lane in range(50)]
                backend = VecEnv.from_states(maps, device="cpu", threads=1, auto_reset=False, episode_ticks=8)
                engines = [CoreEngine(state.copy()) for state in maps]
                episodes = np.zeros(50, dtype=np.int64)
                baseline_lane = league.bridge.lanes[0]
                totals = np.zeros((2, 3), dtype=np.int64)
                with GeneralsTask(config, backend=backend, opponent=league) as task:
                    pool = TestPool()
                    league.attach(pool, stochastic=False, seed=110)
                    def reference_ai():
                        seat = int(task.player_ids[baseline_lane])
                        seed = int(np.random.SeedSequence([349, baseline_lane, int(episodes[baseline_lane])])
                                   .generate_state(1, dtype=np.uint64)[0])
                        kinds = ["random", "random"]
                        kinds[1 - seat] = "aggressive"
                        return make_controllers(kinds, seed=seed)[1 - seat]
                    opponent = reference_ai()
                    recorded = []
                    hook = league.on_result
                    def recording_hook(result, submitted, batch, ids):
                        recorded.append((submitted.copy(), result.moves.copy(), result.done.copy(),
                                         batch.state["tick"].clone(), ids.clone(), batch.state["winner"].clone()))
                        return hook(result, submitted, batch, ids)
                    with patch.object(league, "on_result", recording_hook):
                        for tick in range(32):
                            seats = task.player_ids.clone()
                            learner = torch.zeros(50, dtype=torch.long)
                            # Productive, varied learner actions from the native legal mask.
                            for lane in range(50):
                                legal = torch.where(task.observations["action_mask"][lane])[0]
                                learner[lane] = legal[(tick + lane) % len(legal)]
                            expected_joint = np.zeros((50, 2), dtype=np.int64)
                            expected_joint[np.arange(50), seats.numpy()] = learner.numpy()
                            ai_action = opponent.act(observe(engines[baseline_lane].state,
                                                            1 - int(seats[baseline_lane]), "local"))
                            expected_joint[baseline_lane, 1 - int(seats[baseline_lane])] = ai_action.encode()
                            actual_controller = league.bridge.controllers[0]
                            task.step(learner)
                            submitted, moves, done, hook_ticks, hook_seats, winners = recorded[-1]
                            np.testing.assert_array_equal(submitted, expected_joint)
                            torch.testing.assert_close(hook_seats, seats, rtol=0, atol=0)
                            for lane in range(50):
                                reference = engines[lane].step([Action.decode(int(x)) for x in submitted[lane]])
                                for move in reference.moves:
                                    self.assertEqual(decode_move(moves[lane, move.player_id], move.player_id, move.action), move)
                                    if lane == baseline_lane and move.player_id == 1 - int(seats[lane]):
                                        opponent.on_result(move)
                                self.assertEqual(int(hook_ticks[lane]), engines[lane].state.tick)
                            assert_controller(self, actual_controller, opponent)
                            for lane in range(50):
                                if bool(done[lane].any()):
                                    source = int(lane != baseline_lane)
                                    totals[source] += [1, int(winners[lane] == seats[lane]), int(done[lane, 1] and not done[lane, 0])]
                                    episodes[lane] += 1
                                    engines[lane] = CoreEngine(maps[lane].copy())
                            if bool(done[baseline_lane].any()):
                                opponent = reference_ai()
                            for lane, state in enumerate(task.backend.to_states()):
                                for name in ("army", "owner", "structure", "alive", "general_pos"):
                                    np.testing.assert_array_equal(getattr(state, name), getattr(engines[lane].state, name))
                    np.testing.assert_array_equal(league.totals.numpy(), totals)
                    self.assertEqual(sum(len(x["single_obs"]) for x in pool.snapshots[0].model.inputs), 32 * 50)
                    metrics = league.take_log_metrics()
                    self.assertEqual(metrics["League/aggressive_decisions"], 32)
                    self.assertEqual(metrics["League/historical_decisions"], 32 * 49)
                    self.assertEqual(metrics["League/aggressive_episodes"], int(totals[0, 0]))
                    self.assertEqual(league.take_log_metrics()["League/aggressive_decisions"], 0)
                    self.assertEqual(league.describe()["source_totals"]["historical"]["decisions"], 32 * 49)

    def test_config_rejects_bad_quota_before_training_side_effects(self):
        for kwargs in ({"baseline_envs": True}, {"baseline_envs": 257},
                       {"seed": -1}, {"native_candidates": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                BaselineLeagueConfig(**kwargs)
        for config in (TaskConfig(num_envs=49, opponent="self_play"),
                       TaskConfig(num_envs=50, opponent="random"),
                       TaskConfig(num_envs=50, width=3, height=3, opponent="self_play")):
            with self.assertRaises(ValueError):
                make_league_opponent(config, BaselineLeagueConfig(baseline_envs=1))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            config = {"task": asdict(TaskConfig(num_envs=50, opponent="self_play")),
                      "runner": {"num_steps_per_env": 4},
                      "baseline_league": {"baseline_envs": 2}}
            with self.assertRaisesRegex(ValueError, "98%"):
                train(config, output, iterations=1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
