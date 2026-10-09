"""Original-controller timing, balanced schedules and student information boundary."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from generals import VecEnv
from generals_baselines import AI_KINDS, make_controllers
from generals.core.engine import CoreEngine
from generals.core.state import Action, Direction, GameState, MapLayout, MoveResult
from generals_rl.evaluate_builtin import (
    GameSpec, aggregate_games, battle_batch, decode_move, make_schedule, wilson_interval,
)
from generals_rl.task import TaskConfig


def tiny_state(*, adjacent=False):
    shape = (5, 5)
    positions = (0, 1) if adjacent else (0, 24)
    structure = np.zeros(shape, dtype=np.uint8)
    owner = np.full(shape, -1, dtype=np.int16)
    army = np.zeros(shape, dtype=np.int64)
    for pid, cell in enumerate(positions):
        structure.flat[cell], owner.flat[cell], army.flat[cell] = 2, pid, 1
    return GameState(MapLayout(np.zeros(shape, dtype=np.uint8)), structure, owner, army,
                     np.ones(2, dtype=bool), np.array(positions, dtype=np.int32))


class WaitStudent:
    def __init__(self):
        self.inputs = []

    def act_inference(self, observations):
        assert "critic" not in observations.keys()
        assert set(observations.keys()) == {"policy", "single_obs", "action_mask"}
        self.inputs.append(observations.clone())
        assert observations["action_mask"][:, 0].all()
        return torch.zeros(len(observations), 1, dtype=torch.long)


class BuiltinEvaluationTests(unittest.TestCase):
    def test_schedule_pairs_maps_and_seats_for_all_four_original_kinds(self):
        schedule = make_schedule(maps_per_ai=3, seed=738)
        self.assertEqual(schedule, make_schedule(maps_per_ai=3, seed=738))
        self.assertEqual(len(schedule), 24)
        self.assertEqual([spec.game_id for spec in schedule], list(range(24)))
        for index in range(3):
            same_map = [spec for spec in schedule if spec.map_index == index]
            self.assertEqual(len({spec.map_seed for spec in same_map}), 1)
            self.assertEqual(len({spec.controller_seed for spec in same_map}), 1)
            for kind in AI_KINDS:
                self.assertEqual(sorted(spec.learner_seat for spec in same_map if spec.opponent == kind), [0, 1])
        with self.assertRaises(ValueError):
            make_schedule(opponents=("expansion", "expansion"))

    def test_native_move_adapter_preserves_original_controller_callback_contract(self):
        state = tiny_state(adjacent=True)
        state.army.flat[0], state.army.flat[1] = 10, 2
        actions = [Action.move(0, Direction.RIGHT), Action.move(1, Direction.LEFT)]
        reference = CoreEngine(state.copy()).step(actions)
        with VecEnv.from_states([state], device="cpu", threads=1) as env:
            native = env.step(np.array([[action.encode() for action in actions]], dtype=np.int64))
            move = decode_move(native.moves[0, 0], 0, actions[0])
            self.assertIsInstance(move, MoveResult)
            self.assertEqual(move, reference.moves[0])
            self.assertIsNone(decode_move(native.moves[0, 1], 1, actions[1]))

    def test_original_controllers_receive_pre_growth_local_obs_and_typed_results(self):
        factory_calls, callbacks = [], []

        class RecordingController:
            def __init__(self, original):
                self.original = original
                self.ticks = []

            def act(self, obs):
                self.ticks.append(obs.tick)
                self.assert_local(obs)
                return self.original.act(obs)

            @staticmethod
            def assert_local(obs):
                assert obs.visibility == "local"
                assert obs.stats.cities[1 - obs.player_id] == -1

            def on_result(self, result):
                callbacks.append(result)
                self.original.on_result(result)

        def recording_factory(kinds, seed):
            originals = make_controllers(kinds, seed=seed)
            wrappers = [RecordingController(controller) for controller in originals]
            factory_calls.append((kinds, originals, wrappers))
            return wrappers

        specs = [GameSpec(0, "aggressive", 0, 5, 6, 0), GameSpec(1, "defensive", 0, 5, 6, 1)]
        config = TaskConfig(width=5, height=5, num_envs=2, max_episode_steps=64, history_length=2)
        policy = WaitStudent()
        with patch("generals_rl.evaluate_builtin.make_controllers", side_effect=recording_factory):
            games, timing = battle_batch(specs, [tiny_state(), tiny_state()], policy, config, max_ticks=2)
        self.assertEqual([game["game_id"] for game in games], [0, 1])
        self.assertEqual([game["outcome"] for game in games], ["draw", "draw"])
        self.assertEqual([game["ticks"] for game in games], [2, 2])
        self.assertEqual(factory_calls[0][1][1].__class__.__name__, "AggressiveAI")
        self.assertEqual(factory_calls[1][1][0].__class__.__name__, "DefensiveAI")
        self.assertEqual(factory_calls[0][2][1].ticks, [0, 1])
        self.assertEqual(factory_calls[1][2][0].ticks, [0, 1])
        self.assertEqual(len(callbacks), 4)
        self.assertTrue(all(isinstance(result, MoveResult) and result.valid and result.reason == "WAIT" for result in callbacks))
        self.assertEqual(timing["student_decisions"], 4)
        self.assertEqual(timing["environment_ticks"], 4)
        # Last two features encode absolute seat; the preceding three are clocks.
        # Evaluation cap=2 must not replace the training denominator=64.
        torch.testing.assert_close(policy.inputs[1]["single_obs"][:, -5], torch.full((2,), 1 / 64))
        self.assertEqual(tuple(policy.inputs[0]["policy"].shape), (2, policy.inputs[0]["single_obs"].shape[1] * 2))
        self.assertTrue(all(game["truncated"] and not game["terminated"] for game in games))

    def test_every_scheduled_game_finishes_even_when_other_lanes_end_earlier(self):
        class AttackStudent(WaitStudent):
            def act_inference(self, observations):
                super().act_inference(observations)
                # On an adjacent initial map this captures player 1 immediately;
                # on the separated map the same legal move only expands territory.
                return torch.full((len(observations), 1), Action.move(0, Direction.RIGHT).encode(), dtype=torch.long)

        first, second = tiny_state(adjacent=True), tiny_state()
        first.army.flat[0] = second.army.flat[0] = 10
        first.army.flat[1] = 2
        specs = [GameSpec(0, "random", 0, 5, 6, 0), GameSpec(1, "random", 1, 7, 8, 0)]
        config = TaskConfig(width=5, height=5, num_envs=2, max_episode_steps=64, history_length=1)
        games, _ = battle_batch(specs, [first, second], AttackStudent(), config, max_ticks=3)
        self.assertEqual(len(games), len(specs))
        self.assertEqual(games[0]["outcome"], "win")
        self.assertEqual(games[0]["ticks"], 1)
        self.assertEqual(games[1]["ticks"], 3)
        self.assertEqual(games[1]["outcome"], "draw")

    def test_aggregate_keeps_draws_separate_and_reports_undefined_decisive_rate(self):
        games = [{"outcome": outcome, "ticks": ticks} for outcome, ticks in (("win", 2), ("loss", 4), ("draw", 6))]
        result = aggregate_games(games)
        self.assertEqual((result["wins"], result["losses"], result["draws"]), (1, 1, 1))
        self.assertEqual(result["win_rate"], 1 / 3)
        self.assertEqual(result["decisive_win_rate"], .5)
        self.assertEqual(result["draw_rate"], 1 / 3)
        self.assertEqual(result["mean_ticks"], 4)
        draws = aggregate_games([{"outcome": "draw", "ticks": 2}])
        self.assertIsNone(draws["decisive_win_rate"])
        self.assertIsNone(draws["decisive_win_rate_wilson95"])
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)


if __name__ == "__main__":
    unittest.main()
