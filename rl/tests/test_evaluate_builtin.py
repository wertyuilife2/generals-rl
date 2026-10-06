"""Original-controller timing, balanced schedules and student information boundary."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from generals import VecEnv
from generals.controllers import AI_KINDS, make_controllers
from generals.core.engine import CoreEngine
from generals.core.state import Action, Direction, GameState, MapLayout, MoveResult
from generals_rl.evaluate_builtin import (
    GameSpec, _initial_knowledge, _local_inputs, aggregate_games, battle_batch,
    decode_move, evaluate_builtin, make_schedule, wilson_interval,
)
from generals_rl.config import task_config
from generals_rl.information_ppo import InformationSpatialPPOPolicy
from generals_rl.mdp import GameBatch, ObservationHistory, make_action_space, make_observation_encoder
from generals_rl.mdp.knowledge import KnowledgeState
from generals_rl.task import GeneralsTask, TaskConfig


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

    def action_probabilities(self, observations):
        actions = self.act_inference(observations)
        return torch.zeros_like(observations["action_mask"], dtype=torch.float32).scatter_(1, actions, 1)


class BuiltinEvaluationTests(unittest.TestCase):
    def test_information_checkpoint_report_and_selected_demo_replay_match(self):
        from tensordict import TensorDict

        root = Path(__file__).resolve().parents[2]
        tools = root / "rl/tools"
        module_spec = importlib.util.spec_from_file_location("memory_demo_tool", tools / "analyze_iteration.py")
        analyze = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(analyze)
        baseline = json.loads((root / "rl/configs/self_play_ppo.json").read_text())
        baseline["task"].update(backend="cpu", device="cpu", num_envs=2, map_pool_size=0,
                                opponent="wait", persistent_knowledge=True)
        baseline["runner"]["policy"].update(
            class_name="generals_rl.information_ppo.InformationSpatialPPOPolicy", knowledge_channels=11)
        original_probabilities = InformationSpatialPPOPolicy.action_probabilities

        def local_only(policy, obs, *args, **kwargs):
            self.assertEqual(set(obs.keys()), {"policy", "single_obs", "action_mask", "knowledge"})
            self.assertEqual(obs["knowledge"].shape[-1], 11 * 9 * 9)
            return original_probabilities(policy, obs, *args, **kwargs)

        # Exercise both critic configurations, which must stay absent from inference.
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for critic_information in (False, True):
                with self.subTest(critic_information=critic_information):
                    config = deepcopy(baseline)
                    config["runner"]["policy"]["critic_actor_information"] = critic_information
                    config["runner"]["obs_groups"] = {
                        "policy": ["policy", "knowledge"],
                        "critic": ["critic", "policy", "knowledge"] if critic_information else ["critic"],
                    }
                    run = Path(directory) / str(critic_information)
                    run.mkdir()
                    (run / "run_config.json").write_text(json.dumps(config))
                    checkpoint = run / "snapshot.pt"
                    torch.manual_seed(37)
                    with GeneralsTask(task_config(config["task"])) as task:
                        example = TensorDict(task.get_observations(), batch_size=[task.num_envs])
                        options = deepcopy(config["runner"]["policy"])
                        options.pop("class_name")
                        policy = InformationSpatialPPOPolicy(example, config["runner"]["obs_groups"],
                                                             task.action_space.size, **options)
                        torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
                    with patch.object(InformationSpatialPPOPolicy, "action_probabilities", local_only):
                        report = evaluate_builtin(checkpoint, maps_per_ai=2, seed=73, sampling_seed=419,
                                                  opponents=("aggressive",), batch_size=3, max_ticks=192)
                        repeated = evaluate_builtin(checkpoint, maps_per_ai=2, seed=73, sampling_seed=419,
                                                    opponents=("aggressive",), batch_size=1, max_ticks=192)
                        self.assertTrue(report["persistent_knowledge"])
                        self.assertTrue(report["stochastic"])
                        self.assertEqual(report["sampling_protocol"], "per_game_numpy_uniform_cdf_v1")
                        self.assertEqual(report["games"], repeated["games"])
                        with patch.object(sys, "path", [str(tools), *sys.path]):
                            demos = analyze.demos(checkpoint, report, run / "analysis", render=False)
                    self.assertTrue(demos["cases"])
                    for case in demos["cases"]:
                        self.assertTrue(case["native_core_replay_verified"])
                        original = next(game for game in report["games"] if game["game_id"] == case["game_id"])
                        self.assertEqual(case["match"], original)

    def test_local_memory_is_player_specific_private_and_fresh_per_game(self):
        config = TaskConfig(width=5, height=5, num_envs=2, history_length=2,
                            persistent_knowledge=True)
        ids = torch.tensor([0, 1])
        encoder = make_observation_encoder(config)
        with VecEnv.from_states([tiny_state(), tiny_state()], device="cpu", threads=1) as env:
            batch = GameBatch.from_backend(env, "cpu")
            history = ObservationHistory(config.history_length)
            history.reset(encoder.encode(batch, ids))
            memory = _initial_knowledge(batch, ids, config)
            local = _local_inputs(batch, ids, encoder, history, make_action_space(config), memory)
            self.assertEqual(set(local.keys()), {"policy", "single_obs", "action_mask", "knowledge"})
            self.assertEqual(tuple(local["knowledge"].shape), (2, 11 * 25))
            planes = local["knowledge"].reshape(2, 11, 25)
            torch.testing.assert_close(planes[:, 0].sum(1), torch.tensor([4., 4.]))
            self.assertEqual(float(planes[0, 3, 0]), 1)
            self.assertEqual(float(planes[1, 3, 24]), 1)
            self.assertEqual(int(planes[:, 4].count_nonzero()), 0)  # Enemy generals remain hidden.
            # Hidden terrain/owner/structure mutations cannot reach local memory.
            changed = GameBatch({key: value.clone() for key, value in batch.state.items()},
                                batch.stats.clone(), batch.width, batch.height)
            hidden_cells = torch.tensor([24, 0])
            changed.state["terrain"][torch.arange(2), hidden_cells] = 1
            changed.state["owner"][torch.arange(2), hidden_cells] = -2
            changed.state["structure"][torch.arange(2), hidden_cells] = 1
            changed.state["army"][torch.arange(2), hidden_cells] = 987654
            memory.update(changed, ids)
            torch.testing.assert_close(memory.encode(changed.state["tick"], ids).flatten(1),
                                       local["knowledge"], rtol=0, atol=0)
            # A new lane batch owns fresh memories even when its policy is reused.
            batch.state["owner"][:, 12] = ids
            memory.update(batch, ids)
            expanded = memory.encode(batch.state["tick"], ids).flatten(1)
            self.assertTrue(bool((expanded != local["knowledge"]).any()))
            memory.reset(torch.tensor([True, False]))
            reset = memory.encode(batch.state["tick"], ids).flatten(1)
            self.assertEqual(int(reset[0].count_nonzero()), 0)
            torch.testing.assert_close(reset[1], expanded[1], rtol=0, atol=0)
            fresh = _initial_knowledge(batch, ids, config)
            self.assertIsNot(fresh, memory)
            torch.testing.assert_close(fresh.encode(batch.state["tick"], ids).flatten(1), expanded)

    def test_memory_sampling_and_selected_replay_are_batch_invariant(self):
        class MemoryStudent:
            def action_probabilities(self, obs):
                assert set(obs.keys()) == {"policy", "single_obs", "action_mask", "knowledge"}
                assert "critic" not in obs
                knowledge = obs["knowledge"].reshape(len(obs), 11, 25)
                # Depend on accumulated facts, rather than merely accepting a key.
                wait = .35 + .3 * knowledge[:, 0].mean(1, keepdim=True)
                moves = obs["action_mask"].float()
                moves[:, 0] = 0
                probabilities = moves * (1 - wait) / moves.sum(1, keepdim=True)
                probabilities[:, :1] = wait
                return probabilities

        class Decisions:
            def on_start(self, specs, states):
                self.actions = {spec.game_id: [] for spec in specs}

            def on_tick(self, spec, actions, state, moves, done):
                self.actions[spec.game_id].append(actions[spec.learner_seat].encode())

        specs = [GameSpec(i, "random", i // 2, 5 + i // 2, 20 + i // 2, i % 2)
                 for i in range(24)]
        states = [tiny_state(adjacent=i % 3 == 0) for i in range(len(specs))]
        for state in states:
            state.army[state.owner >= 0] = 20
        config = TaskConfig(width=5, height=5, num_envs=len(specs), history_length=2,
                            persistent_knowledge=True)
        policy, recorder = MemoryStudent(), Decisions()
        rng = torch.get_rng_state()
        games, _ = battle_batch(specs, states, policy, config, max_ticks=8,
                                sampling_seed=491, recorder=recorder)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        selected = [13, 0, 9, 2, 18]
        replay = Decisions()
        reproduced, _ = battle_batch([specs[i] for i in selected], [states[i] for i in selected],
                                     policy, config, max_ticks=8, sampling_seed=491, recorder=replay)
        self.assertEqual(reproduced, [games[i] for i in selected])
        self.assertEqual(replay.actions, {i: recorder.actions[i] for i in selected})

    def test_memory_updates_once_per_completed_tick_and_skips_finished_lanes(self):
        class AttackStudent:
            def act_inference(self, obs):
                assert "knowledge" in obs and "critic" not in obs
                return torch.full((len(obs), 1), Action.move(0, Direction.RIGHT).encode(), dtype=torch.long)

        updates = []

        class ObservedMemory(KnowledgeState):
            def update(self, batch, ids, *, mask=None):
                updates.append((batch.state["tick"].clone(), None if mask is None else mask.clone()))
                return super().update(batch, ids, mask=mask)

        first, second = tiny_state(adjacent=True), tiny_state()
        first.army.flat[0] = second.army.flat[0] = 10
        first.army.flat[1] = 2
        specs = [GameSpec(0, "random", 0, 5, 6, 0), GameSpec(1, "random", 1, 7, 8, 0)]
        config = TaskConfig(width=5, height=5, num_envs=2, history_length=1,
                            persistent_knowledge=True)
        with patch("generals_rl.evaluate_builtin.KnowledgeState", ObservedMemory):
            games, _ = battle_batch(specs, [first, second], AttackStudent(), config,
                                    max_ticks=3, stochastic=False)
        self.assertEqual([game["ticks"] for game in games], [1, 3])
        self.assertEqual(len(updates), 4)  # Initial observation, then three joint ticks.
        self.assertIsNone(updates[0][1])
        for (_, actual), expected in zip(updates[1:], ([True, True], [False, True], [False, True])):
            self.assertEqual(actual.tolist(), expected)

    def test_default_samples_reproducibly_per_game_with_masks_and_independent_rng(self):
        class SampleOnly:
            def action_probabilities(self, obs):
                assert "critic" not in obs
                moves = obs["action_mask"].float()
                moves[:, 0] = 0
                moves *= .4 / moves.sum(-1, keepdim=True)
                moves[:, 0] = .6
                return moves

            def act_inference(self, obs):
                raise AssertionError("default evaluation used greedy")

        class Decisions:
            def on_start(self, specs, states):
                self.actions = {spec.game_id: [] for spec in specs}

            def on_tick(self, spec, actions, state, moves, done):
                self.actions[spec.game_id].append(actions[spec.learner_seat].encode())

        count = 256
        specs = [GameSpec(i, "random", 0, 5, 6, 0) for i in range(count)]
        states = [tiny_state(adjacent=i % 2 == 0) for i in range(count)]
        for state in states:
            state.army.flat[0] = 20
        config = TaskConfig(width=5, height=5, num_envs=count, history_length=2)
        recorder = Decisions()
        rng = torch.get_rng_state()
        numpy_rng = np.random.get_state()
        games, _ = battle_batch(specs, states, SampleOnly(), config, max_ticks=4,
                                sampling_seed=838, recorder=recorder)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        after = np.random.get_state()
        self.assertEqual(numpy_rng[0], after[0]); np.testing.assert_array_equal(numpy_rng[1], after[1])
        self.assertEqual(numpy_rng[2:], after[2:])
        waits = sum(actions[0] == 0 for actions in recorder.actions.values()) / count
        self.assertGreater(waits, .5); self.assertLess(waits, .7)
        self.assertTrue(all(game["learner_invalid_actions"] == 0 for game in games))
        # Reordered selected lanes reproduce full games after others end early.
        selected = [13, 0, 91, 2, 188]
        replay = Decisions()
        reproduced, _ = battle_batch([specs[i] for i in selected], [states[i] for i in selected],
                                     SampleOnly(), config, max_ticks=4, sampling_seed=838, recorder=replay)
        self.assertEqual(reproduced, [games[i] for i in selected])
        self.assertEqual(replay.actions, {i: recorder.actions[i] for i in selected})

    def test_cli_defaults_to_sampling_with_explicit_greedy_override(self):
        from generals_rl.evaluate_builtin import main
        for options, expected in (([], True), (["--stochastic"], True), (["--greedy"], False)):
            with self.subTest(options=options), patch("generals_rl.evaluate_builtin.evaluate_builtin", return_value={}) as evaluate:
                main(["dummy.pt", *options])
                self.assertEqual(evaluate.call_args.kwargs["stochastic"], expected)

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
