"""Model-duel fairness, information boundaries, and real checkpoint loading."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from generals import VecEnv
from generals.core.state import GameState, MapLayout
from generals_rl.evaluate_duel import (
    DuelSpec, _features, duel_batch, evaluate_duel, make_duel_schedule, summarize_duels,
)
from generals_rl.mdp import GameBatch
from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy
from generals_rl.task import TaskConfig


def state(*, adjacent=False, size=5):
    shape = (size, size)
    positions = (0, 1) if adjacent else (0, size * size - 1)
    structure = np.zeros(shape, dtype=np.uint8)
    owner = np.full(shape, -1, dtype=np.int16)
    army = np.zeros(shape, dtype=np.int64)
    for pid, cell in enumerate(positions):
        structure.flat[cell], owner.flat[cell], army.flat[cell] = 2, pid, 20 if pid == 0 else 1
    return GameState(MapLayout(np.zeros(shape, dtype=np.uint8)), structure, owner, army,
                     np.ones(2, dtype=bool), np.array(positions, dtype=np.int32))


class SamplePolicy:
    def __init__(self, length=4, *, knowledge=True):
        self.length, self.knowledge = length, knowledge
        self.inputs = []

    def action_probabilities(self, obs):
        assert set(obs.keys()) == {"policy", "single_obs", "action_mask"} | ({"knowledge"} if self.knowledge else set())
        assert obs["policy"].shape[-1] == self.length * obs["single_obs"].shape[-1]
        self.inputs.append(obs.clone())
        probabilities = obs["action_mask"].float()
        # Memory actually affects sampling, exposing cross-seat/reset errors.
        if self.knowledge:
            probabilities[:, 0] += obs["knowledge"].sum(1) * .01
        return probabilities / probabilities.sum(1, keepdim=True)


class WaitPolicy:
    def act_inference(self, obs):
        return torch.zeros(len(obs), 1, dtype=torch.long)


class AttackPolicy:
    def act_inference(self, obs):
        mask = obs["action_mask"].clone()
        mask[:, 0] = False
        return mask.long().argmax(1, keepdim=True)


class ModelDuelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_schedule_balances_every_map_and_rejects_invalid_arguments(self):
        specs = make_duel_schedule(maps=3, seed=73)
        self.assertEqual(specs, make_duel_schedule(maps=3, seed=73))
        for index in range(3):
            pair = specs[2 * index:2 * index + 2]
            self.assertEqual({spec.candidate_seat for spec in pair}, {0, 1})
            self.assertEqual(pair[0].map_seed, pair[1].map_seed)
        for kwargs in ({"maps": 0}, {"maps": True}, {"seed": -1}, {"seed": True}):
            with self.assertRaises(ValueError):
                make_duel_schedule(**kwargs)

    def test_independent_history_knowledge_and_hidden_state_noninterference(self):
        task = TaskConfig(width=5, height=5, persistent_knowledge=True)
        with VecEnv.from_states([state()], device="cpu", threads=1) as env:
            batch = GameBatch.from_backend(env, "cpu")
            left = _features(batch, torch.tensor([0]), task, capture=False)
            right = _features(batch, torch.tensor([1]), replace(task, history_length=8), capture=False)
            self.assertEqual(left.observations["policy"].shape[1], 4 * (12 * 25 + 13))
            self.assertEqual(right.observations["policy"].shape[1], 8 * (12 * 25 + 13))
            self.assertNotEqual(left.history.frames.data_ptr(), right.history.frames.data_ptr())
            self.assertNotEqual(left.knowledge.owner.data_ptr(), right.knowledge.owner.data_ptr())
            initial = left.observations.clone()
            changed = GameBatch({key: value.clone() for key, value in batch.state.items()},
                                batch.stats.clone(), batch.width, batch.height)
            # Cell24 lies outside seat0's legal visibility; public totals stay fixed.
            for name, value in (("army", 987654), ("terrain", 1), ("owner", -2), ("structure", 1)):
                changed.state[name][:, 24] = value
            left.advance(changed, torch.tensor([True]))
            for name in initial.keys():
                torch.testing.assert_close(left.observations[name], initial[name], rtol=0, atol=0)
            left.knowledge.reset()
            self.assertGreater(int(right.knowledge.encode(batch.state["tick"], right.ids).count_nonzero()), 0)
            fresh = _features(batch, torch.tensor([0]), task, capture=False)
            torch.testing.assert_close(fresh.observations["knowledge"], initial["knowledge"], rtol=0, atol=0)

    def test_stochastic_self_duel_is_exactly_seat_balanced_and_batch_invariant(self):
        specs = make_duel_schedule(maps=4, seed=73)
        states = [state(adjacent=spec.map_index % 2 == 0) for spec in specs]
        task = TaskConfig(width=5, height=5, persistent_knowledge=True)
        rng_state = torch.get_rng_state().clone()
        trace = []
        games, _ = duel_batch(specs, states, SamplePolicy(), SamplePolicy(), task, task,
                              max_ticks=40, sampling_seed=419,
                              recorder=lambda spec, actions, tick: trace.append((spec.game_id, actions.tolist(), tick)))
        separate, separate_trace = [], []
        for spec, initial in zip(specs, states, strict=True):
            completed, _ = duel_batch([spec], [initial], SamplePolicy(), SamplePolicy(), task, task,
                                      max_ticks=40, sampling_seed=419,
                                      recorder=lambda spec, actions, tick: separate_trace.append((spec.game_id, actions.tolist(), tick)))
            separate.extend(completed)
        self.assertEqual(games, separate)
        self.assertEqual(sorted(trace), sorted(separate_trace))
        for index in range(4):
            a, b = games[2 * index:2 * index + 2]
            self.assertEqual(a["ticks"], b["ticks"])
            self.assertEqual(a["winner_seat"], b["winner_seat"])
            self.assertEqual(a["candidate_sampling_seed"], b["reference_sampling_seed"])
        summary = summarize_duels(games)
        self.assertEqual(summary["wins"], summary["losses"])
        self.assertEqual(summary["score"], .5)
        self.assertEqual(summary["score_paired_map_bootstrap95"], [.5, .5])
        torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)

    def test_heterogeneous_history_and_knowledge_settings(self):
        task = TaskConfig(width=5, height=5, persistent_knowledge=True)
        reference_task = replace(task, history_length=8, persistent_knowledge=False)
        games, _ = duel_batch(make_duel_schedule(maps=1), [state(), state()],
                              SamplePolicy(), SamplePolicy(8, knowledge=False), task, reference_task,
                              max_ticks=5)
        self.assertEqual(len(games), 2)
        self.assertTrue(all(game["truncated"] for game in games))

    def test_terminal_wins_at_tick_limit_and_finished_lanes_receive_no_more_decisions(self):
        task = TaskConfig(width=5, height=5)
        specs = [DuelSpec(0, 0, 73, 0), DuelSpec(1, 1, 74, 0)]
        traces = []
        games, _ = duel_batch(specs, [state(adjacent=True), state()], AttackPolicy(), WaitPolicy(),
                              task, task, max_ticks=3, stochastic=False,
                              recorder=lambda spec, actions, tick: traces.append((spec.game_id, tick)))
        self.assertEqual((games[0]["outcome"], games[0]["ticks"]), ("win", 1))
        self.assertEqual((games[1]["outcome"], games[1]["ticks"]), ("draw", 3))
        self.assertEqual([tick for game_id, tick in traces if game_id == 0], [1])
        at_limit, _ = duel_batch(specs[:1], [state(adjacent=True)], AttackPolicy(), WaitPolicy(),
                                 task, task, max_ticks=1, stochastic=False)
        self.assertEqual(at_limit[0]["outcome"], "win")

    def test_invalid_actions_raise_before_simulation(self):
        class BadPolicy:
            def act_inference(self, obs):
                return torch.full((len(obs), 1), 999999, dtype=torch.long)
        task = TaskConfig(width=5, height=5)
        traces = []
        with self.assertRaisesRegex(ValueError, "out of range"):
            duel_batch(make_duel_schedule(maps=1), [state(), state()], BadPolicy(), WaitPolicy(),
                        task, task, stochastic=False, max_ticks=2,
                        recorder=lambda *args: traces.append(args))
        self.assertEqual(traces, [])

    def test_pretick_legal_move_can_be_natively_rejected_after_earlier_seat_captures_source(self):
        initial = state()
        initial.owner.flat[1] = 1  # A normal seat1 tile beside seat0's strong general.
        initial.army.flat[1] = 1
        class FixedPolicy:
            def __init__(self, encoded):
                self.encoded = encoded
            def act_inference(self, obs):
                assert obs["action_mask"][:, self.encoded].all()
                return torch.full((len(obs), 1), self.encoded, dtype=torch.long)
        # Both RIGHT moves are legal at their shared decision boundary.
        # Seat0 captures cell1; seat1's move from cell1 is then NOT_OWNER.
        joint = np.array([[3, 11]], dtype=np.int64)
        with VecEnv.from_states([initial], device="cpu", threads=1, episode_ticks=1) as env:
            result = env.step(joint)
            self.assertEqual(int(result.moves[0, 1, 1]), 0)
            self.assertEqual(int(result.moves[0, 1, 8]), 4)  # Native NOT_OWNER.
        task = TaskConfig(width=5, height=5)
        games, _ = duel_batch([DuelSpec(0, 0, 73, 0)], [initial], FixedPolicy(3), FixedPolicy(11),
                              task, task, stochastic=False, max_ticks=1)
        self.assertEqual(games[0]["candidate_invalid_actions"], 0)
        self.assertEqual(games[0]["reference_invalid_actions"], 1)
        self.assertEqual(games[0]["outcome"], "draw")

    def test_real_vit_h4_resnet_h4_loading_provenance_and_no_privileged_actor_inputs(self):
        base = {
            "task": {"width": 9, "height": 9, "num_envs": 1, "max_episode_steps": 32,
                     "history_length": 4, "opponent": "wait", "persistent_knowledge": True},
            "runner": {
                "obs_groups": {"policy": ["policy", "knowledge"], "critic": ["critic", "policy", "knowledge"]},
                "policy": {"class_name": "generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy",
                           "backbone": "vit", "model_size": "1m", "head_channels": 16,
                           "history_length": 4, "action_distribution": "categorical"}},
        }
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            paths = []
            for role in range(2):
                config = deepcopy(base)
                config["runner"]["policy"]["backbone"] = "vit" if role == 0 else "resnet"
                run = Path(directory) / str(role)
                run.mkdir()
                (run / "run_config.json").write_text(json.dumps(config))
                with VecEnv.from_states([state(size=9)], device="cpu", threads=1) as env:
                    from generals_rl.config import task_config
                    features = _features(GameBatch.from_backend(env, "cpu"), torch.tensor([0]),
                                         task_config(config["task"]), capture=False)
                    observations = features.observations.clone()
                    observations["critic"] = torch.zeros_like(observations["single_obs"])
                    kwargs = dict(config["runner"]["policy"])
                    kwargs.pop("class_name")
                    model = ScaledInformationSpatialPPOPolicy(observations, config["runner"]["obs_groups"], 649, **kwargs)
                    paths.append(run / "snapshot.pt")
                    torch.save({"model_state_dict": model.state_dict()}, paths[-1])
            forward = ScaledInformationSpatialPPOPolicy.action_probabilities
            seen = []
            def local_only(policy, obs, *args, **kwargs):
                self.assertEqual(set(obs.keys()), {"policy", "single_obs", "action_mask", "knowledge"})
                seen.append(policy.history_length)
                return forward(policy, obs, *args, **kwargs)
            output = Path(directory) / "duel.json"
            caller_rng = torch.get_rng_state().clone()
            with patch.object(ScaledInformationSpatialPPOPolicy, "action_probabilities", local_only):
                report = evaluate_duel(*paths, maps=1, max_ticks=2, output=output)
            torch.testing.assert_close(torch.get_rng_state(), caller_rng, rtol=0, atol=0)
            self.assertEqual(set(seen), {4})
            self.assertEqual(report["completed_games"], 2)
            self.assertEqual(report["overall"]["score"], .5)
            self.assertEqual(report, json.loads(output.read_text()))
            self.assertNotEqual(report["candidate"]["checkpoint_sha256"], report["reference"]["checkpoint_sha256"])
            self.assertEqual(len(report["candidate"]["config_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                evaluate_duel(*paths, maps=1, max_ticks=2, output=output)
            original_batch = duel_batch
            def mutate_config(*args, **kwargs):
                result = original_batch(*args, **kwargs)
                path = paths[0].with_name("run_config.json")
                changed = json.loads(path.read_text())
                changed["changed_during_duel"] = True
                path.write_text(json.dumps(changed))
                return result
            with patch("generals_rl.evaluate_duel.duel_batch", mutate_config):
                with self.assertRaisesRegex(RuntimeError, "changed during the duel"):
                    evaluate_duel(*paths, maps=1, max_ticks=2)


if __name__ == "__main__":
    unittest.main()
