"""Persistent memory at task, opponent, terminal and snapshot boundaries."""
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

from generals.core.state import Action, Direction
from generals.vector import VecEnv
from generals_rl.policy.knowledge import KnowledgeState
from generals_rl.self_play_training import prepare_self_play
from generals_rl.task import GeneralsTask, TaskConfig
from test_mdp import capture_state, example_state, hidden_world_variant


torch.set_num_threads(1)


class KnowledgePolicy(torch.nn.Module):
    """Parameter-bearing probe that receives only legal opponent inputs."""
    def __init__(self, obs=None, obs_groups=None, num_actions=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def action_probabilities(self, observations, *, is_teacher=False):
        assert not is_teacher
        assert set(observations.keys()) == {"policy", "single_obs", "action_mask", "knowledge"}
        self.inputs.append({key: value.clone() for key, value in observations.items()})
        weights = observations["action_mask"].float()
        return weights / weights.sum(1, keepdim=True)

    def act_inference(self, observations):
        self.action_probabilities(observations)
        return torch.zeros((len(observations), 1), dtype=torch.long)


@dataclass
class ProbeSnapshot:
    snapshot_id: int
    model: KnowledgePolicy


class KnowledgePool:
    ids = (0,)

    def __init__(self):
        self.snapshot = ProbeSnapshot(0, KnowledgePolicy())

    def sample(self, count, *, generator):
        return [self.snapshot] * count


def memory_planes(task, value):
    return value.reshape(task.num_envs, len(task.knowledge.channels), -1)


class KnowledgeTaskTests(unittest.TestCase):
    def task(self, states, **settings):
        options = dict(random_maps=False, map_pool_size=0, width=states[0].layout.width, height=states[0].layout.height,
                       players=states[0].player_count, num_envs=len(states),
                       opponent="wait", persistent_knowledge=True, history_length=3,
                       max_episode_steps=16, rotate_seats=False)
        options.update(settings)
        config = TaskConfig(**options)
        backend = VecEnv.from_states(states, episode_ticks=config.max_episode_steps,
                                     auto_reset=False, threads=1)
        task = GeneralsTask(config, backend=backend)
        self.addCleanup(task.close)
        return task

    def test_disabled_mode_has_no_memory_allocations_or_behavior_change(self):
        with self.assertRaises(TypeError):
            TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, persistent_knowledge=1)
        self.assertFalse(TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0).persistent_knowledge)
        initial_rng = torch.get_rng_state().clone()
        baseline = self.task([example_state(), example_state()], persistent_knowledge=False,
                             opponent="random")
        repeated = self.task([example_state(), example_state()], persistent_knowledge=False,
                             opponent="random")
        self.assertIsNone(baseline.knowledge)
        self.assertNotIn("knowledge", baseline.get_observations())
        torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)
        for _ in range(3):
            first = baseline.step(torch.zeros(2, dtype=torch.long))
            second = repeated.step(torch.zeros(2, dtype=torch.long))
            self.assertFalse(any(key.startswith("Knowledge/") for key in first.metrics))
            for key in first.observations:
                torch.testing.assert_close(first.observations[key], second.observations[key], rtol=0, atol=0)
            torch.testing.assert_close(first.reward, second.reward, rtol=0, atol=0)
        selfplay = self.task([capture_state()], persistent_knowledge=False, opponent="self_play")
        self.assertIsNone(selfplay.opponent.knowledge)

    def test_task_initial_and_next_memory_do_not_read_hidden_world(self):
        first = example_state()
        task = self.task([first, hidden_world_variant(first)])
        obs = task.get_observations()
        self.assertEqual(tuple(obs["knowledge"].shape), (2, 11 * 25))
        torch.testing.assert_close(obs["knowledge"][0], obs["knowledge"][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(obs["critic"][0], obs["critic"][1]))
        stored = obs["knowledge"].clone()
        result = task.step(torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(result.observations["knowledge"][0],
                                   result.observations["knowledge"][1], rtol=0, atol=0)
        torch.testing.assert_close(obs["knowledge"], stored, rtol=0, atol=0)
        direct = task.knowledge.encode(task.batch().state["tick"], task.player_ids).flatten(1)
        torch.testing.assert_close(result.observations["knowledge"], direct, rtol=0, atol=0)
        planes = memory_planes(task, direct)
        torch.testing.assert_close(result.metrics["Knowledge/explored_fraction"], planes[:, 0].mean(1))
        torch.testing.assert_close(result.metrics["Knowledge/known_enemy_general"], torch.zeros(2))

    def test_terminal_memory_precedes_seat_rotation_and_owns_storage(self):
        task = self.task([capture_state()], rotate_seats=True)
        result = task.step(torch.tensor([Action.move(3, Direction.RIGHT).encode()]))
        self.assertTrue(bool(result.terminated[0]))
        self.assertEqual(int(task.player_ids[0]), 1)
        terminal = memory_planes(task, result.terminal_observations["knowledge"])
        reset = memory_planes(task, result.observations["knowledge"])
        enemy = task.knowledge.channels.index("last_seen_enemy_general")
        city = task.knowledge.channels.index("last_seen_city")
        self.assertEqual(float(terminal[0, enemy, 4]), 0)
        self.assertEqual(float(terminal[0, city, 4]), 1)
        self.assertEqual(float(reset[0, enemy, 4]), 0)
        self.assertEqual(float(reset[0, enemy, 3]), 1)
        self.assertEqual(float(result.metrics["Knowledge/known_enemy_general"][0]), 0)
        preserved = result.terminal_observations["knowledge"].clone()
        task.step(torch.zeros(1, dtype=torch.long))
        torch.testing.assert_close(result.terminal_observations["knowledge"], preserved, rtol=0, atol=0)

    def test_timeout_keeps_exploration_before_memory_reset(self):
        state = example_state()
        state.army[0, 0] = 10
        task = self.task([state], max_episode_steps=2)
        task.step(torch.tensor([Action.move(0, Direction.RIGHT).encode()]))
        result = task.step(torch.zeros(1, dtype=torch.long))
        self.assertTrue(bool(result.truncated[0]))
        self.assertFalse(bool(result.terminated[0]))
        terminal = memory_planes(task, result.terminal_observations["knowledge"])
        reset = memory_planes(task, result.observations["knowledge"])
        self.assertEqual(float(terminal[0, 0, 2]), 1)
        self.assertEqual(float(reset[0, 0, 2]), 0)
        stored = result.terminal_observations["knowledge"].clone()
        task.reset()
        torch.testing.assert_close(result.terminal_observations["knowledge"], stored, rtol=0, atol=0)

    def test_partial_reset_clears_seats_and_preserves_ongoing_task_and_opponent(self):
        state = example_state()
        state.army.flat[0] = 3
        task = self.task([state, state.copy()], opponent="self_play", rotate_seats=True)
        pool = KnowledgePool()
        task.opponent.attach(pool, stochastic=False)
        task.step(torch.tensor([Action.move(0, Direction.RIGHT).encode(), 0]))
        ongoing = task.get_observations()["knowledge"][1].clone()
        opponent_before = [task.opponent.knowledge.encode(task.batch().state["tick"],
                           torch.full_like(task.player_ids, pid)).clone() for pid in range(2)]
        task.reset(torch.tensor([True, False]), rotate=True)
        torch.testing.assert_close(task.get_observations()["knowledge"][1], ongoing, rtol=0, atol=0)
        fresh = KnowledgeState(2, 5, 5, 2)
        fresh.update(task.batch(), task.player_ids)
        torch.testing.assert_close(task.get_observations()["knowledge"][0],
                                   fresh.encode(task.batch().state["tick"], task.player_ids).flatten(1)[0],
                                   rtol=0, atol=0)
        old_seat = torch.zeros(2, dtype=torch.long)
        self.assertEqual(int(task.knowledge.encode(task.batch().state["tick"], old_seat)[0].count_nonzero()), 0)
        for pid in range(2):
            seats = torch.full_like(task.player_ids, pid)
            current = task.opponent.knowledge.encode(task.batch().state["tick"], seats)
            torch.testing.assert_close(current[1], opponent_before[pid][1], rtol=0, atol=0)
            fresh.reset()
            fresh.update(task.batch(), seats)
            torch.testing.assert_close(current[0], fresh.encode(task.batch().state["tick"], seats)[0],
                                       rtol=0, atol=0)

    def test_selfplay_receives_only_own_local_memory_without_rng_side_effects(self):
        first = example_state()
        task = self.task([first, hidden_world_variant(first)], learner_seat=1, opponent="self_play")
        pool = KnowledgePool()
        task.opponent.attach(pool, seed=45)
        global_rng = torch.get_rng_state().clone()
        task.opponent.joint_actions(task.batch(), task.player_ids, task.action_space,
                                    task.observation_encoder)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        inputs = pool.snapshot.model.inputs[0]
        for key in inputs:
            torch.testing.assert_close(inputs[key][0], inputs[key][1], rtol=0, atol=0)
        seats = torch.zeros_like(task.player_ids)
        expected = task.opponent.knowledge.encode(task.batch().state["tick"], seats).flatten(1)
        torch.testing.assert_close(inputs["knowledge"], expected, rtol=0, atol=0)
        self.assertFalse(inputs["knowledge"].requires_grad)
        self.assertNotIn("critic", inputs)

    def test_pool_signature_records_memory_and_rejects_incompatible_resume(self):
        def runner_for(task):
            obs = TensorDict(task.get_observations(), batch_size=[task.num_envs])
            groups = {"policy": ["policy"], "critic": ["critic"]}
            policy = KnowledgePolicy()
            runner = SimpleNamespace(alg=SimpleNamespace(policy=policy), cfg={"obs_groups": groups},
                                     env=SimpleNamespace(get_observations=lambda: obs))
            config = {"runner": {"policy": {"class_name": "unused"}},
                      "self_play": {"pool_size": 2}}
            return runner, config

        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            disabled = self.task([capture_state()], opponent="self_play", persistent_knowledge=False)
            old_runner, config = runner_for(disabled)
            old = prepare_self_play(disabled, old_runner, config, root / "old")
            self.assertNotIn("persistent_knowledge", old.pool.policy_spec["mdp"])
            self.assertNotIn("knowledge_semantics", old.pool.policy_spec)
            checkpoint = root / "old" / "checkpoint.pt"
            torch.save({"model_state_dict": old_runner.alg.policy.state_dict()}, checkpoint)
            enabled = self.task([capture_state()], opponent="self_play")
            new_runner, config = runner_for(enabled)
            with self.assertRaisesRegex(ValueError, "policy_spec"):
                prepare_self_play(enabled, new_runner, config, root / "rejected", resume=checkpoint)
            new = prepare_self_play(enabled, new_runner, config, root / "new")
            spec = new.pool.policy_spec
            self.assertTrue(spec["mdp"]["persistent_knowledge"])
            self.assertEqual(spec["knowledge_semantics"], enabled.knowledge.semantics)
            self.assertEqual(spec["knowledge_channels"], list(enabled.knowledge.channels))
            self.assertEqual(spec["observation_shapes"]["knowledge"], [11 * 9])


if __name__ == "__main__":
    unittest.main()
