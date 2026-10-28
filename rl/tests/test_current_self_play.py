"""Live local self-play batching, state ownership and episode boundaries."""
import unittest
from unittest.mock import patch

import torch
from torch.distributions import Categorical

from generals.core.state import Action, Direction
from generals.vector import VecEnv
from generals_rl.policy.knowledge import KnowledgeState
from generals_rl.opponents import CurrentSelfPlayOpponent
from generals_rl.task import GeneralsTask, TaskConfig
from test_mdp import capture_state, example_state, hidden_world_variant


torch.set_num_threads(1)


class ProbePolicy(torch.nn.Module):
    """A local actor whose live parameter selects WAIT or the last legal move."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-1.0))
        self.distribution = None
        self.inputs = []
        self.validate_masks = []

    def action_probabilities(self, observations, *, is_teacher=False, validate_mask=True):
        assert not is_teacher
        assert set(observations.keys()) <= {"policy", "single_obs", "action_mask", "knowledge"}
        self.inputs.append({key: value.clone() for key, value in observations.items()})
        self.validate_masks.append(validate_mask)
        mask = observations["action_mask"]
        logits = torch.arange(mask.shape[1], device=mask.device) * self.weight
        self.distribution = Categorical(logits=logits.expand_as(mask).masked_fill(~mask, -torch.inf))
        return self.distribution.probs

    def act_inference(self, observations):
        return self.action_probabilities(observations).argmax(1, keepdim=True)


class CurrentSelfPlayTests(unittest.TestCase):
    def task(self, states, **settings):
        options = dict(random_maps=False, map_pool_size=0, width=states[0].layout.width, height=states[0].layout.height,
                       players=states[0].player_count, num_envs=len(states),
                       opponent="current_self_play", history_length=3,
                       max_episode_steps=16, rotate_seats=False)
        options.update(settings)
        config = TaskConfig(**options)
        backend = VecEnv.from_states(states, episode_ticks=config.max_episode_steps,
                                     auto_reset=False, threads=1)
        task = GeneralsTask(config, backend=backend)
        self.addCleanup(task.close)
        self.assertIsInstance(task.opponent, CurrentSelfPlayOpponent)
        return task

    def attach(self, task, **options):
        policy = ProbePolicy()
        task.opponent.attach_policy(policy, **options)
        return policy

    def joint_actions(self, task):
        return task.opponent.joint_actions(task.batch(), task.player_ids, task.action_space,
                                           task.observation_encoder)

    def test_binding_contract_and_live_weights(self):
        with self.assertRaisesRegex(ValueError, "1v1"):
            TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, players=3, opponent="current_self_play")
        state = capture_state()
        state.army.flat[4] = 3
        task = self.task([state])
        with self.assertRaisesRegex(RuntimeError, "attach"):
            task.step(torch.zeros(1, dtype=torch.long))
        self.assertEqual(task.batch().state["tick"].item(), 0)
        policy = self.attach(task, stochastic=False)
        self.assertIs(task.opponent.policy, policy)
        self.assertEqual(self.joint_actions(task).tolist(), [[0, 0]])
        with torch.no_grad():
            policy.weight.fill_(1)
        self.assertGreater(int(self.joint_actions(task)[0, 1]), 0)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            task.opponent.attach_policy(policy)
        self.assertTrue(policy.training)
        self.assertTrue(policy.weight.requires_grad)
        self.assertIsNone(policy.weight.grad)

    def test_one_full_actor_batch_uses_caches_and_preserves_learner_distribution(self):
        task = self.task([capture_state() for _ in range(4)], rotate_seats=True,
                         persistent_knowledge=True)
        policy = self.attach(task, seed=91)
        original_distribution = Categorical(logits=torch.tensor([[1.0, 0.0]]))
        policy.distribution = original_distribution
        initial_global_rng = torch.get_rng_state().clone()
        with patch.object(task.observation_encoder, "encode", wraps=task.observation_encoder.encode) as encode, \
             patch.object(task.action_space, "mask", wraps=task.action_space.mask) as mask, \
             patch.object(task.action_space, "validate", wraps=task.action_space.validate) as validate, \
             patch.object(task.knowledge, "encode", wraps=task.knowledge.encode) as memory_encode:
            random_state = task.opponent.generator.get_state()
            first = self.joint_actions(task)
            task.opponent.generator.set_state(random_state)
            second = self.joint_actions(task)
            self.assertEqual(encode.call_count, 0)
            self.assertEqual(mask.call_count, 0)
            self.assertEqual(memory_encode.call_count, 0)
            self.assertEqual(validate.call_count, 2)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), initial_global_rng, rtol=0, atol=0)
        self.assertIs(policy.distribution, original_distribution)
        self.assertEqual([len(item["policy"]) for item in policy.inputs], [4, 4])
        self.assertEqual(policy.validate_masks, [False, False])
        self.assertTrue((first[torch.arange(4), task.player_ids] == 0).all())
        seats = 1 - task.player_ids
        legal = task.action_space.mask(task.batch(), seats)
        self.assertTrue(legal.gather(1, first[torch.arange(4), seats, None]).all())
        for key, actual in policy.inputs[0].items():
            torch.testing.assert_close(actual, task.opponent.observations[key], rtol=0, atol=0)

    def test_distribution_is_restored_when_actor_raises(self):
        task = self.task([capture_state()])
        policy = self.attach(task)
        original_distribution = Categorical(logits=torch.tensor([[1.0, 0.0]]))
        policy.distribution = original_distribution

        def broken_actor(*args, **kwargs):
            policy.distribution = Categorical(logits=torch.tensor([[0.0, 1.0]]))
            raise RuntimeError("actor failed")

        with patch.object(policy, "action_probabilities", side_effect=broken_actor):
            with self.assertRaisesRegex(RuntimeError, "actor failed"):
                self.joint_actions(task)
        self.assertIs(policy.distribution, original_distribution)

    def test_disabled_memory_and_illegal_actions_are_checked_before_game_tick(self):
        task = self.task([capture_state()])
        policy = self.attach(task, stochastic=False)
        self.assertIsNone(task.knowledge)
        self.assertIsNone(task.opponent.knowledge)
        self.assertNotIn("knowledge", task.opponent.observations)
        # Opponent seat one does not own source cell zero.
        with patch.object(policy, "act_inference", return_value=torch.tensor([[1]])):
            with self.assertRaisesRegex(ValueError, "masked"):
                task.step(torch.zeros(1, dtype=torch.long))
        self.assertEqual(task.batch().state["tick"].item(), 0)

    def test_hidden_world_and_other_seat_memory_never_enter_actor(self):
        initial = example_state()
        task = self.task([initial, hidden_world_variant(initial)], learner_seat=1,
                         persistent_knowledge=True)
        policy = self.attach(task)
        self.joint_actions(task)
        for key, value in policy.inputs[0].items():
            torch.testing.assert_close(value[0], value[1], rtol=0, atol=0)
            self.assertFalse(value.requires_grad)
        self.assertIs(task.knowledge, task.opponent.knowledge)
        self.assertNotIn("critic", policy.inputs[0])
        seats = 1 - task.player_ids
        expected = task.knowledge.encode(task.batch().state["tick"], seats).flatten(1)
        torch.testing.assert_close(policy.inputs[0]["knowledge"], expected, rtol=0, atol=0)

    def test_each_role_encodes_local_observation_and_updates_memory_once_per_tick(self):
        task = self.task([example_state() for _ in range(4)], rotate_seats=True,
                         persistent_knowledge=True)
        self.attach(task, stochastic=False)
        with patch.object(task.observation_encoder, "encode", wraps=task.observation_encoder.encode) as encode, \
             patch.object(task.knowledge, "update", wraps=task.knowledge.update) as update:
            task.step(torch.zeros(4, dtype=torch.long))
        local_calls = [call for call in encode.call_args_list if not call.kwargs.get("privileged", False)]
        self.assertEqual(len(local_calls), 2)
        self.assertEqual(update.call_count, 2)
        for calls in (local_calls, update.call_args_list):
            torch.testing.assert_close(calls[0].args[1], task.player_ids, rtol=0, atol=0)
            torch.testing.assert_close(calls[1].args[1], 1 - task.player_ids, rtol=0, atol=0)

    def test_partial_terminal_reset_rotates_and_clears_only_ended_role_histories(self):
        task = self.task([capture_state() for _ in range(2)], rotate_seats=True,
                         persistent_knowledge=True)
        policy = self.attach(task, stochastic=False)
        retained = {key: value.clone() for key, value in task.get_observations().items()}
        previous_observations = task.get_observations()
        initial_opponent_frames = task.opponent.history.frames.clone()
        task.step(torch.zeros(2, dtype=torch.long))
        ongoing_frames = task.opponent.history.frames[1].clone()
        for key in retained:
            torch.testing.assert_close(previous_observations[key], retained[key], rtol=0, atol=0)
        torch.testing.assert_close(task.opponent.history.frames[:, :2], initial_opponent_frames[:, 1:])
        result = task.step(torch.tensor([Action.move(3, Direction.RIGHT).encode(), 0]))
        self.assertEqual(result.terminated.tolist(), [True, False])
        self.assertEqual(task.player_ids.tolist(), [1, 1])
        self.assertEqual(task.opponent._player_ids.tolist(), [0, 0])
        for key in retained:
            torch.testing.assert_close(previous_observations[key], retained[key], rtol=0, atol=0)
        torch.testing.assert_close(task.opponent.history.frames[1, :2], ongoing_frames[1:])
        current = task.observation_encoder.encode(task.batch(), 1 - task.player_ids)
        torch.testing.assert_close(task.opponent.history.frames[0], current[0].expand(3, -1))
        fresh = KnowledgeState(2, task.config.width, task.config.height, 2)
        for seats in (task.player_ids, 1 - task.player_ids):
            fresh.update(task.batch(), seats)
            expected = fresh.encode(task.batch().state["tick"], seats).flatten(1)
            actual = task.knowledge.encode(task.batch().state["tick"], seats).flatten(1)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        terminal = {key: value.clone() for key, value in result.terminal_observations.items()}
        self.joint_actions(task)
        torch.testing.assert_close(policy.inputs[-1]["single_obs"], current)
        task.step(torch.zeros(2, dtype=torch.long))
        for key in terminal:
            torch.testing.assert_close(result.terminal_observations[key], terminal[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
