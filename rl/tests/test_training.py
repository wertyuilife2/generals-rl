"""CTS unit tests: legal discrete policies, teacher lanes, and timeout bootstrap."""

import contextlib
import io
import types
import unittest

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MoECTS
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS


def make_observations(num_envs=8, num_actions=13):
    """Every lane has a different legal subset and at least one legal action."""
    mask = torch.zeros(num_envs, num_actions, dtype=torch.bool)
    mask[:, 0] = True
    for lane in range(num_envs):
        mask[lane, 1 + lane % (num_actions - 1)] = True
        mask[lane, 1 + (lane + 3) % (num_actions - 1)] = True
    return TensorDict({"policy": torch.randn(num_envs, 24),
                       "single_obs": torch.randn(num_envs, 12),
                       "critic": torch.randn(num_envs, 18),
                       "action_mask": mask}, batch_size=[num_envs])


def make_algorithm(obs, num_steps=4, ratio=0.75):
    num_envs = obs.batch_size[0]
    num_actions = obs["action_mask"].shape[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCriticMoECTS(
            obs, {"policy": ["policy"], "critic": ["critic"]}, num_actions,
            action_distribution="categorical", actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8], teacher_encoder_hidden_dims=[16],
            student_encoder_hidden_dims=[16, 8, 8], expert_num=2, latent_dim=8,
        )
    teacher_envs = MoECTS.teacher_environment_count(num_envs, ratio)
    storage = RolloutStorageCTS("rl", num_envs, teacher_envs, num_steps,
                                obs, [1], distribution_shape=[num_actions],
                                action_dtype=torch.long, store_action_std=False)
    return MoECTS(policy, storage, num_envs, teacher_env_ratio=ratio,
                  num_learning_epochs=1, num_mini_batches=2, gamma=0.9,
                  learning_rate=0.003, student_encoder_learning_rate=0.003)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def assert_legal(self, actions, obs):
        self.assertEqual(actions.dtype, torch.long)
        self.assertEqual(tuple(actions.shape), (obs.batch_size[0], 1))
        self.assertTrue(obs["action_mask"].gather(1, actions).all())

    def test_categorical_samples_and_student_priors_respect_mask(self):
        obs = make_observations()
        algorithm = make_algorithm(obs)
        policy = algorithm.policy
        with torch.inference_mode():
            for is_teacher in (True, False):
                for _ in range(12):
                    self.assert_legal(policy.act(obs, is_teacher), obs)
                probabilities = policy.action_probabilities(obs, is_teacher)
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(8))
                self.assertTrue((probabilities[~obs["action_mask"]] == 0).all())
            self.assert_legal(policy.act_inference(obs), obs)
            # Deployment and search priors consume the student's legal local inputs.
            local_obs = obs.exclude("critic")
            torch.testing.assert_close(policy.action_probabilities(local_obs),
                                       policy.action_probabilities(obs))
            torch.testing.assert_close(policy.act_inference(local_obs),
                                       policy.act_inference(obs))

    def test_timeout_bootstraps_terminal_next_state_in_correct_lane_order(self):
        obs = make_observations()
        obs["critic"][:, 0] = 1
        terminal_obs = obs.clone()
        terminal_obs["critic"][:, 0] = torch.arange(8, dtype=torch.float32) + 7
        algorithm = make_algorithm(obs, num_steps=1)

        def state_value(_policy, value_obs, is_teacher, **kwargs):
            return value_obs["critic"][:, :1]

        algorithm.policy.evaluate = types.MethodType(state_value, algorithm.policy)
        rewards = torch.arange(8, dtype=torch.float32)
        # A mix of teacher and student lanes time out; the other dones truly terminate.
        timeouts = torch.tensor([True, False, True, False, False, True, False, True])
        with torch.inference_mode():
            algorithm.act(obs)
            algorithm.process_env_step(obs, rewards, torch.ones(8, dtype=torch.bool),
                                       {"time_outs": timeouts, "terminal_observations": terminal_obs})
            algorithm.compute_returns(obs)
        order = torch.cat((algorithm.teacher_env_idxs, algorithm.student_env_idxs))
        expected = rewards + 0.9 * terminal_obs["critic"][:, 0] * timeouts
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], expected[order])
        # Dones stop GAE across resets: terminal transitions include no reset-state value.
        torch.testing.assert_close(algorithm.storage.returns[0, :, 0], expected[order])
        torch.testing.assert_close(rewards, torch.arange(8, dtype=torch.float32))

    def test_non_integral_teacher_ratio_keeps_both_environment_groups(self):
        obs = make_observations(num_envs=7)
        algorithm = make_algorithm(obs, ratio=0.6)
        teachers, students = algorithm.teacher_env_idxs, algorithm.student_env_idxs
        self.assertEqual(len(teachers), 4)
        self.assertEqual(len(students), 3)
        self.assertEqual(torch.unique(torch.cat((teachers, students))).numel(), 7)
        self.assertEqual(set(teachers.tolist()) & set(students.tolist()), set())
        with torch.inference_mode():
            self.assert_legal(algorithm.act(obs), obs)

if __name__ == "__main__":
    unittest.main()
