"""Lossless history storage across resets, rollover and partial final rollouts."""
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorageCTS


class HistoryStorageTests(unittest.TestCase):
    def storage(self, obs, histories=None):
        return RolloutStorageCTS('rl', len(obs), 2, 8, obs, [1], action_dtype=torch.long,
                                 store_action_mean=False, store_action_std=False,
                                 history_lengths=histories)

    def transition(self, obs, done, step):
        value = RolloutStorageCTS.Transition()
        value.observations = obs
        value.actions = torch.arange(len(obs))[:, None]
        value.rewards = torch.arange(len(obs)).float() + step
        value.dones = done
        value.values = torch.arange(len(obs)).float()[:, None] - step
        value.actions_log_prob = torch.full((len(obs),), -.3)
        return value

    def assert_same(self, left, right):
        self.assertEqual(set(left.keys()), set(right.keys()))
        for key in left.keys():
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_exact_history_and_minibatches_across_async_resets_and_rollout_boundary(self):
        generator = torch.Generator().manual_seed(27)
        frames = torch.randn(8, 4, 13, generator=generator)
        def observations():
            return TensorDict(dict(policy=frames.flatten(1), aux=frames[:, -2:].flatten(1),
                critic=torch.randn(8, 7, generator=generator), action_mask=torch.ones(8, 5, dtype=torch.bool)), [8])
        obs = observations()
        dense, compact = self.storage(obs), self.storage(obs, dict(policy=4, aux=2))
        for active in (8, 3):
            dense.num_transitions_per_env = compact.num_transitions_per_env = active
            expected = []
            for step in range(active):
                done = torch.tensor([step == 0, step == 2, step % 2 == 0, step == active-1,
                                     step < 3, False, True, step % 3 == 1])
                expected.append(obs.clone())
                transition = self.transition(obs, done, step)
                dense.add_transition(transition)
                compact.add_transition(transition)
                new = torch.randn(8, 13, generator=generator)
                frames = torch.cat((frames[:, 1:], new[:, None]), dim=1)
                frames[done] = new[done, None].expand(-1, 4, -1)
                obs = observations()
            time = torch.arange(active)[:, None].expand(active, 8)
            env = torch.arange(8)[None].expand(active, 8)
            self.assert_same(dense.observation_batch(time, env), compact.observation_batch(time, env))
            for step, sample in enumerate(expected):
                self.assert_same(sample, compact.observation_batch(torch.full((8,), step), torch.arange(8)))
            plan = dense.mini_batch_indices()
            dense.advantages.normal_()
            compact.advantages.copy_(dense.advantages)
            dense.returns.normal_()
            compact.returns.copy_(dense.returns)
            for a, b in zip(dense.mini_batch_generator(2, 3, plan), compact.mini_batch_generator(2, 3, plan), strict=True):
                self.assert_same(a[0], b[0])
                for left, right in zip(a[1:8], b[1:8]):
                    if left is not None:
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
            dense.clear()
            compact.clear()

    def test_one_frame_history_and_selected_probe_keys(self):
        obs = TensorDict(dict(policy=torch.arange(24).reshape(8, 3).float(),
                              critic=torch.zeros(8, 2)), [8])
        storage = self.storage(obs, dict(policy=1))
        storage.add_transition(self.transition(obs, torch.ones(8, dtype=torch.bool), 0))
        value = storage.observation_batch(torch.zeros(8, dtype=torch.long), torch.arange(8), keys=['policy'])
        torch.testing.assert_close(value['policy'], obs['policy'], rtol=0, atol=0)
        self.assertEqual(list(value.keys()), ['policy'])

    def test_only_prefix_and_current_frames_are_allocated(self):
        obs = TensorDict(dict(policy=torch.zeros(8, 4*13), critic=torch.zeros(8, 7)), [8])
        dense, compact = self.storage(obs), self.storage(obs, dict(policy=4))
        self.assertEqual(compact._history_frames['policy'].shape, (11, 8, 13))
        self.assertEqual(compact.observations['policy'].shape, (8, 8, 13))
        self.assertEqual(compact.observations['policy'].untyped_storage().data_ptr(),
                         compact._history_frames['policy'].untyped_storage().data_ptr())
        self.assertLess(compact._history_frames['policy'].numel(), dense.observations['policy'].numel() / 2)


if __name__ == '__main__':
    unittest.main()
