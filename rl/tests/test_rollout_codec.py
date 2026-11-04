"""Exact binary packing, continuous bits and histories with asynchronous resets."""
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorageCTS

from generals_rl.rollout_codec import GeneralsObservationCodec


class RolloutCodecTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(79)
        self.n, self.cells, self.history = 8, 6, 4

    def frame(self):
        maps = torch.randint(2, (self.n, 13, self.cells), generator=self.generator).float()
        maps[:, 9:12] = torch.randn(self.n, 3, self.cells, generator=self.generator) * 1000
        return torch.cat((maps.flatten(1), torch.randn(self.n, 13, generator=self.generator)), -1)

    def observations(self, frames):
        knowledge = torch.randint(2, (self.n, 11, self.cells), generator=self.generator).float()
        knowledge[:, 10] = torch.randint(2001, (self.n, self.cells), generator=self.generator).float() / 2000
        return TensorDict(dict(policy=frames.flatten(1), critic=self.frame(), knowledge=knowledge.flatten(1),
            action_mask=torch.randint(2, (self.n, 1 + self.cells * 8), generator=self.generator).bool(),
            valid_cells=torch.randint(2, (self.n, self.cells), generator=self.generator).bool(),
            auxiliary=torch.arange(self.n, dtype=torch.int32)[:, None]), [self.n])

    def assert_same(self, a, b):
        self.assertEqual(set(a.keys()), set(b.keys()))
        for key in a.keys():
            self.assertEqual(a[key].dtype, b[key].dtype)
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            if a[key].dtype == torch.float32:
                self.assertTrue(torch.equal(a[key].contiguous().view(torch.int32), b[key].contiguous().view(torch.int32)))

    def test_complete_rollouts_probes_minibatches_and_short_final_rollout(self):
        for history in (None, {'policy': self.history}):
            with self.subTest(history=history):
                frames = torch.stack([self.frame() for _ in range(self.history)], 1)
                obs = self.observations(frames)
                options = dict(action_dtype=torch.long, store_action_mean=False,
                               store_action_std=False, history_lengths=history)
                reference = RolloutStorageCTS('rl', self.n, 2, 5, obs, [1], **options)
                compact = RolloutStorageCTS('rl', self.n, 2, 5, obs, [1],
                    observation_codec=GeneralsObservationCodec(obs), **options)
                for steps in (5, 2):
                    reference.num_transitions_per_env = compact.num_transitions_per_env = steps
                    for step in range(steps):
                        done = torch.tensor([step == 0, step % 2 == 0, True, False,
                                             step == steps-1, step < 3, False, step == 1])
                        transition = RolloutStorageCTS.Transition()
                        transition.observations = obs
                        transition.actions = torch.zeros(self.n, 1, dtype=torch.long)
                        transition.rewards = torch.arange(self.n).float()
                        transition.dones = done
                        transition.values = torch.ones(self.n, 1)
                        transition.actions_log_prob = torch.full((self.n,), -.4)
                        reference.add_transition(transition)
                        compact.add_transition(transition)
                        new = self.frame()
                        frames = torch.cat((frames[:, 1:], new[:, None]), 1)
                        frames[done] = new[done, None].expand(-1, self.history, -1)
                        obs = self.observations(frames)
                    times, envs = torch.arange(steps)[:, None], torch.arange(self.n)[None]
                    self.assert_same(reference.observation_batch(times, envs), compact.observation_batch(times, envs))
                    self.assert_same(reference.observation_batch(times, envs, keys=['policy', 'action_mask']),
                                     compact.observation_batch(times, envs, keys=['policy', 'action_mask']))
                    plan = reference.mini_batch_indices()
                    for a, b in zip(reference.mini_batch_generator(2, 2, plan),
                                    compact.mini_batch_generator(2, 2, plan), strict=True):
                        self.assert_same(a[0], b[0])
                    reference.clear()
                    compact.clear()
                tensors = list(compact.observations.values(include_nested=True, leaves_only=True))
                self.assertIn(torch.int16, [v.dtype for v in tensors])
                self.assertEqual(compact.observations['action_mask'].shape[-1], 7)
                self.assertLess(sum(v.untyped_storage().nbytes() for v in tensors),
                                sum(v.untyped_storage().nbytes() for v in reference.observations.values()) / 2)

    def test_all_flag_combinations_and_signed_zero_are_exact(self):
        flags = torch.arange(1024, dtype=torch.int16)
        maps = torch.zeros(1024, 13, 1)
        for bit, channel in enumerate((*range(9), 12)):
            maps[:, channel, 0] = ((flags >> bit) & 1).float()
        maps[:, 9:12] = -0.
        public = torch.randn(1024, 13, generator=self.generator)
        original = torch.cat((maps.flatten(1), public), -1)
        obs = TensorDict(dict(policy=original, action_mask=torch.ones(1024, 9, dtype=torch.bool)), [1024])
        codec = GeneralsObservationCodec(obs)
        encoded = codec.encode('policy', original).unsqueeze(0)
        actual = codec.gather('policy', encoded, torch.zeros(1024, dtype=torch.long), torch.arange(1024))
        self.assertTrue(torch.equal(original.view(torch.int32), actual.view(torch.int32)))


if __name__ == '__main__':
    unittest.main()
