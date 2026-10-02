"""Growth caches must follow ownership changes, resets and external state edits."""
import unittest

import numpy as np

from generals.backends.cpu import CPUEnv
from generals.core.state import NEUTRAL
from test_gpu_env import make_state, torch, _CUDA_AVAILABLE


class OwnedGrowthTests(unittest.TestCase):
    def compare(self, gpu):
        state = make_state(2, 5, generals=[0, 4, 9])
        state.army.flat[0] = 20
        state.army.flat[4] = 2
        state.structure.flat[1] = 1
        state.owner.flat[1] = NEUTRAL
        state.structure.flat[3] = 1
        state.owner.flat[3] = 1
        state.army.flat[3] = 6
        state.owner.flat[8] = 1
        state.army.flat[8] = 5
        reference = CPUEnv.from_states([state], auto_reset=True, episode_ticks=53)
        if gpu:
            from generals.backends.gpu import GPUEnv
            candidate = GPUEnv.from_states([state], auto_reset=True, episode_ticks=53)
        else:
            candidate = CPUEnv.from_states([state], auto_reset=True, episode_ticks=53)
        def host(value):
            return value.cpu().numpy() if gpu else value
        def advance(actions):
            # Rebuild the reference index every tick to catch stale warm-cache entries.
            reference.invalidate_cache()
            reference.step(actions)
            candidate.step(torch.from_numpy(actions).cuda() if gpu else actions)
            reference.check_errors(); candidate.check_errors()
            for name in reference.state:
                np.testing.assert_array_equal(reference.state[name], host(candidate.state[name]))
            for name in ('stats', 'last_moves', 'done', 'errors', 'counters'):
                np.testing.assert_array_equal(getattr(reference, name), host(getattr(candidate, name)))
        try:
            for tick in range(60):
                actions = np.zeros((1, 3), np.int64)
                if tick < 4:
                    actions[0, 0] = 1 + tick * 8 + 2  # Acquire neutral city, enemy city, then general.
                advance(actions)
                if tick == 3:
                    self.assertFalse(reference.state['alive'][0, 1])
                    self.assertEqual(reference.stats[0, 0, 2], 3)
            # Supported direct edits need not preserve the general/alive
            # relationship. Count actual owned structures, not cities + alive.
            for env in (reference, candidate):
                env.state['structure'][0, 0] = 0
                env.state['structure'][0, 6] = 1
                env.state['owner'][0, 6] = NEUTRAL
                env.state['army'][0, 6] = 5
                env.state['structure'][0, 7] = 1
                env.state['owner'][0, 7] = 2
                env.state['tick'][0] = 1
                env.invalidate_cache()
            advance(np.zeros((1, 3), np.int64))
        finally:
            reference.close(); candidate.close()

    def test_cpu_owned_growth_lifetime(self):
        self.compare(False)

    @unittest.skipUnless(_CUDA_AVAILABLE, 'requires CUDA')
    def test_gpu_owned_growth_lifetime(self):
        self.compare(True)
