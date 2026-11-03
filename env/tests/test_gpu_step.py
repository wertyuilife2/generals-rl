"""CUDA boundary cases, thread layouts and caches match CPU tick outputs."""
import unittest
import numpy as np

from test_gpu_env import _CUDA_AVAILABLE, make_state, torch
from generals.backends.cpu import CPUEnv
from generals.core.config import MapConfig
from generals.core.mapgen import generate_map
from generals.benchmarks.contracts import BenchmarkCase
from generals.benchmarks.step import measure_step_case


@unittest.skipUnless(_CUDA_AVAILABLE, "requires CUDA and nvcc")
class CUDAStepTests(unittest.TestCase):
    def pair(self, states, **kwargs):
        from generals.backends.gpu import GPUEnv
        cpu = CPUEnv.from_states(states, seed=891, threads=2, **kwargs)
        gpu = GPUEnv.from_states(states, seed=891, **kwargs)
        for env in (cpu, gpu): self.addCleanup(env.close)
        return cpu, gpu

    def assert_equal(self, pair):
        cpu, gpu = pair
        for name in cpu.state:
            np.testing.assert_array_equal(cpu.state[name], gpu.state[name].cpu().numpy(), err_msg=name)
        for name in ('rng_state', 'counters', 'stats', 'last_moves', 'errors', 'done'):
            np.testing.assert_array_equal(getattr(cpu, name), getattr(gpu, name).cpu().numpy(), err_msg=name)

    def advance(self, pair, actions):
        pair[0].step(actions)
        pair[1].step(torch.from_numpy(actions).cuda())
        self.assert_equal(pair)

    def test_transient_large_statistics_are_exact_after_combat(self):
        state = make_state()
        state.tick = 1
        state.army.flat[0] = np.iinfo(np.int64).max - 2
        state.owner.flat[1], state.structure.flat[1], state.army.flat[1] = 0, 1, 2
        state.owner.flat[7], state.army.flat[7] = 1, 10
        pair = self.pair([state])
        self.advance(pair, np.array([[5, 0]], np.int64))
        for env in pair: env.check_errors()
        self.assertGreater(pair[0].stats[0, 0, 0], 2**62)

    def test_large_statistics_growth_merge_and_tick_are_exact(self):
        states = [make_state() for _ in range(5)]
        for state in states: state.owner.flat[1:4] = 0
        pair = self.pair(states)
        large = 2**54 + 3
        actions = np.zeros((5, 2), np.int64)
        self.advance(pair, actions)
        actions[1, 0] = 3
        for env in pair:
            env.state['tick'][:] = 0
            env.state['army'][0, 1:4] = large
            env.state['army'][1, 0:2] = large
            env.state['tick'][2] = 1
            env.state['army'][2, 0] = large
            env.state['tick'][3] = large
            env.invalidate_cache()
        self.advance(pair, actions)
        self.assertEqual(pair[0].errors.tolist(), [0] * 5)
        self.assertEqual(pair[0].state['tick'][3], large + 1)
        self.advance(pair, actions)

    def test_truncation_freezes_until_explicit_unbounded_step(self):
        states = [make_state() for _ in range(3)]
        states[0].tick, states[1].tick, states[2].tick = 0, 49, 50
        pair = self.pair(states, episode_ticks=50)
        actions = np.zeros((3, 2), np.int64)
        for _ in range(8): self.advance(pair, actions)
        self.assertEqual(pair[0].state['tick'].tolist(), [8, 50, 50])
        for env in pair: env.episode_ticks = 0
        self.advance(pair, actions)
        self.assertEqual(pair[0].state['tick'].tolist(), [9, 51, 51])

    def test_malformed_int64_actions_preserve_diagnostics(self):
        values = [-1, -2, -9, -(2**63), 2**63 - 1, 100000]
        states = [make_state() for _ in values]
        for state in states: state.army.flat[0] = 8
        pair = self.pair(states)
        self.advance(pair, np.array([[value, 0] for value in values], np.int64))
        self.assertEqual(pair[0].last_moves[:, 0, 1].tolist(), [0] * len(values))
        self.assertEqual(pair[0].last_moves[:, 0, 8].tolist(), [3] * len(values))
        self.assertEqual(pair[0].state['army'][:, 0].tolist(), [8] * len(values))

    def test_auto_reset_rebuilds_growth_index_after_external_edits(self):
        state = make_state()
        state.owner.flat[1:3], state.army.flat[1:3] = 0, 7
        state.structure.flat[1] = 1
        pair = self.pair([state], auto_reset=True, episode_ticks=5)
        for env in pair:
            env.state['structure'][0, 1] = 0
            env.state['structure'][0, 2] = 1
            env.invalidate_cache()
        for _ in range(7): self.advance(pair, np.zeros((1, 2), np.int64))
        self.assertEqual(pair[0].state['army'][0, 1:3].tolist(), [8, 7])

    def test_groups_partial_blocks_growth_and_resets(self):
        for threads, groups in ((4, 8), (8, 16), (16, 8), (16, 16), (32, 1), (32, 4),
                                (32, 8), (32, 16), (64, 1), (128, 1), (256, 1), (512, 1)):
            with self.subTest(threads=threads, groups=groups):
                states = [generate_map(MapConfig(25, 25, 8), seed) for seed in range(11)]
                for state in states: state.army.ravel()[state.general_pos] = 88
                pair = self.pair(states, auto_reset=True, episode_ticks=53)
                pair[1].configure_step(group_threads=threads, envs_per_block=groups)
                rng = np.random.default_rng(971)
                for _ in range(111):
                    self.advance(pair, rng.integers(-9, 65 * 8, size=(11, 8), dtype=np.int64))
                for env in pair: env.close()

    def test_lines_and_maximum_board(self):
        configs = [(1, 7, 2, 16, 8), (7, 1, 3, 16, 8), (64, 64, 8, 16, 8),
                   (1, 7, 2, 64, 1), (7, 1, 3, 128, 1)]
        configs += [(64, 64, 8, threads, 1) for threads in (64, 128, 256, 512)]
        for w, h, p, threads, groups in configs:
            with self.subTest(width=w, height=h, threads=threads):
                if min(w, h) == 1:
                    states = [make_state(h, w, generals=list(range(p))) for _ in range(3)]
                else:
                    states = [generate_map(MapConfig(w, h, p, .1, .1), seed) for seed in (12, 19, 43)]
                pair = self.pair(states, auto_reset=True, episode_ticks=61)
                pair[1].configure_step(group_threads=threads, envs_per_block=groups)
                for _ in range(67): self.advance(pair, np.zeros((3, p), np.int64))
                for env in pair: env.close()

    def test_multiwarp_groups_require_separate_blocks(self):
        pair = self.pair([make_state()])
        for threads in (64, 128, 256, 512):
            with self.assertRaisesRegex(ValueError, 'one environment per block'):
                pair[1].configure_step(group_threads=threads, envs_per_block=2)

    def test_prepared_trajectory_matches_cpu(self):
        case = BenchmarkCase(MapConfig(25, 25, 4), batch_size=7, threads=2, episode_ticks=17)
        result = measure_step_case(case, backend='gpu', steps=61, warmup=2, repeats=1)
        self.assertTrue(result['samples'][0]['verification']['state_statistics_counters_match'])
        self.assertTrue(result['samples'][0]['verification']['rng_unchanged'])


if __name__ == '__main__': unittest.main()
