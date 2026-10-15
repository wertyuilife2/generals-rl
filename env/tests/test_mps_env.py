"""Real Metal execution against independent CPU rules and prepared trajectories."""
import unittest

import numpy as np

from generals import MapConfig, VecEnv
from generals.core.batch import generate_states
from generals.core.observation import legal_action_mask, observe
from generals.benchmarks.contracts import BenchmarkCase
from generals.benchmarks.workload import prepare_workload
from generals.benchmarks.step import measure_step_case
from test_gpu_env import make_state, torch

MPS_AVAILABLE = torch is not None and torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader')


@unittest.skipUnless(MPS_AVAILABLE, 'requires an Apple GPU and MPS-enabled PyTorch')
class MPSEnvTests(unittest.TestCase):
    def pair(self, states, **kwargs):
        cpu = VecEnv.from_states(states, threads=2, seed=891, **kwargs)
        gpu = VecEnv.from_states(states, device='mps', seed=891, **kwargs)
        self.addCleanup(cpu.close)
        self.addCleanup(gpu.close)
        return cpu, gpu

    def assert_equal(self, cpu, gpu):
        cpu, gpu = cpu.unwrapped, gpu.unwrapped
        gpu.check_errors()
        for name in cpu.state:
            np.testing.assert_array_equal(cpu.state[name], gpu.state[name].cpu().numpy(), err_msg=name)
        for name in ('rng_state', 'stats', 'last_moves', 'done', 'counters', 'errors'):
            np.testing.assert_array_equal(getattr(cpu, name), getattr(gpu, name).cpu().numpy(), err_msg=name)

    def advance(self, pair, actions):
        cpu, gpu = pair
        cpu.step(actions)
        gpu.step(torch.from_numpy(actions).to('mps'))
        self.assert_equal(cpu, gpu)

    def test_prepared_trajectories_and_auto_reset(self):
        for players in (2, 4, 8):
            case = BenchmarkCase(MapConfig(9, 7, players), batch_size=5, threads=2, episode_ticks=53)
            work = prepare_workload(case, 161, 0)
            cpu = VecEnv(work.initial, width=9, height=7, auto_reset=True, episode_ticks=53)
            gpu = VecEnv({k: torch.from_numpy(v) for k, v in work.initial.items()},
                         width=9, height=7, device='mps', auto_reset=True, episode_ticks=53)
            with cpu, gpu:
                for actions in work.history:
                    self.advance((cpu, gpu), actions)

    def test_large_integer_growth_combat_and_malformed_actions(self):
        bad = [-1, -2, -9, -(2**63), 2**63-1, 100000]
        states = [make_state() for _ in bad]
        for state in states:
            state.owner.flat[1:4] = 0
            state.army.flat[:4] = 2**54+3
        pair = self.pair(states)
        self.advance(pair, np.array([[v, 0] for v in bad], np.int64))
        self.advance(pair, np.full((6, 2), 3, np.int64))
        for env in pair:
            env.state['tick'][0] = 2**54+3
            env.invalidate_cache()
        self.advance(pair, np.zeros((6, 2), np.int64))

    def test_capture_inheritance_and_freeze(self):
        state = make_state(height=1, width=7, generals=[0, 2, 6])
        state.army.flat[0] = 100
        state.owner.flat[1] = 0
        state.army.flat[1] = 100
        state.owner.flat[3:5] = 1
        state.army.flat[3:5] = [5, 0]
        pair = self.pair([state])
        self.advance(pair, np.array([[11, 0, 0]], np.int64))
        self.assertFalse(pair[0].state['alive'][0, 1])
        self.assertEqual(pair[0].state['army'][0, 3:5].tolist(), [2, 1])
        for env in pair: env.unwrapped.episode_ticks = 2
        for _ in range(4): self.advance(pair, np.zeros((1, 3), np.int64))
        for env in pair: env.unwrapped.episode_ticks = 0
        self.advance(pair, np.zeros((1, 3), np.int64))

    def test_partial_reset_edits_and_inference_tensors(self):
        with torch.inference_mode():
            cpu, gpu = self.pair(generate_states(MapConfig(9, 7, 4), 3), auto_reset=True, episode_ticks=7)
            for _ in range(9): self.advance((cpu, gpu), np.zeros((3, 4), np.int64))
            cpu.reset(np.array([True, False, True]))
            gpu.reset(torch.tensor([True, False, True], device='mps'))
            self.assert_equal(cpu, gpu)
            for env in (cpu, gpu):
                env.state['army'][0, 0] = 0
                env.invalidate_cache()
            self.advance((cpu, gpu), np.zeros((3, 4), np.int64))

    def test_external_edits_automatically_invalidate_cache(self):
        pair = self.pair([make_state()])
        self.advance(pair, np.zeros((1, 2), np.int64))
        for env in pair:
            env.state['army'][0, 0] = 700
        pair[0].invalidate_cache()  # CPU numpy writes have no version counter.
        self.advance(pair, np.array([[3, 0]], np.int64))

    def test_packed_state_edits_rebuild_owned_and_structure_indices(self):
        pair = self.pair([make_state(), make_state()], auto_reset=True, episode_ticks=57)
        self.advance(pair, np.zeros((2, 2), np.int64))
        for env in pair:
            env.state['owner'][0, 1] = 0
            env.state['army'][0, 1] = 101
            env.state['structure'][0, 1] = 1
            env.state['terrain'][1, 2] = 1
        pair[0].invalidate_cache()
        for _ in range(61): self.advance(pair, np.zeros((2, 2), np.int64))

    def test_recapturing_owned_land_does_not_duplicate_growth(self):
        state = make_state(height=1, width=3, generals=[0, 2])
        state.owner.flat[1] = 1
        state.army.flat[:] = [100, 1, 200]
        pair = self.pair([state])
        self.advance(pair, np.array([[3, 23]], np.int64))
        for _ in range(51): self.advance(pair, np.zeros((1, 2), np.int64))

    def test_full_owned_index_capacity_and_large_growth(self):
        state = make_state(height=64, width=64)
        state.owner.flat[:-1] = 0
        state.army.flat[:-1] = 2**40 + 1
        pair = self.pair([state])
        for _ in range(51): self.advance(pair, np.zeros((1, 2), np.int64))

    def test_group_reconfiguration_preserves_live_trajectory(self):
        case = BenchmarkCase(MapConfig(9, 7, 4), batch_size=37, threads=2, episode_ticks=17)
        work = prepare_workload(case, 61, 0)
        with VecEnv(work.initial, width=9, height=7, auto_reset=True, episode_ticks=17) as cpu, \
             VecEnv({k: torch.from_numpy(v) for k, v in work.initial.items()},
                    width=9, height=7, device='mps', auto_reset=True, episode_ticks=17) as gpu:
            for index, actions in enumerate(work.history):
                if index % 20 == 0:
                    gpu.unwrapped.configure_step(threads_per_group=(32, 64, 128, 256)[index // 20])
                self.advance((cpu, gpu), actions)
            with self.assertRaises(ValueError): gpu.unwrapped.configure_step(threads_per_group=0)
            with self.assertRaises(ValueError): gpu.unwrapped.configure_step(threads_per_group=2**20)

    def test_prepared_step_benchmark_and_inference_output_edits(self):
        case = BenchmarkCase(MapConfig(9, 7, 4), batch_size=5, threads=2, episode_ticks=17, device='mps')
        result = measure_step_case(case, backend='mps', steps=61, warmup=2, repeats=1)
        self.assertTrue(result['samples'][0]['verification']['state_statistics_counters_match'])
        self.assertTrue(result['samples'][0]['verification']['rng_unchanged'])
        with torch.inference_mode():
            pair = self.pair([make_state()])
            self.advance(pair, np.zeros((1, 2), np.int64))
            pair[1].stats.zero_()  # Public outputs must not corrupt private totals.
            self.advance(pair, np.zeros((1, 2), np.int64))

    def test_observation_masks_and_export(self):
        states = generate_states(MapConfig(9, 7, 4), 3)
        cpu, gpu = self.pair(states)
        for pid in range(4):
            actual = gpu.unwrapped.observe(pid)
            moves, wait = gpu.unwrapped.legal_action_mask(pid)
            for lane, state in enumerate(states):
                expected = observe(state, pid)
                for key in ('terrain', 'structure', 'owner', 'army', 'visible'):
                    np.testing.assert_array_equal(actual[key][lane].cpu(), getattr(expected, key))
                expected_moves, expected_wait = legal_action_mask(expected)
                np.testing.assert_array_equal(moves[lane].cpu(), expected_moves)
                self.assertEqual(bool(wait[lane]), expected_wait)
        for state in gpu.to_states(): state.validate()

    def test_lines_max_board_noncontiguous_actions_and_validation(self):
        for width, height in ((1, 7), (7, 1), (64, 64)):
            cpu, gpu = self.pair(generate_states(MapConfig(width, height, 2), 3))
            gpu.unwrapped.configure_step(threads_per_group=32)
            actions = torch.zeros((2, 3), dtype=torch.int64, device='mps').t()
            for _ in range(51):
                cpu.step(np.zeros((3, 2), np.int64)); gpu.step(actions)
            self.assert_equal(cpu, gpu)
        with self.assertRaises(TypeError): gpu.step(actions.float())
        with self.assertRaises(ValueError): gpu.step(actions.cpu())
        gpu.state['army'] = gpu.state['army'].clone()
        with self.assertRaises(ValueError): gpu.step(actions)
        gpu.close()
        with self.assertRaises(RuntimeError): gpu.step(actions)


if __name__ == '__main__': unittest.main()
