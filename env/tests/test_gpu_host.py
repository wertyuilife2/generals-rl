"""Native CUDA launch validates inputs and preserves the caller's stream order."""
import unittest
import numpy as np

from test_gpu_env import make_state, torch, _CUDA_AVAILABLE


@unittest.skipUnless(_CUDA_AVAILABLE, 'requires CUDA and nvcc')
class NativeHostTests(unittest.TestCase):
    def pair(self):
        from generals.backends.cpu import CPUEnv
        from generals.backends.gpu import GPUEnv
        states = [make_state() for _ in range(3)]
        cpu = CPUEnv.from_states(states, auto_reset=True, episode_ticks=7)
        gpu = GPUEnv.from_states(states, auto_reset=True, episode_ticks=7)
        gpu.configure_step()
        for env in (cpu, gpu): self.addCleanup(env.close)
        return cpu, gpu

    def test_nondefault_stream_noncontiguous_actions_and_mutations(self):
        cpu, gpu = self.pair()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actions = torch.zeros((3, 4), dtype=torch.int64, device='cuda')[:, ::2]
            self.assertFalse(actions.is_contiguous())
            for tick in range(15):
                if tick == 3:
                    cpu.state['army'][:, 0] = gpu.state['army'][:, 0] = 20
                    cpu.invalidate_cache()  # GPU detects normal tensor edits automatically.
                if tick == 5: cpu.episode_ticks = gpu.episode_ticks = 11
                cpu.step(np.zeros((3, 2), np.int64))
                self.assertIs(gpu.step(actions), gpu._step_result)
                # A consumer on the same stream must see the just-completed tick.
                np.testing.assert_array_equal(cpu.stats, gpu.stats.clone().cpu().numpy())
        torch.cuda.current_stream().wait_stream(stream)
        for name in cpu.state:
            np.testing.assert_array_equal(cpu.state[name], gpu.state[name].cpu().numpy())

    def test_input_and_storage_errors(self):
        _, env = self.pair()
        for actions, error in ((np.zeros((3, 2), np.int64), TypeError),
                               (torch.zeros((3, 2), device='cuda'), TypeError),
                               (torch.zeros((2, 3), dtype=torch.int64, device='cuda'), ValueError),
                               (torch.zeros((3, 2), dtype=torch.int64), ValueError)):
            with self.assertRaises(error): env.step(actions)
        env.state['army'].resize_(3, 22)
        with self.assertRaisesRegex(ValueError, 'storage'): env.step(env._wait_actions)

    def test_reconfiguring_threads_rebuilds_cache_without_resetting_state(self):
        cpu, gpu = self.pair()
        for tick in range(3):
            if tick == 1: gpu.configure_step(group_threads=8, envs_per_block=4)
            cpu.step(np.zeros((3, 2), np.int64))
            gpu.step(gpu._wait_actions)
            np.testing.assert_array_equal(cpu.stats, gpu.stats.cpu().numpy())
            np.testing.assert_array_equal(cpu.state['tick'], gpu.state['tick'].cpu().numpy())
