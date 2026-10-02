"""Optimized caches are checked against the independent CPU rules."""
import unittest
import numpy as np
from test_gpu_env import _CUDA_AVAILABLE as _AVAILABLE
from test_gpu_env import make_state, torch


@unittest.skipUnless(_AVAILABLE, 'requires CUDA and nvcc')
class OptimizedCacheTests(unittest.TestCase):
    def test_packed_cache_statistics_edits_inference_and_capture_order(self):
        from generals import VecEnv
        fields = ('stats', 'last_moves', 'done', 'errors', 'counters', 'rng_state')
        for players, inference, packed in (
            (2,False,True), (4,True,True), (8,False,True), (8,True,False),
        ):
            with self.subTest(players=players,inference=inference,packed=packed):
                states = []
                for lane in range(37):
                    s = make_state(3,11,generals=list(range(players)))
                    s.army.flat[0] = 2**54 + 3 if lane % 3 else 1
                    s.owner.flat[-1] = 1; s.army.flat[-1] = 17; s.structure.flat[-1] = 1
                    states.append(s)
                cpu = VecEnv.from_states(states,device='cpu',threads=2,auto_reset=True,episode_ticks=53)
                with torch.inference_mode(inference):
                    gpu = VecEnv.from_states(states,device='cuda',auto_reset=True,episode_ticks=53)
                    try:
                        gpu.unwrapped.configure_step(group_threads=4,envs_per_block=32,packed_metadata=packed)
                        for tick in range(115):
                            actions = np.zeros((37,players),np.int64)
                            if tick < players-1:
                                actions[:,0] = 1 + tick*8 + 2
                                actions[:,1] = 1 + 8 + 4
                            if tick == 12:
                                for env in (cpu,gpu):
                                    env.state['tick'][:] = 49
                                    env.state['structure'][:,-1] = 0
                                    env.invalidate_cache()
                            if tick == 20:
                                mask=np.arange(37)%2==0
                                cpu.reset(mask); gpu.reset(torch.from_numpy(mask).cuda())
                            if tick == 60:
                                actions[:,0] = np.resize(np.array([-1,-2,-9,-(2**63),2**63-1,99999],np.int64),37)
                            if tick % 7 == 0:
                                # Public diagnostics must never become a new input.
                                gpu.stats.fill_(-123)
                            cpu.step(actions);gpu.step(torch.from_numpy(actions).cuda())
                            for name,value in cpu.state.items():
                                np.testing.assert_array_equal(value,gpu.state[name].cpu().numpy(),err_msg=f'{tick} {name}')
                            for name in fields:
                                np.testing.assert_array_equal(getattr(cpu.unwrapped,name),getattr(gpu.unwrapped,name).cpu().numpy(),err_msg=f'{tick} {name}')
                        gpu.check_errors()
                    finally:gpu.close();cpu.close()


if __name__ == '__main__': unittest.main()
