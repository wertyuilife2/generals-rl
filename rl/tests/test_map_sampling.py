"""Map bank reset/storage contracts; CUDA reset operations run on CPU tensors."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from generals import Action, MapConfig, make_env
from generals.backends.gpu.env import GPUEnv
from generals.core.engine import CoreEngine
from generals.core.recording import state_digest
from generals_rl.map_sampling import MapSampler
from generals_rl.task import GeneralsTask, TaskConfig


class MapSamplingTests(unittest.TestCase):
    config = MapConfig(18, 18, 2, .1, .05)

    def env(self):
        env = make_env(self.config, batch_size=4, threads=1, auto_reset=False, episode_ticks=2)
        self.addCleanup(env.close)
        return env

    def test_option_disabled_allocates_no_pool_and_unsupported_caches_fail_before_generation(self):
        with patch("generals_rl.map_sampling._build_bank") as build:
            with GeneralsTask(TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, num_envs=4, opponent="wait")) as task:
                self.assertIsNone(task.map_sampler)
            build.assert_not_called()
            env = self.env()
            unsupported = SimpleNamespace(**env.unwrapped.__dict__)
            del unsupported._initial_stats
            with self.assertRaisesRegex(RuntimeError, "reset caches"):
                MapSampler(unsupported, self.config, 32, seed=19)
            incompatible = SimpleNamespace(**env.unwrapped.__dict__)
            incompatible._initial_stats = None
            with self.assertRaisesRegex(RuntimeError, "cache structure"):
                MapSampler(incompatible, self.config, 32, seed=19)
            build.assert_not_called()
        for invalid in (-1, True, 1.5):
            with self.subTest(size=invalid), self.assertRaises(ValueError):
                TaskConfig(width=9, height=9, random_maps=False, map_pool_size=invalid)

    def test_cpu_ended_lanes_preserve_native_bindings_and_next_tick_matches_core(self):
        env = self.env()
        sampler = MapSampler(env, self.config, 64, seed=19)
        env.state["tick"][[0, 2]] = 1
        env.step(np.zeros((4, 2), dtype=np.int64))
        mask = env.done.any(axis=1).copy()
        np.testing.assert_array_equal(mask, [True, False, True, False])
        native = env.unwrapped
        refs = list(native.state.values()) + list(native.initial.values()) + [
            native.stats, native._initial_stats, native.rng_state, native._initial_rng,
            native.last_moves, native.counters, native.done, native._cache_valid]
        pointers = [(id(value), value.ctypes.data) for value in refs]
        before = [value.copy() for value in refs]
        self.assertIs(sampler.reset(mask), env.state)
        self.assertEqual(pointers, [(id(value), value.ctypes.data) for value in refs])
        for value, previous in zip(refs, before):
            np.testing.assert_array_equal(value[~mask], previous[~mask])
        self.assertTrue((env.state["tick"][mask] == 0).all())
        engines = [CoreEngine(state) for state in env.to_states(validate=True)]
        env.step(np.zeros((4, 2), dtype=np.int64))
        for state, engine in zip(env.to_states(validate=True), engines):
            engine.step([Action.wait(), Action.wait()])
            self.assertEqual(state_digest(state), state_digest(engine.state))

    def test_private_rng_repeatability_empty_mask_and_new_version_seed(self):
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = deepcopy(np.random.get_state())
        first, second = self.env(), self.env()
        a, b = MapSampler(first, self.config, 64, seed=19), MapSampler(second, self.config, 64, seed=19)
        a.reset(np.zeros(4, dtype=bool))
        a.reset()
        b.reset()
        for name in first.state:
            np.testing.assert_array_equal(first.state[name], second.state[name])
        changed = MapSampler(self.env(), self.config, 64, seed=20)
        self.assertFalse(np.array_equal(a.arrays["general_pos"], changed.arrays["general_pos"]))
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        current = np.random.get_state()
        np.testing.assert_array_equal(current[1], numpy_rng[1])
        self.assertEqual(current[0], numpy_rng[0])
        self.assertEqual(current[2:], numpy_rng[2:])

    def test_tensor_reset_preserves_other_lanes_and_uint32_rng_bits(self):
        cpu = self.env().unwrapped
        native = GPUEnv.__new__(GPUEnv)  # No CUDA construction/launch.
        native.width, native.height, native.players, native.batch_size = 18, 18, 2, 4
        native.device, native.auto_reset, native._closed, native._step_kernel = torch.device("cpu"), False, False, None
        native.initial = {name: torch.from_numpy(value.copy()) for name, value in cpu.initial.items()}
        native.state = {name: value.clone() for name, value in native.initial.items()}
        for name in ("stats", "_initial_stats", "rng_state", "_initial_rng", "counters", "last_moves", "errors", "done"):
            setattr(native, name, torch.from_numpy(getattr(cpu, name).copy()))
        sampler = MapSampler(native, self.config, 32, seed=19)
        sampler.rng.fill_(-1)
        refs = list(native.state.values()) + list(native.initial.values()) + [native.stats, native.rng_state]
        pointers = [(id(value), value.data_ptr()) for value in refs]
        before = [value.numpy().copy() for value in refs]
        mask = torch.tensor([True, False, True, False])
        sampler.reset(mask)
        self.assertEqual(pointers, [(id(value), value.data_ptr()) for value in refs])
        for value, previous in zip(refs, before):
            np.testing.assert_array_equal(value.numpy()[~mask.numpy()], previous[~mask.numpy()])
        np.testing.assert_array_equal(native.rng_state.numpy()[mask.numpy()], [0xffffffff, 0xffffffff])

    def test_task_initial_history_and_partial_terminal_reset_use_sampled_maps(self):
        settings = TaskConfig(random_maps=False, width=18, height=18, num_envs=4, max_episode_steps=2,
                              history_length=3, opponent="wait", map_pool_size=64, seed=19)
        with GeneralsTask(settings) as task:
            initial = task.get_observations()["single_obs"]
            torch.testing.assert_close(task.history.frames, initial[:, None].expand(-1, 3, -1))
            task.backend.state["tick"][[0, 2]] = 1
            seats = task.player_ids.clone()
            outcome = task.step(torch.zeros((4, 1), dtype=torch.long))
            done = outcome.truncated
            torch.testing.assert_close(done, torch.tensor([True, False, True, False]))
            self.assertTrue((task.episode_length_buf[done] == 0).all())
            self.assertTrue((task.episode_length_buf[~done] == 1).all())
            self.assertTrue((task.player_ids[done] == 1 - seats[done]).all())
            terminal_history = outcome.terminal_observations["policy"].reshape(4, 3, -1)
            torch.testing.assert_close(task.history.frames[~done], terminal_history[~done])
            fresh = task.get_observations()["single_obs"]
            torch.testing.assert_close(task.history.frames[done], fresh[done, None].expand(-1, 3, -1))


if __name__ == "__main__":
    unittest.main()
