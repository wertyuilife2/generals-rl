"""Public modes, dependency boundaries and direct hot-path binding."""
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from generals import BatchStep, MapConfig, VecEnv, make_env
from generals.protocol.vector import VectorBackend
from generals.runtime.play import PlaySession


class PublicAPITests(unittest.TestCase):
    def test_lazy_imports_and_cpu_creation_do_not_load_optional_or_benchmark_modules(self):
        code = '''
import sys
from generals import MapConfig, make_env, VecEnv
assert not any(n.startswith(('torch', 'tkinter', 'generals.core')) for n in sys.modules)
with make_env(MapConfig(15,15,2), batch_size=2, threads=1) as env:
    assert isinstance(env, VecEnv)
    assert not any(n.startswith(('torch', 'tkinter', 'generals.benchmarks')) for n in sys.modules)
    assert not any(n.startswith('generals.benchmarks') for n in sys.modules)
session = make_env(MapConfig(15,15,2), mode='realtime')
assert 'tkinter' not in sys.modules
'''
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
        result = subprocess.run([sys.executable, '-c', code], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_modes_devices_and_cross_mode_options_are_rejected(self):
        config = MapConfig(15, 15, 2)
        for options in ({'mode': 'unknown'}, {'device': 'gpu'}, {'device': 'cuda:no'},
                        {'mode': 'realtime', 'device': 'cuda'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_env(config, **options)
        with self.assertRaises(TypeError):
            make_env(config, mode='realtime', batch_size=4)
        with self.assertRaises(TypeError):
            make_env('not a config')

    def test_realtime_session_delegates_to_existing_launch_without_changing_arguments(self):
        fake = types.ModuleType('generals.gui.app')
        with patch.object(fake, 'launch', create=True, return_value='closed') as launch:
            with patch.dict(sys.modules, {'generals.gui.app': fake}):
                config = MapConfig(15, 15, 2)
                for watch in (False, True):
                    session = make_env(config, mode='realtime', watch=watch, seed=81,
                                       human_seat=1, visibility='full', kinds=['random','defensive'], speed=5)
                    self.assertIsInstance(session, PlaySession)
                    self.assertEqual(session.run(), 'closed')
                    launch.assert_called_with(config, seed=81, mode='watch' if watch else 'play',
                                              human_seat=1, visibility='full',
                                              kinds=('random', 'defensive'), speed=5)

    def test_cli_play_watch_use_realtime_mode(self):
        from generals.cli import main
        with patch('generals.api.make_env') as factory:
            for command in ('play', 'watch'):
                self.assertEqual(main([command, '--players', '2', '--size', '15']), 0)
                self.assertEqual(factory.call_args.kwargs['mode'], 'realtime')
                self.assertEqual(factory.call_args.kwargs['watch'], command == 'watch')
                factory.return_value.run.assert_called()

    def exercise_vector(self, device):
        config = MapConfig(15, 15, 2)
        options = {'threads': 1} if device == 'cpu' else {'kernel_backend': 'cuda'}
        if device == 'cpu':
            action = np.zeros((3,2), np.int64)
            mask = np.array([True,False,True])
            host = lambda v: v
        else:
            import torch
            action = torch.zeros((3,2), dtype=torch.int64, device=device)
            mask = torch.tensor([True,False,True], device=device)
            host = lambda v: v.cpu().numpy()
        with make_env(config, device=device, batch_size=3, auto_reset=True,
                      episode_ticks=2, seed=42, **options) as env:
            self.assertIsInstance(env, VecEnv)
            self.assertIsInstance(env.unwrapped, VectorBackend)
            self.assertIsInstance(env, VectorBackend)
            self.assertEqual(env.num_envs, 3)
            for name in ('initial', 'counters', 'rng_state', 'errors', 'last_moves'):
                self.assertFalse(hasattr(env, name))
                self.assertIsNotNone(getattr(env.unwrapped, name))
            if device != 'cpu':
                self.assertEqual(env.unwrapped.step_metadata()['step_implementation'], 'cuda')
            for name in ('step','synchronize','reset','check_errors','invalidate_cache','close'):
                bound = getattr(env,name)
                self.assertIs(bound.__self__, env.unwrapped)
                self.assertIs(bound.__func__, getattr(type(env.unwrapped),name))
            first = env.step(action)
            env.synchronize()
            self.assertIsInstance(first, BatchStep)
            self.assertIs(first.stats, env.stats)
            self.assertIs(env.state, env.unwrapped.state)
            result = env.step(action)
            env.synchronize()
            np.testing.assert_array_equal(host(result.done), [[False,True]]*3)
            np.testing.assert_array_equal(host(env.state['tick']), [0,0,0])
            env.step(action)
            env.reset(mask)
            env.synchronize()
            np.testing.assert_array_equal(host(env.state['tick']), [0,1,0])
            saved = env.to_states()
            with VecEnv.from_states(saved, device=device, **options) as restored:
                restored.synchronize()
                for name, value in env.state.items():
                    np.testing.assert_array_equal(host(value), host(restored.state[name]), err_msg=name)
            with VecEnv(env.unwrapped.initial, width=env.width, height=env.height, device=device, **options) as raw:
                raw.step(action)
                raw.synchronize()
                np.testing.assert_array_equal(host(raw.state['tick']), [1,1,1])
            env.check_errors()
        with self.assertRaises(RuntimeError):
            env.step(action)

    @unittest.skipUnless(sys.platform in ('linux', 'darwin'), 'native OpenMP backend needs Linux/macOS')
    def test_cpu_vector_contract_and_lifecycle(self):
        self.exercise_vector('cpu')

    def test_cuda_vector_contract_and_lifecycle(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch dependency is unavailable')
        if not torch.cuda.is_available():
            self.skipTest('CUDA device is unavailable')
        self.exercise_vector('cuda')


if __name__ == '__main__':
    unittest.main()
