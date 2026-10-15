"""The native Metal simulator composes with the actual two-stage learner."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from generals_rl.train import train
from generals_rl.task import TaskConfig
from rl_test_utils import small_training_config


class MPSConfigTests(unittest.TestCase):
    def test_mps_backend_requires_matching_device(self):
        with self.assertRaises(ValueError): TaskConfig(backend='mps', device='cpu')
        with self.assertRaises(ValueError): TaskConfig(backend='mps', device='cuda')
        TaskConfig(backend='gpu', device='mps')
        TaskConfig(backend='mps', device='mps')


@unittest.skipUnless(torch.backends.mps.is_available(), 'requires Apple GPU')
class MPSTrainingTests(unittest.TestCase):
    def test_native_mps_self_play_and_two_stage_updates(self):
        config = small_training_config(task_overrides=dict(backend='mps', device='mps'))
        config['evaluation'] = {'enabled': False}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            summary = train(config, directory, iterations=2)
            self.assertEqual(summary['environment_steps'], 32)
            self.assertGreater(summary['truncated_episodes'], 0)
            self.assertTrue((Path(directory)/'checkpoint.pt').is_file())
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in summary['last_losses'].values()))


if __name__ == '__main__': unittest.main()
