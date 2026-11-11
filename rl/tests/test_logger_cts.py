"""Episode metrics can first appear partway through a collected rollout."""
from unittest.mock import Mock

import pytest
import torch

from rsl_rl.utils.logger_cts import LoggerCTS


def test_episode_metrics_after_first_step_are_logged(tmp_path):
    logger = LoggerCTS(None, {"num_steps_per_env": 2, "algorithm": {"rnd_cfg": None}}, {},
                       2, False, 1, 0, torch.empty(0, dtype=torch.long), "cpu")
    logger.log_dir = str(tmp_path)
    logger.logger_type = "tensorboard"
    logger.writer = Mock()
    logger.cur_episode_length[0] = 69
    logger.process_env_step(torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                            {"log": {"Reward/tick": torch.full((2,), -.001)}})
    logger.process_env_step(torch.zeros(2), torch.tensor([True, False]),
                            {"log": {"Reward/tick": torch.full((2,), -.001),
                                     "Episode/length": torch.tensor([71.]),
                                     "Episode/win": torch.tensor([0.]),
                                     "Episode/truncated": torch.tensor([0.])}})
    logger.log(0, 0, 2, .01, .01, {}, .001, torch.ones(1), None)
    scalars = {(call.args[0], call.args[2]): float(call.args[1])
               for call in logger.writer.add_scalar.call_args_list}
    assert scalars["Episode/length", 0] == 71.
    assert scalars["Episode/win", 0] == scalars["Episode/truncated", 0] == 0.
    assert scalars["Train/mean_student_episode_length", 0] == 71.
    assert scalars["Reward/tick", 0] == pytest.approx(-.001)
    assert not logger.ep_extras

    logger.writer.reset_mock()
    logger.process_env_step(torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                            {"log": {"Reward/tick": torch.full((2,), -.001)}})
    logger.log(1, 0, 2, .01, .01, {}, .001, torch.ones(1), None)
    assert not any(call.args[0].startswith("Episode/")
                   for call in logger.writer.add_scalar.call_args_list)
