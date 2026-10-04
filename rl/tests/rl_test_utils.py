"""Small fixtures shared by the training integration tests."""
from dataclasses import asdict

import torch

from generals_rl.cts import runner_config
from generals_rl.task import TaskConfig


def small_training_config(*, task_overrides=None, runner_overrides=None):
    task = dict(width=3, height=3, num_envs=4, max_episode_steps=3,
                history_length=2, opponent="self_play", seed=19,
                mountain_density=0, city_density=0, threads=1)
    task.update(task_overrides or {})
    runner = dict(rollout_steps=4, learning_epochs=2, mini_batches=2,
                  hidden_dim=16, expert_num=2)
    runner.update(runner_overrides or {})
    return {"task": asdict(TaskConfig(**task)),
            "runner": runner_config(**runner), "torch_threads": 1}


def assert_nested_equal(test, actual, expected):
    """Compare full optimizer/model states, including non-tensor metadata."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_nested_equal(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            assert_nested_equal(test, a, e)
    else:
        test.assertEqual(actual, expected)
