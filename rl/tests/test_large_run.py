"""Training budgets and final checkpoints; storage details are tested in vendor."""
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rsl_rl.algorithms import MoECTS

from generals_rl.train import train
from rl_test_utils import small_training_config


class ExactBudgetTests(unittest.TestCase):
    def run_budget(self, *, steps_per_env=None, iterations=None):
        config = small_training_config(task_overrides={"max_episode_steps": 8, "opponent": "random"})
        update_lengths = []
        original_update = MoECTS.update

        def update(algorithm):
            self.assertEqual(algorithm.storage.step, algorithm.storage.num_transitions_per_env)
            update_lengths.append(algorithm.storage.num_transitions_per_env)
            return original_update(algorithm)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            with patch.object(MoECTS, "update", update), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = train(config, output, iterations=iterations, steps_per_env=steps_per_env)
            checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        # Two epochs and two minibatches advance both Adam optimizers four times
        # per update, including a shorter final rollout.
        for name in ("optimizer_state_dict", "optimizer_stu_enc_state_dict"):
            states = checkpoint[name]["state"]
            self.assertTrue(states)
            self.assertEqual({int(state["step"]) for state in states.values()}, {4 * len(update_lengths)})
        self.assertEqual(checkpoint["iter"], len(update_lengths) - 1)
        for key in ("environment_transitions_per_second", "wall_environment_transitions_per_second"):
            value = summary["performance"][key]
            self.assertTrue(math.isfinite(value) and value > 0)
        return summary, update_lengths

    def test_tick_and_iteration_budgets(self):
        cases = (({"steps_per_env": 10}, [4, 4, 2]),
                 ({"steps_per_env": 2}, [2]),
                 ({"iterations": 1}, [4]))
        for budget, expected_lengths in cases:
            with self.subTest(budget=budget):
                summary, lengths = self.run_budget(**budget)
                self.assertEqual(lengths, expected_lengths)
                self.assertEqual(summary["iterations"], len(expected_lengths))
                self.assertEqual(summary["steps_per_environment"], sum(expected_lengths))
                self.assertEqual(summary["environment_steps"], 4 * sum(expected_lengths))

    def test_budget_arguments_are_positive_and_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            for budget in ({}, {"iterations": 1, "steps_per_env": 4},
                           {"iterations": 0}, {"steps_per_env": 0}):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    train(small_training_config(), output, **budget)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
