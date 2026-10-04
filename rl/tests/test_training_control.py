"""Stopping at update boundaries and retaining usable final/best checkpoints."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from generals_rl.train import train
from generals_rl.training_control import TrainingControlConfig, TrainingController
from rl_test_utils import small_training_config


class TrainingControlTests(unittest.TestCase):
    def test_plateau_needs_successful_points_minimum_time_and_meaningful_progress(self):
        controller = TrainingController(TrainingControlConfig(
            min_seconds=10, max_seconds=100, plateau_evaluations=3, min_improvement=0.02))
        point = {"selection_score": [0.4, 0.4, 0.6]}
        for _ in range(3):
            controller.on_evaluation(point)
        self.assertIsNone(controller.stop_reason(20))
        controller.on_evaluation(point)
        self.assertIsNone(controller.stop_reason(9))
        self.assertIsNone(controller.stop_reason(20, evaluation_healthy=False))
        self.assertEqual(controller.stop_reason(20), "evaluation_plateau")
        controller.on_evaluation({"selection_score": [0.4, 0.45, 0.6]})
        self.assertIsNone(controller.stop_reason(20))
        self.assertEqual(controller.stop_reason(100, evaluation_healthy=False), "max_seconds")

    def test_timed_stop_saves_a_complete_update_and_does_not_require_the_full_tick_budget(self):
        config = small_training_config()
        config["evaluation"] = {"enabled": False}
        config["training_control"] = {"max_seconds": 1e-9, "keep_checkpoints": 2}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            summary = train(config, output, steps_per_env=40)
            checkpoint = torch.load(output / "checkpoint.pt", weights_only=False, map_location="cpu")
            self.assertEqual(summary["stop_reason"], "max_seconds")
            self.assertEqual(summary["steps_per_environment"], 4)
            self.assertEqual(summary["iterations"], 1)
            self.assertEqual(checkpoint["iter"], 0)
            for key in ("optimizer_state_dict", "optimizer_stu_enc_state_dict"):
                self.assertEqual({int(state["step"]) for state in checkpoint[key]["state"].values()}, {4})
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)

    def test_runner_checkpoints_have_a_fixed_total_count(self):
        config = small_training_config()
        config["runner"]["save_interval"] = 1
        config["evaluation"] = {"enabled": False}
        config["training_control"] = {"keep_checkpoints": 2}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            train(config, output, iterations=5)
            self.assertEqual({path.name for path in output.glob("*.pt")}, {"model_3.pt", "checkpoint.pt"})

    def test_invalid_control_settings_fail_before_creating_output(self):
        cases = ({"max_seconds": 0}, {"max_seconds": float("nan")}, {"min_seconds": -1},
                 {"min_seconds": 2, "max_seconds": 1}, {"plateau_evaluations": True},
                 {"keep_checkpoints": 0}, {"min_improvement": -0.1})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            for settings in cases:
                config = small_training_config()
                config["training_control"] = settings
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    train(config, output, iterations=1)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
