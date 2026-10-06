"""Actual PPO rollout boundaries, TB values, and curriculum resume contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from generals_rl.self_play_training import prepare_self_play
from generals_rl.train import train


class CurriculumTrainingTests(unittest.TestCase):
    def config(self):
        config = json.loads((Path(__file__).parents[1] / "configs/v22.json").read_text())
        config["task"].update(num_envs=8, backend="cpu", device="cpu",
                              max_episode_steps=8, map_pool_size=0)
        config["runner"]["num_steps_per_env"] = 4
        config["runner"]["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
        config["self_play"].update(pool_size=3, snapshot_interval=1)
        config["evaluation"] = {"enabled": False}
        config["training_control"] = {}
        return config

    def test_real_rollouts_returns_and_updates_share_settings_and_resume_absolute_step(self):
        config = self.config()
        config["curriculum"] = {"tick_decay": True, "gamma_final": .999,
                                "gamma_end_iteration": 2}
        records = []

        def capture(task, runner, *args, **kwargs):
            algorithm = runner.alg
            stage = {"act": [], "returns": [], "update": []}
            records.append(stage)
            coefficients = deepcopy(task.reward_model.config)
            for name, method_name in (("act", "act"), ("returns", "compute_returns"),
                                      ("update", "update")):
                original = getattr(algorithm, method_name)

                def observed(*call_args, _name=name, _original=original, **call_kwargs):
                    reward = task.reward_model.config
                    self.assertEqual(algorithm.gamma, reward.potential_gamma)
                    for key in ("territory", "cities", "army", "win", "loss"):
                        self.assertEqual(getattr(reward, key), getattr(coefficients, key))
                    stage[_name].append((algorithm.gamma, reward.tick))
                    return _original(*call_args, **call_kwargs)

                setattr(algorithm, method_name, observed)
            return prepare_self_play(task, runner, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch("generals_rl.train.prepare_self_play", capture):
            first = Path(directory) / "first"
            train(config, first, iterations=3)
            resumed = Path(directory) / "resumed"
            train(config, resumed, iterations=2, resume=str(first / "checkpoint.pt"))
            for directory, steps in ((first, range(3)), (resumed, range(3, 5))):
                events = EventAccumulator(str(directory)).Reload()
                for tag in ("gamma", "tick_scale", "tick"):
                    values = events.Scalars(f"Curriculum/{tag}")
                    self.assertEqual([value.step for value in values], list(steps))
                    for value in values:
                        expected = {"gamma": .995 + .004 * min(value.step / 2, 1),
                                    "tick_scale": 1 - value.step / 1000,
                                    "tick": -.001 * (1 - value.step / 1000)}[tag]
                        self.assertAlmostEqual(value.value, expected, places=7)
        for stage, steps in zip(records, (range(3), range(3, 5)), strict=True):
            expected = [(.999 if step >= 2 else .995 + .004 * step / 2,
                         -.001 * (1 - step / 1000)) for step in steps]
            self.assertEqual(stage["returns"], expected)
            self.assertEqual(stage["update"], expected)
            self.assertEqual(stage["act"], [value for value in expected for _ in range(4)])

    def test_default_does_not_emit_curriculum_metrics_and_invalid_config_creates_no_run(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "default"
            train(config, output, iterations=1)
            tags = EventAccumulator(str(output)).Reload().Tags()["scalars"]
            self.assertFalse(any(tag.startswith("Curriculum/") for tag in tags))
            config["curriculum"] = {"gamma_final": .999}
            invalid = Path(directory) / "invalid"
            with self.assertRaises(ValueError):
                train(config, invalid, iterations=1)
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
