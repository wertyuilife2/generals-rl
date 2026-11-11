"""Actual PPO rollout boundaries, TB values, and curriculum resume contracts."""
from dataclasses import replace
from types import SimpleNamespace
import torch
from generals_rl.curriculum import CurriculumConfig, RolloutCurriculum
from generals_rl.mdp.rewards import RewardConfig, ShapedReward
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
        config = json.loads((Path(__file__).parents[1] / "configs/v32.json").read_text())
        config.pop("curriculum", None)
        config["task"].update(num_envs=8, backend="cpu", device="cpu",
                              max_episode_steps=8, map_pool_size=0)
        config["runner"]["num_steps_per_env"] = 4
        config["runner"]["algorithm"].update(num_learning_epochs=1, num_mini_batches=2)
        config["self_play"].update(pool_size=3, snapshot_interval=1)
        config["evaluation"] = {"enabled": False}
        config["training_control"] = {}
        return config

    def test_real_rollouts_returns_and_updates_share_settings_and_resume_absolute_step(self):
        for tick_end_iteration in (None, 4):
            with self.subTest(tick_end_iteration=tick_end_iteration):
                self._assert_real_rollouts_and_resume(tick_end_iteration)

    def test_ablation_gamma099_to0999_preserves_rollout_boundaries_after_resume(self):
        self._assert_real_rollouts_and_resume(None, initial_gamma=.99)

    def _assert_real_rollouts_and_resume(self, tick_end_iteration, initial_gamma=.995):
        config = self.config()
        config["runner"]["algorithm"]["gamma"] = initial_gamma
        config["task"]["reward"]["potential_gamma"] = initial_gamma
        config["curriculum"] = {"tick_decay": True, "gamma_final": .999,
                                "gamma_end_iteration": 2}
        if tick_end_iteration is not None:
            config["curriculum"]["tick_end_iteration"] = tick_end_iteration
        endpoint = 1000 if tick_end_iteration is None else tick_end_iteration
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
                        tick_scale = max(0., 1 - value.step / endpoint)
                        expected = {"gamma": initial_gamma + (.999 - initial_gamma) * min(value.step / 2, 1),
                                    "tick_scale": tick_scale,
                                    "tick": -.001 * tick_scale}[tag]
                        self.assertAlmostEqual(value.value, expected, places=7)
        for stage, steps in zip(records, (range(3), range(3, 5)), strict=True):
            expected = [(.999 if step >= 2 else initial_gamma + (.999 - initial_gamma) * step / 2,
                         -.001 * max(0., 1 - step / endpoint)) for step in steps]
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
            config["curriculum"] = {"tick_end_iteration": 4}
            disabled = Path(directory) / "disabled"
            train(config, disabled, iterations=1)
            tags = EventAccumulator(str(disabled)).Reload().Tags()["scalars"]
            self.assertFalse(any(tag.startswith("Curriculum/") for tag in tags))
            for index, settings in enumerate(({"gamma_final": .999}, {"tick_end_iteration": 0})):
                config["curriculum"] = settings
                invalid = Path(directory) / f"invalid_{index}"
                with self.assertRaises(ValueError):
                    train(config, invalid, iterations=1)
                self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()


def test_potential_and_tick_decay_reconstruct_from_original_coefficients():
    base = RewardConfig(territory=.1, cities=.2, army=.01, tick=-.01, potential_gamma=.99)
    cfg = CurriculumConfig(tick_decay=True, potential_decay=True, tick_end_iteration=4,
                           gamma_final=1., gamma_end_iteration=2)
    reward = ShapedReward(base)
    alg = SimpleNamespace(gamma=.99)
    course = RolloutCurriculum(cfg, alg, reward)
    for i in (0, 2, 4, 8, 2):
        metrics = course.apply(i)
        scale = max(0., 1-i/4)
        assert metrics['potential_scale'] == scale
        assert reward.config == replace(base, tick=base.tick*scale, territory=base.territory*scale,
                                         cities=base.cities*scale, army=base.army*scale,
                                         potential_gamma=.99+.01*min(i/2, 1))
        assert reward.config.win == 1. and reward.config.loss == -1.
        assert alg.gamma == reward.config.potential_gamma



def test_sparse_reward_at_gamma_one_has_only_terminal_outcomes():
    reward = ShapedReward(RewardConfig(territory=0., cities=0., army=0., tick=0., potential_gamma=1.))
    done = torch.tensor([False, True, True, False])
    alive = torch.tensor([True, True, False, True])
    result = reward.evaluate(torch.rand(4, 3)*100, torch.rand(4, 3)*100, done,
        torch.tensor([False, False, False, True]), torch.tensor([-1, 0, 1, -1]), alive,
        torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(result.total, torch.tensor([0., 1., -1., 0.]), rtol=0, atol=0)
