"""CPU contracts for absolute-iteration, tick-only and PBRS gamma courses."""
from dataclasses import asdict
import random
from types import SimpleNamespace
import unittest

import torch

from generals_rl.curriculum import CurriculumConfig, RolloutCurriculum
from generals_rl.mdp.rewards import RewardConfig, ShapedReward


def reward_config():
    return RewardConfig(territory=.1, cities=.5, army=.01, win=3., loss=-4.,
                        tick=-.02, potential_gamma=.995)


class CurriculumTests(unittest.TestCase):
    def make(self, **settings):
        algorithm = SimpleNamespace(gamma=.995)
        reward = ShapedReward(reward_config())
        course = RolloutCurriculum(CurriculumConfig(**settings), algorithm, reward)
        return course, algorithm, reward

    def test_defaults_do_not_touch_reward_algorithm_or_rng(self):
        course, algorithm, reward = self.make()
        original = reward.config
        before_torch, before_python = torch.get_rng_state(), random.getstate()
        for iteration in (0, 500, 1000, 2500):
            self.assertEqual(course.apply(iteration), {})
            self.assertIs(reward.config, original)
            self.assertEqual(algorithm.gamma, .995)
        torch.testing.assert_close(before_torch, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(before_python, random.getstate())
        # Disabled integration is also valid for custom reward/algorithm types;
        # it must not even require accessing their gamma/config attributes.
        class Untouched:
            def __getattr__(self, name):
                raise AssertionError(f"disabled curriculum accessed {name}")
        self.assertEqual(RolloutCurriculum(CurriculumConfig(), Untouched(), Untouched()).apply(0), {})
        self.assertEqual(RolloutCurriculum(CurriculumConfig(tick_end_iteration=4),
                                          Untouched(), Untouched()).apply(4), {})

    def test_default_tick_schedule_matches_legacy_at_every_iteration(self):
        course, _, reward = self.make(tick_decay=True)
        self.assertEqual(course.settings.tick_end_iteration, 1000)
        for iteration in range(2001):
            scale = max(0.0, 1.0 - iteration / 1000)
            self.assertEqual(course.apply(iteration)["tick_scale"], scale)
            self.assertEqual(reward.config.tick, -.02 * scale)

    def test_custom_tick_endpoint_is_independent_and_reconstructs_on_resume(self):
        settings = dict(tick_decay=True, tick_end_iteration=4,
                        gamma_final=.999, gamma_end_iteration=8)
        course, algorithm, reward = self.make(**settings)
        for iteration, scale in ((0, 1.), (2, .5), (3, .25), (4, 0.),
                                 (9, 0.), (2, .5)):
            with self.subTest(iteration=iteration):
                metrics = course.apply(iteration)
                self.assertEqual(metrics["tick_scale"], scale)
                self.assertEqual(reward.config.tick, -.02 * scale)
                self.assertEqual(algorithm.gamma, .995 + .004 * min(iteration / 8, 1))
                resumed, resumed_algorithm, resumed_reward = self.make(**settings)
                self.assertEqual(metrics, resumed.apply(iteration))
                self.assertEqual(algorithm.gamma, resumed_algorithm.gamma)
                self.assertEqual(reward.config, resumed_reward.config)
        self.assertEqual(course.base_reward.tick, -.02)

    def test_tick_endpoint_requires_a_positive_integer_even_when_disabled(self):
        for enabled in (False, True):
            for endpoint in (0, -1, True, False, 1., "1000", None, float("inf")):
                with self.subTest(enabled=enabled, endpoint=endpoint), \
                        self.assertRaisesRegex(ValueError, "tick_end_iteration must be a positive integer"):
                    CurriculumConfig(tick_decay=enabled, tick_end_iteration=endpoint)

    def test_tick_boundaries_preserve_all_pbrs_and_terminal_coefficients(self):
        course, algorithm, reward = self.make(tick_decay=True)
        original = reward.config
        for iteration, scale in ((0, 1.), (500, .5), (1000, 0.), (2500, 0.)):
            with self.subTest(iteration=iteration):
                metrics = course.apply(iteration)
                self.assertEqual(metrics["tick_scale"], scale)
                self.assertEqual(reward.config.tick, original.tick * scale)
                self.assertEqual(algorithm.gamma, .995)
                for key in ("territory", "cities", "army", "win", "loss", "potential_gamma"):
                    self.assertEqual(getattr(reward.config, key), getattr(original, key))
        self.assertEqual(asdict(original), asdict(reward_config()))

    def test_gamma_boundaries_are_synchronized_and_tick_ramp_is_independent(self):
        course, algorithm, reward = self.make(gamma_final=.999, gamma_end_iteration=2000,
                                               tick_decay=True)
        for iteration, gamma, tick_scale in ((0, .995, 1.), (500, .996, .5),
                                            (1000, .997, 0.), (2000, .999, 0.), (9000, .999, 0.)):
            with self.subTest(iteration=iteration):
                metrics = course.apply(iteration)
                self.assertAlmostEqual(algorithm.gamma, gamma)
                self.assertEqual(reward.config.potential_gamma, algorithm.gamma)
                self.assertEqual(metrics["gamma"], algorithm.gamma)
                self.assertEqual(metrics["tick_scale"], tick_scale)
                self.assertEqual(reward.config.tick, -.02 * tick_scale)

    def test_repeated_or_reconstructed_resume_uses_original_not_scaled_coefficients(self):
        settings = dict(tick_decay=True, gamma_final=.999, gamma_end_iteration=2000)
        course, algorithm, reward = self.make(**settings)
        for iteration in (0, 500, 500, 1000, 2000, 500, 1500):
            metrics = course.apply(iteration)
            resumed, resumed_algorithm, resumed_reward = self.make(**settings)
            self.assertEqual(metrics, resumed.apply(iteration))
            self.assertEqual(algorithm.gamma, resumed_algorithm.gamma)
            self.assertEqual(reward.config, resumed_reward.config)
        self.assertEqual(course.base_reward.tick, -.02)
        self.assertEqual(course.base_reward.potential_gamma, .995)

    def test_terminal_and_timeout_pbrs_use_scheduled_gamma_without_decaying_win_loss(self):
        course, algorithm, reward = self.make(tick_decay=True, gamma_final=.999,
                                               gamma_end_iteration=1000)
        before = torch.tensor([[10., 2., 1.]]).expand(4, 3)
        after = torch.tensor([[20., 3., 2.]]).expand(4, 3)
        terminated = torch.tensor([False, True, True, False])
        truncated = torch.tensor([False, False, False, True])
        alive = torch.tensor([True, True, False, True])
        for iteration in (0, 500, 1000, 2000):
            with self.subTest(iteration=iteration):
                metrics = course.apply(iteration)
                result = reward.evaluate(before, after, terminated, truncated,
                                         torch.tensor([-1, 0, 1, -1]), alive,
                                         torch.zeros(4, dtype=torch.long))
                phi_before, phi_after = reward.potential(before).flatten(), reward.potential(after).flatten()
                shaping = sum(result.terms[key] for key in ("territory", "cities", "army"))
                expected = algorithm.gamma * phi_after.masked_fill(terminated | ~alive, 0) - phi_before
                torch.testing.assert_close(shaping, expected)
                torch.testing.assert_close(result.terms["terminal"], torch.tensor([0., 3., -4., 0.]))
                torch.testing.assert_close(result.terms["tick"], torch.full((4,), metrics["tick"]))
                # Timeout keeps Phi(final), and the matching shaped-value
                # bootstrap cancels it. A true terminal has no bootstrap.
                base_value_after = 7.
                timeout_return = result.total[3] + algorithm.gamma * (base_value_after - phi_after[3])
                torch.testing.assert_close(timeout_return,
                                           metrics["tick"] - phi_before[3] + algorithm.gamma * base_value_after)
                self.assertAlmostEqual(float(result.total[1]), 3. - float(phi_before[1]) + metrics["tick"], places=6)
                self.assertAlmostEqual(float(result.total[2]), -4. - float(phi_before[2]) + metrics["tick"], places=6)

    def test_gamma_only_does_not_scale_tick_or_other_original_coefficients(self):
        course, algorithm, reward = self.make(gamma_final=.999, gamma_end_iteration=1000)
        course.apply(1500)
        expected = asdict(reward_config())
        expected["potential_gamma"] = .999
        self.assertEqual(asdict(reward.config), expected)
        self.assertEqual(algorithm.gamma, .999)

    def test_ablation_gamma_course_reaches_final_at_update112_and_preserves_it_through224(self):
        settings = CurriculumConfig(gamma_final=.999, gamma_end_iteration=112)
        algorithm = SimpleNamespace(gamma=.99)
        reward = ShapedReward(RewardConfig(potential_gamma=.99))
        course = RolloutCurriculum(settings, algorithm, reward)
        for iteration in range(224):
            metrics = course.apply(iteration)
            expected = .999 if iteration >= 112 else .99 + .009 * iteration / 112
            self.assertAlmostEqual(metrics["gamma"], expected, places=14)
            self.assertEqual(algorithm.gamma, reward.config.potential_gamma)
            self.assertEqual(reward.config.tick, course.base_reward.tick)
            # Resume reconstruction uses configured initial gamma, not the
            # previously scheduled checkpoint discount.
            resumed_algorithm = SimpleNamespace(gamma=.99)
            resumed_reward = ShapedReward(RewardConfig(potential_gamma=.99))
            resumed = RolloutCurriculum(settings, resumed_algorithm, resumed_reward)
            self.assertEqual(resumed.apply(iteration), metrics)

    def test_enabled_requires_original_matching_potential_gamma(self):
        settings = CurriculumConfig(tick_decay=True)
        for potential in (None, .99):
            with self.subTest(potential_gamma=potential), self.assertRaisesRegex(ValueError, "initial PPO gamma"):
                RolloutCurriculum(settings, SimpleNamespace(gamma=.995),
                                  ShapedReward(RewardConfig(potential_gamma=potential)))
        for gamma in (True, None, float("nan"), 1.1):
            with self.subTest(gamma=gamma), self.assertRaisesRegex(ValueError, "algorithm.gamma"):
                RolloutCurriculum(settings, SimpleNamespace(gamma=gamma), ShapedReward(reward_config()))
        with self.assertRaisesRegex(TypeError, "ShapedReward"):
            RolloutCurriculum(settings, SimpleNamespace(gamma=.995), object())

    def test_invalid_schedule_and_absolute_iterations_fail(self):
        for options in ({"tick_decay": 1}, {"gamma_final": .999}, {"gamma_end_iteration": 1000},
                        {"gamma_final": True, "gamma_end_iteration": 1000},
                        {"gamma_final": float("inf"), "gamma_end_iteration": 1000},
                        {"gamma_final": 1.1, "gamma_end_iteration": 1000},
                        {"gamma_final": .999, "gamma_end_iteration": 0},
                        {"gamma_final": .999, "gamma_end_iteration": True}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                CurriculumConfig(**options)
        course, _, _ = self.make(tick_decay=True)
        for iteration in (-1, True, 500., "500"):
            with self.subTest(iteration=iteration), self.assertRaisesRegex(ValueError, "absolute TB iteration"):
                course.apply(iteration)


if __name__ == "__main__":
    unittest.main()
