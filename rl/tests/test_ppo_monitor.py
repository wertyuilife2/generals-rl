"""Categorical KL math, LR response, bounded probes and monitoring isolation."""
from copy import deepcopy
import math
from types import MethodType
import unittest

import torch
from torch.distributions import Categorical, kl_divergence

from generals_rl.ppo_monitor import categorical_changes
from rl_test_utils import assert_nested_equal
from test_student_ppo import algorithm, collect, observation


class PPOMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_kl_estimator_expectation_and_exact_wait_move_decomposition(self):
        old = torch.tensor([[.6, .3, .1], [1., 0, 0], [.2, .8, 0]], dtype=torch.float64)
        new = torch.tensor([[.4, .1, .5], [1., 0, 0], [.3, .7, 0]], dtype=torch.float64)
        log_ratio = new.log() - old.log()
        estimate = torch.where(old > 0, old * (torch.expm1(log_ratio) - log_ratio), 0).sum(-1)
        exact = kl_divergence(Categorical(probs=old), Categorical(probs=new))
        torch.testing.assert_close(estimate, exact, rtol=1e-12, atol=1e-12)
        changes = categorical_changes(old.log(), new.log())
        torch.testing.assert_close(changes["exact_update_kl"], exact, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(changes["wait_move_gate_kl"] + changes["weighted_move_kl"], exact,
                                   rtol=1e-12, atol=1e-12)
        conditional = (.3 * math.log((.3 / .4) / (.1 / .6)) +
                       .1 * math.log((.1 / .4) / (.5 / .6)))
        self.assertAlmostEqual(changes["weighted_move_kl"][0].item(), conditional)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in changes.values()))
        self.assertEqual(changes["exact_update_kl"][1].item(), 0)
        identical = categorical_changes(old.log(), old.log())
        self.assertTrue(bool(identical["exact_update_kl"].eq(0).all()))
        extreme = categorical_changes(torch.tensor([[0., -1000., -torch.inf]]),
                                      torch.tensor([[-1000., 0., -torch.inf]]))
        self.assertAlmostEqual(extreme["exact_update_kl"].item(), 1000)

    def test_default_adaptive_and_lr_shrink_grow_bounds_without_mu_cache(self):
        for old_probability, initial_lr, expected_lr in (
            (.8, 3e-4, 3e-4 / 1.5 ** 2),
            (.2 * math.exp(-.01), 3e-4, 3e-4 * 1.5 ** 2),
            (.8, 1e-5, 1e-5),
            (.2 * math.exp(-.01), .01, .01),
            (.2, 3e-4, 3e-4),
        ):
            with self.subTest(old_probability=old_probability, lr=initial_lr):
                torch.manual_seed(83)
                obs = observation()
                alg = algorithm(obs, learning_rate=initial_lr)
                self.assertEqual((alg.schedule, alg.desired_kl, alg.monitor.enabled), ("adaptive", .02, True))
                # Keep the actor uniformly distributed; only the critic updates.
                with torch.no_grad():
                    for parameter in alg.policy.actor.parameters():
                        parameter.zero_()
                        parameter.requires_grad_(False)
                collect(alg, obs)
                alg.storage.actions.zero_()
                alg.storage.actions_log_prob.fill_(math.log(old_probability))
                alg.update()
                self.assertAlmostEqual(alg.learning_rate, expected_lr, places=10)
                self.assertEqual(alg.optimizer.param_groups[0]["lr"], alg.learning_rate)
                self.assertIsNone(alg.storage.mu)
                self.assertIsNone(alg.storage.sigma)
                self.assertAlmostEqual(alg.update_metrics["exact_update_kl"], 0, places=12)
                self.assertTrue(all(math.isfinite(value) for value in alg.update_metrics.values()))

    def test_monitor_on_off_matches_full_model_adam_gradients_and_rng(self):
        torch.manual_seed(419)
        obs = observation()
        enabled = algorithm(obs, schedule="fixed", desired_kl=None, ppo_probe_samples=7,
                            ppo_probe_batch_size=3)
        disabled = algorithm(obs, schedule="fixed", desired_kl=None, ppo_monitor=False)
        disabled.policy.load_state_dict(enabled.policy.state_dict())
        rng = torch.get_rng_state()
        collect(enabled, obs)
        torch.set_rng_state(rng)
        collect(disabled, obs)
        rng = torch.get_rng_state()
        rollout = {name: getattr(enabled.storage, name).clone()
                   for name in ("returns", "values", "advantages")}
        losses = enabled.update()
        final_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        self.assertEqual(losses, disabled.update())
        torch.testing.assert_close(final_rng, torch.get_rng_state(), rtol=0, atol=0)
        assert_nested_equal(self, enabled.policy.state_dict(), disabled.policy.state_dict())
        assert_nested_equal(self, enabled.optimizer.state_dict(), disabled.optimizer.state_dict())
        for left, right in zip(enabled.policy.parameters(), disabled.policy.parameters()):
            assert_nested_equal(self, left.grad, right.grad)
        self.assertEqual(disabled.update_metrics, {})
        self.assertEqual(enabled.update_metrics["probe_samples"], 7)
        self.assertEqual(enabled.update_metrics["minibatches"], 2)
        for name, value in rollout.items():
            assert_nested_equal(self, getattr(enabled.storage, name), value)
            assert_nested_equal(self, getattr(disabled.storage, name), value)
        self.assertAlmostEqual(enabled.update_metrics["returns_mean"], rollout["returns"].mean().item(), places=6)
        self.assertAlmostEqual(enabled.update_metrics["raw_advantage_std"],
                               (rollout["returns"] - rollout["values"]).std(correction=0).item(), places=6)
        self.assertIsNone(enabled.monitor.observations)
        self.assertIsNone(enabled.monitor.old_logits)
        self.assertIsNone(enabled.monitor.rollout_statistics)
        # Disabled diagnostics must not access even an absent storage field.
        disabled.monitor.begin(object())
        self.assertEqual(disabled.monitor.finish(object()), {})

    def test_probe_restores_distribution_rng_and_uses_only_filled_prefix(self):
        torch.manual_seed(13)
        obs = observation(4)
        # Mixture of forced WAIT and ordinary decisions.
        obs["action_mask"][0, 1:] = False
        alg = algorithm(obs, ppo_probe_samples=100, ppo_probe_batch_size=3)
        alg.storage.num_transitions_per_env = 2
        collect(alg, obs)
        for tensor in alg.storage.observations.values():
            if tensor.is_floating_point():
                tensor[2:] = float("nan")
        for tensor in (alg.storage.returns, alg.storage.values):
            tensor[2:] = float("nan")
        distribution = alg.policy.distribution
        state = deepcopy(alg.policy.state_dict())
        rng = torch.get_rng_state()
        alg.monitor.begin(alg)
        self.assertIs(alg.policy.distribution, distribution)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        assert_nested_equal(self, alg.policy.state_dict(), state)
        self.assertEqual(len(alg.monitor.observations), 8)
        self.assertTrue(bool(torch.isfinite(alg.monitor.old_logits).all()))
        self.assertNotIn("critic", alg.monitor.observations)
        alg.monitor.release()
        alg.update()
        self.assertEqual(alg.update_metrics["probe_samples"], 8)
        self.assertTrue(all(math.isfinite(value) for value in alg.update_metrics.values()))

    def test_rollout_statistics_match_analytic_population_moments(self):
        for per_minibatch in (False, True):
            with self.subTest(per_minibatch=per_minibatch):
                obs = observation(4)
                alg = algorithm(obs, schedule="fixed", desired_kl=None,
                                normalize_advantage_per_mini_batch=per_minibatch)
                alg.storage.num_transitions_per_env = 2
                collect(alg, obs)
                returns = torch.tensor([0., 1., 2., 3.]).repeat(2).reshape(2, 4, 1)
                values = torch.tensor([0., 0., 1., 1.]).repeat(2).reshape(2, 4, 1)
                raw = returns - values
                with torch.inference_mode():
                    alg.storage.returns[:2].copy_(returns)
                    alg.storage.values[:2].copy_(values)
                    alg.storage.advantages.copy_(raw if per_minibatch else
                                                 (raw - raw.mean()) / (raw.std() + 1e-8))
                alg.update()
                expected = {"returns_mean": 1.5, "returns_std": math.sqrt(1.25),
                            "raw_advantage_mean": 1., "raw_advantage_std": math.sqrt(.5),
                            "buffer_advantage_mean": 1. if per_minibatch else 0.,
                            "buffer_advantage_std": math.sqrt(.5 if per_minibatch else 7 / 8),
                            "critic_explained_variance_valid": 1., "critic_explained_variance": .6}
                for name, value in expected.items():
                    self.assertAlmostEqual(alg.update_metrics[name], value, places=6)

    def test_constant_return_omits_explained_variance_and_ignores_unused_tail(self):
        obs = observation(4)
        alg = algorithm(obs, schedule="fixed", desired_kl=None)
        alg.storage.num_transitions_per_env = 2
        collect(alg, obs)
        alg.storage.returns[:2].fill_(2)
        values = torch.tensor([-1., 0., 1., 2.]).repeat(2).reshape(2, 4, 1)
        alg.storage.values[:2].copy_(values)
        raw = 2 - values
        with torch.inference_mode():
            alg.storage.advantages.copy_((raw - raw.mean()) / (raw.std() + 1e-8))
        for tensor in (alg.storage.returns, alg.storage.values):
            tensor[2:] = float("nan")
        alg.update()
        self.assertEqual(alg.update_metrics["returns_mean"], 2)
        self.assertEqual(alg.update_metrics["returns_std"], 0)
        self.assertAlmostEqual(alg.update_metrics["raw_advantage_mean"], 1.5)
        self.assertAlmostEqual(alg.update_metrics["raw_advantage_std"], math.sqrt(1.25), places=6)
        self.assertEqual(alg.update_metrics["critic_explained_variance_valid"], 0)
        self.assertNotIn("critic_explained_variance", alg.update_metrics)
        self.assertTrue(all(math.isfinite(value) for value in alg.update_metrics.values()))

    def test_probe_rng_is_preserved_even_with_stochastic_forward(self):
        obs = observation()
        alg = algorithm(obs)
        collect(alg, obs)
        original = alg.policy.action_probabilities
        def random_forward(self, inputs, **kwargs):
            torch.rand(11)  # A future stochastic actor must not consume rollout RNG.
            return original(inputs, **kwargs)
        alg.policy.action_probabilities = MethodType(random_forward, alg.policy)
        rng = torch.get_rng_state()
        alg.monitor.begin(alg)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        alg.monitor.release()


if __name__ == "__main__":
    unittest.main()
