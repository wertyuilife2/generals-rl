"""Logical PPO batch equivalence, RNG ordering and microbatch error guards."""
import math
import unittest
from unittest.mock import patch

import torch

from generals_rl.student_ppo import StudentPPO
from rl_test_utils import assert_nested_equal
from test_student_ppo import algorithm, collect, observation, policy
from rsl_rl.storage import RolloutStorageCTS


class StudentMicrobatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _pair(self, *, per_minibatch=False, schedule="fixed", physical=6,
              learning_epochs=2):
        torch.manual_seed(773)
        obs = observation(17)
        options = dict(normalize_advantage_per_mini_batch=per_minibatch,
                       schedule=schedule, desired_kl=.02 if schedule == "adaptive" else None,
                       ppo_probe_samples=13, ppo_probe_batch_size=5)
        reference = algorithm(obs, **options)
        candidate = algorithm(obs, physical_minibatch_size=physical, **options)
        candidate.policy.load_state_dict(reference.policy.state_dict())
        for alg in (reference, candidate):
            alg.num_learning_epochs = learning_epochs
            # 17 environments * 2 active ticks / 2 batches = logical batch 17;
            # physical size 6 produces unequal 6/6/5 chunks.
            alg.storage.num_transitions_per_env = 2
        rng = torch.get_rng_state()
        collect(reference, obs)
        torch.set_rng_state(rng)
        collect(candidate, obs)
        for name in ("actions", "values", "actions_log_prob", "returns", "advantages"):
            torch.testing.assert_close(getattr(reference.storage, name), getattr(candidate.storage, name),
                                       rtol=0, atol=0)
        return reference, candidate

    def _compare_updates(self, reference, candidate):
        before = torch.get_rng_state()
        reference_losses = reference.update()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        with (patch("generals_rl.student_ppo.torch.nn.utils.clip_grad_norm_",
                    wraps=torch.nn.utils.clip_grad_norm_) as clipping,
              patch.object(candidate.optimizer, "step", wraps=candidate.optimizer.step) as optimizer_steps,
              patch.object(candidate.monitor, "minibatch", wraps=candidate.monitor.minibatch) as monitoring):
            candidate_losses = candidate.update()
        torch.testing.assert_close(torch.get_rng_state(), after, rtol=0, atol=0)
        for key in reference_losses:
            self.assertAlmostEqual(reference_losses[key], candidate_losses[key], delta=2e-6)
        self.assertEqual(reference.learning_rate, candidate.learning_rate)
        for name, tensor in reference.policy.state_dict().items():
            torch.testing.assert_close(tensor, candidate.policy.state_dict()[name],
                                       rtol=2e-5, atol=2e-7)
        left, right = reference.optimizer.state_dict(), candidate.optimizer.state_dict()
        self.assertEqual(left["param_groups"], right["param_groups"])
        self.assertEqual(left["state"].keys(), right["state"].keys())
        for index in left["state"]:
            for key in left["state"][index]:
                torch.testing.assert_close(left["state"][index][key], right["state"][index][key],
                                           rtol=3e-5, atol=2e-7)
        expected_updates = reference.num_learning_epochs * reference.num_mini_batches
        self.assertEqual(clipping.call_count, expected_updates)
        self.assertEqual(optimizer_steps.call_count, expected_updates)
        self.assertEqual(monitoring.call_count, expected_updates)
        self.assertTrue(all(call.args[1] == candidate.max_grad_norm for call in clipping.call_args_list))
        self.assertEqual({int(value["step"]) for value in right["state"].values()},
                         {expected_updates})
        self.assertEqual(candidate.update_metrics["minibatches"], expected_updates)
        self.assertEqual(candidate.storage.step, 0)
        for key in ("approx_kl", "clip_fraction", "entropy", "grad_norm", "exact_update_kl",
                    "wait_move_gate_kl", "weighted_move_kl"):
            self.assertAlmostEqual(reference.update_metrics[key], candidate.update_metrics[key], delta=2e-5)
        self.assertTrue(all(math.isfinite(value) for value in candidate.update_metrics.values()))
        self.assertIsNone(candidate.monitor.observations)

    def test_unequal_chunks_preserve_losses_adam_rng_and_normalization_scope(self):
        for per_minibatch in (False, True):
            with self.subTest(per_minibatch=per_minibatch):
                reference, candidate = self._pair(per_minibatch=per_minibatch)
                self._compare_updates(reference, candidate)
                self.assertEqual(candidate.update_metrics["logical_minibatch_size"], 17)
                self.assertEqual(candidate.update_metrics["physical_minibatch_size"], 6)
                self.assertEqual(candidate.update_metrics["microbatches"], 12)

    def test_physical_single_sample_and_larger_than_logical_batch(self):
        for physical in (1, 1000):
            with self.subTest(physical=physical):
                reference, candidate = self._pair(physical=physical, learning_epochs=1)
                self._compare_updates(reference, candidate)
                self.assertEqual(candidate.update_metrics["microbatches"], 34 if physical == 1 else 2)

    def test_clipping_applies_to_accumulated_logical_gradient_once(self):
        reference, candidate = self._pair()
        reference.max_grad_norm = candidate.max_grad_norm = .01
        self._compare_updates(reference, candidate)
        for alg in (reference, candidate):
            norm = torch.linalg.vector_norm(torch.cat(
                [parameter.grad.flatten() for parameter in alg.policy.parameters()
                 if parameter.grad is not None]))
            self.assertLessEqual(float(norm), .0100001)

    def test_unclipped_value_loss_and_disabled_monitor(self):
        reference, candidate = self._pair()
        for alg in (reference, candidate):
            alg.use_clipped_value_loss = False
            alg.monitor.enabled = False
        before = torch.get_rng_state()
        losses = reference.update()
        torch.set_rng_state(before)
        chunk_losses = candidate.update()
        for key in losses:
            self.assertAlmostEqual(losses[key], chunk_losses[key], delta=2e-6)
        self.assertNotIn("approx_kl", candidate.update_metrics)
        self.assertEqual(candidate.update_metrics["microbatches"], 12)

    def test_adaptive_kl_uses_one_complete_logical_batch_and_matches_bounds(self):
        for probability in (.8, .2 * math.exp(-.01), .2):
            with self.subTest(probability=probability):
                reference, candidate = self._pair(schedule="adaptive", learning_epochs=1)
                for alg in (reference, candidate):
                    # Uniform frozen actor allows an exact, controlled k3 value.
                    with torch.no_grad():
                        for parameter in alg.policy.actor.parameters():
                            parameter.zero_()
                            parameter.requires_grad_(False)
                    alg.storage.actions.zero_()
                    alg.storage.actions_log_prob.fill_(math.log(probability))
                self._compare_updates(reference, candidate)
                multiplier = 1/1.5 if probability == .8 else 1.5 if probability != .2 else 1
                self.assertAlmostEqual(candidate.learning_rate, 3e-4 * multiplier**2, places=10)
        # Deliberately mix large and zero divergences so physical chunk KLs
        # differ: adaptation must still equal the unchunked logical mean.
        reference, candidate = self._pair(schedule="adaptive", learning_epochs=1)
        for alg in (reference, candidate):
            with torch.no_grad():
                for parameter in alg.policy.actor.parameters():
                    parameter.zero_()
                    parameter.requires_grad_(False)
            alg.storage.actions.zero_()
            alg.storage.actions_log_prob.fill_(math.log(.2))
            alg.storage.actions_log_prob[:, ::4] = math.log(.8)
        self._compare_updates(reference, candidate)

        for probability, initial_lr, expected_lr in ((.8, 1.1e-5, 1e-5),
                                                     (.2 * math.exp(-.01), 9e-3, 1e-2)):
            with self.subTest(initial_lr=initial_lr):
                reference, candidate = self._pair(schedule="adaptive", learning_epochs=1)
                for alg in (reference, candidate):
                    with torch.no_grad():
                        for parameter in alg.policy.actor.parameters():
                            parameter.zero_()
                            parameter.requires_grad_(False)
                    alg.learning_rate = initial_lr
                    for group in alg.optimizer.param_groups:
                        group["lr"] = initial_lr
                    alg.storage.actions.zero_()
                    alg.storage.actions_log_prob.fill_(math.log(probability))
                self._compare_updates(reference, candidate)
                self.assertEqual(candidate.learning_rate, expected_lr)

    def test_policy_amp_flag_must_match_without_changing_legacy_default_update(self):
        torch.manual_seed(841)
        obs = observation()
        reference = algorithm(obs, schedule="fixed", desired_kl=None)
        candidate_policy = policy(obs)
        candidate_policy.amp_bfloat16 = False
        candidate_policy.load_state_dict(reference.policy.state_dict())
        storage = RolloutStorageCTS("rl", len(obs), 0, 4, obs, [1], distribution_shape=[5],
                                    action_dtype=torch.long, store_action_std=False,
                                    store_action_mean=False)
        candidate = StudentPPO(candidate_policy, storage, len(obs),
                               num_learning_epochs=1, num_mini_batches=2,
                               schedule="fixed", desired_kl=None, amp_bfloat16=False)
        self.assertFalse(candidate._chunked_update)
        before = torch.get_rng_state()
        collect(reference, obs)
        torch.set_rng_state(before)
        collect(candidate, obs)
        before = torch.get_rng_state()
        losses = reference.update()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        self.assertEqual(losses, candidate.update())
        torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
        assert_nested_equal(self, candidate.policy.state_dict(), reference.policy.state_dict())
        assert_nested_equal(self, candidate.optimizer.state_dict(), reference.optimizer.state_dict())
        self.assertNotIn("microbatches", candidate.update_metrics)
        for policy_amp, update_amp in ((False, True), (True, False)):
            candidate_policy.amp_bfloat16 = policy_amp
            with self.assertRaisesRegex(ValueError, "amp_bfloat16 must match"):
                StudentPPO(candidate_policy, storage, len(obs), amp_bfloat16=update_amp)
        candidate_policy.amp_bfloat16 = True
        with self.assertRaisesRegex(ValueError, "CUDA"):
            StudentPPO(candidate_policy, storage, len(obs), amp_bfloat16=True)

    def test_disabled_options_delegate_to_upstream_without_extra_metrics(self):
        reference, candidate = self._pair()
        candidate.physical_minibatch_size = None
        candidate._chunked_update = False
        before = torch.get_rng_state()
        losses = reference.update()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        self.assertEqual(losses, candidate.update())
        torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
        assert_nested_equal(self, reference.policy.state_dict(), candidate.policy.state_dict())
        assert_nested_equal(self, reference.optimizer.state_dict(), candidate.optimizer.state_dict())
        self.assertNotIn("microbatches", candidate.update_metrics)

    def test_invalid_chunk_amp_and_unsupported_execution_fail_before_training(self):
        obs = observation()
        for size in (True, 0, -1, 3.0, "4"):
            with self.assertRaisesRegex(ValueError, "physical_minibatch_size"):
                algorithm(obs, physical_minibatch_size=size)
        with self.assertRaisesRegex(TypeError, "amp_bfloat16"):
            algorithm(obs, amp_bfloat16=1)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            algorithm(obs, physical_minibatch_size=2, amp_bfloat16=True)
        with self.assertRaisesRegex(ValueError, "single-device"):
            algorithm(obs, physical_minibatch_size=2, multi_gpu_cfg={"global_rank": 0, "world_size": 1})
        with self.assertRaisesRegex(ValueError, "RND or symmetry"):
            algorithm(obs, physical_minibatch_size=2, symmetry_cfg={})
        recurrent = policy(obs)
        recurrent.is_recurrent = True
        storage = RolloutStorageCTS("rl", len(obs), 0, 4, obs, [1])
        with self.assertRaisesRegex(ValueError, "feedforward"):
            StudentPPO(recurrent, storage, len(obs), physical_minibatch_size=2)
        for unsupported in (torch.nn.BatchNorm1d(16), torch.nn.Dropout(.1)):
            invalid = policy(obs)
            invalid.actor.add_module("unsupported", unsupported)
            with self.assertRaisesRegex(ValueError, "sample-independent"):
                StudentPPO(invalid, storage, len(obs), physical_minibatch_size=2)


if __name__ == "__main__":
    unittest.main()
