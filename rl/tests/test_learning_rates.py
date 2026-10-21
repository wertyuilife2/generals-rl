"""Actor-KL scheduling, independent critic LR, and checkpoint resume contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from generals_rl.train import train
from rl_test_utils import assert_nested_equal
from test_student_ppo import algorithm, collect, observation, student_config


def distributed_fixed_critic_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(52 + rank)
        obs = observation()
        alg = algorithm(obs, learning_rate=1e-4, adaptive_lr_max=1e-3, critic_learning_rate=1e-4,
                        multi_gpu_cfg={"global_rank": rank, "world_size": 2})
        alg.broadcast_parameters()
        for local_kl, expected in ((rank * .1, 1e-5), (1e-6 * (rank + 1), 1e-3)):
            for _ in range(30):
                alg._adapt_learning_rate(torch.tensor(local_kl))
            assert alg.learning_rate == expected
            assert alg.optimizer.param_groups[0]["lr"] == expected
            assert alg.optimizer.param_groups[1]["lr"] == 1e-4
        alg.restore_learning_rates()
        collect(alg, obs)
        alg.update()
        flat = torch.cat([parameter.detach().flatten() for parameter in alg.policy.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        assert alg.optimizer.param_groups[1]["lr"] == 1e-4
        assert alg.update_metrics["critic_learning_rate"] == 1e-4
    finally:
        dist.destroy_process_group()


class LearningRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_one_group_preserves_original_parameters_and_explicit_default_settings(self):
        torch.manual_seed(97)
        obs = observation()
        original = algorithm(obs)
        explicit = algorithm(obs, adaptive_lr_min=1e-5, adaptive_lr_max=1e-2,
                             critic_learning_rate=None)
        explicit.policy.load_state_dict(original.policy.state_dict())
        self.assertEqual(len(original.optimizer.param_groups), 1)
        self.assertFalse(original._custom_learning_rates)
        rng = torch.get_rng_state()
        collect(original, obs)
        torch.set_rng_state(rng)
        collect(explicit, obs)
        rng = torch.get_rng_state()
        losses = original.update()
        final_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        self.assertEqual(losses, explicit.update())
        assert_nested_equal(self, original.policy.state_dict(), explicit.policy.state_dict())
        assert_nested_equal(self, original.optimizer.state_dict(), explicit.optimizer.state_dict())
        torch.testing.assert_close(torch.get_rng_state(), final_rng, rtol=0, atol=0)

    def test_fixed_critic_groups_cover_parameters_once_and_actor_kl_does_not_change_critic_lr(self):
        for critic_lr in (None, 1e-4):
            with self.subTest(critic_lr=critic_lr):
                alg = algorithm(observation(), learning_rate=1e-4,
                                adaptive_lr_min=1e-5, adaptive_lr_max=1e-3,
                                critic_learning_rate=critic_lr)
                owned = [id(parameter) for group in alg.optimizer.param_groups for parameter in group["params"]]
                self.assertEqual(len(owned), len(set(owned)))
                self.assertEqual(set(owned), {id(parameter) for parameter in alg.policy.parameters()
                                             if parameter.requires_grad})
                if critic_lr is not None:
                    self.assertEqual({id(parameter) for parameter in alg.optimizer.param_groups[1]["params"]},
                                     {id(parameter) for parameter in alg.policy.critic.parameters()})
                for kl, expected in ((.00001, 1e-3), (.1, 1e-5)):
                    for _ in range(30):
                        alg._adapt_learning_rate(torch.tensor(kl))
                    self.assertEqual(alg.learning_rate, expected)
                    self.assertEqual(alg.optimizer.param_groups[0]["lr"], expected)
                    if critic_lr is not None:
                        self.assertEqual(alg.optimizer.param_groups[1]["lr"], 1e-4)
                for kl in (0., .02):
                    alg._adapt_learning_rate(torch.tensor(kl))
                    self.assertEqual(alg.learning_rate, 1e-5)

    def test_full_update_uses_custom_bounds_and_reports_both_actual_lrs(self):
        for critic_lr in (None, 1e-4):
            for old_probability, initial_lr, expected_lr in (
                (.8, 1e-5, 1e-5), (.2 * math.exp(-.01), .001, .001),
                (.8, .001, .001 / 1.5 ** 2),
            ):
                with self.subTest(critic_lr=critic_lr, old_probability=old_probability, initial_lr=initial_lr):
                    obs = observation()
                    alg = algorithm(obs, learning_rate=initial_lr, adaptive_lr_max=.001,
                                    critic_learning_rate=critic_lr)
                    with torch.no_grad():
                        for parameter in alg.policy.actor.parameters():
                            parameter.zero_()
                            parameter.requires_grad_(False)
                    collect(alg, obs)
                    alg.storage.actions.zero_()
                    alg.storage.actions_log_prob.fill_(math.log(old_probability))
                    alg.update()
                    self.assertAlmostEqual(alg.learning_rate, expected_lr)
                    self.assertEqual(alg.update_metrics["actor_learning_rate"], alg.learning_rate)
                    self.assertEqual(alg.update_metrics["critic_learning_rate"],
                                     alg.learning_rate if critic_lr is None else critic_lr)

    def test_equal_fixed_rates_match_single_group_one_chunk_adam_and_rng(self):
        torch.manual_seed(151)
        obs = observation()
        original = algorithm(obs, physical_minibatch_size=32, schedule="fixed", desired_kl=None)
        candidate = algorithm(obs, critic_learning_rate=3e-4, schedule="fixed", desired_kl=None)
        candidate.policy.load_state_dict(original.policy.state_dict())
        rng = torch.get_rng_state()
        collect(original, obs)
        torch.set_rng_state(rng)
        collect(candidate, obs)
        rng = torch.get_rng_state()
        losses = original.update()
        final_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        self.assertEqual(losses, candidate.update())
        assert_nested_equal(self, original.policy.state_dict(), candidate.policy.state_dict())
        for left, right in zip(original.policy.parameters(), candidate.policy.parameters()):
            assert_nested_equal(self, original.optimizer.state.get(left), candidate.optimizer.state.get(right))
        torch.testing.assert_close(torch.get_rng_state(), final_rng, rtol=0, atol=0)

    def test_loaded_adam_restores_scheduler_and_rejects_changed_lr_semantics(self):
        obs = observation()
        source = algorithm(obs, learning_rate=1e-4, adaptive_lr_max=1e-3, critic_learning_rate=1e-4)
        collect(source, obs)
        source.update()
        source._adapt_learning_rate(torch.tensor(.1))
        candidate = algorithm(obs, learning_rate=1e-4, adaptive_lr_max=1e-3, critic_learning_rate=1e-4)
        candidate.optimizer.load_state_dict(deepcopy(source.optimizer.state_dict()))
        candidate.restore_learning_rates()
        self.assertEqual(candidate.learning_rate, source.learning_rate)
        self.assertEqual(candidate.optimizer.param_groups[1]["lr"], 1e-4)
        changed = algorithm(obs, critic_learning_rate=2e-4)
        changed.optimizer.load_state_dict(deepcopy(source.optimizer.state_dict()))
        with self.assertRaisesRegex(ValueError, "fixed critic LR differs"):
            changed.restore_learning_rates()
        candidate.optimizer.param_groups[0]["lr"] = .01
        with self.assertRaisesRegex(ValueError, "outside configured"):
            candidate.restore_learning_rates()

    def test_real_training_resume_retains_actual_actor_and_fixed_critic_lrs(self):
        config = student_config()
        config["runner"]["algorithm"].update(schedule="adaptive", desired_kl=.01,
                                             learning_rate=1e-4, adaptive_lr_max=.001,
                                             critic_learning_rate=1e-4,
                                             student_encoder_learning_rate=None)
        config["evaluation"] = {"enabled": False}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            first, resumed = Path(directory) / "first", Path(directory) / "resumed"
            train(config, first, iterations=2)
            saved = torch.load(first / "checkpoint.pt", map_location="cpu", weights_only=False)
            expected_lr = saved["optimizer_state_dict"]["param_groups"][0]["lr"]
            observed = []
            from generals_rl.student_ppo import StudentPPO
            original_restore = StudentPPO.restore_learning_rates

            def restore(algorithm):
                original_restore(algorithm)
                observed.append(algorithm.learning_rate)

            with patch.object(StudentPPO, "restore_learning_rates", restore):
                train(config, resumed, iterations=1, resume=str(first / "checkpoint.pt"))
            self.assertEqual(observed, [expected_lr])
            final = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
            groups = final["optimizer_state_dict"]["param_groups"]
            self.assertEqual([group["role"] for group in groups], ["actor", "critic"])
            self.assertEqual(groups[1]["lr"], 1e-4)
            events = EventAccumulator(str(resumed)).Reload()
            self.assertEqual([event.step for event in events.Scalars("PPO/actor_learning_rate")], [2])
            self.assertAlmostEqual(events.Scalars("PPO/critic_learning_rate")[0].value, 1e-4, places=10)

    def test_invalid_bounds_or_fixed_rate_fail_before_training(self):
        cases = ({"adaptive_lr_min": None}, {"adaptive_lr_max": float("inf")},
                 {"adaptive_lr_min": True}, {"adaptive_lr_max": 0},
                 {"critic_learning_rate": False}, {"critic_learning_rate": float("nan")},
                 {"adaptive_lr_min": .01, "adaptive_lr_max": .001},
                 {"adaptive_lr_max": 1e-4})
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                algorithm(observation(), **options)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is unavailable")
    def test_two_ranks_share_actor_schedule_and_keep_critic_lr_fixed(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}):
            mp.spawn(distributed_fixed_critic_worker,
                     args=(str(Path(directory) / "process_group"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
