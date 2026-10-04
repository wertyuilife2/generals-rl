"""Student task gradients and the unchanged upstream PPO update/lifecycle."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from generals.controllers import AI_KINDS
from generals_rl.self_play_training import prepare_self_play
from generals_rl.student_ppo import StudentPPO, StudentPPOPolicy
from generals_rl.train import train
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS
from rl_test_utils import assert_nested_equal, small_training_config


def observation(count=8):
    return TensorDict({"policy": torch.randn(count, 12), "single_obs": torch.randn(count, 6),
                       "critic": torch.randn(count, 6),
                       "action_mask": torch.ones(count, 5, dtype=torch.bool)}, batch_size=[count])


def policy(obs, norm="l2norm", normalization=(False, False)):
    with redirect_stdout(io.StringIO()):
        return StudentPPOPolicy(obs, {"policy": ["policy"], "critic": ["critic"]}, 5,
                                action_distribution="categorical", latent_dim=8, expert_num=2,
                                actor_hidden_dims=[16], critic_hidden_dims=[16],
                                teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16],
                                actor_obs_normalization=normalization[0],
                                critic_obs_normalization=normalization[1],
                                norm_type=norm)


def algorithm(obs, norm="l2norm", normalization=(False, False), **kwargs):
    storage = RolloutStorageCTS("rl", len(obs), 0, 4, obs, [1], distribution_shape=[5],
                                action_dtype=torch.long, store_action_std=False,
                                store_action_mean=False)
    return StudentPPO(policy(obs, norm, normalization), storage, len(obs),
                      num_learning_epochs=1, num_mini_batches=2, **kwargs)


def collect(alg, obs):
    with torch.inference_mode():
        for step in range(alg.storage.num_transitions_per_env):
            alg.act(obs)
            reward = torch.arange(len(obs), dtype=torch.float32) / 10 + step / 5
            done = torch.zeros(len(obs), dtype=torch.bool)
            alg.process_env_step(obs, reward, done, {"time_outs": done, "terminal_observations": obs})
        alg.compute_returns(obs)


def student_config():
    config = small_training_config(runner_overrides={"learning_epochs": 1})
    config["runner"]["policy"]["class_name"] = "generals_rl.student_ppo.StudentPPOPolicy"
    config["runner"]["algorithm"].update(class_name="generals_rl.student_ppo.StudentPPO",
                                             teacher_env_ratio=0, schedule="fixed", desired_kl=None,
                                             load_balance_coef=0)
    config["self_play"] = {"pool_size": 3, "snapshot_interval": 1}
    # Omit enabled: exercise the default real subprocess evaluation path.
    config["evaluation"] = {"every_n_snapshots": 2, "maps_per_ai": 1, "max_ticks": 3, "batch_size": 8}
    return config


class StudentPPOTests(unittest.TestCase):
    def test_local_student_gets_task_gradients_in_one_optimizer_and_critic_latent_stays_detached(self):
        torch.manual_seed(41)
        obs = observation()
        alg = algorithm(obs)
        self.assertIsNone(alg.storage.mu)
        self.assertFalse(StudentPPO.requires_old_distribution)
        priors = alg.policy.action_probabilities(obs).detach().clone()
        changed = obs.clone()
        changed["critic"] += 100
        torch.testing.assert_close(priors, alg.policy.action_probabilities(changed), rtol=0, atol=0)
        self.assertFalse(torch.equal(alg.policy.evaluate(obs), alg.policy.evaluate(changed)))
        alg.policy.evaluate(obs).sum().backward()
        self.assertTrue(all(p.grad is None for p in alg.policy.student_moe_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in alg.policy.actor.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in alg.policy.critic.parameters()))
        alg.policy.zero_grad(set_to_none=True)
        collect(alg, obs)
        before = deepcopy(alg.policy.state_dict())
        alg.update()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in alg.policy.student_moe_encoder.parameters()))
        for prefix in ("student_moe_encoder.", "actor.", "critic."):
            self.assertTrue(any(not torch.equal(before[name], value)
                                for name, value in alg.policy.named_parameters() if name.startswith(prefix)))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in alg.policy.teacher_encoder.parameters()))
        for name, value in alg.policy.teacher_encoder.state_dict().items():
            torch.testing.assert_close(value, before["teacher_encoder." + name], rtol=0, atol=0)
        owned = [id(p) for group in alg.optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(set(owned), {id(p) for p in alg.policy.parameters() if p.requires_grad})
        self.assertIsNone(alg.optimizer_stu_enc)
        self.assertEqual((alg.teacher_num_envs, alg.student_num_envs), (0, len(obs)))

    def test_update_matches_original_ppo_complete_parameters_adam_and_rng(self):
        for normalization in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(normalization=normalization):
                torch.manual_seed(123)
                obs = observation()
                candidate = algorithm(obs, normalization=normalization)
                reference_policy = policy(obs, normalization=normalization)
                reference_policy.load_state_dict(candidate.policy.state_dict())
                # Original encoder-with-grad then latent.detach value path.
                reference_policy.evaluate = MethodType(
                    lambda self, inputs, **kw: ActorCriticMoECTS.evaluate(self, inputs, False, **kw),
                    reference_policy)
                cached = RolloutStorageCTS("rl", len(obs), 0, 4, obs, [1], distribution_shape=[5],
                                          action_dtype=torch.long, store_action_std=False)
                reference = PPO(reference_policy, cached, num_learning_epochs=1,
                                num_mini_batches=2, schedule="fixed", desired_kl=None, learning_rate=3e-4)
                reference.optimizer = torch.optim.Adam(
                    (p for p in reference_policy.parameters() if p.requires_grad), lr=3e-4)
                rng = torch.get_rng_state()
                collect(candidate, obs)
                torch.set_rng_state(rng)
                collect(reference, obs)
                self.assertIsNone(candidate.storage.mu)
                self.assertEqual(cached.mu.shape, (4, len(obs), 5))
                for field in ("actions", "rewards", "values", "returns", "advantages", "actions_log_prob"):
                    torch.testing.assert_close(getattr(candidate.storage, field), getattr(cached, field), rtol=0, atol=0)
                rng = torch.get_rng_state()
                losses = candidate.update()
                final_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                self.assertEqual(losses, reference.update())
                torch.testing.assert_close(final_rng, torch.get_rng_state(), rtol=0, atol=0)
                assert_nested_equal(self, candidate.policy.state_dict(), reference_policy.state_dict())
                assert_nested_equal(self, candidate.optimizer.state_dict(), reference.optimizer.state_dict())
                for left, right in zip(candidate.policy.parameters(), reference_policy.parameters()):
                    assert_nested_equal(self, left.grad, right.grad)
        self.assertIs(StudentPPO.update, PPO.update)

    def test_timeout_and_partial_prefix_gae_ignore_unused_nan_tail(self):
        obs = observation(4)
        obs["critic"][:, 0] = torch.arange(1., 5.)
        alg = algorithm(obs)
        alg.storage.num_transitions_per_env = 2
        def values(self, inputs, is_teacher=False, **kwargs):
            return inputs["critic"][:, :1]
        alg.policy.evaluate = MethodType(values, alg.policy)
        terminal = obs.clone()
        terminal["critic"][:, 0] *= 2
        for step in range(2):
            alg.act(obs)
            timeout = torch.tensor([False, step == 0, False, False])
            done = torch.tensor([False, step == 0, step == 0, False])
            alg.process_env_step(obs, torch.ones(4), done,
                                 {"time_outs": timeout, "terminal_observations": terminal})
        torch.testing.assert_close(alg.storage.rewards[0, :, 0],
                                   torch.tensor([1., 1. + alg.gamma * 4, 1., 1.]))
        for field in (alg.storage.rewards, alg.storage.values, alg.storage.returns):
            field[2:] = float("nan")
        alg.compute_returns(obs)
        reward, value = alg.storage.rewards[:2], alg.storage.values[:2]
        alive = 1 - alg.storage.dones[:2].float()
        next_value = torch.cat((value[1:], obs["critic"][:, :1].unsqueeze(0)))
        residual = reward + alg.gamma * alive * next_value - value
        advantage = torch.stack((residual[0] + alg.gamma * alg.lam * alive[0] * residual[1], residual[1]))
        torch.testing.assert_close(alg.storage.returns[:2], value + advantage)
        expected = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        torch.testing.assert_close(alg.storage.advantages, expected)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in alg.update().values()))

    def test_config_and_active_sample_guards_and_simnorm_empty_teacher(self):
        cases = ({"schedule": "adaptive"}, {"desired_kl": .02}, {"load_balance_coef": .01},
                 {"teacher_env_ratio": .75}, {"student_encoder_learning_rate": .001},
                 {"rnd_cfg": {}}, {"symmetry_cfg": {}})
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                algorithm(observation(), **options)
        for count, normalized in ((1, False), (2, True)):
            obs = observation(count)
            alg = algorithm(obs, normalize_advantage_per_mini_batch=normalized)
            alg.storage.num_transitions_per_env = 1
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "active PPO rollout"):
                alg.compute_returns(obs)
        obs = observation()
        alg = algorithm(obs, norm="simnorm")
        self.assertEqual(alg.policy.evaluate(obs[:0], is_teacher=True).shape, (0, 1))
        collect(alg, obs)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in alg.update().values()))
        alg.act(obs)
        with self.assertRaisesRegex(ValueError, "terminal_observations"):
            alg.process_env_step(obs, torch.ones(len(obs)), torch.ones(len(obs), dtype=torch.bool),
                                 {"time_outs": torch.ones(len(obs), dtype=torch.bool)})

    def test_real_self_play_cts_warm_start_resume_and_default_periodic_subprocess_eval(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = student_config()
            source_config = deepcopy(config)
            source_config["runner"]["policy"]["class_name"] = "ActorCriticMoECTS"
            source_config["runner"]["algorithm"].update(class_name="MoECTS", teacher_env_ratio=.75)
            train(source_config, root / "cts", iterations=1)
            source = torch.load(root / "cts/checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertTrue(source["optimizer_stu_enc_state_dict"]["state"])
            summary = train(config, root / "warm", steps_per_env=10, warm_start=root / "cts/checkpoint.pt")
            seed = torch.load(root / "warm/history_pool/snapshot_00000000.pt", map_location="cpu", weights_only=False)
            assert_nested_equal(self, seed["model_state_dict"], source["model_state_dict"])
            self.assertEqual(seed["source"]["kind"], "warm_start_weights")
            saved = torch.load(root / "warm/checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["iter"], 2)
            self.assertIsNone(saved["optimizer_stu_enc_state_dict"])
            self.assertEqual({int(state["step"]) for state in saved["optimizer_state_dict"]["state"].values()}, {6})
            self.assertEqual((summary["teacher_envs"], summary["student_envs"]), (0, 4))
            self.assertEqual(summary["environment_steps"], 40)
            observed = {}
            def capture(task, runner, *args, **kwargs):
                self.assertIsNone(runner.alg.storage.mu)
                observed["optimizer"] = deepcopy(runner.alg.optimizer.state_dict())
                observed["model"] = deepcopy(runner.alg.policy.state_dict())
                observed["iteration"] = runner.current_learning_iteration
                return prepare_self_play(task, runner, *args, **kwargs)
            with patch("generals_rl.train.prepare_self_play", capture):
                resumed = train(config, root / "resume", iterations=1,
                                resume=str(root / "warm/checkpoint.pt"))
            assert_nested_equal(self, observed["optimizer"], saved["optimizer_state_dict"])
            assert_nested_equal(self, observed["model"], saved["model_state_dict"])
            self.assertEqual(observed["iteration"], 3)
            last = torch.load(root / "resume/checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual({int(state["step"]) for state in last["optimizer_state_dict"]["state"].values()}, {8})
            for run, evidence in ((root / "warm", summary), (root / "resume", resumed)):
                self.assertEqual((evidence["evaluation"]["completed"], evidence["evaluation"]["failed"]), (1, 0))
                jobs = json.loads((run / "evaluations/index.json").read_text())["jobs"]
                self.assertEqual(jobs[0]["status"], "completed")
                report = json.loads(Path(jobs[0]["report"]).read_text())
                self.assertEqual((report["completed_games"], report["device"]), (8, "cpu"))
                self.assertEqual(set(report["by_ai"]), set(AI_KINDS))
                self.assertTrue(all(metrics["games"] == 2 for metrics in report["by_ai"].values()))
                self.assertEqual(report["policy"], "student_local")


if __name__ == "__main__":
    unittest.main()
