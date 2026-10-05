"""Spatial privacy/layout/value semantics and the real training CLI lifecycle."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from generals_rl.config import merge_config
from generals_rl.cts import runner_config
from generals_rl.self_play_training import SelfPlayConfig
from generals_rl.spatial_ppo import SpatialPPOPolicy, policy_config
from generals_rl.task import TaskConfig
from generals_rl.train import main
from generals_rl.training_evaluation import TrainingEvalConfig
from rl_test_utils import assert_nested_equal


ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"policy": ["policy"], "critic": ["critic"]}


def observations(count=4):
    return TensorDict({"policy": torch.randn(count, 3940), "single_obs": torch.randn(count, 985),
                      "critic": torch.randn(count, 985), "critic_potential": torch.rand(count, 1),
                      "action_mask": torch.ones(count, 649, dtype=torch.bool)}, [count])


def construct(obs, offset=False):
    options = policy_config(value_potential_offset=offset)
    options.pop("class_name")
    return SpatialPPOPolicy(obs, GROUPS, 649, **options)


def cli_config():
    return {"task": {"width": 9, "height": 9, "players": 2, "history_length": 4,
                     "num_envs": 4, "backend": "cpu", "device": "cpu", "threads": 1,
                     "seed": 71, "max_episode_steps": 3, "combat_features": True,
                     "productive_actions": True, "critic_potential": True,
                     "opponent": "self_play", "map_pool_size": 8,
                     "reward": {"potential_gamma": .995}},
            "runner": {"num_steps_per_env": 2,
                       "policy": policy_config(value_potential_offset=True),
                       "algorithm": {"class_name": "generals_rl.student_ppo.StudentPPO",
                                     "teacher_env_ratio": 0, "num_mini_batches": 2,
                                     "gamma": .995, "learning_rate": 1e-4,
                                     "student_encoder_learning_rate": 1e-4,
                                     "schedule": "fixed", "desired_kl": None, "load_balance_coef": 0}},
            "self_play": {"pool_size": 2, "snapshot_interval": 1},
            "evaluation": {"every_n_snapshots": 2, "maps_per_ai": 1,
                           "max_ticks": 3, "batch_size": 8}, "torch_threads": 1}


class SpatialPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actor_privacy_action_layout_and_detached_critic_offset(self):
        obs = observations()
        base, shifted = construct(obs), construct(obs, True)
        shifted.load_state_dict(base.state_dict(), strict=True)
        assert_nested_equal(self, shifted.state_dict(), base.state_dict())
        prior = base.action_probabilities(obs).detach().clone()
        changed = obs.clone()
        changed["critic"] += 100
        changed["critic_potential"] += 100
        changed["single_obs"] += 100
        torch.testing.assert_close(prior, shifted.action_probabilities(changed), rtol=0, atol=0)
        head = base.evaluate(obs)
        obs["critic_potential"].requires_grad_(True)
        torch.testing.assert_close(shifted.evaluate(obs), head - obs["critic_potential"])
        shifted.evaluate(obs).sum().backward()
        self.assertIsNone(obs["critic_potential"].grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in shifted.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in shifted.actor.parameters()))
        shifted.zero_grad(set_to_none=True)
        shifted.action_probabilities(obs.exclude("critic", "critic_potential"))
        shifted.get_actions_log_prob(torch.zeros(4, 1, dtype=torch.long)).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in shifted.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in shifted.critic.parameters()))
        self.assertEqual(shifted.evaluate(obs[:0], is_teacher=True).shape, (0, 1))
        with self.assertRaisesRegex(ValueError, "critic_potential"):
            shifted.evaluate(obs.exclude("critic_potential"))
        # The eight logits belong to each row-major source, then direction/mode.
        hook = base.actor.cell_head.register_forward_hook(
            lambda module, args, output: torch.arange(648.).reshape(9, 9, 8).permute(2, 0, 1)[None].expand(4, -1, -1, -1))
        raw = base.actor(obs["policy"])
        torch.testing.assert_close(raw[:, 1:], torch.arange(648.)[None].expand(4, -1), rtol=0, atol=0)
        obs["action_mask"].zero_()
        obs["action_mask"][:, [0, 1, 17, 648]] = True
        probabilities = base.action_probabilities(obs)
        expected = raw.masked_fill(~obs["action_mask"], torch.finfo(raw.dtype).min).softmax(-1)
        torch.testing.assert_close(probabilities, expected)
        self.assertEqual(int((probabilities[~obs["action_mask"]] != 0).sum()), 0)
        hook.remove()
        obs["action_mask"].zero_(); obs["action_mask"][:, 0] = True
        self.assertTrue(bool((base.act_inference(obs) == 0).all()))
        torch.testing.assert_close(base.entropy, torch.zeros(4), rtol=0, atol=0)
        for kwargs in ({"actor_obs_normalization": True}, {"players": 3}, {"actor_hidden_dims": [16]},
                       {"value_potential_offset": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                SpatialPPOPolicy(obs, GROUPS, 649, **kwargs)

    def test_cli_policy_replacement_is_explicit_and_algorithm_defaults_survive(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            path = Path(directory) / "config.json"
            overrides = {"runner": {"policy": {"actor_hidden_dims": [24]},
                                     "algorithm": {"num_mini_batches": 6}}}
            path.write_text(json.dumps(overrides))
            original = {"task": asdict(TaskConfig(opponent="self_play")), "runner": runner_config(),
                        "self_play": asdict(SelfPlayConfig()), "evaluation": asdict(TrainingEvalConfig()),
                        "torch_threads": 1}
            with patch("generals_rl.train.train") as train:
                main(["--config", str(path), "--iterations", "1", "--output", str(Path(directory)/"old")])
                self.assertEqual(train.call_args.args[0], merge_config(original, overrides))
            overrides = {"runner": {"policy": policy_config(),
                                     "algorithm": {"class_name": "generals_rl.student_ppo.StudentPPO"}}}
            path.write_text(json.dumps(overrides))
            with patch("generals_rl.train.train") as train:
                main(["--config", str(path), "--replace-policy", "--iterations", "1", "--output", str(Path(directory)/"new")])
                cfg = train.call_args.args[0]
                self.assertEqual(cfg["runner"]["policy"], policy_config())
                self.assertEqual(cfg["runner"]["algorithm"]["num_learning_epochs"], 4)
                self.assertEqual(cfg["runner"]["algorithm"]["num_mini_batches"], 4)
                self.assertEqual(cfg["runner"]["algorithm"]["lam"], .95)
            with self.assertRaises(SystemExit):
                main(["--replace-policy"])

    def test_real_cli_two_updates_worker_eval_reload_resume_and_fresh_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / "candidate.json"; cfg.write_text(json.dumps(cli_config()))
            def run(*args):
                result = subprocess.run([sys.executable, "-m", "generals_rl.train", *map(str, args)],
                                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cold = root / "cold"
            run("--config", cfg, "--replace-policy", "--iterations", 2, "--output", cold)
            config = json.loads((cold/"run_config.json").read_text())
            self.assertEqual(config["runner"]["policy"], policy_config(value_potential_offset=True))
            self.assertEqual(config["runner"]["algorithm"]["num_learning_epochs"], 4)
            summary = json.loads((cold/"summary.json").read_text())
            self.assertEqual((summary["iterations"], summary["steps_per_environment"], summary["teacher_envs"]), (2, 4, 0))
            self.assertEqual((summary["evaluation"]["completed"], summary["evaluation"]["failed"]), (1, 0))
            jobs = json.loads((cold/"evaluations/index.json").read_text())["jobs"]
            report = json.loads(Path(jobs[0]["report"]).read_text())
            self.assertEqual(report["completed_games"], 8)
            self.assertEqual(set(report["by_ai"]), {"aggressive", "expansion", "defensive", "random"})
            self.assertEqual(report["policy"], "student_local")
            saved = torch.load(cold/"checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(saved["iter"], 1)
            self.assertEqual({int(s["step"]) for s in saved["optimizer_state_dict"]["state"].values()}, {16})
            self.assertIsNone(saved["optimizer_stu_enc_state_dict"])
            self.assertTrue(all(k.startswith(("actor.", "critic.")) for k in saved["model_state_dict"]))
            resumed = root/"resume"
            run("--resume", cold/"checkpoint.pt", "--iterations", 1, "--no-eval", "--output", resumed)
            resume = torch.load(resumed/"checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(resume["iter"], 2)
            self.assertEqual({int(s["step"]) for s in resume["optimizer_state_dict"]["state"].values()}, {24})
            warm = root/"warm"
            run("--config", cfg, "--replace-policy", "--warm-start", cold/"checkpoint.pt",
                "--iterations", 1, "--no-eval", "--output", warm)
            seed = torch.load(warm/"history_pool/snapshot_00000000.pt", weights_only=True, map_location="cpu")
            assert_nested_equal(self, seed["model_state_dict"], saved["model_state_dict"])
            self.assertEqual(seed["source"]["kind"], "warm_start_weights")
            warmed = torch.load(warm/"checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual(warmed["iter"], 0)
            self.assertEqual({int(s["step"]) for s in warmed["optimizer_state_dict"]["state"].values()}, {8})
            manifest = json.loads((warm/"history_pool/manifest.json").read_text())
            self.assertEqual(manifest["policy_spec"]["policy_class"], "generals_rl.spatial_ppo.SpatialPPOPolicy")
            self.assertLessEqual(len(manifest["snapshots"]), 2)


if __name__ == "__main__":
    unittest.main()
