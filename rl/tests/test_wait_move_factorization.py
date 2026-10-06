"""Joint categorical math and the fresh Light PPO/history/eval wiring."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from generals_rl.evaluate_builtin import _load_student, evaluate_builtin
from generals_rl.self_play import HistoricalPolicyPool
from generals_rl.spatial_ppo import LightSpatialPPOPolicy, policy_config
from generals_rl.train import train
from rl_test_utils import assert_nested_equal, small_training_config
from test_spatial_ppo import GROUPS, observations


def construct(obs, **options):
    return LightSpatialPPOPolicy(obs, GROUPS, 649, own_king_geometry=True, **options)


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits)

    def forward(self, inputs):
        return self.logits[:inputs.shape[0]]


class WaitMoveFactorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_flat_math_initialization_and_strict_flag(self):
        obs = observations()
        rng = torch.get_rng_state()
        default = construct(obs)
        torch.set_rng_state(rng)
        explicit = construct(obs, wait_move_factorization=False)
        torch.set_rng_state(rng)
        factored = construct(obs, wait_move_factorization=True)
        assert_nested_equal(self, explicit.state_dict(), default.state_dict())
        assert_nested_equal(self, factored.state_dict(), default.state_dict())
        logits = default.actor(obs["policy"])
        reference = Categorical(logits=logits.masked_fill(~obs["action_mask"], torch.finfo(logits.dtype).min), validate_args=False)
        torch.testing.assert_close(default.action_probabilities(obs), reference.probs, rtol=0, atol=0)
        torch.testing.assert_close(default.entropy, reference.entropy(), rtol=0, atol=0)
        torch.testing.assert_close(default.act_inference(obs), reference.probs.argmax(-1, keepdim=True), rtol=0, atol=0)
        rng = torch.get_rng_state()
        expected = reference.sample().unsqueeze(-1)
        torch.set_rng_state(rng)
        torch.testing.assert_close(explicit.act(obs), expected, rtol=0, atol=0)
        torch.testing.assert_close(explicit.get_actions_log_prob(expected), reference.log_prob(expected.squeeze(-1)), rtol=0, atol=0)
        prior = factored.action_probabilities(obs).detach().clone()
        changed = obs.clone()
        changed["critic"] += 100; changed["single_obs"] += 100
        torch.testing.assert_close(factored.action_probabilities(changed), prior, rtol=0, atol=0)
        for invalid in (1, 0, "true", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "must be boolean"):
                construct(obs, wait_move_factorization=invalid)

    def test_joint_analytic_oracle_mask_gradients_and_shared_group_mode(self):
        obs = observations(count=6)
        logits = torch.zeros(6, 649, dtype=torch.float64)
        logits[:2, :3] = torch.tensor([[-math.log(3), math.log(2), math.log(3)], [5., math.log(2), math.log(3)]], dtype=torch.float64)
        logits[2, 0] = -1000; logits[3, 0] = -math.log(3); logits[5, 0] = -1
        obs["action_mask"].zero_()
        obs["action_mask"][0, :3] = True; obs["action_mask"][1, 1:3] = True
        obs["action_mask"][2, 0] = True; obs["action_mask"][3, :5] = True
        obs["action_mask"][4:, :3] = True
        policy = construct(obs, wait_move_factorization=True)
        policy.actor = FixedLogits(logits)
        # Stochastic policy construction/sampling must not compute group argmax.
        with patch.object(torch.Tensor, "argmax", side_effect=AssertionError("unexpected greedy computation")):
            probabilities = policy.action_probabilities(obs)
            actions = policy.act(obs)
        self.assertTrue(bool(obs["action_mask"].gather(1, actions).all()))
        torch.testing.assert_close(probabilities[0, :3], torch.tensor([.25, .30, .45], dtype=torch.float64))
        torch.testing.assert_close(probabilities[1, :3], torch.tensor([0., .4, .6], dtype=torch.float64))
        self.assertEqual(probabilities[2, 0].item(), 1.)
        entropy = -.25 * math.log(.25) - .75 * math.log(.75) + .75 * (-.4 * math.log(.4) - .6 * math.log(.6))
        self.assertAlmostEqual(policy.entropy[0].item(), entropy)
        torch.testing.assert_close(kl_divergence(policy.distribution, policy.distribution), torch.zeros(6, dtype=torch.float64), rtol=0, atol=0)
        greedy = policy.act_inference(obs)
        torch.testing.assert_close(greedy[:, 0], torch.tensor([2, 2, 0, 1, 0, 1]), rtol=0, atol=0)
        self.assertEqual(probabilities[3].argmax().item(), 0)  # JointMAP differs.
        chosen = torch.tensor([[2], [2], [0], [1], [0], [1]])
        logp = policy.get_actions_log_prob(chosen)
        self.assertAlmostEqual(logp[0].item(), math.log(.45))
        (-logp[:3].sum()).backward()
        torch.testing.assert_close(policy.actor.logits.grad[0, :3], torch.tensor([.25, .4, -.4], dtype=torch.float64))
        self.assertEqual(policy.actor.logits.grad[1, 0].item(), 0.)
        self.assertTrue(bool((policy.actor.logits.grad[2] == 0).all()))
        self.assertTrue(bool((policy.actor.logits.grad[~obs["action_mask"]] == 0).all()))

    def test_finite_float32_edges_and_actual_empty_light_actor(self):
        obs = observations(count=3)
        obs["action_mask"].zero_(); obs["action_mask"][:, :3] = True
        logits = torch.zeros(3, 649)
        logits[:, :3] = torch.tensor([[-1000., 1000., -1000.], [1000., 1000., -1000.], [0., 1000., 1000.]])
        policy = construct(obs, wait_move_factorization=True)
        policy.actor = FixedLogits(logits)
        chosen = policy.act_inference(obs)
        loss = -policy.get_actions_log_prob(chosen).sum() + .01 * policy.entropy.sum()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss) & torch.isfinite(policy.actor.logits.grad).all()))
        empty_policy = construct(obs, wait_move_factorization=True)
        empty = obs[:0]
        self.assertEqual(empty_policy.action_probabilities(empty).shape, (0, 649))
        self.assertEqual(empty_policy.act_inference(empty).shape, (0, 1))
        empty_policy.entropy.sum().backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in empty_policy.parameters()))

    def test_actual_two_update_light_PPO_pool_signature_and_fresh_CPU_eval(self):
        cfg = small_training_config(task_overrides={"width": 9, "height": 9, "players": 2,
                "history_length": 4, "combat_features": True, "king_threat_distance": True,
                "productive_actions": True, "critic_potential": False, "map_pool_size": 0})
        cfg["runner"]["policy"] = policy_config(light_cnn=True)
        cfg["runner"]["policy"].update(own_king_geometry=True, wait_move_factorization=True)
        cfg["runner"]["algorithm"].update(class_name="generals_rl.student_ppo.StudentPPO", teacher_env_ratio=0,
                load_balance_coef=0, schedule="fixed", desired_kl=None)
        cfg["runner"]["storage_obs_keys"] = ["policy", "critic", "action_mask"]
        cfg["self_play"] = {"pool_size": 2, "snapshot_interval": 1}
        cfg["evaluation"] = {"enabled": False}
        made, loaded = [], []
        original = HistoricalPolicyPool._make_model
        def tracked_factory(pool, state):
            model = original(pool, state); made.append(model.wait_move_factorization); return model
        def tracked_loader(*args, **kwargs):
            model = _load_student(*args, **kwargs); loaded.append(model.wait_move_factorization); return model
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            with patch.object(HistoricalPolicyPool, "_make_model", tracked_factory):
                summary = train(cfg, run, iterations=2)
            self.assertEqual((summary["iterations"], summary["environment_steps"], summary["teacher_envs"]), (2, 32, 0))
            self.assertTrue(made and all(made))
            checkpoint = torch.load(run / "checkpoint.pt", weights_only=True, map_location="cpu")
            self.assertEqual({int(s["step"]) for s in checkpoint["optimizer_state_dict"]["state"].values()}, {8})
            self.assertIsNone(checkpoint["optimizer_stu_enc_state_dict"])
            self.assertTrue(all(bool(torch.isfinite(t).all()) for t in checkpoint["model_state_dict"].values()))
            manifest = json.loads((run / "history_pool/manifest.json").read_text())
            signature = deepcopy(manifest["policy_spec"])
            self.assertTrue(signature["policy_config"]["wait_move_factorization"])
            signature["policy_config"]["wait_move_factorization"] = False
            with self.assertRaisesRegex(ValueError, "policy_spec differs"):
                HistoricalPolicyPool(run / "history_pool", policy_factory=lambda: None, policy_spec=signature, max_size=2)
            with patch("generals_rl.evaluate_builtin._load_student", side_effect=tracked_loader):
                report = evaluate_builtin(run / "checkpoint.pt", maps_per_ai=1, seed=19, batch_size=8, max_ticks=3)
            self.assertEqual(report["completed_games"], 8)
            self.assertEqual(loaded, [True])


if __name__ == "__main__":
    unittest.main()
