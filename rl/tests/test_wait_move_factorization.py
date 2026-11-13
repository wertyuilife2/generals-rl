"""Joint categorical math and the ViT PPO/history/eval wiring."""
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
from torch.distributions import kl_divergence

from generals import MapConfig
from generals_arena.competitions.evaluation import evaluate
from generals_arena.competitions.evaluation import make_profile
from generals_rl.export import export_policy
from generals_rl.self_play import HistoricalPolicyPool
from generals_rl.policy.distributions import make_distribution, greedy_actions
from generals_rl.train import train
from rl_test_utils import small_training_config
from rl_test_utils import spatial_observations as observations


class WaitMoveFactorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)


    def test_joint_analytic_oracle_mask_gradients_and_shared_group_mode(self):
        obs = observations(count=6)
        logits = torch.zeros(6, 649, dtype=torch.float64)
        logits[:2, :3] = torch.tensor([[-math.log(3), math.log(2), math.log(3)], [5., math.log(2), math.log(3)]], dtype=torch.float64)
        logits[2, 0] = -1000; logits[3, 0] = -math.log(3); logits[5, 0] = -1
        obs["action_mask"].zero_()
        obs["action_mask"][0, :3] = True; obs["action_mask"][1, 1:3] = True
        obs["action_mask"][2, 0] = True; obs["action_mask"][3, :5] = True
        obs["action_mask"][4:, :3] = True
        logits.requires_grad_()
        # Stochastic policy construction/sampling must not compute group argmax.
        with patch.object(torch.Tensor, "argmax", side_effect=AssertionError("unexpected greedy computation")):
            distribution = make_distribution(logits, obs["action_mask"], factorized=True)
            probabilities = distribution.probs
            actions = distribution.sample().unsqueeze(-1)
        self.assertTrue(bool(obs["action_mask"].gather(1, actions).all()))
        torch.testing.assert_close(probabilities[0, :3], torch.tensor([.25, .30, .45], dtype=torch.float64))
        torch.testing.assert_close(probabilities[1, :3], torch.tensor([0., .4, .6], dtype=torch.float64))
        self.assertEqual(probabilities[2, 0].item(), 1.)
        entropy = -.25 * math.log(.25) - .75 * math.log(.75) + .75 * (-.4 * math.log(.4) - .6 * math.log(.6))
        self.assertAlmostEqual(distribution.entropy()[0].item(), entropy)
        torch.testing.assert_close(kl_divergence(distribution, distribution), torch.zeros(6, dtype=torch.float64), rtol=0, atol=0)
        greedy = greedy_actions(probabilities, factorized=True)
        torch.testing.assert_close(greedy[:, 0], torch.tensor([2, 2, 0, 1, 0, 1]), rtol=0, atol=0)
        self.assertEqual(probabilities[3].argmax().item(), 0)  # JointMAP differs.
        chosen = torch.tensor([[2], [2], [0], [1], [0], [1]])
        logp = distribution.log_prob(chosen.squeeze(-1))
        self.assertAlmostEqual(logp[0].item(), math.log(.45))
        (-logp[:3].sum()).backward()
        torch.testing.assert_close(logits.grad[0, :3], torch.tensor([.25, .4, -.4], dtype=torch.float64))
        self.assertEqual(logits.grad[1, 0].item(), 0.)
        self.assertTrue(bool((logits.grad[2] == 0).all()))
        self.assertTrue(bool((logits.grad[~obs["action_mask"]] == 0).all()))

    def test_finite_float32_edges(self):
        obs = observations(count=3)
        obs["action_mask"].zero_(); obs["action_mask"][:, :3] = True
        logits = torch.zeros(3, 649)
        logits[:, :3] = torch.tensor([[-1000., 1000., -1000.], [1000., 1000., -1000.], [0., 1000., 1000.]])
        logits.requires_grad_()
        distribution = make_distribution(logits, obs["action_mask"], factorized=True)
        chosen = greedy_actions(distribution.probs, factorized=True)
        loss = -distribution.log_prob(chosen.squeeze(-1)).sum() + .01 * distribution.entropy().sum()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss) & torch.isfinite(logits.grad).all()))

    def test_actual_two_update_vit_PPO_pool_signature_and_fresh_CPU_eval(self):
        cfg = small_training_config(task_overrides={"width": 9, "height": 9, "players": 2,
                "history_length": 4, "combat_features": True, "king_threat_distance": True,
                "critic_potential": False, "map_pool_size": 0})
        cfg["self_play"] = {"pool_size": 2, "snapshot_interval": 1}
        cfg["evaluation"] = {"enabled": False}
        made = []
        original = HistoricalPolicyPool._make_model
        def tracked_factory(pool, state):
            model = original(pool, state); made.append(model.wait_move_factorization); return model
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
            bundle = export_policy(run / "checkpoint.pt", run/"bundle")
            profile = make_profile(MapConfig(**{key: cfg["task"][key] for key in (
                "width", "height", "players", "mountain_density", "city_density")}),
                maps_per_opponent=1, seed=19, max_ticks=3)
            report = evaluate(bundle, profile, batch_size=8)
            self.assertEqual(report["games"], 10)


if __name__ == "__main__":
    unittest.main()
