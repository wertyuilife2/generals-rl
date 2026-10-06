"""CPU contracts for broadcasting the latest local own-general embedding."""
import unittest

import torch
from torch import nn

from generals_rl.spatial_ppo import FRAME_DIM, LightSpatialPPOPolicy
from rl_test_utils import assert_nested_equal
from test_spatial_ppo import GROUPS, observations


def construct(obs, **options):
    torch.manual_seed(137)
    return LightSpatialPPOPolicy(obs, GROUPS, 649, own_king_geometry=True,
                                 wait_move_factorization=True, **options)


class OwnKingContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_old_forward_rng_and_enabled_independent_initialization(self):
        obs = observations()
        default = construct(obs)
        default_rng = torch.get_rng_state().clone()
        disabled = construct(obs, own_king_context=False)
        torch.testing.assert_close(torch.get_rng_state(), default_rng, rtol=0, atol=0)
        assert_nested_equal(self, default.state_dict(), disabled.state_dict())
        # The pre-feature actor formula, using freshly initialized weights.
        actor = default.actor
        frames = obs['policy'].reshape(4, 4, FRAME_DIM)
        spatial = actor.trunk(frames[..., :972].reshape(4, 48, 9, 9))
        pooled = spatial.mean((-2, -1))
        context = actor.context(torch.cat((pooled, frames[:, -1, -13:]), -1))
        cells = actor.cell_head(nn.functional.elu(spatial + context[..., None, None]))
        reference = torch.cat((actor.wait_head(torch.cat((pooled, context), -1)),
                               cells.permute(0, 2, 3, 1).reshape(4, 648)), -1)
        torch.testing.assert_close(actor(obs['policy']), reference, rtol=0, atol=0)
        enabled = construct(obs, own_king_context=True)
        for part in ('critic', 'actor.trunk', 'actor.cell_head', 'actor.wait_head'):
            assert_nested_equal(self, enabled.get_submodule(part).state_dict(),
                                default.get_submodule(part).state_dict())
        torch.testing.assert_close(enabled.evaluate(obs), default.evaluate(obs), rtol=0, atol=0)
        self.assertEqual((default.actor.context[0].in_features, enabled.actor.context[0].in_features), (29, 45))
        self.assertEqual(sum(p.numel() for p in enabled.parameters()) -
                         sum(p.numel() for p in default.parameters()), 256)
        self.assertEqual(enabled.state_dict().keys(), default.state_dict().keys())
        for invalid in (1, 'true', None):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                construct(obs, own_king_context=invalid)

    def test_home_embedding_broadcast_oracle_and_gradient(self):
        policy = construct(observations(1), own_king_context=True)
        actor = policy.actor
        # Isolate context wiring from learned convolutions: channel0 is current
        # army, and only the gathered home channel enters context/MOVE/WAIT.
        actor.trunk = nn.Conv2d(48, 16, 1, bias=False)
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.trunk.weight[0, 45, 0, 0] = 1
            actor.context[0].weight[0, 16] = 1
            actor.cell_head.weight[0, 0, 0, 0] = 1
            actor.wait_head[0].weight[0, 16] = 1
            actor.wait_head[2].weight[0, 0] = 1
        frames = torch.zeros(1, 4, FRAME_DIM)
        maps = frames[..., :972].unflatten(-1, (12, 9, 9))
        maps[0, -1, 4:6, 0, 0] = 1
        maps[0, -1, 9, 0, 0] = 2
        maps[0, 0, 4:6, 8, 8] = 1  # Old marker cannot locate the context.
        history = frames.flatten(1).detach().requires_grad_()
        logits = actor(history)
        far_move = 1 + 80 * 8
        torch.testing.assert_close(logits[0, [0, far_move]], torch.tensor([2., 2.]), rtol=0, atol=0)
        gradient = torch.autograd.grad(logits[0, 0] + logits[0, far_move], history)[0]
        expected = torch.zeros_like(history)
        expected[0, 3 * FRAME_DIM + 9 * 81] = 2
        expected[0, 3 * FRAME_DIM + 9 * 81 + 80] = 1  # Far MOVE's local path.
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)

    def test_local_privacy_missing_ambiguous_dtype_and_empty_batch(self):
        for dtype in (torch.float32, torch.float64):
            obs = observations(3).to(dtype=dtype)
            frames = obs['policy'].reshape(3, 4, FRAME_DIM)
            maps = frames[..., :972].unflatten(-1, (12, 9, 9))
            maps[:, :, 4:6] = 0
            maps[:, 0, 4:6, 0, 0] = 1
            maps[0, -1, 4:6, 4, 4] = 1
            maps[2, -1, 4:6, 1, 1] = maps[2, -1, 4:6, 7, 7] = 1
            policy = construct(obs, own_king_context=True).to(dtype=dtype)
            captured = []
            hook = policy.actor.context[0].register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().clone()))
            prior = policy.action_probabilities(obs).detach().clone()
            hook.remove()
            self.assertEqual(captured[0].dtype, dtype)
            torch.testing.assert_close(captured[0][1:, 16:32], torch.zeros(2, 16, dtype=dtype), rtol=0, atol=0)
            changed = obs.clone()
            for key in ('critic', 'single_obs', 'critic_potential'):
                changed[key] += 100
            torch.testing.assert_close(policy.action_probabilities(changed), prior, rtol=0, atol=0)
            policy.get_actions_log_prob(torch.zeros(3, 1, dtype=torch.long)).sum().backward()
            self.assertTrue(bool(policy.actor.context[0].weight.grad[:, 16:32].abs().sum() > 0))
            self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
            policy.zero_grad(set_to_none=True)
            policy.evaluate(obs).sum().backward()
            self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
            self.assertEqual(tuple(policy.action_probabilities(obs[:0]).shape), (0, 649))
            self.assertEqual(tuple(policy.actor(obs[:0]['policy']).shape), (0, 649))
