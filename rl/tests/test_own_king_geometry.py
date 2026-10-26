"""Optional actor geometry: local inputs, legacy weights and pool compatibility."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn
from tensordict import TensorDict

from generals import VecEnv
from generals_rl.mdp import GameBatch, make_observation_encoder
from generals_rl.policy.actions import make_action_space
from generals_rl.self_play import HistoricalPolicyPool
from generals_rl.self_play_training import prepare_self_play
from generals_rl.policy.cnn import OwnKingRelativeGeometry
from generals_rl.spatial_ppo import LightSpatialPPOPolicy, SpatialPPOPolicy
from generals_rl.task import TaskConfig
from test_king_threat_distance import distance_state
from rl_test_utils import light_policy_config
from rl_test_utils import GROUPS
from rl_test_utils import assert_nested_equal


class LegacyLightPolicy(SpatialPPOPolicy):
    """The original two-convolution layout, independent of the new option."""
    @staticmethod
    def trunk_factory(inputs, channels, _blocks):
        return nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.ELU(),
                             nn.Conv2d(channels, channels, 3, padding=1), nn.ELU())


def construct(obs, **options):
    torch.manual_seed(137)
    return LightSpatialPPOPolicy(obs, GROUPS, 649, **options)


class OwnKingGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def local_observations(self):
        first = distance_state()
        hidden = first.copy()
        for array in (hidden.owner, hidden.army, hidden.structure):
            array.flat[80], array.flat[8] = array.flat[8], array.flat[80]
            array.flat[79], array.flat[9] = array.flat[9], array.flat[79]
        hidden.general_pos[1] = 8
        hidden.validate()
        backend = VecEnv.from_states([first, hidden], device='cpu', threads=1, auto_reset=False)
        self.addCleanup(backend.close)
        batch = GameBatch.from_backend(backend, 'cpu')
        cfg = TaskConfig(backend='cpu', device='cpu', opponent='self_play',
                         combat_features=True, king_threat_distance=True, productive_actions=True)
        encoder = make_observation_encoder(cfg)
        ids = torch.zeros(2, dtype=torch.long)
        local = encoder.encode(batch, ids)
        obs = TensorDict({'policy': local[:, None].expand(-1, 4, -1).reshape(2, -1),
                          'single_obs': local.clone(),
                          'critic': encoder.encode(batch, ids, privileged=True),
                          'action_mask': make_action_space(cfg).mask(batch, ids)}, [2])
        return obs, cfg, encoder

    def test_relative_xy_latest_local_frame_guards_and_dtype(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                maps = torch.zeros(2, 48, 9, 9, dtype=dtype)
                for lane, (row, column) in enumerate(((7, 3), (1, 6))):
                    maps[lane, 40:42, row, column] = 1
                    # A visible enemy general and a self-owned plain cell are
                    # insufficient individually: both markers must intersect.
                    maps[lane, 40, 0, 8] = 1
                    maps[lane, 41, 8, 0] = 1
                geometry = OwnKingRelativeGeometry().to(device=maps.device, dtype=dtype)
                actual = geometry(maps)
                self.assertEqual((actual.dtype, actual.device), (maps.dtype, maps.device))
                self.assertEqual(tuple(actual.shape), (2, 50, 9, 9))
                torch.testing.assert_close(actual[:, :48], maps, rtol=0, atol=0)
                rows, columns = torch.meshgrid(torch.arange(9, dtype=dtype),
                                               torch.arange(9, dtype=dtype), indexing='ij')
                expected = torch.stack([torch.stack(((columns-column)/8, (rows-row)/8))
                                        for row, column in ((7, 3), (1, 6))])
                torch.testing.assert_close(actual[:, -2:], expected, rtol=0, atol=0)
                changed = maps.clone()
                changed[:, :36] = torch.randn_like(changed[:, :36])
                torch.testing.assert_close(geometry(changed)[:, -2:], expected, rtol=0, atol=0)
                # Latest missing marker must not fall back to any old frame.
                changed[:, 4:6, 4, 4] = 1
                changed[:, 40] = 0
                self.assertTrue(bool(geometry(changed)[:, -2:].eq(0).all()))
                changed = maps.clone()
                changed[:, 40:42, 4, 4] = 1
                self.assertTrue(bool(geometry(changed)[:, -2:].eq(0).all()))
                self.assertFalse(geometry.state_dict())
                self.assertEqual(sum(p.numel() for p in geometry.parameters()), 0)
        with self.assertRaises(ValueError):
            OwnKingRelativeGeometry()(torch.zeros(1, 50, 9, 9))
        with self.assertRaises(TypeError):
            OwnKingRelativeGeometry()(torch.zeros(1, 48, 9, 9, dtype=torch.long))

    def test_default_preserves_legacy_weights_logits_value_and_enabled_critic(self):
        obs, _, _ = self.local_observations()
        torch.manual_seed(137)
        legacy = LegacyLightPolicy(obs, GROUPS, 649, spatial_channels=16, residual_blocks=1)
        default = construct(obs)
        disabled = construct(obs, own_king_geometry=False)
        for policy in (default, disabled):
            assert_nested_equal(self, policy.state_dict(), legacy.state_dict())
            torch.testing.assert_close(policy.actor(obs['policy']), legacy.actor(obs['policy']), rtol=0, atol=0)
            torch.testing.assert_close(policy.action_probabilities(obs), legacy.action_probabilities(obs), rtol=0, atol=0)
            torch.testing.assert_close(policy.evaluate(obs), legacy.evaluate(obs), rtol=0, atol=0)
            self.assertEqual(policy.semantics, LightSpatialPPOPolicy.semantics)
        base = construct(obs)
        enabled = construct(obs, own_king_geometry=True)
        assert_nested_equal(self, enabled.critic.state_dict(), base.critic.state_dict())
        torch.testing.assert_close(enabled.evaluate(obs), base.evaluate(obs), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in enabled.parameters()) -
                         sum(p.numel() for p in base.parameters()), 288)
        self.assertIsInstance(enabled.actor.trunk[0], OwnKingRelativeGeometry)
        self.assertEqual(enabled.actor.trunk[1].in_channels, 50)
        self.assertEqual(enabled.actor.trunk[3].dilation, (1, 1))
        self.assertIs(enabled.actor.forward.__func__, base.actor.forward.__func__)
        self.assertTrue(enabled.semantics.endswith('_own_king_relative_xy_v1'))
        with self.assertRaises(RuntimeError):
            enabled.load_state_dict(base.state_dict(), strict=True)
        for invalid in (1, 'true', None):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                construct(obs, own_king_geometry=invalid)

    def test_hidden_world_privacy_legal_probabilities_and_new_conv_gradients(self):
        obs, _, _ = self.local_observations()
        torch.testing.assert_close(obs['policy'][0], obs['policy'][1], rtol=0, atol=0)
        torch.testing.assert_close(obs['action_mask'][0], obs['action_mask'][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(obs['critic'][0], obs['critic'][1]))
        policy = construct(obs, own_king_geometry=True)
        probabilities = policy.action_probabilities(obs)
        torch.testing.assert_close(probabilities[0], probabilities[1], rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(probabilities).all()))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2), rtol=0, atol=1e-6)
        self.assertTrue(bool(probabilities[~obs['action_mask']].eq(0).all()))
        changed = obs.clone()
        changed['critic'].fill_(1e6)
        changed['single_obs'].fill_(-1e6)
        torch.testing.assert_close(policy.action_probabilities(changed), probabilities, rtol=0, atol=0)
        torch.testing.assert_close(policy.action_probabilities(obs.exclude('critic', 'single_obs')),
                                   probabilities, rtol=0, atol=0)
        self.assertTrue(bool(obs['action_mask'].gather(1, policy.act_inference(obs)).all()))
        policy.get_actions_log_prob(torch.zeros(2, 1, dtype=torch.long)).sum().backward()
        gradient = policy.actor.trunk[1].weight.grad[:, -2:]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool(gradient.abs().sum() > 0))
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))

    def test_explicit_geometry_config_isolates_real_history_pool_signature(self):
        obs, task_cfg, encoder = self.local_observations()
        task = SimpleNamespace(config=task_cfg, observation_encoder=encoder, device=torch.device('cpu'),
                               action_space=SimpleNamespace(size=649), opponent=SimpleNamespace(attach=Mock()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = []
            for name, enabled in (('legacy', False), ('geometry', True)):
                options = light_policy_config()
                if enabled:
                    options['own_king_geometry'] = True
                factory_options = deepcopy(options)
                factory_options.pop('class_name')
                policy = construct(obs, **factory_options)
                runner = SimpleNamespace(alg=SimpleNamespace(policy=policy), cfg={'obs_groups': GROUPS},
                                         env=SimpleNamespace(get_observations=lambda: obs))
                sessions.append(prepare_self_play(task, runner, {'runner': {'policy': options}}, root/name))
            legacy, geometry = sessions
            self.assertNotIn('own_king_geometry', legacy.pool.policy_spec['policy_config'])
            self.assertIs(geometry.pool.policy_spec['policy_config']['own_king_geometry'], True)
            self.assertEqual(legacy.pool.policy_spec['observation_shapes'],
                             geometry.pool.policy_spec['observation_shapes'])
            self.assertNotEqual(legacy.pool.policy_spec_sha256, geometry.pool.policy_spec_sha256)
            restored = geometry.pool.get(0).model
            torch.testing.assert_close(restored.action_probabilities(obs),
                                       geometry.policy.action_probabilities(obs), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'policy_spec'):
                HistoricalPolicyPool(root/'mismatched', policy_factory=geometry.pool.policy_factory,
                                     policy_spec=geometry.pool.policy_spec, resume_from=legacy.pool.directory)


if __name__ == '__main__':
    unittest.main()
