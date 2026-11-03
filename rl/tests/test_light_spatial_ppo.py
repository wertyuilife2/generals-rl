"""LightCNN action layout, actor privacy and a cold self-play lifecycle."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from rsl_rl.runners import OnPolicyRunnerCTS
from rsl_rl.utils import resolve_callable
from generals_rl.cts import runner_config
from generals import MapConfig
from generals_arena.competitions.evaluation import evaluate
from generals_arena.competitions.evaluation import make_profile
from generals_rl.export import export_policy
from generals_rl.opponents import SelfPlayOpponent
from generals_rl.spatial_ppo import LightSpatialPPOPolicy
from generals_rl.information_ppo import InformationSpatialPPOPolicy
from generals_rl.train import train
from rl_test_utils import (GROUPS, assert_nested_equal, light_policy_config,
                           small_training_config, spatial_observations as observations)


class LightSpatialPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_layout_mask_actor_privacy_and_independent_critic(self):
        obs=observations()
        options=light_policy_config()
        cls=resolve_callable(options.pop('class_name'))
        policy=cls(obs,GROUPS,649,**options)
        self.assertIs(type(policy),LightSpatialPPOPolicy)
        actor_convs=[m for m in policy.actor.trunk if isinstance(m,torch.nn.Conv2d)]
        critic_convs=[m for m in policy.critic.trunk if isinstance(m,torch.nn.Conv2d)]
        self.assertEqual([(m.in_channels,m.out_channels,m.kernel_size,m.groups) for m in actor_convs],
                         [(56,16,(3,3),1),(16,16,(3,3),1)])
        self.assertEqual([(m.in_channels,m.out_channels,m.kernel_size,m.groups) for m in critic_convs],
                         [(14,16,(3,3),1),(16,16,(3,3),1)])
        original=policy.action_probabilities(obs).detach().clone()
        changed=obs.clone()
        for key in ('critic','single_obs'):
            changed[key]+=100
        torch.testing.assert_close(original,policy.action_probabilities(changed),rtol=0,atol=0)
        policy.action_probabilities(obs.exclude('critic','single_obs'))
        torch.testing.assert_close(policy.evaluate(obs),policy.critic(obs['critic']))
        policy.evaluate(obs).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        policy.zero_grad(set_to_none=True)
        policy.action_probabilities(obs)
        policy.get_actions_log_prob(torch.zeros(4,1,dtype=torch.long)).sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.actor.parameters()))
        hook=policy.actor.cell_head.register_forward_hook(lambda module,args,output:
             torch.arange(648.).reshape(9,9,8).permute(2,0,1)[None].expand(4,-1,-1,-1))
        raw=policy.actor(obs['policy'])
        torch.testing.assert_close(raw[:,1:],torch.arange(648.)[None].expand(4,-1),rtol=0,atol=0)
        obs['action_mask'].zero_(); obs['action_mask'][:,[0,1,17,648]]=True
        probs=policy.action_probabilities(obs)
        torch.testing.assert_close(probs,raw.masked_fill(~obs['action_mask'],torch.finfo(raw.dtype).min).softmax(-1))
        self.assertEqual(int(torch.count_nonzero(probs[~obs['action_mask']])),0)
        hook.remove()
        obs['action_mask'].zero_(); obs['action_mask'][:,0]=True
        self.assertTrue(bool((policy.act_inference(obs)==0).all()))
        torch.testing.assert_close(policy.entropy,torch.zeros(4),rtol=0,atol=0)
        self.assertEqual(tuple(policy.evaluate(obs[:0],is_teacher=True).shape),(0,1))
    def test_cold_self_play_seeds_exact_random_policy_without_baseline_training(self):
        cfg = small_training_config()
        cfg['task']['king_threat_distance'] = True
        cfg['runner'] = runner_config(rollout_steps=2, learning_epochs=2, mini_batches=2)
        cfg['runner']['obs_groups'] = GROUPS.copy()
        cfg['runner'].pop('storage_obs_keys', None)
        cfg['runner']['policy'] = light_policy_config()
        cfg['runner']['policy']['class_name'] = 'generals_rl.information_ppo.InformationSpatialPPOPolicy'
        cfg['runner']['algorithm'].update(
            class_name='generals_rl.student_ppo.StudentPPO', teacher_env_ratio=0,
            gamma=.995, schedule='fixed', desired_kl=None)
        cfg['self_play'] = dict(pool_size=3, snapshot_interval=1, retain_seed=True, stochastic=True, seed=917, sampling='uniform')
        cfg['evaluation'] = {'enabled': False}
        initial = {}

        def capture_runner(env, *args, **kwargs):
            self.assertIs(type(env.task.opponent), SelfPlayOpponent)
            self.assertIsNone(env.task.opponent.pool)
            runner = OnPolicyRunnerCTS(env, *args, **kwargs)
            self.assertIs(type(runner.alg.policy), InformationSpatialPPOPolicy)
            self.assertEqual(runner.alg.optimizer.state, {})
            initial.update({name: value.detach().clone()
                            for name, value in runner.alg.policy.state_dict().items()})
            return runner

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            # Evaluation keeps the original controllers, but collecting training
            # rollouts must never construct one, even with threat observations.
            with patch('generals_rl.train.OnPolicyRunnerCTS', side_effect=capture_runner), \
                    patch('generals.controllers.make_controllers', side_effect=AssertionError('baseline training')), \
                    patch('generals_baselines.base.BaseAI.__init__', side_effect=AssertionError('baseline training')):
                summary = train(cfg, root / 'cold', iterations=2)
            seed = torch.load(root / 'cold/history_pool/snapshot_00000000.pt',
                              weights_only=True, map_location='cpu')
            assert_nested_equal(self, seed['model_state_dict'], initial)
            self.assertEqual(seed['source'], {'kind': 'random_initial_policy'})
            manifest = json.loads((root / 'cold/history_pool/manifest.json').read_text())
            self.assertEqual(manifest['policy_spec']['encoder_semantics'],
                             'official_31_4_5_combat_p2_v1_king_threat_distance_v1')
            self.assertIsNone(manifest['resume_source'])
            self.assertEqual(summary['self_play']['pool_ids'], [0, 1, 2])
            self.assertGreaterEqual(summary['self_play']['episode_assignments'], 4)
            self.assertEqual((summary['teacher_envs'], summary['student_envs']), (0, 4))
            self.assertIsNone(summary['resume'])
            self.assertNotIn('warm_start', summary)
            self.assertNotIn('baseline_league', summary)
            saved = torch.load(root / 'cold/checkpoint.pt', weights_only=True, map_location='cpu')
            self.assertEqual({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}, {8})
            self.assertTrue(any(not torch.equal(initial[name], saved['model_state_dict'][name])
                                for name in initial))
            bundle = export_policy(root / 'cold/checkpoint.pt', root/'bundle')
            profile = make_profile(MapConfig(**{key: cfg['task'][key] for key in (
                'width', 'height', 'players', 'mountain_density', 'city_density')}),
                maps_per_opponent=1, max_ticks=3)
            evaluation = evaluate(bundle, profile, batch_size=8, cpu_threads=1)
            self.assertEqual(evaluation['games'], 10)

    def test_legacy_baseline_config_rejected_before_any_training_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            for league in ({}, {'baseline_envs': 0}, {'baseline_envs': 128}):
                with self.subTest(league=league), self.assertRaisesRegex(ValueError, 'frozen source'):
                    train({'baseline_league': league}, output, iterations=1)
                self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
