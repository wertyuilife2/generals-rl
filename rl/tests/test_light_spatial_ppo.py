"""Opt-in light CNN contracts; reuse the existing bounded distillation/PPO CLI."""
from contextlib import redirect_stdout
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
from rsl_rl.runners import OnPolicyRunnerCTS
from rsl_rl.utils import resolve_callable

from generals_rl.cts import runner_config
from generals_rl.config import task_config
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.opponents import SelfPlayOpponent
from generals_rl.self_play_training import prepare_self_play
from generals_rl.spatial_ppo import LightSpatialPPOPolicy, SpatialPPOPolicy, policy_config
from generals_rl.task import GeneralsTask
from generals_rl.train import train
from test_spatial_ppo import GROUPS, ROOT, cli_config, observations
from test_spatial_distillation import tool
from rl_test_utils import assert_nested_equal
from types import SimpleNamespace


class LightSpatialPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_layout_mask_actor_privacy_and_detached_phi(self):
        obs=observations()
        options=policy_config(value_potential_offset=True,light_cnn=True)
        cls=resolve_callable(options.pop('class_name'))
        policy=cls(obs,GROUPS,649,**options)
        self.assertIs(type(policy),LightSpatialPPOPolicy)
        actor_convs=[m for m in policy.actor.trunk if isinstance(m,torch.nn.Conv2d)]
        critic_convs=[m for m in policy.critic.trunk if isinstance(m,torch.nn.Conv2d)]
        self.assertEqual([(m.in_channels,m.out_channels,m.kernel_size,m.groups) for m in actor_convs],
                         [(48,16,(3,3),1),(16,16,(3,3),1)])
        self.assertEqual([(m.in_channels,m.out_channels,m.kernel_size,m.groups) for m in critic_convs],
                         [(12,16,(3,3),1),(16,16,(3,3),1)])
        original=policy.action_probabilities(obs).detach().clone()
        changed=obs.clone()
        for key in ('critic','critic_potential','single_obs'):
            changed[key]+=100
        torch.testing.assert_close(original,policy.action_probabilities(changed),rtol=0,atol=0)
        policy.action_probabilities(obs.exclude('critic','critic_potential','single_obs'))
        obs['critic_potential'].requires_grad_(True)
        torch.testing.assert_close(policy.evaluate(obs),policy.critic(obs['critic'])-obs['critic_potential'])
        policy.evaluate(obs).sum().backward()
        self.assertIsNone(obs['critic_potential'].grad)
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
        with self.assertRaisesRegex(ValueError,'critic_potential'):
            policy.evaluate(obs.exclude('critic_potential'))
        with self.assertRaisesRegex(ValueError,'residual_blocks'):
            LightSpatialPPOPolicy(obs,GROUPS,649,residual_blocks=2)
        defaults=policy_config()
        self.assertEqual(defaults['class_name'],'generals_rl.spatial_ppo.SpatialPPOPolicy')
        self.assertEqual((defaults['spatial_channels'],defaults['residual_blocks']),(32,4))
        default_cls=resolve_callable(defaults.pop('class_name'))
        self.assertIs(type(default_cls(obs,GROUPS,649,**defaults)),SpatialPPOPolicy)

    def test_real_distill_cli_fresh_ppo_resume_evaluator_and_pool_factory(self):
        cfg=cli_config()
        cfg['runner']=runner_config(hidden_dim=16,expert_num=2,rollout_steps=2,mini_batches=2)
        cfg['runner']['policy'].update(class_name='generals_rl.student_ppo.StudentPPOPolicy',
                                      value_potential_offset=True,latent_dim=8)
        cfg['runner']['algorithm']['gamma']=.995
        cfg['evaluation']={'enabled':False}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory); source=root/'source'; source.mkdir()
            with GeneralsTask(task_config(cfg['task'])) as task:
                obs=TensorDict(task.get_observations(),[4])
                teacher=tool.teacher_factory(cfg,obs)()
                runner=SimpleNamespace(alg=SimpleNamespace(policy=teacher),cfg={'obs_groups':cfg['runner']['obs_groups']},
                                       env=SimpleNamespace(get_observations=lambda:obs))
                prepare_self_play(task,runner,cfg,source)
                torch.save({'model_state_dict':teacher.state_dict()},source/'checkpoint.pt')
                (source/'run_config.json').write_text(json.dumps(cfg))
            def run(command):
                result=subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run([ROOT/'rl/tools/distill_spatial.py','--checkpoint',source/'checkpoint.pt','--output',root/'light',
                 '--light-cnn','--num-envs','4','--batch-size','2','--max-teacher-models','1',
                 '--ticks','2','--time-budget','30','--no-tensorboard'])
            generated=json.loads((root/'light/run_config.json').read_text())
            expected='generals_rl.spatial_ppo.LightSpatialPPOPolicy'
            self.assertEqual(generated['runner']['policy'],policy_config(value_potential_offset=True,light_cnn=True))
            self.assertEqual(generated['self_play'],cfg['self_play'])
            report=json.loads((root/'light/distillation_report.json').read_text())
            self.assertEqual((report['samples'],report['Adam_steps']),(8,4))
            self.assertEqual(report['teacher_branch'],'deployed_student_local_only_is_teacher_False')
            initial=torch.load(root/'light/checkpoint.pt',weights_only=True,map_location='cpu')
            # Load through the same dynamic factory used for read-only opponents.
            restored=tool.load_teacher(root/'light/checkpoint.pt',tool.teacher_factory(generated,obs),'cpu')
            self.assertIs(type(restored),LightSpatialPPOPolicy)
            assert_nested_equal(self,restored.state_dict(),initial['model_state_dict'])
            run(['-m','generals_rl.train','--config',root/'light/run_config.json','--replace-policy',
                 '--warm-start',root/'light/checkpoint.pt','--iterations','1','--no-eval','--output',root/'ppo'])
            saved=torch.load(root/'ppo/checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()},{8})
            seed=torch.load(root/'ppo/history_pool/snapshot_00000000.pt',weights_only=True,map_location='cpu')
            assert_nested_equal(self,seed['model_state_dict'],initial['model_state_dict'])
            manifest=json.loads((root/'ppo/history_pool/manifest.json').read_text())
            self.assertEqual(manifest['policy_spec']['policy_class'],expected)
            self.assertLessEqual(len(manifest['snapshots']),2)
            run(['-m','generals_rl.train','--resume',root/'ppo/checkpoint.pt','--iterations','1',
                 '--no-eval','--output',root/'resume'])
            resumed=torch.load(root/'resume/checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual((saved['iter'],resumed['iter']),(0,1))
            self.assertEqual({int(v['step']) for v in resumed['optimizer_state_dict']['state'].values()},{16})
            evaluation=evaluate_builtin(root/'resume/checkpoint.pt',maps_per_ai=1,max_ticks=3,batch_size=8,cpu_threads=1)
            self.assertEqual(evaluation['completed_games'],8)

    def test_cold_self_play_seeds_exact_random_policy_without_baseline_training(self):
        cfg = cli_config()
        cfg['task']['king_threat_distance'] = True
        cfg['runner'] = runner_config(rollout_steps=2, mini_batches=2)
        cfg['runner']['policy'] = policy_config(value_potential_offset=True, light_cnn=True)
        cfg['runner']['algorithm'].update(
            class_name='generals_rl.student_ppo.StudentPPO', teacher_env_ratio=0,
            gamma=.995, load_balance_coef=0, schedule='fixed', desired_kl=None)
        cfg['self_play'].update(pool_size=3, stochastic=True, seed=917, sampling='uniform')
        cfg['evaluation'] = {'enabled': False}
        initial = {}

        def capture_runner(env, *args, **kwargs):
            self.assertIs(type(env.task.opponent), SelfPlayOpponent)
            self.assertIsNone(env.task.opponent.pool)
            runner = OnPolicyRunnerCTS(env, *args, **kwargs)
            self.assertIs(type(runner.alg.policy), LightSpatialPPOPolicy)
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
                    patch('generals_rl.evaluate_builtin.make_controllers', side_effect=AssertionError('baseline training')):
                summary = train(cfg, root / 'cold', iterations=2)
            seed = torch.load(root / 'cold/history_pool/snapshot_00000000.pt',
                              weights_only=True, map_location='cpu')
            assert_nested_equal(self, seed['model_state_dict'], initial)
            self.assertEqual(seed['source'], {'kind': 'random_initial_policy'})
            manifest = json.loads((root / 'cold/history_pool/manifest.json').read_text())
            self.assertEqual(manifest['policy_spec']['encoder_semantics'],
                             'combat_p2_985_v1_king_threat_distance_v1')
            self.assertIsNone(manifest['resume_source'])
            self.assertEqual(summary['self_play']['pool_ids'], [0, 1, 2])
            self.assertGreaterEqual(summary['self_play']['episode_assignments'], 4)
            self.assertEqual((summary['teacher_envs'], summary['student_envs']), (0, 4))
            self.assertIsNone(summary['resume'])
            self.assertNotIn('warm_start', summary)
            self.assertNotIn('baseline_league', summary)
            saved = torch.load(root / 'cold/checkpoint.pt', weights_only=True, map_location='cpu')
            self.assertEqual({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}, {16})
            self.assertTrue(any(not torch.equal(initial[name], saved['model_state_dict'][name])
                                for name in initial))
            evaluation = evaluate_builtin(root / 'cold/checkpoint.pt', maps_per_ai=1,
                                          max_ticks=3, batch_size=8, cpu_threads=1)
            self.assertEqual(evaluation['completed_games'], 8)

    def test_legacy_baseline_config_rejected_before_any_training_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            for league in ({}, {'baseline_envs': 0}, {'baseline_envs': 128}):
                with self.subTest(league=league), self.assertRaisesRegex(ValueError, 'frozen source'):
                    train({'baseline_league': league}, output, iterations=1)
                self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
