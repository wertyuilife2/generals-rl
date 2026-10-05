"""Opt-in light CNN contracts; reuse the existing bounded distillation/PPO CLI."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
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
from rsl_rl.utils import resolve_callable

from generals_rl.cts import runner_config
from generals_rl.config import task_config
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.self_play_training import prepare_self_play
from generals_rl.spatial_ppo import LightSpatialPPOPolicy, SpatialPPOPolicy, policy_config
from generals_rl.task import GeneralsTask
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

    def test_king_distance_local_labels_and_formal_batched_league_survive_distill(self):
        cfg=cli_config()
        cfg['task'].update(num_envs=100,king_threat_distance=True)
        cfg['runner']=runner_config(hidden_dim=16,expert_num=2,rollout_steps=2,mini_batches=2)
        cfg['runner']['policy'].update(class_name='generals_rl.student_ppo.StudentPPOPolicy',
                                      value_potential_offset=True,latent_dim=8)
        cfg['runner']['algorithm'].update(class_name='generals_rl.student_ppo.StudentPPO',teacher_env_ratio=0,
                                         gamma=.995,load_balance_coef=0,schedule='fixed',desired_kl=None)
        cfg['self_play'].update(pool_size=8,snapshot_interval=50,stochastic=True,seed=917,sampling='uniform')
        cfg['evaluation']={'enabled':False}
        cfg['baseline_league']={'baseline_envs':1,'seed':918,'native_candidates':True,'batched_observations':True}
        semantics='combat_p2_985_v1_king_threat_distance_v1'
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory); source=root/'source'; source.mkdir()
            with GeneralsTask(task_config(cfg['task'])) as task:
                obs=TensorDict(task.get_observations(),[100])
                self.assertEqual(task.observation_encoder.semantics,semantics)
                teacher=tool.teacher_factory(cfg,obs)().eval().requires_grad_(False)
                original=teacher.action_probabilities
                def local_only(inputs,**kwargs):
                    self.assertEqual(set(inputs.keys()),{'policy','single_obs','action_mask'})
                    self.assertFalse(kwargs['is_teacher'])
                    return original(inputs,**kwargs)
                with patch.object(teacher,'action_probabilities',side_effect=local_only):
                    labels,values=tool.teacher_labels(teacher,obs)
                teacher.value_potential_offset=False
                torch.testing.assert_close(values,teacher.evaluate(obs)-obs['critic_potential'])
                teacher.value_potential_offset=True
                runner=SimpleNamespace(alg=SimpleNamespace(policy=teacher),cfg={'obs_groups':cfg['runner']['obs_groups']},
                                       env=SimpleNamespace(get_observations=lambda:obs))
                prepare_self_play(task,runner,cfg,source)
                torch.save({'model_state_dict':teacher.state_dict(),'optimizer_state_dict':{'sentinel':True}},source/'checkpoint.pt')
            (source/'run_config.json').write_text(json.dumps(cfg))
            immutable={p.relative_to(source).as_posix():sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob('*') if p.is_file()}
            def run(arguments):
                result=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            initial=root/'light'
            run([ROOT/'rl/tools/distill_spatial.py','--checkpoint',source/'checkpoint.pt','--output',initial,
                 '--light-cnn','--device','cpu','--num-envs','4','--batch-size','2','--ticks','4',
                 '--max-teacher-models','1','--time-budget','30','--no-tensorboard'])
            generated=json.loads((initial/'run_config.json').read_text())
            report=json.loads((initial/'distillation_report.json').read_text())
            self.assertEqual(generated['task'],cfg['task'])
            self.assertEqual(generated['self_play'],cfg['self_play'])
            self.assertEqual(generated['baseline_league'],cfg['baseline_league'])
            self.assertEqual(generated['runner']['algorithm'],cfg['runner']['algorithm'])
            self.assertEqual((report['ticks'],report['samples'],report['Adam_steps']),(4,16,8))
            self.assertEqual(report['observation_semantics'],semantics)
            self.assertEqual(report['actor_inputs'],['policy','single_obs','action_mask'])
            self.assertTrue(report['value_potential_offset'])
            self.assertEqual(report['limits']['saved_datasets'],0)
            self.assertEqual({p.relative_to(source).as_posix():sha256(p.read_bytes()).hexdigest()
                              for p in source.rglob('*') if p.is_file()},immutable)
            initial_state=torch.load(initial/'checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(set(initial_state),{'model_state_dict','migration'})
            light_class='generals_rl.spatial_ppo.LightSpatialPPOPolicy'
            initial_pool=json.loads((initial/'history_pool/manifest.json').read_text())
            self.assertEqual(initial_pool['policy_spec']['policy_class'],light_class)
            self.assertEqual(initial_pool['policy_spec']['encoder_semantics'],semantics)
            source_pool=json.loads((source/'history_pool/manifest.json').read_text())
            self.assertNotEqual(initial_pool['policy_spec_sha256'],source_pool['policy_spec_sha256'])
            for key,replacement in (('policy_class',source_pool['policy_spec']['policy_class']),
                                    ('encoder_semantics','combat_p2_985_v1')):
                incompatible=deepcopy(initial_pool['policy_spec']); incompatible[key]=replacement
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,'policy_spec'):
                    tool.ReadOnlyTeacherPool(initial/'history_pool',factory=lambda:None,
                                             signature=incompatible,device='cpu',max_models=1,scratch=root)
            run(['-m','generals_rl.train','--config',initial/'run_config.json','--replace-policy',
                 '--warm-start',initial/'checkpoint.pt','--iterations','1','--no-eval','--output',root/'ppo'])
            saved=torch.load(root/'ppo/checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(saved['iter'],0)
            self.assertEqual({int(v['step']) for v in saved['optimizer_state_dict']['state'].values()},{8})
            self.assertIsNone(saved['optimizer_stu_enc_state_dict'])
            seed=torch.load(root/'ppo/history_pool/snapshot_00000000.pt',weights_only=True,map_location='cpu')
            assert_nested_equal(self,seed['model_state_dict'],initial_state['model_state_dict'])
            ppo_pool=json.loads((root/'ppo/history_pool/manifest.json').read_text())
            self.assertEqual(ppo_pool['policy_spec_sha256'],initial_pool['policy_spec_sha256'])
            summary=json.loads((root/'ppo/summary.json').read_text())
            self.assertTrue(summary['baseline_league']['batched_observations_active'])
            self.assertEqual(summary['baseline_league']['config'],cfg['baseline_league'])
            self.assertEqual(summary['baseline_league']['native_fallback_calls'],0)
            run(['-m','generals_rl.train','--resume',root/'ppo/checkpoint.pt','--iterations','1',
                 '--no-eval','--output',root/'resume'])
            resumed=torch.load(root/'resume/checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(resumed['iter'],1)
            self.assertEqual({int(v['step']) for v in resumed['optimizer_state_dict']['state'].values()},{16})
            resumed_pool=json.loads((root/'resume/history_pool/manifest.json').read_text())
            self.assertEqual(resumed_pool['policy_spec_sha256'],initial_pool['policy_spec_sha256'])
            self.assertIsNotNone(resumed_pool['resume_source'])
            evaluation=evaluate_builtin(root/'resume/checkpoint.pt',maps_per_ai=1,max_ticks=3,batch_size=8,cpu_threads=1)
            self.assertEqual(evaluation['completed_games'],8)


if __name__=='__main__':
    unittest.main()
