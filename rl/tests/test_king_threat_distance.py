"""King-distance privacy/default contracts and dense seat-weight migration."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from generals import VecEnv
from generals.core.state import GameState, MapLayout
from generals_rl.combat_migration import COMBAT_SEMANTICS, FRAME_DIM, INPUT_LAYERS, migrate_checkpoint
from generals_rl.cts import runner_config
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.king_threat_migration import fold_king_threat_input_weights
from generals_rl.mdp import GameBatch, RewardConfig, make_observation_encoder
from generals_rl.mdp.observations import LocalObservationEncoder, KingThreatDistanceEncoder
from generals_rl.student_ppo import StudentPPOPolicy
from generals_rl.task import TaskConfig
from generals_rl.train import train
from rsl_rl.modules import ActorCriticMoECTS


def distance_state(tick=0, army_count=5, city=False):
    shape=(9,9)
    owner=np.full(shape,-1,np.int16)
    army=np.zeros(shape,np.int64)
    structure=np.zeros(shape,np.uint8)
    for cell,pid,count,kind in ((27,0,8,2),(29,0,2,0),(30,1,army_count,int(city)),
                                (80,1,5,2),(79,1,7,0)):
        owner.flat[cell],army.flat[cell],structure.flat[cell]=pid,count,kind
    terrain=np.zeros(shape,np.uint8)
    terrain.flat[28]=1  # Geometry ignores the wall between King27 and enemy30.
    state=GameState(MapLayout(terrain),structure,owner,army,np.ones(2,bool),np.array([27,80],np.int32),tick)
    state.validate()
    return state


def configuration(class_name='generals_rl.student_ppo.StudentPPOPolicy'):
    cfg={'task':asdict(TaskConfig(num_envs=4,max_episode_steps=3,combat_features=True,
              critic_potential=True,reward=RewardConfig(potential_gamma=.995),opponent='self_play')),
         'runner':runner_config(hidden_dim=16,expert_num=2,rollout_steps=4,learning_epochs=1,mini_batches=2),
         'self_play':{'pool_size':2,'snapshot_interval':1},'evaluation':{'enabled':False},'torch_threads':1}
    cfg['runner']['policy'].update(class_name=class_name,latent_dim=32)
    if class_name.endswith('StudentPPOPolicy'):
        cfg['runner']['policy']['value_potential_offset']=True
    cfg['runner']['algorithm'].update(class_name='generals_rl.student_ppo.StudentPPO',teacher_env_ratio=0,
         schedule='fixed',desired_kl=None,load_balance_coef=0,gamma=.995)
    return cfg


def observations(count=32):
    frames=torch.randn(count,4,FRAME_DIM)
    frames[..., -2]=torch.randint(0,2,(count,4)).float()
    frames[..., -1]=1-frames[..., -2]
    mask=torch.rand(count,649)>.2
    mask[:,0]=True
    old=TensorDict({'policy':frames.flatten(1),'single_obs':frames[:,-1].clone(),
         'critic':frames[:,0].clone(),'critic_potential':torch.rand(count,1),'action_mask':mask},[count])
    new=old.clone()
    for key in ('policy','single_obs','critic'):
        new[key].reshape(count,-1,FRAME_DIM)[...,-1]=torch.rand(count,new[key].shape[-1]//FRAME_DIM)
    return old,new


def policy(cfg,obs):
    opts=deepcopy(cfg['runner']['policy'])
    name=opts.pop('class_name')
    cls=StudentPPOPolicy if name.endswith('StudentPPOPolicy') else ActorCriticMoECTS
    with redirect_stdout(io.StringIO()):
        return cls(obs[:1],cfg['runner']['obs_groups'],649,**opts).eval()


class KingThreatDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self,states):
        backend=VecEnv.from_states(states,device='cpu',auto_reset=False,threads=1)
        self.addCleanup(backend.close)
        return GameBatch.from_backend(backend,'cpu')

    def test_default_distance_growth_privileged_and_guards(self):
        for combat in (False,True):
            default=make_observation_encoder(TaskConfig(combat_features=combat))
            enabled=make_observation_encoder(TaskConfig(combat_features=combat,king_threat_distance=True))
            self.assertIsInstance(enabled,KingThreatDistanceEncoder)
            self.assertIs(type(enabled.base),type(default))
            states=[distance_state(),distance_state(army_count=1),distance_state(1,1,True),distance_state(49,1)]
            batch=self.batch(states)
            ids=torch.zeros(4,dtype=torch.long)
            original=default.encode(batch,ids)
            actual=enabled.encode(batch,ids)
            torch.testing.assert_close(actual[:,:-1],original[:,:-1],rtol=0,atol=0)
            torch.testing.assert_close(actual[:,-1],torch.tensor([3/16,1,3/16,3/16]),rtol=0,atol=0)
            self.assertEqual(tuple(actual.shape),(4,985))
            self.assertTrue(enabled.semantics.endswith('_king_threat_distance_v1'))
            # Seat1 uses its own King80; hostile own0 cell29 is visible from its cell30.
            ids.fill_(1)
            self.assertEqual(enabled.encode(batch,ids)[0,-1].item(),11/16)
            self.assertEqual(enabled.encode(batch,ids,privileged=True)[0,-1].item(),11/16)
            # Defensive guards for a malformed injected batch; no arbitrary cell0 fallback.
            batch.state['alive'][0,1]=False
            self.assertEqual(enabled.encode(batch,ids,privileged=True)[0,-1].item(),1)
            batch.state['structure'][1,80]=0
            self.assertEqual(enabled.encode(batch,ids,privileged=True)[1,-1].item(),1)
            batch.state['structure'][2,79]=2
            self.assertEqual(enabled.encode(batch,ids,privileged=True)[2,-1].item(),1)
            batch.state['alive'][3,0]=False
            self.assertEqual(enabled.encode(batch,ids,privileged=True)[3,-1].item(),1)
        self.assertIs(type(make_observation_encoder(TaskConfig())),LocalObservationEncoder)
        for updates in ({'players':3},):
            with self.assertRaises(ValueError):
                TaskConfig(king_threat_distance=True,**updates)
        with self.assertRaises(TypeError):
            TaskConfig(king_threat_distance=1)

    def test_hidden_army_and_enemy_general_position_cannot_affect_local(self):
        first=distance_state()
        second=first.copy()
        for array in (second.owner,second.army,second.structure):
            array.flat[80],array.flat[8]=array.flat[8],array.flat[80]
            array.flat[79],array.flat[9]=array.flat[9],array.flat[79]
        second.general_pos[1]=8
        second.validate()
        batch=self.batch([first,second])
        ids=torch.zeros(2,dtype=torch.long)
        encoder=make_observation_encoder(TaskConfig(combat_features=True,king_threat_distance=True))
        local=encoder.encode(batch,ids)
        torch.testing.assert_close(local[0],local[1],rtol=0,atol=0)
        privileged=encoder.encode(batch,ids,privileged=True)
        self.assertFalse(torch.equal(privileged[0],privileged[1]))
        torch.testing.assert_close(encoder.encode(batch,ids),local,rtol=0,atol=0)

    def test_dense_h4_latent32_bias_fold_preserves_function_and_source(self):
        for name in ('rsl_rl.modules.ActorCriticMoECTS','generals_rl.student_ppo.StudentPPOPolicy'):
            with self.subTest(policy=name):
                torch.manual_seed(29)
                cfg=configuration(name)
                old,new=observations()
                source,target=policy(cfg,old),policy(cfg,new)
                before=deepcopy(source.state_dict())
                folded,report=fold_king_threat_input_weights(source.state_dict(),cfg['runner']['policy'],cfg['runner']['obs_groups'])
                target.load_state_dict(folded,strict=True)
                self.assertEqual(len(report['changed_layers']),5)
                branches=(False,) if isinstance(source,StudentPPOPolicy) else (False,True)
                with torch.inference_mode():
                    for branch in branches:
                        left=source.action_probabilities(old,is_teacher=branch)
                        right=target.action_probabilities(new,is_teacher=branch)
                        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
                        torch.testing.assert_close(left.argmax(-1),right.argmax(-1),rtol=0,atol=0)
                        torch.testing.assert_close(source.evaluate(old,is_teacher=branch),target.evaluate(new,is_teacher=branch),rtol=1e-5,atol=1e-6)
                changed={entry['weight']:entry for entry in report['changed_layers']}
                changed_bias={entry['bias'] for entry in report['changed_layers']}
                for key,value in before.items():
                    torch.testing.assert_close(source.state_dict()[key],value,rtol=0,atol=0)
                    if key not in changed and key not in changed_bias:
                        torch.testing.assert_close(folded[key],value,rtol=0,atol=0)
                    if key in changed:
                        spec=changed[key]
                        unchanged=torch.ones(value.shape[1],dtype=torch.bool)
                        for frame in range(spec['frames']):
                            seat0=spec['latent_prefix']+frame*985+983
                            unchanged[seat0:seat0+2]=False
                            self.assertEqual(int(torch.count_nonzero(folded[key][:,seat0+1])),0)
                        torch.testing.assert_close(folded[key][:,unchanged],value[:,unchanged],rtol=0,atol=0)
                        if spec['latent_prefix']:
                            self.assertEqual(spec['latent_prefix'],32)
        invalid=deepcopy(cfg['runner']['policy'])
        invalid['actor_obs_normalization']=True
        with self.assertRaises(ValueError):
            fold_king_threat_input_weights(before,invalid,cfg['runner']['obs_groups'])
        malformed=deepcopy(before)
        malformed['actor.network.0.bias']=torch.zeros(1)
        with self.assertRaises(ValueError):
            fold_king_threat_input_weights(malformed,cfg['runner']['policy'],cfg['runner']['obs_groups'])

    def test_checkpoint_metadata_guard_and_legal_combat_then_distance(self):
        cfg=configuration()
        old,_=observations(4)
        weights=policy(cfg,old).state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            checkpoint=root/'snapshot.pt'
            config_path=root/'run_config.json'
            config_path.write_text(json.dumps(cfg))
            # Checkpoint metadata is authoritative even if an old config is supplied.
            for index,metadata in enumerate((
                    {'observation_semantics':'unknown_985_layout'},
                    {'observation_semantics':COMBAT_SEMANTICS+'_king_threat_distance_v1'},
                    {'observation_semantics':'local_p2_985_v1'}, {}, None)):
                with self.subTest(metadata=metadata):
                    torch.save({'model_state_dict':weights,'migration':metadata},checkpoint)
                    original_bytes=checkpoint.read_bytes()
                    output=root/f'rejected_{index}'
                    with self.assertRaisesRegex(ValueError,'checkpoint migration observation semantics'):
                        migrate_checkpoint(checkpoint,output,king_threat_distance=True)
                    self.assertFalse(output.exists())
                    self.assertEqual(checkpoint.read_bytes(),original_bytes)
            # The original default migration can still be composed with the new fold.
            original_cfg=deepcopy(cfg)
            original_cfg['task']['combat_features']=False
            config_path.write_text(json.dumps(original_cfg))
            torch.save({'model_state_dict':weights},checkpoint)
            migrate_checkpoint(checkpoint,root/'combat')
            combat_checkpoint=root/'combat/checkpoint.pt'
            combat_bytes=combat_checkpoint.read_bytes()
            report=migrate_checkpoint(combat_checkpoint,root/'distance',king_threat_distance=True)
            self.assertEqual(report['observation_semantics'],COMBAT_SEMANTICS+'_king_threat_distance_v1')
            self.assertEqual(combat_checkpoint.read_bytes(),combat_bytes)
            # Supplying the pre-combat config for combat weights must also fail.
            with self.assertRaisesRegex(ValueError,'checkpoint migration observation semantics'):
                migrate_checkpoint(combat_checkpoint,root/'mismatched',config_path=config_path,
                                   king_threat_distance=True)
            self.assertFalse((root/'mismatched').exists())
            # Reusing an old matching base config cannot repeat distance migration.
            with self.assertRaisesRegex(ValueError,'checkpoint migration observation semantics'):
                migrate_checkpoint(root/'distance/checkpoint.pt',root/'repeat',
                                   config_path=root/'combat/run_config.json',king_threat_distance=True)
            self.assertFalse((root/'repeat').exists())

    def test_weights_only_cli_and_real_cpu_ppo_smoke(self):
        torch.manual_seed(37)
        cfg=configuration()
        old,_=observations(4)
        model=policy(cfg,old)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'source'; source.mkdir()
            checkpoint=source/'snapshot.pt'
            torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':{'old':True},'iter':99},checkpoint)
            checkpoint.with_name('run_config.json').write_text(json.dumps(cfg))
            original_bytes=checkpoint.read_bytes()
            script=Path(__file__).resolve().parents[1]/'tools/migrate_combat_observations.py'
            result=subprocess.run([sys.executable,str(script),'--checkpoint',str(checkpoint),'--output',str(root/'migrated'),
                                   '--king-threat-distance'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            weights=torch.load(root/'migrated/checkpoint.pt',weights_only=True,map_location='cpu')
            self.assertEqual(set(weights),{'model_state_dict','migration'})
            self.assertEqual(weights['migration']['source']['checkpoint_sha256'],sha256(original_bytes).hexdigest())
            self.assertEqual(checkpoint.read_bytes(),original_bytes)
            migrated=json.loads((root/'migrated/run_config.json').read_text())
            self.assertTrue(migrated['task']['combat_features'] and migrated['task']['king_threat_distance'])
            self.assertTrue(migrated['runner']['policy']['value_potential_offset'])
            with redirect_stdout(io.StringIO()):
                summary=train(migrated,root/'train',iterations=1,warm_start=root/'migrated/checkpoint.pt')
            self.assertEqual((summary['environment_steps'],summary['teacher_envs']),(16,0))
            manifest=json.loads((root/'train/history_pool/manifest.json').read_text())
            self.assertTrue(manifest['policy_spec']['encoder_semantics'].endswith('_king_threat_distance_v1'))
            self.assertEqual(manifest['policy_spec']['observation_shapes']['policy'],[3940])
            with redirect_stdout(io.StringIO()):
                report=evaluate_builtin(root/'train/checkpoint.pt',maps_per_ai=1,max_ticks=3,batch_size=8,cpu_threads=1)
            self.assertEqual(report['completed_games'],8)
            # Already-distance config must reject a second fold before any output.
            again=subprocess.run([sys.executable,str(script),'--checkpoint',str(root/'migrated/checkpoint.pt'),
                     '--output',str(root/'again'),'--king-threat-distance'],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(again.returncode,0)
            self.assertFalse((root/'again').exists())
            invalid=deepcopy(cfg)
            invalid['observation_semantics']='unknown_985_layout'
            checkpoint.with_name('run_config.json').write_text(json.dumps(invalid))
            invalid_result=subprocess.run([sys.executable,str(script),'--checkpoint',str(checkpoint),
                     '--output',str(root/'unknown'),'--king-threat-distance'],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(invalid_result.returncode,0)
            self.assertFalse((root/'unknown').exists())


if __name__=='__main__':
    unittest.main()
