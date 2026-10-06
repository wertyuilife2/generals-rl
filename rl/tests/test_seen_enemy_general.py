"""Targeted isolated candidate contracts using existing fixed Core fixtures."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from tensordict import TensorDict
from generals import VecEnv
from generals.core.batch import pack_states
from generals.core.engine import CoreEngine
from generals.core.observation import observe
from generals.core.state import Action, Structure
from generals_rl.mdp import GameBatch, ObservationHistory, make_observation_encoder
from generals_rl.mdp.seen_general import SeenEnemyGeneralEncoder
from generals_rl.evaluate_builtin import _local_inputs
from generals_rl.self_play import HistoricalPolicyPool
from generals_rl.self_play_training import prepare_self_play, SelfPlaySession
from generals_rl.spatial_ppo import LightSpatialPPOPolicy, OwnKingRelativeGeometry
from generals_rl.task import GeneralsTask, TaskConfig
from test_king_threat_distance import distance_state
from test_mdp import example_state, hidden_world_variant, capture_state
from test_self_play_task import TestPool

ROOT=Path('/home/yuhangwang/code/generals-rl')


def existing_variant(*, seen=False):
    state=distance_state()
    if seen:
        # Deterministic visibility variant, no generated map or new seed.
        for cell,pid in ((70,0),(18,1)):
            state.owner.flat[cell],state.army.flat[cell]=pid,2
    state.validate()
    return state


def batch(states):
    arrays,w,h=pack_states(states)
    fields={k:torch.from_numpy(v) for k,v in arrays.items()}
    stats=torch.zeros(len(states),2,3,dtype=torch.long)
    for pid in (0,1):
        own=fields['owner']==pid
        stats[:,pid]=torch.stack(((fields['army']*own).sum(1),own.sum(1),(own&(fields['structure']==1)).sum(1)),1)
    return GameBatch(fields,stats,w,h)


def config(n=2,**extra):
    return TaskConfig(num_envs=n,opponent='self_play',backend='cpu',device='cpu',history_length=4,
                      max_episode_steps=1024,combat_features=True,king_threat_distance=True,
                      productive_actions=True,remember_enemy_general=True,**extra)


def learn_both(encoder, b):
    for pid in (0,1):
        encoder.encode(b,torch.full((b.num_envs,),pid,dtype=torch.long))


class SeenEnemyGeneralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    def test_current_core_visibility_oracle_both_seats_mixed_lanes(self):
        seen,unknown=existing_variant(seen=True),existing_variant()
        b=batch([seen,unknown,seen,unknown]);encoder=make_observation_encoder(config(4));encoder.reset(b)
        self.assertEqual(encoder.positions.tolist(),[[-1,-1]]*4)
        ids=torch.tensor([0,0,1,1]);out=encoder.encode(b,ids)
        self.assertEqual(encoder.positions.tolist(),[[80,-1],[-1,-1],[-1,27],[-1,-1]])
        self.assertEqual(tuple(out.shape),(4,985))
        for lane,pid in enumerate(ids.tolist()):
            obs=observe([seen,unknown,seen,unknown][lane],pid)
            true_visible_enemy=(obs.visible & (obs.structure==Structure.GENERAL) & (obs.owner==1-pid))
            target=80 if pid==0 else 27
            self.assertEqual(bool(true_visible_enemy.flat[target]),lane%2==0)
            self.assertEqual(int(encoder.positions[lane,pid]),target if lane%2==0 else -1)
        encoder.encode(b,1-ids)
        self.assertEqual(encoder.positions.tolist(),[[80,27],[-1,-1],[80,27],[-1,-1]])

    def test_four_pushes_forget_base_but_keep_landmark_and_hidden_fields(self):
        encoder=make_observation_encoder(config());seen=batch([existing_variant(seen=True)]*2)
        ids=torch.tensor([0,1]);encoder.reset(seen);history=ObservationHistory(4);history.reset(encoder.encode(seen,ids))
        hidden=batch([existing_variant()]*2);targets=torch.tensor([80,27])
        for tick in range(1,6):
            hidden.state['tick'][:]=tick;current=encoder.encode(hidden,ids);history.push(current)
            base=encoder.base.encode(hidden,ids)
            self.assertTrue(bool(base[torch.arange(2),4*81+targets].eq(0).all()))
            expected=base.clone();expected[torch.arange(2),4*81+targets]=1
            torch.testing.assert_close(current,expected,rtol=0,atol=0)
            self.assertTrue(bool(current[torch.arange(2),targets].eq(0).all()))  # visible stays hidden
            self.assertTrue(bool(current[torch.arange(2),6*81+targets].eq(0).all()))  # current enemy stays hidden
        self.assertTrue(bool(history.frames[:,:,4*81+targets[0]][0].eq(1).all()))
        self.assertTrue(bool(history.frames[:,:,4*81+targets[1]][1].eq(1).all()))
        self.assertEqual(tuple(history.flattened().shape),(2,3940))

    def test_never_seen_privileged_and_hidden_world_perturbation_do_not_update(self):
        cfg=TaskConfig(width=5,height=5,num_envs=2,remember_enemy_general=True)
        encoder=make_observation_encoder(cfg);first=example_state();second=hidden_world_variant(first)
        b=batch([first,second]);encoder.reset(b);ids=torch.zeros(2,dtype=torch.long)
        self.assertEqual(encoder.positions[:,0].tolist(),[-1,-1])
        before=encoder.positions.clone();local=encoder.encode(b,ids)
        torch.testing.assert_close(local[0],local[1],rtol=0,atol=0)
        torch.testing.assert_close(local,encoder.base.encode(b,ids),rtol=0,atol=0)
        critic=encoder.encode(b,ids,privileged=True)
        self.assertFalse(torch.equal(critic[0],critic[1]))
        torch.testing.assert_close(encoder.positions,before,rtol=0,atol=0)
        torch.testing.assert_close(critic,encoder.base.encode(b,ids,privileged=True),rtol=0,atol=0)
        # Equal observed histories but changed hidden king after losing vision:
        e=make_observation_encoder(config());s=batch([existing_variant(seen=True)]*2);e.reset(s);e.encode(s,ids)
        hidden=existing_variant();variant=hidden.copy()
        for field in ('owner','army','structure'):
            values=getattr(variant,field);values.flat[80],values.flat[8]=values.flat[8],values.flat[80]
        variant.general_pos[1]=8;variant.validate()
        h=batch([hidden,variant]);out=e.encode(h,ids)
        torch.testing.assert_close(out[0],out[1],rtol=0,atol=0)
        self.assertEqual(e.positions[:,0].tolist(),[80,80])

    def test_partial_reset_clone_and_repeat_reads_are_independent(self):
        e=make_observation_encoder(config());ids=torch.tensor([0,1]);seen=batch([existing_variant(seen=True)]*2);e.reset(seen)
        learn_both(e,seen)
        h=batch([existing_variant()]*2);history=ObservationHistory(4);history.reset(e.encode(h,ids))
        frames=history.frames.clone();positions=e.positions.clone()
        for _ in range(3):e.encode(h,ids);e.encode(h,ids,privileged=True)
        torch.testing.assert_close(frames,history.frames,rtol=0,atol=0)
        torch.testing.assert_close(positions,e.positions,rtol=0,atol=0)
        fork=e.clone();fork.reset(h,torch.tensor([True,False]))
        self.assertEqual(fork.positions.tolist(),[[-1,-1],[80,27]])
        torch.testing.assert_close(e.positions,positions,rtol=0,atol=0)
        e.reset(h,torch.tensor([False,True]));self.assertEqual(e.positions.tolist(),[[80,27],[-1,-1]])
        self.assertEqual(e.positions.dtype,torch.int16)
        self.assertEqual(e.positions.numel()*e.positions.element_size(),8)

    def test_task_opponent_eval_share_marker_without_advancing_history_on_read(self):
        states=[existing_variant(seen=True)]*2;cfg=config();backend=VecEnv.from_states(states,threads=1,auto_reset=False,episode_ticks=1024)
        with GeneralsTask(cfg,backend=backend) as task:
            pool=TestPool();task.opponent.attach(pool,stochastic=False,seed=100)  # existing test seed
            # Remove only fixed explorers from current state; initial map remains
            # unchanged. Rebuild native caches before next real WAIT tick.
            for cell in (70,18):task.backend.state['owner'][:,cell]=-1;task.backend.state['army'][:,cell]=0
            task.backend.invalidate_cache()
            with patch.object(task.observation_encoder.base,'encode',wraps=task.observation_encoder.base.encode) as encoded:
                for _ in range(5):task.step(torch.zeros(2,dtype=torch.long))
            # Existing lifecycle only: joint_actions2 + learnerpush1 +
            # opponentpush2 + single_obs1, and one privileged critic per tick.
            self.assertEqual(sum(not c.kwargs.get('privileged',False) for c in encoded.call_args_list),30)
            self.assertEqual(sum(c.kwargs.get('privileged',False) for c in encoded.call_args_list),5)
            obs=task.get_observations();self.assertEqual(tuple(obs['policy'].shape),(2,3940));self.assertEqual(tuple(obs['critic'].shape),(2,985))
            positions=task.observation_encoder.positions.clone();lh=task.history.frames.clone();oh=[h.frames.clone() for h in task.opponent.histories]
            local=_local_inputs(task.batch(),task.player_ids,task.observation_encoder,task.history,task.action_space)
            self.assertEqual(set(local.keys()),{'policy','single_obs','action_mask'})
            task.opponent.joint_actions(task.batch(),task.player_ids,task.action_space,task.observation_encoder)
            torch.testing.assert_close(lh,task.history.frames,rtol=0,atol=0)
            for before,h in zip(oh,task.opponent.histories):torch.testing.assert_close(before,h.frames,rtol=0,atol=0)
            torch.testing.assert_close(positions,task.observation_encoder.positions,rtol=0,atol=0)
            for pid,h in enumerate(task.opponent.histories):
                target=80 if pid==0 else 27;self.assertTrue(bool(h.frames[:,-1,4*81+target].eq(1).all()))
            # Reset lane0 to a fixed never-seen state with rotation; bothbanks
            # must clear. Replacing initial arrays is fixture setup, no map seed.
            arrays,_,_=pack_states([existing_variant()]*2)
            for k,v in arrays.items():task.backend.unwrapped.initial[k][0]=v[0]
            lane1=task.observation_encoder.positions[1].clone();task.reset(torch.tensor([True,False]),rotate=True)
            self.assertEqual(task.observation_encoder.positions[0].tolist(),[-1,-1])
            torch.testing.assert_close(task.observation_encoder.positions[1],lane1,rtol=0,atol=0)
            self.assertEqual(task.player_ids.tolist(),[1,1])

    def test_eval_lifecycle_uses_same_memory_no_schedule_or_models(self):
        # Drive actual battle_batch with existing fixed states/spec seeds and a
        # WAIT interface. This is a six-tick unit fixture, not a baseline eval.
        from generals_rl.evaluate_builtin import battle_batch,GameSpec
        from test_evaluate_builtin import WaitStudent
        seen=existing_variant(seen=True);student=WaitStudent();created=[];encoded=[]
        class WaitController:
            def act(self,obs):return Action.wait()
            def on_result(self,result):pass
        def encoder_factory(cfg):
            e=make_observation_encoder(cfg);e.base.encode=Mock(wraps=e.base.encode);created.append(e);encoded.append(e.base.encode);return e
        from generals.backends.cpu.env import CPUEnv
        original_step=CPUEnv.step
        def step(env,actions):
            result=original_step(env,actions)
            if env.state['tick'][0]==1:
                for cell in (70,18):env.state['owner'][:,cell]=-1;env.state['army'][:,cell]=0
                env.invalidate_cache()
            return result
        specs=[GameSpec(0,'random',0,5,6,0),GameSpec(1,'random',0,5,6,1)]  # existing eval fixture seeds
        with patch('generals_rl.evaluate_builtin.make_observation_encoder',side_effect=encoder_factory),patch('generals_rl.evaluate_builtin.make_controllers',side_effect=lambda kinds,seed:[WaitController(),WaitController()]),patch.object(CPUEnv,'step',step):
            games,_=battle_batch(specs,[seen,seen.copy()],student,config(),max_ticks=6)
        self.assertEqual(len(games),2);self.assertEqual(len(student.inputs),6)
        for frame in student.inputs[1:]:
            self.assertEqual(set(frame.keys()),{'policy','single_obs','action_mask'})
            for lane,target in enumerate((80,27)):
                self.assertEqual(frame['single_obs'][lane,4*81+target].item(),1)
                self.assertEqual(frame['single_obs'][lane,target].item(),0)
        # Eval need not encode the controller's observer bank at all.
        self.assertEqual(created[0].positions.tolist(),[[80,-1],[-1,27]])
        self.assertEqual(len(encoded[0].call_args_list),13)  # initial1 +6*(input1+push1)
        self.assertTrue(all(not c.kwargs.get('privileged',False) for c in encoded[0].call_args_list))
        self.assertTrue(all(torch.equal(c.args[1],torch.tensor([0,1])) for c in encoded[0].call_args_list))

    def test_light_actor_privacy_and_shapes_critic_exact_no_old_weights(self):
        e=make_observation_encoder(config());b=batch([existing_variant(seen=True)]*2);e.reset(b);ids=torch.tensor([0,1]);e.encode(b,ids);b=batch([existing_variant()]*2);current=e.encode(b,ids)
        obs=TensorDict({'policy':current[:,None].expand(-1,4,-1).reshape(2,3940),'single_obs':current,'critic':e.encode(b,ids,privileged=True),'action_mask':torch.ones(2,649,dtype=torch.bool)},[2])
        torch.random.default_generator.manual_seed(137)  # existing own-king fixture seed
        policy=LightSpatialPPOPolicy(obs,{'policy':['policy'],'critic':['critic']},649,own_king_geometry=True,actor_obs_normalization=False,critic_obs_normalization=False)
        before=policy.action_probabilities(obs).detach().clone();changed=obs.clone();changed['critic']+=100;changed['single_obs']+=100
        torch.testing.assert_close(before,policy.action_probabilities(changed),rtol=0,atol=0)
        torch.testing.assert_close(before,policy.action_probabilities(obs.select('policy','action_mask')),rtol=0,atol=0)
        self.assertEqual(len(policy.state_dict()),22);self.assertEqual(sum(p.numel() for p in policy.parameters()),15786)
        geometry=OwnKingRelativeGeometry();maps=obs['policy'].reshape(2,4,985)[:,:,:972].reshape(2,48,9,9)
        base=e.base.encode(b,ids);basemaps=base[:,None,:972].expand(-1,4,-1).reshape(2,48,9,9)
        torch.testing.assert_close(geometry(maps)[:,-2:],geometry(basemaps)[:,-2:],rtol=0,atol=0)
        self.assertEqual(tuple(policy.evaluate(obs).shape),(2,1))

    def test_default_config_unchanged_and_real_pool_signature_guards(self):
        from test_combat_observations import small_runner
        from generals_rl.mdp.observations import LocalObservationEncoder
        first=example_state();b=batch([first]);disabled=make_observation_encoder(TaskConfig(width=5,height=5))
        torch.testing.assert_close(disabled.encode(b,torch.tensor([0])),LocalObservationEncoder(512).encode(b,torch.tensor([0])),rtol=0,atol=0)
        with self.assertRaises(TypeError):TaskConfig(remember_enemy_general=1)
        with self.assertRaises(ValueError):TaskConfig(players=3,remember_enemy_general=True)
        with tempfile.TemporaryDirectory() as d:
            pools=[]
            for enabled in (False,True):
                cfg=TaskConfig(width=5,height=5,num_envs=1,opponent='self_play',remember_enemy_general=enabled)
                backend=VecEnv.from_states([first],threads=1,auto_reset=False,episode_ticks=512)
                with GeneralsTask(cfg,backend=backend) as task:
                    runner,settings=small_runner(task)
                    with patch.object(SelfPlaySession,'publish'),patch.object(task.opponent,'attach'):
                        session=prepare_self_play(task,runner,settings,Path(d)/str(enabled))
                    pools.append(session.pool)
            self.assertNotEqual(pools[0].policy_spec_sha256,pools[1].policy_spec_sha256)
            self.assertTrue(pools[1].policy_spec['encoder_semantics'].endswith('_seen_enemy_general_v1'))
            factory=Mock(side_effect=AssertionError('factory cannot run before signature rejection'))
            with self.assertRaisesRegex(ValueError,'spec'):
                HistoricalPolicyPool(Path(d)/'reload',policy_factory=factory,policy_spec=pools[1].policy_spec,device='cpu',max_size=2,resume_from=pools[0].directory)
            factory.assert_not_called()

    def test_default_injected_legacy_encoder_keeps_original_interface(self):
        from generals_rl.mdp.observations import LocalObservationEncoder
        class LegacyEncoder:
            def __init__(self):self.base=LocalObservationEncoder(4)
            def encode(self,*args,**kwargs):return self.base.encode(*args,**kwargs)
            def visibility(self,*args,**kwargs):return self.base.visibility(*args,**kwargs)
        state=example_state();cfg=TaskConfig(width=5,height=5,num_envs=1,max_episode_steps=4)
        backend=VecEnv.from_states([state],threads=1,auto_reset=False,episode_ticks=4)
        with GeneralsTask(cfg,backend=backend,observation_encoder=LegacyEncoder()) as task:
            task.step(torch.tensor([0]));task.reset()
            self.assertEqual(task.batch().state['tick'].item(),0)
            self.assertFalse(hasattr(task.observation_encoder,'reset'))
            self.assertFalse(hasattr(task.observation_encoder,'observe'))

    def test_core_enemy_general_capture_is_terminal_and_public_death_clears(self):
        state=capture_state();result=CoreEngine(state.copy()).step([Action.move(3,1),Action.wait()]);self.assertTrue(result.terminated)
        b=batch([state]);encoder=make_observation_encoder(TaskConfig(width=3,height=3,remember_enemy_general=True));encoder.reset(b)
        encoder.encode(b,torch.tensor([0]))
        self.assertEqual(encoder.positions[0,0].item(),4)
        engine=CoreEngine(state.copy());engine.step([Action.move(3,1),Action.wait()]);encoder.encode(batch([engine.state]),torch.tensor([0]))
        self.assertEqual(encoder.positions[0,0].item(),-1)

    def test_reuses_one_local_encode_reset_zero_and_privileged_never_observes(self):
        e=make_observation_encoder(config());b=batch([existing_variant(seen=True)]*2);ids=torch.tensor([0,1])
        with patch.object(e.base,'encode',wraps=e.base.encode) as encoded:
            e.reset(b);self.assertEqual(encoded.call_count,0)
            e.encode(b,ids);self.assertEqual(encoded.call_count,1)
            self.assertEqual(e.positions.tolist(),[[80,-1],[-1,27]])
            positions=e.positions.clone();e.encode(b,ids,privileged=True)
            self.assertEqual(encoded.call_count,2)
            torch.testing.assert_close(e.positions,positions,rtol=0,atol=0)
            e.encode(b,ids);self.assertEqual(encoded.call_count,3)
            torch.testing.assert_close(e.positions,positions,rtol=0,atol=0)
            e.reset(b,torch.tensor([True,False]));self.assertEqual(encoded.call_count,3)
            self.assertEqual(e.positions.tolist(),[[-1,-1],[-1,27]])


if __name__=='__main__':unittest.main()
