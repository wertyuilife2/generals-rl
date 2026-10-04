"""Small native contracts for the optional combat observation layout."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

from generals import VecEnv
from generals.core.engine import CoreEngine
from generals.core.state import Action, GameState, MapLayout
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.mdp import (CombatObservationEncoder, GameBatch, LocalObservationEncoder,
                             make_observation_encoder)
from generals_rl.self_play_training import prepare_self_play
from generals_rl.student_ppo import StudentPPOPolicy
from generals_rl.task import GeneralsTask, TaskConfig

torch.set_num_threads(1)


def combat_state(tick=0):
    shape=(9,9)
    structure=np.zeros(shape,np.uint8)
    owner=np.full(shape,-1,np.int16)
    army=np.zeros(shape,np.int64)
    for cell,pid,count,kind in ((27,0,2,2),(66,1,9,2),(28,1,4,0),
                                (29,0,9,0),(37,0,4,0),(75,0,3,0),(80,1,8,1)):
        structure.flat[cell],owner.flat[cell],army.flat[cell]=kind,pid,count
    state=GameState(MapLayout(np.zeros(shape,np.uint8)),structure,owner,army,
                    np.ones(2,bool),np.array([27,66],np.int32),tick)
    state.validate()
    return state


def small_runner(task):
    groups={"policy":["policy"],"critic":["critic"]}
    options={"class_name":"generals_rl.student_ppo:StudentPPOPolicy",
             "action_distribution":"categorical","actor_hidden_dims":[16],
             "critic_hidden_dims":[16],"teacher_encoder_hidden_dims":[16],
             "student_encoder_hidden_dims":[16],"latent_dim":8,"expert_num":2,
             "actor_obs_normalization":False,"critic_obs_normalization":False}
    obs=TensorDict(task.get_observations(),batch_size=[task.num_envs])
    kwargs={key:value for key,value in options.items() if key!="class_name"}
    with redirect_stdout(io.StringIO()):
        policy=StudentPPOPolicy(obs,groups,task.action_space.size,**kwargs)
    runner=SimpleNamespace(alg=SimpleNamespace(policy=policy),cfg={"obs_groups":groups},
                           env=SimpleNamespace(get_observations=lambda:obs))
    return runner,{"runner":{"policy":options,"obs_groups":groups},
                   "self_play":{"pool_size":2,"stochastic":False},"torch_threads":1}


class CombatObservationTests(unittest.TestCase):
    def backend(self,states,episode_ticks=0):
        env=VecEnv.from_states(states,device="cpu",threads=1,auto_reset=False,
                               episode_ticks=episode_ticks)
        self.addCleanup(env.close)
        return env

    def task(self,*,combat,opponent="wait",episode_ticks=8):
        cfg=TaskConfig(num_envs=2,max_episode_steps=episode_ticks,history_length=4,
                       opponent=opponent,combat_features=combat,
                       mountain_density=0,city_density=0)
        task=GeneralsTask(cfg,backend=self.backend([combat_state(),combat_state()],episode_ticks))
        self.addCleanup(task.close)
        return task

    def test_factory_defaults_and_geometry_validation(self):
        original=make_observation_encoder(TaskConfig())
        self.assertIs(type(original),LocalObservationEncoder)
        self.assertIs(type(make_observation_encoder(TaskConfig(combat_features=True))),
                      CombatObservationEncoder)
        for updates in ({"players":3},{"width":8},{"height":8}):
            with self.assertRaises(ValueError):
                TaskConfig(combat_features=True,**updates)
        with self.assertRaises(TypeError):
            TaskConfig(combat_features=1)
        batch=GameBatch.from_backend(self.backend([combat_state()]),"cpu")
        ids=torch.tensor([0])
        torch.testing.assert_close(original.encode(batch,ids),
                                   LocalObservationEncoder(512).encode(batch,ids),rtol=0,atol=0)

    def test_potential_uses_core_growth_and_correct_combat_targets(self):
        states=[combat_state(tick) for tick in (1,49,184)]
        batch=GameBatch.from_backend(self.backend(states),"cpu")
        encoded=CombatObservationEncoder(1024).encode(batch,torch.zeros(3,dtype=torch.long))
        self.assertEqual(tuple(encoded.shape),(3,985))
        for lane,state in enumerate(states):
            engine=CoreEngine(state.copy())
            engine.begin_tick()
            army=engine.state.army.ravel()
            risk=max(0,int(army[28])-1)
            attack=max(0,int(army[29])-1)-int(army[28])
            own_margin=int(army[27])-risk
            torch.testing.assert_close(encoded[lane,810+27],torch.log1p(torch.tensor(float(risk)))/10)
            torch.testing.assert_close(encoded[lane,891+27],
                torch.tensor(float(np.sign(own_margin)))*torch.log1p(torch.tensor(float(abs(own_margin))))/10)
            torch.testing.assert_close(encoded[lane,891+28],
                torch.tensor(float(np.sign(attack)))*torch.log1p(torch.tensor(float(abs(attack))))/10)
        # The recorded-failure geometry: own general at risk, a winning adjacent
        # defense at28, and an underpowered attack on enemy general66.
        torch.testing.assert_close(encoded[2,891+28],torch.log1p(torch.tensor(4.))/10)
        torch.testing.assert_close(encoded[2,891+66],-torch.log1p(torch.tensor(7.))/10)

    def test_hidden_world_changes_do_not_leak_or_persist_after_privileged_encode(self):
        first=combat_state()
        second=first.copy()
        for array in (second.structure,second.owner,second.army):
            array.flat[79],array.flat[80]=array.flat[80],array.flat[79]
        terrain=second.layout.terrain.copy()
        terrain.flat[8]=1
        second.layout=MapLayout(terrain)
        second.validate()
        batch=GameBatch.from_backend(self.backend([first,second]),"cpu")
        ids=torch.tensor([0,0])
        encoder=CombatObservationEncoder(1024)
        local=encoder.encode(batch,ids)
        torch.testing.assert_close(local[0],local[1],rtol=0,atol=0)
        privileged=encoder.encode(batch,ids,privileged=True)
        self.assertFalse(torch.equal(privileged[0],privileged[1]))
        torch.testing.assert_close(encoder.encode(batch,ids),local,rtol=0,atol=0)

    def test_native_task_history_partial_reset_and_reward_keep_same_boundary(self):
        original=self.task(combat=False)
        combat=self.task(combat=True)
        self.assertIsInstance(combat.observation_encoder,CombatObservationEncoder)
        for _ in range(2):
            before=original.step(torch.zeros(2,dtype=torch.long))
            after=combat.step(torch.zeros(2,dtype=torch.long))
            torch.testing.assert_close(before.reward,after.reward,rtol=0,atol=0)
            self.assertEqual(tuple(after.observations["policy"].shape),(2,3940))
            self.assertEqual(tuple(after.observations["action_mask"].shape),(2,649))
        continuing=combat.history.frames[1].clone()
        combat.reset(torch.tensor([True,False]),rotate=True)
        torch.testing.assert_close(combat.history.frames[1],continuing,rtol=0,atol=0)
        current=combat.observation_encoder.encode(combat.batch(),combat.player_ids)
        torch.testing.assert_close(combat.history.frames[0],current[0].expand(4,-1),rtol=0,atol=0)

    def test_self_play_encoder_signature_and_real_evaluator_factory(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            original=self.task(combat=False,opponent="self_play")
            old_runner,config=small_runner(original)
            old=prepare_self_play(original,old_runner,config,root/"original")
            old_checkpoint=root/"original/checkpoint.pt"
            torch.save({"model_state_dict":old_runner.alg.policy.state_dict()},old_checkpoint)
            combat=self.task(combat=True,opponent="self_play")
            runner,config=small_runner(combat)
            with self.assertRaisesRegex(ValueError,"policy_spec"):
                prepare_self_play(combat,runner,config,root/"rejected",resume=old_checkpoint)
            session=prepare_self_play(combat,runner,config,root/"combat")
            self.assertNotEqual(old.pool.policy_spec["encoder"],session.pool.policy_spec["encoder"])
            self.assertNotIn("encoder_semantics",old.pool.policy_spec)
            self.assertEqual(session.pool.policy_spec["encoder_semantics"],CombatObservationEncoder.semantics)
            for pid,history in enumerate(combat.opponent.histories):
                expected=combat.observation_encoder.encode(combat.batch(),torch.full_like(combat.player_ids,pid))
                torch.testing.assert_close(history.frames,expected[:,None].expand(-1,4,-1),rtol=0,atol=0)
            combat.step(torch.zeros(2,dtype=torch.long))  # Actual historical policy opponent.
            destination=root/"combat/checkpoint.pt"
            torch.save({"model_state_dict":runner.alg.policy.state_dict()},destination)
            with patch.object(CombatObservationEncoder,"semantics","combat_p2_985_v2"), \
                    self.assertRaisesRegex(ValueError,"policy_spec"):
                prepare_self_play(combat,runner,config,root/"semantic_rejected",resume=destination)
            config["task"]=asdict(combat.config)
            destination.with_name("run_config.json").write_text(json.dumps(config))
            with patch("generals_rl.evaluate_builtin.make_observation_encoder",
                       wraps=make_observation_encoder) as factory,redirect_stdout(io.StringIO()):
                report=evaluate_builtin(destination,maps_per_ai=1,batch_size=8,max_ticks=2,
                                        seed=91234,device="cpu",cpu_threads=1)
            self.assertTrue(report["combat_features"])
            self.assertEqual(report["completed_games"],8)
            self.assertEqual(set(report["by_ai"]),{"random","defensive","expansion","aggressive"})
            self.assertEqual(factory.call_count,2)  # Bootstrap shapes and native matches.
            self.assertTrue(all(call.args[0].combat_features for call in factory.call_args_list))


if __name__=="__main__":
    unittest.main()
