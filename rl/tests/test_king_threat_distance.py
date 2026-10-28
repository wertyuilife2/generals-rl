"""King-distance observation semantics, growth boundaries and information isolation."""
import unittest
import numpy as np
import torch
from generals import VecEnv
from generals.core.state import GameState, MapLayout
from generals_rl.mdp import GameBatch, make_observation_encoder
from generals_rl.mdp.observations import LocalObservationEncoder, KingThreatDistanceEncoder
from generals_rl.task import TaskConfig


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
            default=make_observation_encoder(TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, combat_features=combat))
            enabled=make_observation_encoder(TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, combat_features=combat,king_threat_distance=True))
            self.assertIsInstance(enabled,KingThreatDistanceEncoder)
            self.assertIs(type(enabled.base),type(default))
            states=[distance_state(),distance_state(army_count=1),distance_state(1,1,True),distance_state(49,1)]
            batch=self.batch(states)
            ids=torch.zeros(4,dtype=torch.long)
            original=default.encode(batch,ids)
            actual=enabled.encode(batch,ids)
            torch.testing.assert_close(actual[:,:-1],original[:,:-1],rtol=0,atol=0)
            torch.testing.assert_close(actual[:,-1],torch.tensor([3/16,1,1,1]),rtol=0,atol=0)
            self.assertEqual(tuple(actual.shape),(4,1147))
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
        self.assertIs(type(make_observation_encoder(TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0))),LocalObservationEncoder)
        for updates in ({'players':3},):
            with self.assertRaises(ValueError):
                TaskConfig(king_threat_distance=True,**updates)
        with self.assertRaises(TypeError):
            TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, king_threat_distance=1)

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
        encoder=make_observation_encoder(TaskConfig(width=9, height=9, random_maps=False, map_pool_size=0, combat_features=True,king_threat_distance=True))
        local=encoder.encode(batch,ids)
        torch.testing.assert_close(local[0],local[1],rtol=0,atol=0)
        privileged=encoder.encode(batch,ids,privileged=True)
        self.assertFalse(torch.equal(privileged[0],privileged[1]))
        torch.testing.assert_close(encoder.encode(batch,ids),local,rtol=0,atol=0)





if __name__=='__main__':
    unittest.main()
