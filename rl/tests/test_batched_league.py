"""Focused integration contracts; unchanged frozen v6 is the disabled reference."""
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from generals_rl.baseline_league import BaselineLeagueConfig, make_league_opponent
from generals_rl.batched_local_observation import BatchedObservationBridge
from generals_rl.league_bridge import OriginalBaselineBridge
from generals_rl.task import GeneralsTask, TaskConfig


class LegalPolicy:
    def action_probabilities(self, observations, *, is_teacher=False):
        assert not is_teacher
        values = observations["action_mask"].float()
        return values / values.sum(1, keepdim=True)


@dataclass
class Snapshot:
    snapshot_id: int
    model: LegalPolicy


class Pool:
    ids = (0,)
    def __init__(self): self.snapshot = Snapshot(0, LegalPolicy())
    def sample(self, count, *, generator): return [self.snapshot] * count


def controller_equal(a, b):
    assert a.player_id == b.player_id and a.rng.bit_generator.state == b.rng.bit_generator.state
    for name in ("terrain", "structure", "owner", "army", "last_seen"):
        x,y=getattr(a,name),getattr(b,name)
        if x is None or y is None: assert x is y
        else: np.testing.assert_array_equal(x,y)
    for name in ("path", "path_strategy", "path_tick", "_pending"):
        assert getattr(a,name)==getattr(b,name)


class BatchedLeagueContracts(unittest.TestCase):
    def test_config_bool_default_and_actual_bridge_selection(self):
        self.assertIs(BaselineLeagueConfig().batched_observations,False)
        for value in (1,None,"true"):
            with self.assertRaises(TypeError):BaselineLeagueConfig(batched_observations=value)
        task=TaskConfig(num_envs=50,opponent="self_play")
        self.assertIsNone(make_league_opponent(task,BaselineLeagueConfig(batched_observations=True)))
        for flag,expected in ((False,OriginalBaselineBridge),(True,BatchedObservationBridge)):
            opponent=make_league_opponent(task,BaselineLeagueConfig(baseline_envs=1,batched_observations=flag))
            self.assertIs(type(opponent.bridge),expected)
            self.assertIs(bool(getattr(opponent.bridge,"batched_observations",False)),flag)

    def test_frozen_native_disabled_and_enabled_trajectory_rng_exact(self):
        import generals_rl.baseline_league as module
        # Optional independent archived implementation; normal repo tests remain
        # portable and still compare the original bridge path with the new path.
        if os.environ.get("FROZEN_V6_SOURCE"):
            frozen=Path(os.environ["FROZEN_V6_SOURCE"])/"generals_rl/baseline_league.py"
            name="generals_rl._frozen_v6_league_reference"
            spec=importlib.util.spec_from_file_location(name,frozen);module=importlib.util.module_from_spec(spec)
            sys.modules[name]=module;spec.loader.exec_module(module)
        settings=TaskConfig(num_envs=50,opponent="self_play",max_episode_steps=8,history_length=2,
                            productive_actions=True,seed=437)
        old=module.make_league_opponent(settings,module.BaselineLeagueConfig(baseline_envs=1,seed=349,native_candidates=True))
        off=make_league_opponent(settings,BaselineLeagueConfig(baseline_envs=1,seed=349,native_candidates=True))
        on=make_league_opponent(settings,BaselineLeagueConfig(baseline_envs=1,seed=349,native_candidates=True,batched_observations=True))
        with GeneralsTask(settings,opponent=old) as a,GeneralsTask(settings,opponent=off) as b,GeneralsTask(settings,opponent=on) as c:
            for task in (a,b,c):task.opponent.attach(Pool(),stochastic=True,seed=110)
            for tick in range(32):
                actions=torch.empty(50,dtype=torch.long)
                for lane,mask in enumerate(a.observations["action_mask"]):
                    legal=torch.where(mask)[0];actions[lane]=legal[(tick+lane)%len(legal)]
                results=[task.step(actions) for task in (a,b,c)]
                for other,result in zip((b,c),results[1:]):
                    for field in ("reward","terminated","truncated"):
                        torch.testing.assert_close(getattr(results[0],field),getattr(result,field),rtol=0,atol=0)
                    for name in results[0].observations:
                        torch.testing.assert_close(results[0].observations[name],result.observations[name],rtol=0,atol=0)
                    for name in a.backend.state:np.testing.assert_array_equal(a.backend.state[name],other.backend.state[name])
                    torch.testing.assert_close(a.generator.get_state(),other.generator.get_state(),rtol=0,atol=0)
                    torch.testing.assert_close(a.opponent.historical.generator.get_state(),other.opponent.historical.generator.get_state(),rtol=0,atol=0)
                    self.assertEqual(a.opponent.bridge.lanes,other.opponent.bridge.lanes)
                    np.testing.assert_array_equal(a.opponent.bridge.episode,other.opponent.bridge.episode)
                    np.testing.assert_array_equal(a.opponent.totals.numpy(),other.opponent.totals.numpy())
                    for left,right in zip(a.opponent.bridge.controllers,other.opponent.bridge.controllers):controller_equal(left,right)
            for task,active in ((b,False),(c,True)):
                report=task.opponent.describe();self.assertIs(report["batched_observations_active"],active)
                self.assertEqual(report["bridge_class"],"BatchedObservationBridge" if active else "OriginalBaselineBridge")
                self.assertEqual(report["source_totals"],a.opponent.describe()["source_totals"])
                self.assertEqual(report["native_candidate_calls"],a.opponent.describe()["native_candidate_calls"])
                self.assertEqual(report["native_fallback_calls"],0)
                self.assertEqual(task.opponent.take_log_metrics()["League/batched_observations_active"],int(active))


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
