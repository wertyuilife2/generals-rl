import unittest
from unittest.mock import patch

import numpy as np

from generals.backends.cpu import CPUEnv
from generals.core.config import MapConfig
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.state import Action, Direction, MoveMode, compute_stats
from generals.benchmarks.contracts import BenchmarkCase
from generals.benchmarks.step import measure_step_case
from generals.benchmarks.measurement import timed_steps
from test_gpu_env import make_state


class NativeStepTests(unittest.TestCase):
    def test_cpu_matches_independent_python_rules(self):
        for players in (2, 4, 8):
            states = [generate_map(MapConfig(25, 25, players), seed) for seed in (21, 47, 56)]
            for state in states:
                state.army.ravel()[state.general_pos] = 57
            env = CPUEnv.from_states(states, threads=2)
            rng = np.random.default_rng(79)
            try:
                for _ in range(61):
                    actions=[]
                    for state in states:
                        actions.append([Action.move(int(rng.integers(65)),Direction(int(rng.integers(4))),
                            MoveMode(int(rng.integers(2)))) for _ in range(players)])
                    env.step(np.array([[a.encode() for a in row] for row in actions],np.int64))
                    env.check_errors()
                    for i,(state,row) in enumerate(zip(states,actions)):
                        if not state.terminated:
                            CoreEngine(state).step(row)
                        for name in ('owner','structure','army','alive','general_pos'):
                            np.testing.assert_array_equal(env.state[name][i],getattr(state,name).ravel())
                        stats=compute_stats(state)
                        np.testing.assert_array_equal(env.stats[i],np.stack([stats.army,stats.territory,stats.cities],axis=1))
            finally:
                env.close()

    def test_zero_moves_half_inheritance_and_frozen_records(self):
        states = [make_state(2, 4, generals=[0, 1, 4, 5]) for _ in range(3)]
        for state in states[1:]:
            state.army.flat[0] = 10
            state.army.flat[4] = 30
            state.army.flat[5] = 2
            state.owner.flat[2], state.army.flat[2] = 1, 5
        env = CPUEnv.from_states(states)
        self.addCleanup(env.close)
        actions = np.array([[3, 11, 33, 43]] * 3, np.int64)
        for tick in range(3):
            env.step(actions)
            for lane, state in enumerate(states):
                if not state.terminated:
                    CoreEngine(state).step([Action.decode(int(a)) if state.alive[pid] else Action.wait()
                                            for pid, a in enumerate(actions[lane])])
                for name in ('owner', 'structure', 'army', 'alive'):
                    np.testing.assert_array_equal(env.state[name][lane], getattr(state, name).ravel())
                self.assertEqual(env.state['terminated'][lane], state.terminated)
            if tick == 0:
                self.assertEqual(env.last_moves[0, 0, 4], 0)
                self.assertFalse(env.last_moves[0, 0, 1])
                self.assertEqual(env.last_moves[0, 0, 8], 6)

    def test_external_edits_invalidate_persistent_cache_and_reset(self):
        env=CPUEnv.from_states([make_state()],auto_reset=True,episode_ticks=3)
        try:
            actions=np.zeros((1,2),np.int64)
            env.step(actions)
            env.state['owner'][0,1]=0;env.state['army'][0,1]=19
            env.state['structure'][0,1]=1
            env.invalidate_cache();env.step(actions)
            self.assertEqual(env.stats[0,0,0],22)
            env.step(actions)
            self.assertTrue(env.done[0,1]);self.assertEqual(env.state['tick'][0],0)
            self.assertEqual(env.stats[0,0,0],22)
            env.step(actions)
            self.assertEqual(env.stats[0,0,0],1)
        finally:env.close()

    def test_benchmark_times_each_step_and_completion(self):
        calls=[]
        class Env:
            def step(self,a):calls.append(('step',a))
            def synchronize(self):calls.append(('complete',))
        with patch('generals.benchmarks.measurement.perf_counter_ns',side_effect=[1,11,20,50]):
            times=timed_steps(Env(),[4,9])
        self.assertEqual(times,[10,30])
        self.assertEqual(calls,[('step',4),('complete',),('step',9),('complete',)])

    def test_step_benchmark_exact_trajectory_across_repeats(self):
        case = BenchmarkCase(MapConfig(25, 25, 4), batch_size=4, threads=2, episode_ticks=17)
        result = measure_step_case(case, backend='cpu', steps=61, warmup=2, repeats=2)
        self.assertEqual(result['samples'][0]['counters'], result['samples'][1]['counters'])
        self.assertEqual(len(result['samples'][0]['step_latency_ns']), 61)


    def test_negative_actions_are_invalid_and_cannot_create_armies(self):
        values = list(range(-1,-18,-1)) + [-(2**63)]
        states = [make_state() for _ in values]
        for state in states: state.army.flat[0] = 8
        env = CPUEnv.from_states(states)
        try:
            env.step(np.array([[value,0] for value in values],np.int64))
            np.testing.assert_array_equal(env.last_moves[:,0,1],0)
            np.testing.assert_array_equal(env.last_moves[:,0,8],3)
            np.testing.assert_array_equal(env.state['army'][:,0],8)
            np.testing.assert_array_equal(env.counters[:,4],1)
        finally: env.close()

    def test_native_overflow_priorities_and_storage_lifetime(self):
        states = [make_state() for _ in range(5)]
        for state in states: state.owner.flat[1:4] = 0
        env = CPUEnv.from_states(states)
        self.addCleanup(env.close)
        maximum = np.iinfo(np.int64).max
        actions = np.zeros((5, 2), np.int64)
        actions[1, 0] = 3
        env.step(np.zeros_like(actions))
        env.state['tick'][:] = 0
        env.state['army'][0, 1:4] = maximum
        env.state['army'][1, 0] = 2
        env.state['army'][1, 1] = maximum
        env.state['tick'][2] = 1
        env.state['army'][2, 0] = maximum
        env.state['tick'][3] = maximum
        env.invalidate_cache()
        for _ in range(2):
            env.step(actions)
            self.assertEqual(env.errors.tolist(), [3, 3, 1, 4, 0])
        env.reset()
        exported = env.to_states()
        exported[0].army.flat[0] = 123
        self.assertEqual(env.state['army'][0, 0], 1)
        env.state['army'] = env.state['army'].copy()
        with self.assertRaisesRegex(ValueError, 'storage'):
            env.step(actions)


if __name__=='__main__':unittest.main()
