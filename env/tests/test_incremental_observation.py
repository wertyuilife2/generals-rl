"""Native incremental observations equal complete public snapshots at every tick."""
import ctypes as C
from contextlib import closing
import numpy as np
import pytest

from generals import MapConfig
from generals.backends.cpu.env import CPUEnv, _Observation
from generals.core.mapgen import generate_map
from test_official_edges import CASES, initial


class Bound:
    def __init__(self, env):
        self.env, self.binding, self.outputs = env, env.native_binding(), {}
        self.observe_call = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p)(self.binding.abi.observe)
        self.step_call = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(self.binding.abi.step)
        self.depart_call = C.CFUNCTYPE(None, C.c_void_p, C.c_int, C.c_void_p)(self.binding.abi.depart)

    def check(self, ids, slot, selected=None):
        ids = np.full(self.env.batch_size, ids, np.int64) if np.isscalar(ids) else np.array(ids, np.int64)
        if slot not in self.outputs:
            raw = self.env.observer(0)()
            arrays = {**raw['state'], **{k: raw[k] for k in ('visible','stats','general_pos','active')}}
            out = _Observation(*(arrays[k].ctypes.data for k,_ in _Observation._fields_))
            self.outputs[slot] = arrays, out
        arrays, out = self.outputs[slot]
        before = {k:v.copy() for k,v in arrays.items()}
        mask = None if selected is None else np.array(selected, bool)
        self.observe_call(self.binding.abi.context, ids.ctypes.data,
                          None if mask is None else mask.ctypes.data, C.byref(out))
        full = bool(np.all(ids == -1))
        expected = self.env.observer(-1 if full else ids, full=full)()
        expected = {**expected['state'], **{k:expected[k] for k in ('visible','stats','general_pos','active')}}
        for key,value in arrays.items():
            np.testing.assert_array_equal(value if mask is None else value[mask],
                expected[key] if mask is None else expected[key][mask], err_msg=key)
            if mask is not None and key != 'active':
                np.testing.assert_array_equal(value[~mask],before[key][~mask],err_msg='inactive '+key)
        if mask is not None:
            assert not arrays['active'][~mask].any()

    def step(self, actions):
        actions = np.ascontiguousarray(actions, np.int64)
        self.step_call(self.binding.abi.context, actions.ctypes.data)
        self.env.check_errors()

    def depart(self, seat, lanes):
        mask = np.array(lanes, bool)
        self.depart_call(self.binding.abi.context, seat, mask.ctypes.data)


@pytest.mark.parametrize('players',[2,4,8])
def test_masks_growth_skipped_lanes_observer_switching_and_departures(players):
    states = [generate_map(MapConfig(35,33,players,0,0),seed) for seed in (3,7)]
    for state in states:
        state.army.flat[state.general_pos] = 80
    with closing(CPUEnv.from_states(states, threads=2)) as env:
        bound = Bound(env)
        rng = np.random.default_rng(135)
        for tick in range(180):
            for seat in range(players):
                bound.check([seat,(seat+1)%players],seat,selected=[tick%7!=0,True])
            # Reuse a scratch buffer across observers and spectator queries.
            bound.check(tick%players,'scratch')
            if tick%9==0:
                bound.check(-1,'scratch')
                bound.check(tick%players,'scratch')
            actions=np.zeros((2,players),np.int64)
            for lane in range(2):
                for seat in range(players):
                    if not env.state['alive'][lane,seat]:continue
                    candidates=np.flatnonzero((env.state['owner'][lane]==seat)&(env.state['army'][lane]>1))
                    if len(candidates):
                        source=int(rng.choice(candidates));direction=int(rng.integers(4))
                        actions[lane,seat]=1+source*8+direction*2+int(rng.integers(2))
            bound.step(actions)
            if tick==31:bound.depart(1,[True,False])
            if tick==90 and players>2:bound.depart(2,[False,True])
        for seat in range(players):bound.check(seat,seat)


@pytest.mark.parametrize('case',CASES,ids=lambda case:case.get('name',case.get('id','edge')))
def test_official_capture_trade_and_visibility_edges(case):
    with closing(CPUEnv.from_states([initial(case)],threads=1)) as env:
        bound=Bound(env)
        for seat in range(env.players):bound.check(seat,seat)
        bound.step([case['actions']])
        for tick in range(52):
            for seat in range(env.players):bound.check(seat,seat)
            bound.check(-1,'spectator')
            bound.step(np.zeros((1,env.players),np.int64))


def test_prepared_initial_states_own_storage_and_reset_without_revalidation(monkeypatch):
    from generals.core.batch import PreparedState
    from generals.core.state import GameState
    state=generate_map(MapConfig(18,18,2),29)
    state.army.flat[state.general_pos[0]]=40
    prepared=PreparedState.from_state(state)
    original=prepared.arrays['army'].copy()
    # Editing the caller's original state cannot change the prepared map.
    state.army.flat[state.general_pos[0]]=400
    def repeated_validation(self):
        raise AssertionError('a prepared map was reconstructed and validated again')
    monkeypatch.setattr(GameState,'validate',repeated_validation)
    with closing(CPUEnv.from_states([prepared,prepared],threads=1)) as env:
        first=env.state['army'].copy()
        source=int(state.general_pos[0])
        env.step(np.array([[1+source*8,0],[0,0]],np.int64))
        env.reset()
        np.testing.assert_array_equal(env.state['army'],first)
        np.testing.assert_array_equal(prepared.arrays['army'],original)
        assert not np.shares_memory(env.state['army'],prepared.arrays['army'])
