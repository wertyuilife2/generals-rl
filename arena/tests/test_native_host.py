"""The fast host must pass the complete original SDK observation/previous move."""
import ctypes as C
from dataclasses import replace

import numpy as np

from generals import Action, MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import MoveResult
from generals_arena import native


def reference_call(controller, obs):
    kinds = [C.c_uint8, C.c_uint8, C.c_int16, C.c_int64, C.c_uint8, C.c_uint8, C.c_int64, C.c_int64, C.c_int64]
    values = [obs.terrain, obs.structure, obs.owner, obs.army, obs.visible, obs.alive,
              obs.stats.army, obs.stats.territory, obs.stats.cities]
    arrays = [np.ascontiguousarray(v, dtype=np.dtype(k)) for v, k in zip(values, kinds)]
    move = native.Move(*controller.previous) if controller.previous is not None else None
    bound = native.Observation(obs.width, obs.height, len(obs.alive), obs.player_id, obs.tick, obs.general_pos,
        *(a.ctypes.data_as(C.POINTER(k)) for a, k in zip(arrays, kinds)), C.pointer(move) if move else None)
    return Action.decode(controller.lib.arena_host_act(controller.instance, C.byref(bound),
                         controller.values.ctypes.data_as(C.POINTER(C.c_int64))))


def test_native_and_ctypes_hosts_match_with_strides_large_values_and_previous_move(tmp_path):
    source = tmp_path/'policy.cpp'
    source.write_text('''#include "arena_policy.h"
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
extern "C" std::int64_t arena_act(void *p, const ArenaObservation *o) {
    std::uint64_t h = ++*static_cast<int *>(p);
    auto add = [&](std::int64_t x) { h = (h ^ std::uint64_t(x)) * 1099511628211ULL; };
    add(o->width); add(o->height); add(o->players); add(o->player); add(o->tick); add(o->general_pos);
    for(int i=0;i<o->width*o->height;++i) {
        add(o->terrain[i]); add(o->structure[i]); add(o->owner[i]); add(o->army[i]); add(o->visible[i]);
    }
    for(int i=0;i<o->players;++i) { add(o->alive[i]); add(o->total_army[i]); add(o->total_land[i]); add(o->total_cities[i]); }
    if(o->previous) { auto m=o->previous;
        add(m->player); add(m->action); add(m->valid); add(m->source); add(m->target);
        add(m->moved); add(m->captured); add(m->eliminated); add(m->target_owned);
    }
    return h % (o->width*o->height*8 + 1);
}''')
    bundle = tmp_path/'policy.zip'
    native.build_bundle(source, bundle)
    left, right = native.NativeController(bundle=bundle), native.NativeController(bundle=bundle)
    state = generate_map(MapConfig(15, 17, 2), 17)
    state.army.flat[state.general_pos[0]] = 2**53 + 37
    try:
        left.reset(0, 42); right.reset(0, 42)
        for tick in range(10):
            state.tick = tick
            obs = observe(state, 0)
            if tick % 2:
                obs = replace(obs, **{k: np.asfortranarray(getattr(obs, k)) for k in
                                     ('terrain', 'structure', 'owner', 'army', 'visible')})
            action = left.act(obs)
            assert action == reference_call(right, obs)
            move = MoveResult(0, action, True, 'MOVED', 5, 6, 2**53 + tick,
                              bool(tick % 2), 1 if tick % 3 else None, bool(tick % 3))
            left.on_result(move); right.on_result(move)
        assert left.values[0] == right.values[0] == 10
        assert left.values[4] == right.values[4] == 0
        assert left.values[5] == right.values[5] == 0
    finally:
        left.close(); right.close()
