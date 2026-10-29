"""The fast host must pass the complete original SDK observation/previous move."""
import ctypes as C
from dataclasses import replace

import numpy as np

from generals import Action, MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import MoveResult
from generals_arena.policies import native


class Move(C.Structure):
    _fields_ = [(key, C.c_int64) for key in
                ('player', 'action', 'valid', 'source', 'target', 'moved', 'captured', 'eliminated', 'target_owned')]


class Observation(C.Structure):
    _fields_ = [(key, C.c_int) for key in ('width', 'height', 'players', 'player')] + [
        ('tick', C.c_int64), ('general_pos', C.c_int)] + [
        (key, C.POINTER(kind)) for key, kind in (
            ('terrain', C.c_uint8), ('structure', C.c_uint8), ('owner', C.c_int16),
            ('army', C.c_int64), ('visible', C.c_uint8), ('alive', C.c_uint8),
            ('total_army', C.c_int64), ('total_land', C.c_int64), ('total_cities', C.c_int64),
            ('fog_obstacles', C.c_uint8), ('has_kill', C.c_uint8), ('warn_afk', C.c_uint8), ('neutralize_in', C.c_double))] + [
        ('previous', C.POINTER(Move))]


def reference_call(controller, obs):
    kinds = [C.c_uint8, C.c_uint8, C.c_int16, C.c_int64, C.c_uint8, C.c_uint8, C.c_int64, C.c_int64, C.c_int64, C.c_uint8, C.c_uint8, C.c_uint8, C.c_double]
    values = [obs.terrain, obs.structure, obs.owner, obs.army, obs.visible, obs.alive,
              obs.stats.army, obs.stats.territory, obs.stats.cities, obs.fog_obstacles, obs.has_kill, obs.warn_afk, obs.neutralize_in]
    arrays = [np.ascontiguousarray(v, dtype=np.dtype(k)) for v, k in zip(values, kinds)]
    move = Move(*controller.previous) if controller.previous is not None else None
    bound = Observation(obs.width, obs.height, len(obs.alive), obs.player_id, obs.tick, obs.general_pos,
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
        add(o->terrain[i]); add(o->structure[i]); add(o->owner[i]); add(o->army[i]); add(o->visible[i]); add(o->fog_obstacles[i]);
    }
    for(int i=0;i<o->players;++i) { add(o->alive[i]); add(o->total_army[i]); add(o->total_land[i]); add(o->total_cities[i]); add(o->has_kill[i]); add(o->warn_afk[i]); add(std::int64_t(o->neutralize_in[i]*1000)); }
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


def test_native_cache_recovers_from_complete_and_partial_eviction(tmp_path):
    import shutil
    from pathlib import Path
    from generals_arena.util import file_digest
    bundle = tmp_path/'policy.zip'
    native.build_bundle(Path(__file__).resolve().parents[1]/'examples/cpp_wait', bundle)
    expected = file_digest(bundle)
    root = native._unpack(bundle, expected)
    library = (root/'policy.so').read_bytes()
    (root/'policy.so').unlink()
    assert (native._unpack(bundle, expected)/'policy.so').read_bytes() == library
    (root/'params.json').write_text('incorrect')
    assert (native._unpack(bundle, expected)/'params.json').read_text() == '{}'
    shutil.rmtree(root)
    assert (native._unpack(bundle, expected)/'policy.so').read_bytes() == library
