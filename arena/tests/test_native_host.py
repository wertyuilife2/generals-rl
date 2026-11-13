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


def test_registered_native_load_reads_bundle_once_and_rejects_changes(tmp_path, monkeypatch):
    from pathlib import Path
    import pytest
    from generals_arena.policies.entries import prepare_entrant, load_controller
    from generals_arena.matches.config import MatchConfig

    bundle = tmp_path/'policy.zip'
    native.build_bundle(Path(__file__).resolve().parents[1]/'examples/cpp_wait', bundle)
    entry = prepare_entrant(dict(name='wait', kind='cpp', bundle=str(bundle)))
    original = Path.open
    reads = []

    def counted_open(path, *args, **kwargs):
        if path == bundle and (not args or 'r' in args[0]):
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', counted_open)
    controller = load_controller(entry, MatchConfig())
    try:
        assert len(reads) == 1
        controller.reset_batch([0, 1], [1, 2])
        controller.reset_batch([1, 0], [3, 4])
        assert len(reads) == 1
    finally:
        controller.close()
    with original(bundle, 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='policy bundle changed'):
        load_controller(entry, MatchConfig())


def test_unpack_extracts_verified_snapshot(tmp_path, monkeypatch):
    from pathlib import Path
    from generals_arena.util import file_digest

    bundle = tmp_path/'policy.zip'
    native.build_bundle(Path(__file__).resolve().parents[1]/'examples/cpp_wait', bundle)
    expected = file_digest(bundle)
    cache = tmp_path/'cache'
    cache.mkdir()
    monkeypatch.setattr(native, '_private_cache', lambda: cache)
    metadata = native._bundle_metadata

    def replace_after_read(archive):
        result = metadata(archive)
        bundle.write_bytes(b'replaced after snapshot')
        return result

    monkeypatch.setattr(native, '_bundle_metadata', replace_after_read)
    root = native._unpack(bundle, expected)
    assert root.name == expected
    assert (root/'policy.so').is_file()
    assert (root/'params.json').read_text() == '{}'


def test_worker_shares_code_across_maps_and_slots_but_resets_instances(tmp_path, monkeypatch):
    from pathlib import Path
    from generals_arena.matches.cache import RuntimeCache
    from generals_arena.matches.batch import _PolicyGroup
    from generals_arena.matches.config import MatchConfig
    from generals_arena.policies.entries import prepare_entrant

    source = tmp_path/'counter.cpp'
    source.write_text('''#include "arena_policy.h"
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
extern "C" std::int64_t arena_act(void *p, const ArenaObservation *) { return ++*static_cast<int *>(p); }
''')
    bundle = tmp_path/'counter.zip'
    native.build_bundle(source, bundle)
    entry = prepare_entrant(dict(name='counter', kind='cpp', bundle=str(bundle)))
    original = Path.read_bytes
    reads = []
    def read(path):
        if path == bundle:
            reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    cache = RuntimeCache()
    first_rules = MatchConfig(MapConfig(15, 17, 2))
    second_rules = MatchConfig(MapConfig(25, 25, 2))
    groups = []
    try:
        for rules, slot in [(first_rules,0),(first_rules,1),(second_rules,0)]:
            groups.append(_PolicyGroup(entry, rules, [0], [42], cache=cache, slot=slot))
        children = [group.controller._children[0] for group in groups]
        assert len({id(child._resources) for child in children}) == 1
        assert len({child.instance for child in children}) == 3
        obs = observe(generate_map(first_rules.map, 17), 0)
        assert [child.act(obs).encode() for child in children] == [1,1,1]
        assert children[0].act(obs).encode() == 2
        assert children[1].act(obs).encode() == 2
        for group in groups:
            group.close()
        groups.clear()
        # Already-loaded resources survive controller eviction and disk cache loss.
        cache.trim(0)
        import shutil
        shutil.rmtree(children[0].root)
        bundle.unlink()
        group = _PolicyGroup(entry, first_rules, [0], [42], cache=cache, slot=3)
        groups.append(group)
        assert group.controller._children[0].act(obs).encode() == 1
        assert len(reads) == 1
    finally:
        for group in groups:
            group.close()
        cache.close()
    assert not cache.resources
