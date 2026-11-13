"""Existing policy submissions run in the current native loop without re-registration."""
from dataclasses import asdict

from generals import MapConfig
from generals_arena.matches.batch import execute_batch
from generals_arena.policies import native
from generals_arena.policies.entries import prepare_entrant


def test_old_adapter_rebuilds_automatically_and_runs_current_loop(tmp_path, monkeypatch):
    sdk = tmp_path/'old-sdk'
    sdk.mkdir()
    (sdk/'arena_policy.h').write_bytes((native.SDK/'arena_policy.h').read_bytes())
    adapter = (native.SDK/'adapter.cpp').read_text()
    start = adapter.index('// Native Arena supplies')
    end = adapter.index('EXPORT void arena_host_destroy', start)
    (sdk/'adapter.cpp').write_text(adapter[:start]+adapter[end:])
    source = tmp_path/'policy.cpp'
    source.write_text('''#include "arena_policy.h"
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }
''')
    bundle = tmp_path/'policy.zip'
    with monkeypatch.context() as old:
        old.setattr(native, 'SDK', sdk)
        native.build_bundle(source, bundle)
    entry = prepare_entrant(dict(name='legacy', kind='cpp', bundle=str(bundle)))
    plan = dict(game_id='legacy', entrant_ids=[entry['entrant_id']]*2,
                instance_ids=[0,1], controller_seeds=[1,2], map_seed=3,
                map=asdict(MapConfig(15,15,2)))
    results, timing = execute_batch({entry['entrant_id']:entry}, [plan], max_ticks=4)
    result, _ = results[0]
    assert 'native_bookkeeping_seconds' in timing
    assert result['ticks'] == 4
    assert all(item['calls'] == 4 and item['timeouts'] == 0 for item in result['decision_timing'])
