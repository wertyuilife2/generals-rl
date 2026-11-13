"""Native batches preserve public inputs, state, receipts and per-lane failures."""
from dataclasses import asdict

import pytest

from generals import MapConfig
from generals_arena.policies.entries import prepare_entrant
from generals_arena.policies.native import build_bundle
from generals_arena.matches.runner import MatchRunner
from generals_arena.matches.replay import verified_replay


def policy(tmp_path, body):
    source = tmp_path / 'batch.cpp'
    source.write_text('''#include "arena_policy.h"
#include <chrono>
#include <thread>
struct State { std::uint64_t seed, hash; };
extern "C" void *arena_create(int, std::uint64_t seed, const char *) { return new State{seed,seed}; }
extern "C" void arena_destroy(void *p) { delete static_cast<State *>(p); }
extern "C" std::int64_t arena_act(void *p, const ArenaObservation *o) {
    auto &s = *static_cast<State *>(p);
''' + body + '\n}\n')
    bundle = tmp_path / 'batch.zip'
    build_bundle(source, bundle)
    return prepare_entrant(dict(name='native', kind='cpp', bundle=str(bundle)))


def plans(entry, players, seeds):
    return [dict(game_id=str(i), map=asdict(MapConfig(25, 25, players)), map_seed=700+i,
        entrant_ids=[entry['entrant_id']]*players, instance_ids=list(range(players)),
        controller_seeds=row) for i, row in enumerate(seeds)]


def run(entry, schedule, ticks, scalar=False, batch_size=128):
    with MatchRunner({entry['entrant_id']: entry}, max_ticks=ticks, scalar=scalar,
                     batch_size=batch_size, record_replays=True) as runner:
        results, _ = runner.run(schedule)
        modes = runner.execution_modes
    return results, modes


@pytest.mark.parametrize('players', [2, 4])
def test_native_batches_match_scalar_and_chunked_replay_and_decision_counts(tmp_path, players):
    entry = policy(tmp_path, '''
    auto add = [&](std::int64_t value) { s.hash = (s.hash ^ std::uint64_t(value)) * 1099511628211ULL; };
    add(o->player); add(o->tick); add(o->general_pos);
    for(int i=0;i<o->width*o->height;++i) {
        add(o->terrain[i]); add(o->structure[i]); add(o->owner[i]); add(o->army[i]);
        add(o->visible[i]); add(o->fog_obstacles[i]);
    }
    for(int i=0;i<o->players;++i) {
        add(o->alive[i]); add(o->total_army[i]); add(o->total_land[i]); add(o->total_cities[i]);
        add(o->has_kill[i]); add(o->warn_afk[i]); add(std::int64_t(o->neutralize_in[i]*1000));
    }
    if(o->previous) {
        const auto &m=*o->previous;
        add(m.player); add(m.action); add(m.valid); add(m.source); add(m.target);
        add(m.moved); add(m.captured); add(m.eliminated); add(m.target_owned);
    }
    int source=o->general_pos;
    for(int i=0;i<o->width*o->height;++i)
        if(o->owner[i]==o->player && o->army[i]>o->army[source]) source=i;
    return arena_move(source, s.hash%4, (s.hash>>4)%2);
''')
    schedule = plans(entry, players, [[2**48+i*10+p for p in range(players)] for i in range(3)])
    batch, modes = run(entry, schedule, 20)
    reference, _ = run(entry, schedule, 20, scalar=True)
    chunked, _ = run(entry, schedule, 20, batch_size=1)
    assert modes == {'isolated_batch'}
    for rows in (batch, reference, chunked):
        for result in rows:
            timing = result.pop('decision_timing')
            assert {m['seat'] for m in timing} == set(range(players))
            assert all(m['calls'] == 20 and m['errors'] == m['timeouts'] == 0 for m in timing)
    assert batch == reference == chunked


def test_hard_timeout_only_forfeits_its_lane_and_seat_and_other_matches_complete(tmp_path):
    entry = policy(tmp_path, 'if(s.seed==42 && o->tick==2) { for(;;) {} } return 0;')
    schedule = plans(entry, 3, [[10,42,12], [20,21,22], [42,31,32]])
    captured = []
    with MatchRunner({entry['entrant_id']: entry}, max_ticks=6, record_replays=True) as runner:
        results, _ = runner.run(schedule, on_result=lambda result, blob: captured.append((result, blob)))
    assert len(results) == 3
    assert [result['game_id'] for result in results] == ['0', '1', '2']
    for index, seat in ((0,1), (2,0)):
        row = results[index]
        assert row['reason'] != 'policy_error'
        assert row['elimination_ticks'][seat] == 2
        assert not row['final_alive'][seat]
        assert len(row['forfeits']) == 1
        assert row['forfeits'][0]['seat'] == seat
        assert row['forfeits'][0]['reason'] == 'decision_timeout'
        stats = {m['seat']: m for m in row['decision_timing']}
        assert stats[seat]['timeouts'] == 1
        assert stats[seat]['calls'] == 3
        assert all(stats[p]['calls'] == 6 for p in range(3) if p != seat)
    assert all(results[1]['final_alive'])
    assert not results[1].get('forfeits')
    for result, blob in captured:
        assert len(list(verified_replay(blob, result))) == result['ticks']+1


@pytest.mark.parametrize('execution', ['isolated', 'batch'])
def test_training_budget_survives_timeout_recovery_without_changing_arena_default(tmp_path, execution):
    entry = policy(tmp_path, '''
    if(s.seed==42 && o->tick==2) { for(;;) {} }
    if(s.seed==7) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    return 0;
''')
    schedule = plans(entry, 3, [[7,42,12], [7,21,22]])
    entries = {entry['entrant_id']: entry}
    # The slow survivor must retain 50ms while its peer forfeits and the match
    # is restored. A subsequent ordinary Arena run must still use 10ms.
    for limit, options in ((50, {'decision_limit_ms': 50}), (10, {})):
        with MatchRunner(entries, max_ticks=6, execution=execution, **options) as runner:
            results, _ = runner.run(schedule)
        assert all('policy_error' not in row for row in results)
        assert [e['seat'] for e in results[0]['forfeits']] == ([1] if limit == 50 else [0, 1])
        assert [e['seat'] for e in results[1].get('forfeits', [])] == ([] if limit == 50 else [0])
        for row in results:
            assert all(e['limit_ns'] == limit * 1_000_000 for e in row.get('forfeits', []))
            assert all(m['limit_ns'] == limit * 1_000_000 for m in row['decision_timing'])
        if limit == 50:
            assert results[0]['ticks'] == results[1]['ticks'] == 6
            slow = next(m for m in results[0]['decision_timing'] if m['seat'] == 0)
            assert slow['calls'] == 6 and slow['max_ns'] > 10_000_000 and slow['timeouts'] == 0


def test_invalid_native_action_fails_only_its_match(tmp_path):
    entry = policy(tmp_path, 'return s.seed==42 ? -1 : 0;')
    schedule = plans(entry, 2, [[42,11], [20,21]])
    results, _ = run(entry, schedule, 3)
    assert results[0]['reason'] == 'policy_error'
    assert results[0]['error_detail']['player'] == 0
    assert results[1]['reason'] == 'tick_limit'


def test_worker_reuses_journal_across_shapes_and_restart(tmp_path):
    entry = policy(tmp_path, 'return arena_move(o->general_pos, s.seed%4, 0);')
    entries = {entry['entrant_id']: entry}
    paths, processes = set(), []
    with MatchRunner(entries, max_ticks=8, record_replays=True) as runner:
        for trial, (count, players) in enumerate(((1, 2), (4, 4), (2, 2), (1, 2))):
            schedule = plans(entry, players, [[100+i*8+p for p in range(players)] for i in range(count)])
            captured = []
            actual, _ = runner.run(schedule, on_result=lambda result, blob: captured.append((result, blob)))
            worker = runner._batch_worker
            paths.add(worker._journal.path)
            processes.append(worker.process.pid)
            for result, blob in captured:
                assert all(row['calls'] == 8 for row in result.pop('decision_timing'))
                assert len(list(verified_replay(blob, result))) == 9
            expected, _ = run(entry, schedule, 8)
            for result in expected:
                result.pop('decision_timing')
            assert actual == expected
            if trial == 2:
                worker._stop_process()  # A restarted child attaches the retained mapping.
        assert len(paths) == 1
        assert processes[0] == processes[1] == processes[2] != processes[3]
    assert not next(iter(paths)).exists()
