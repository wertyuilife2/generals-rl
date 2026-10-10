"""Native ABI, enforced deadlines, immutable registration and managed experiments."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import pytest

from generals import Action, MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals_arena import research
from generals_arena.league.config import LeagueConfig
from generals_arena.league.game import replay_frames
from generals_arena.league.store import LeagueStore
from generals_arena.native import build_bundle, NativeController, PolicyError, DECISION_NS


def source(root, body='return 0;'):
    root.mkdir(parents=True, exist_ok=True)
    (root/'policy.cpp').write_text('''#include "arena_policy.h"
#include <thread>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
struct Policy { int pid; };
extern "C" void *arena_create(int pid, std::uint64_t, const char *) { return new Policy{pid}; }
extern "C" std::int64_t arena_act(void *p, const ArenaObservation *o) {
''' + body + '''
}
extern "C" void arena_destroy(void *p) { delete static_cast<Policy *>(p); }
''')
    return root


@pytest.fixture
def setup(tmp_path):
    config = LeagueConfig(player_counts=(2, 4), min_side=15, max_side=15, side_cap=15,
                          max_ticks=4, mountain_min=0, mountain_max=0, city_min=0, city_max=0)
    league = LeagueStore.create(tmp_path/'official', config)
    directory = tmp_path/'research'
    research.initialize(directory, league.directory)
    return directory, league


def native(tmp_path, body):
    bundle = tmp_path/'policy.zip'
    build_bundle(source(tmp_path/'source', body), bundle)
    controller = NativeController(bundle=bundle)
    controller.reset(0, 42)
    return controller


def test_native_receives_only_observation_and_previous_result(tmp_path):
    body = '''
if(o->player!=static_cast<Policy *>(p)->pid) throw std::runtime_error("pid");
for(int c=0;c<o->height*o->width;++c)
    if(!o->visible[c] && (o->terrain[c]!=255 || o->army[c]!=-1 || o->owner[c]!=-3))
        throw std::runtime_error("hidden information");
for(int i=0;i<o->players;++i)
    if(i!=o->player && o->total_cities[i]!=-1) throw std::runtime_error("cities");
if(o->tick>0 && (!o->previous || !o->previous->valid)) throw std::runtime_error("result");
return 0;
'''
    controller = native(tmp_path, body)
    from generals import make_engine
    state = generate_map(MapConfig(15, 17, 2), 8)
    engine = make_engine(state)
    try:
        for _ in range(4):
            action = controller.act(observe(state, 0))
            assert action == Action.wait()
            result = engine.step([action, Action.wait()])
            controller.on_result(result.moves[0])
        assert controller.decision_metrics['calls'] == 4
        assert controller.decision_metrics['max_ns'] <= DECISION_NS
        assert controller.decision_metrics['timeouts'] == 0
    finally:
        controller.close()
        engine.close()


def test_multiple_cpp_files_link_into_one_policy(tmp_path):
    code = source(tmp_path/'multiple', 'return helper();')
    path = code/'policy.cpp'
    path.write_text('int helper();\n'+path.read_text())
    (code/'helper.cpp').write_text('int helper() { return 0; }')
    bundle = tmp_path/'multiple.zip'
    build_bundle(code,bundle)
    controller = NativeController(bundle=bundle)
    try:
        controller.reset(0,1)
        assert controller.act(observe(generate_map(MapConfig(15,15,2),1),0)) == Action.wait()
    finally:
        controller.close()


def test_all_cpp_seats_share_one_serial_match_process(setup, tmp_path):
    directory, _ = setup
    witness = tmp_path/'pids'
    code = source(tmp_path/'same_process',
        'FILE *f=fopen(' + json.dumps(str(witness)) + ',"a"); fprintf(f,"%d\\n",int(getpid())); fclose(f); return 0;')
    path = code/'policy.cpp'
    path.write_text('#include <cstdio>\n'+path.read_text())
    research.register(directory,code,'v1')
    report = research.evaluate(directory,'v1',opponents=['self'],players=[2],games=2)
    assert report['ok']
    pids = witness.read_text().splitlines()
    assert len(pids) == 16 and len(set(pids)) == 1
    with pytest.raises(ValueError, match='serially'):
        research.evaluate(directory,'v1',games=2,workers=2)


@pytest.mark.parametrize('body,kind', [
    ('std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;', 'timeout'),
    ('for(;;) { asm volatile("" ::: "memory"); }', 'timeout'),
    ('_exit(42);', 'policy_error'),
    ('throw std::runtime_error("broken");', 'policy_error'),
    ('return -1;', 'policy_error'),
])
def test_native_rejects_late_crashed_and_malformed_results(setup, tmp_path, body, kind):
    directory, _ = setup
    research.register(directory,source(tmp_path/'fault',body),'fault')
    report = research.evaluate(directory,'fault',games=1,players=[2])
    assert not report['ok'] and report['completed_games'] == 0
    assert report['violations'][0]['error_detail']['kind'] == kind
    assert report['timing']['calls'] == 1
    assert report['timing']['timeouts' if kind == 'timeout' else 'errors'] == 1


def test_registration_freezes_code_parameters_and_rejects_tampering(setup, tmp_path):
    directory, _ = setup
    code = source(tmp_path/'source')
    original = research.register(directory, code, 'v1', {'aggression': .7})
    assert research.register(directory, code, 'v1', {'aggression': .7}) == original
    source(code, 'return -1;')
    with pytest.raises(ValueError, match='immutable'):
        research.register(directory, code, 'v1')
    spec, _ = research.registered(directory, 'v1')
    controller = NativeController(bundle=spec['bundle'])
    try:
        controller.reset(0, 2)
        assert controller.act(observe(generate_map(MapConfig(15,15,2), 1),0)) == Action.wait()
    finally:
        controller.close()
    with Path(spec['bundle']).open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        research.registered(directory, 'v1')


def test_private_trials_repeat_seeds_and_do_not_touch_official_rankings(setup, tmp_path):
    directory, official = setup
    code = source(tmp_path/'source')
    research.register(directory, code, 'v1')
    research.register(directory, code, 'v2', {'variant': 2})
    before = official.leaderboard()
    left = research.evaluate(directory, 'v1', opponents=['baseline:random','self'], games=8, seed=7)
    right = research.evaluate(directory, 'v2', opponents=['baseline:random','self'], games=8, seed=7)
    assert left['ok'] and right['ok']
    assert official.leaderboard() == before
    assert left['timing']['calls'] > 0
    assert left['timing']['max_ns'] <= DECISION_NS
    def plans(report):
        store = LeagueStore(report['league_directory'])
        with store.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT plan FROM games ORDER BY rowid')]
    for a,b in zip(plans(left), plans(right)):
        for key in ('map','map_seed','controller_seeds','instance_ids'):
            assert a[key] == b[key]
    repeated = research.evaluate(directory, 'v1', opponents=['baseline:random','self'], games=8, seed=7)
    assert repeated['completed_games'] == 8
    game_id = plans(left)[0]['game_id']
    result, blob = LeagueStore(left['league_directory']).replay(game_id)
    assert len(replay_frames(blob,result)) == result['ticks']+1
    self_play = research.evaluate(directory, 'v1', opponents=['self'], games=2, players=[2])
    assert self_play['ok'] and self_play['macro_pair_score'] is None


def test_timeout_is_persisted_not_scored_and_cannot_be_published(setup, tmp_path):
    directory, official = setup
    research.register(directory, source(tmp_path/'slow',
        'std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;'), 'slow')
    report = research.evaluate(directory,'slow',games=4,players=[2])
    assert not report['ok'] and report['timing']['timeouts'] >= 1
    assert report['candidate_violations'] == report['timing']['timeouts']
    assert report['job']['status'] == 'failed' and report['completed_games'] == 0
    store = LeagueStore(report['league_directory'])
    for row in store.matches()['matches']:
        assert row['status'] == 'failed' and row['error_detail']['kind'] == 'timeout'
        result, blob = store.replay(row['game_id'])
        assert len(replay_frames(blob,result)) == 1
    with pytest.raises(ValueError,match='violation'):
        research.publish(directory,'slow')
    assert official.leaderboard()['completed_games'] == 0
    assert research.status(directory)['remaining_publications'] == 3
    from generals_arena.league.service import eligible_schedule
    _, slow = research.registered(directory, 'slow')
    assert slow['entrant_id'] in store.disabled_policy_ids()
    plans = eligible_schedule(store, {'entrant_id':store.anchor,'id':'later','done':0,'budget':2})
    assert all(slow['entrant_id'] not in plan['entrant_ids'] for plan in plans)


def test_violation_as_an_opponent_also_prevents_publication(setup,tmp_path):
    directory, _ = setup
    research.register(directory,source(tmp_path/'fast'),'fast')
    research.register(directory,source(tmp_path/'bad','return -1;'),'bad')
    report = research.evaluate(directory,'fast',opponents=['bad'],games=2,players=[2])
    assert not report['ok'] and report['candidate_violations'] == 0
    with pytest.raises(ValueError,match='violation'):
        research.publish(directory,'bad')


def test_broken_destructor_is_attributed_before_the_next_match(setup,tmp_path):
    directory, _ = setup
    code = source(tmp_path/'destructor')
    path = code/'policy.cpp'
    path.write_text(path.read_text().replace('delete static_cast<Policy *>(p);',
        'throw std::runtime_error("destruction");'))
    research.register(directory,code,'bad_destroy')
    report = research.evaluate(directory,'bad_destroy',opponents=['baseline:random'],games=2,players=[2])
    assert not report['ok'] and report['completed_games'] == 0
    assert 'destruction' in report['violations'][0]['policy_error']
    assert report['violations'][0]['ticks'] == 4
    store = LeagueStore(report['league_directory'])
    result,blob = store.replay(report['violations'][0]['game_id'])
    assert len(replay_frames(blob,result)) == 5


def test_signal_timeout_recovers_committed_moves_from_shared_journal(setup,tmp_path):
    directory, _ = setup
    research.register(directory,source(tmp_path/'late',
        'if(o->tick==2) std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;'),'late')
    report = research.evaluate(directory,'late',opponents=['baseline:random'],games=1,players=[2])
    assert not report['ok'] and report['timing']['timeouts'] == 1
    violation = report['violations'][0]
    assert violation['ticks'] == 2 and violation['error_detail']['phase'] == 'decision'
    store = LeagueStore(report['league_directory'])
    result,blob = store.replay(violation['game_id'])
    frames = replay_frames(blob,result)
    assert len(frames) == 3 and frames[-1]['tick'] == 2
    assert report['completed_games'] == 0


def test_human_match_preserves_cpp_initialization_failure(setup, tmp_path, monkeypatch):
    directory, official = setup
    code = source(tmp_path/'broken_create')
    path = code/'policy.cpp'
    path.write_text(path.read_text().replace('return new Policy{pid};', 'throw std::runtime_error("create");'))
    research.register(directory, code, 'broken')
    spec, entrant = research.registered(directory, 'broken')
    official.register(spec)
    human = official.account('test human')['entrant']['entrant_id']
    plan = research._plans(official, human, [entrant['entrant_id']], 1, 8, 'human-init')[0]
    plan['human'] = True
    from generals_arena.league import service
    monkeypatch.setattr(service, 'eligible_schedule', lambda *a, **kw: [plan])
    live = service.LiveArena(official)
    with pytest.raises(PolicyError, match='initialization'):
        live.create_game(human, len(plan['entrant_ids']))
    rows = official.matches()['matches']
    assert len(rows) == 1 and rows[0]['status'] == 'failed'
    result, blob = official.replay(rows[0]['game_id'])
    assert len(replay_frames(blob, result)) == 1
    assert official.leaderboard()['completed_games'] == 0
    assert entrant['entrant_id'] in official.disabled_policy_ids()


@pytest.mark.parametrize('body,failed', [('return 0;',False),
    ('std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;',True)])
def test_human_cpp_match_uses_same_native_deadline_and_replay(setup,tmp_path,monkeypatch,body,failed):
    directory, official = setup
    research.register(directory,source(tmp_path/'human_cpp',body),'bot')
    spec, entrant = research.registered(directory,'bot')
    official.register(spec)
    human = official.account('human cpp')['entrant']['entrant_id']
    plan = research._plans(official,human,[entrant['entrant_id']],1,8,'human-cpp')[0]
    plan['human'] = True
    from generals_arena.league import service
    monkeypatch.setattr(service,'eligible_schedule',lambda *a,**kw:[plan])
    live = service.LiveArena(official)
    live.start(run_worker=False)
    try:
        view = live.create_game(human,len(plan['entrant_ids']))
        deadline = time.monotonic()+5
        while not view['done'] and time.monotonic()<deadline:
            time.sleep(.02)
            view = live.view(view['game_id'],human)
        assert view['done'] and bool(view['error']) == failed
        result, blob = official.replay(view['game_id'])
        assert len(replay_frames(blob,result)) == result['ticks']+1
        assert bool(result.get('policy_error')) == failed
        assert result['decision_timing']
        assert official.leaderboard()['completed_games'] == (0 if failed else 1)
    finally:
        live.close()


def test_publication_runs_full_budget_is_idempotent_and_limits_versions(setup,tmp_path,monkeypatch):
    directory, official = setup
    code = source(tmp_path/'source')
    # Small integration budget; production CLI has no override for 1000.
    monkeypatch.setattr(research,'PUBLICATION_GAMES',3)
    for i in range(4):
        research.register(directory,code,f'v{i}',{'version':i})
    for i in range(3):
        report = research.publish(directory,f'v{i}')
        assert report['ok'] and report['completed_games'] == 3 and report['slot'] == i+1
        assert report['timing']['max_ns'] <= DECISION_NS
    assert official.leaderboard()['completed_games'] == 9
    assert research.publish(directory,'v0')['completed_games'] == 3
    with pytest.raises(ValueError,match='3 checkpoint'):
        research.publish(directory,'v3')
    assert official.leaderboard()['completed_games'] == 9
    assert research.status(directory)['remaining_publications'] == 0
    from generals_arena.league.service import process_lock
    from types import SimpleNamespace
    with process_lock(SimpleNamespace(directory=directory), '.publish-v0.lock'):
        with pytest.raises(RuntimeError, match='another process'):
            research.publish(directory, 'v0')
    # Formal opponents belong to the league, not a disposable development folder.
    entries = [e for e in official.entrants().values() if e['kind']=='cpp']
    assert all(Path(e['artifact_paths']['bundle']).is_relative_to(official.directory) for e in entries)
    for path in (directory/'strategies').glob('*/policy.zip'):
        path.unlink()
    from generals_arena.registry import load_controller
    from types import SimpleNamespace
    for entry in entries:
        controller = load_controller(entry, SimpleNamespace(map=MapConfig(15,15,2),visibility='local'))
        try:
            controller.reset(0,42)
            assert controller.act(observe(generate_map(MapConfig(15,15,2),1),0)) == Action.wait()
        finally:
            controller.close()
