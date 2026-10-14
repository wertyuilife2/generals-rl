"""Native ABI, enforced deadlines, immutable registration and managed experiments."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import pytest

from generals import Action, MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals_arena import research
from generals_arena.league.config import LeagueConfig
from generals_arena.execution import replay_frames
from generals_arena.league.store import LeagueStore
from generals_arena.native import build_bundle, NativeController, PolicyError, DECISION_NS


def test_legacy_decision_timeouts_remain_eligible():
    from generals_arena.execution import decision_timeout
    assert decision_timeout({'kind':'timeout', 'limit_ns':10_000_000})
    assert decision_timeout({'kind':'timeout', 'phase':'decision', 'limit_ns':10_000_000})
    assert not decision_timeout({'kind':'timeout', 'limit_ns':2_000_000_000})
    assert not decision_timeout({'kind':'timeout', 'phase':'initialization', 'limit_ns':2_000_000_000})
    assert not decision_timeout({'kind':'policy_error', 'phase':'decision', 'limit_ns':10_000_000})


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
    config = LeagueConfig(player_counts=(2, 4), map_generation=None, min_side=15, max_side=15, side_cap=15,
                          max_ticks=4, mountain_min=0, mountain_max=0, city_min=0, city_max=0)
    league = LeagueStore.create(tmp_path/'official', config)
    return league.directory, league


def native(tmp_path, body):
    bundle = tmp_path/'policy.zip'
    build_bundle(source(tmp_path/'source', body), bundle)
    controller = NativeController(bundle=bundle)
    controller.reset(0, 42)
    return controller


def test_official_default_maps_run_through_native_arena(tmp_path):
    from generals import MapGenerationConfig
    from generals_arena.league.game import replay_frames
    rules = LeagueConfig(max_ticks=3, map_generation=MapGenerationConfig(city_army_min=50, city_army_max=50))
    official = LeagueStore.create(tmp_path/'official', rules)
    for count in (2, 8):
        report = research.run(['baseline/random'] * count, league=official.directory, games=2, workers=2)
        assert report['ok'] and report['completed_games'] == 2
        private = LeagueStore(report['league_directory'])
        for row in private.matches()['matches']:
            result, blob = private.replay(row['game_id'])
            assert result['map']['generation']['city_army_min'] == 50
            assert replay_frames(blob, result)[-1]['tick'] == 3
    report = research.run(['baseline/random'] * 2, league=official.directory, size='20x25')
    private = LeagueStore(report['league_directory'])
    result = private.matches()['matches'][0]
    assert result['map']['width'] == 20 and result['map']['height'] == 25


def test_native_reextracts_policy_after_temporary_cache_is_removed(tmp_path):
    bundle = tmp_path/'policy.zip'
    build_bundle(source(tmp_path/'source', 'return o->tick == 123456789 ? 0 : 0;'), bundle)
    first = NativeController(bundle=bundle)
    root = first.root
    first.close()
    (root/'policy.so').unlink()
    second = NativeController(bundle=bundle)
    try:
        assert second.root == root
        assert (root/'policy.so').is_file()
    finally:
        second.close()
    shutil.rmtree(root)
    third = NativeController(bundle=bundle)
    try:
        assert third.root == root
        assert (root/'policy.so').is_file()
    finally:
        third.close()


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


def test_multiple_cpp_seats_share_one_serial_match_process(setup, tmp_path):
    directory, _ = setup
    witness = tmp_path/'pids'
    code = source(tmp_path/'same_process',
        'FILE *f=fopen(' + json.dumps(str(witness)) + ',"a"); fprintf(f,"%d\\n",int(getpid())); fclose(f); return 0;')
    path = code/'policy.cpp'
    path.write_text('#include <cstdio>\n'+path.read_text())
    report = research.run([code, code], league=directory, games=2)
    assert report['ok']
    pids = witness.read_text().splitlines()
    assert len(pids) == 16 and len(set(pids)) == 1
    assert report['players'][0]['seats'] == [0, 1]
    assert report['players'][0]['macro_pair_score'] is None


@pytest.mark.parametrize('body,kind', [
    ('_exit(42);', 'policy_error'),
    ('throw std::runtime_error("broken");', 'policy_error'),
    ('return -1;', 'policy_error'),
])
def test_native_rejects_late_crashed_and_malformed_results(setup, tmp_path, body, kind):
    directory, _ = setup
    report = research.run([source(tmp_path/'fault',body), 'baseline/random'], league=directory)
    assert not report['ok'] and report['completed_games'] == 0
    assert report['violations'][0]['error_detail']['kind'] == kind
    assert report['timing']['calls'] == 1
    assert report['timing']['timeouts' if kind == 'timeout' else 'errors'] == 1


def test_single_cpp_with_local_headers_and_snapshot_independence(setup, tmp_path):
    directory, official = setup
    code = source(tmp_path/'source', 'return decision();')
    path = code/'agent.cpp'
    (code/'policy.cpp').rename(path)
    path.write_text('#include "nested/logic.hpp"\n'+path.read_text())
    (code/'nested').mkdir()
    (code/'nested/logic.hpp').write_text('#include "constant.h"\ninline int decision() { return WAIT_VALUE; }')
    (code/'nested/constant.h').write_text('constexpr int WAIT_VALUE = 0;')
    (code/'unrelated.cpp').write_text('this must not be compiled')
    original = research.snapshot_policy(official, path, {'aggression': .7})
    assert research.snapshot_policy(official, path, {'aggression': .7}) == original
    selector = 'policy/'+original['version']
    report = research.run([selector,'baseline/random'], league=directory)
    assert report['ok']
    (code/'nested/constant.h').write_text('constexpr int WAIT_VALUE = -1;')
    assert research.snapshot_policy(official, path, {'aggression': .7})['version'] != original['version']
    assert research.run([selector,'baseline/random'], league=directory,seed=13)['ok']
    with Path(original['bundle']).open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError,match='changed'):
        research.run([selector,'baseline/random'], league=directory)


def test_private_trials_seats_seeds_sizes_exports_and_no_official_changes(setup, tmp_path):
    directory, official = setup
    first, second = source(tmp_path/'first'), source(tmp_path/'second')
    (second/'policy.cpp').write_text((second/'policy.cpp').read_text()+'\n// another version\n')
    before = official.leaderboard()
    left = research.run([first,'baseline/random',first],league=directory,games=6,seed=7,size='15:18',output=tmp_path/'export')
    right = research.run([second,'baseline/random',second],league=directory,games=6,seed=7,size='15:18')
    assert left['ok'] and right['ok'] and len(left['players']) == 2
    assert official.leaderboard() == before
    assert left['timing']['calls'] > 0 and left['timing']['max_ns'] <= DECISION_NS
    def plans(report):
        store = LeagueStore(report['league_directory'])
        with store.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT plan FROM games ORDER BY rowid')]
    for a,b in zip(plans(left),plans(right)):
        for key in ('map','map_seed','controller_seeds','instance_ids'):
            assert a[key] == b[key]
        assert 15 <= a['map']['width'] <= 18 and 15 <= a['map']['height'] <= 18
    assert plans(left)[0]['instance_ids'] == [0,1,2]
    assert plans(left)[1]['instance_ids'] == [1,2,0]
    repeated = research.run([first,'baseline/random',first],league=directory,games=6,seed=7,size='15:18',output=tmp_path/'export')
    assert repeated['completed_games'] == 6 and repeated['run_id'] == left['run_id']
    exported = LeagueStore(tmp_path/'export')
    result,blob = exported.replay(plans(left)[0]['game_id'])
    assert len(replay_frames(blob,result)) == result['ticks']+1
    with pytest.raises(ValueError, match='different run'):
        research.run([second,'baseline/random'],league=directory,output=tmp_path/'export')
    with pytest.raises(ValueError, match='outside'):
        research.run([first,'baseline/random'],league=directory,output=directory)
    fixed = research.run(['baseline/aggressive.cpp','baseline/random'],league=directory,size='20x25')
    assert plans(fixed)[0]['map']['width'] == 20 and plans(fixed)[0]['map']['height'] == 25
    assert not (directory/'commits').exists()


@pytest.mark.parametrize('players,size', [(['baseline/random'],'auto'),
    (['baseline/random']*9,'auto'), (['baseline/random']*2,'35:15'),
    (['baseline/random']*2,'2x40'), (['baseline/random']*2,'65x65')])
def test_invalid_run_inputs_fail_before_recording(setup,players,size):
    directory, official = setup
    with pytest.raises(ValueError):
        research.run(players,league=directory,size=size)
    assert official.leaderboard()['completed_games'] == 0
    assert not (directory/'runs').exists()


@pytest.mark.parametrize('body', [
    'std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;',
    'for(;;) { asm volatile("" ::: "memory"); }',
])
def test_timeout_surrenders_scores_and_can_be_committed(setup, tmp_path, monkeypatch, body):
    directory, official = setup
    slow = source(tmp_path/'slow', body)
    report = research.run([slow,'baseline/random'],league=directory,games=4,workers=2)
    assert report['ok'] and report['timing']['timeouts'] == 4
    assert report['job']['status'] == 'complete' and report['completed_games'] == 4
    assert report['violations'] == []
    store = LeagueStore(report['league_directory'])
    entrant = next(p for p in report['players'] if p['kind']=='cpp')
    for row in store.matches()['matches']:
        assert row['status'] == 'complete' and row['reason'] == 'surrender'
        result, blob = store.replay(row['game_id'])
        event = result['forfeits'][0]
        assert event['reason'] == 'decision_timeout' and event['tick'] == 0
        assert result['elimination_ticks'][event['seat']] == 0
        assert result['ranks'][event['seat']] == 2
        assert result['winner_seat'] != event['seat']
        assert not result['final_alive'][event['seat']]
        assert result['kills'] == [0,0]
        assert len(replay_frames(blob,result)) == 1
        assert result['rating_changes'][entrant['entrant_id']]['delta'] < 0
    assert entrant['entrant_id'] not in store.disabled_policy_ids()
    from generals_arena.league.statistics import LeaderboardStatistics
    stats = LeaderboardStatistics(store).leaderboard()
    loser = next(p for p in stats['ranking'] if p['entrant_id'] == entrant['entrant_id'])
    assert loser['survival_index'] == loser['kill_index'] == loser['win_rate'] == 0
    monkeypatch.setattr(research, 'PUBLICATION_GAMES', 4)
    publication = research.commit(slow,league=directory,name='slow',workers=2)
    assert publication['ok'] and publication['completed_games'] == 4
    assert all(board['completed_games'] == 2 for board in official.leaderboards().values())


def test_violation_as_an_opponent_also_prevents_commit(setup,tmp_path):
    directory, _ = setup
    fast, bad = source(tmp_path/'fast'), source(tmp_path/'bad','return -1;')
    report = research.run([fast,bad],league=directory,games=2)
    assert not report['ok'] and report['players'][0]['candidate_violations'] == 0
    with pytest.raises(ValueError,match='violation'):
        research.commit(bad,league=directory)


def test_broken_destructor_is_attributed_before_the_next_match(setup,tmp_path):
    directory, _ = setup
    code = source(tmp_path/'destructor')
    path = code/'policy.cpp'
    path.write_text(path.read_text().replace('delete static_cast<Policy *>(p);',
        'throw std::runtime_error("destruction");'))
    report = research.run([code,'baseline/random'],league=directory,games=2)
    assert not report['ok'] and report['completed_games'] == 0
    assert 'destruction' in report['violations'][0]['policy_error']
    assert report['violations'][0]['ticks'] == 4
    store = LeagueStore(report['league_directory'])
    result,blob = store.replay(report['violations'][0]['game_id'])
    assert len(replay_frames(blob,result)) == 5


def test_signal_timeout_recovers_committed_moves_from_shared_journal(setup,tmp_path):
    directory, _ = setup
    code = source(tmp_path/'late','if(o->tick==2) std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;')
    report = research.run([code,'baseline/random'],league=directory)
    assert report['ok'] and report['timing']['timeouts'] == 1
    store = LeagueStore(report['league_directory'])
    match = store.matches()['matches'][0]
    result, blob = store.replay(match['game_id'])
    assert result['ticks'] == 2 and result['forfeits'][0]['phase'] == 'decision'
    frames = replay_frames(blob,result)
    assert len(frames) == 3 and frames[-1]['tick'] == 2
    assert report['completed_games'] == 1
    assert sum(frames[-1]['alive']) == 1


def test_multiplayer_timeout_restores_survivor_memory_and_neutralizes_land(setup, tmp_path):
    import gzip
    from generals_arena.league.statistics import replay_kills
    directory, _ = setup
    slow = source(tmp_path/'late_multi',
        'if(o->tick==2) for(;;) { asm volatile("" ::: "memory"); } return 0;')
    survivor = source(tmp_path/'survivor', '''
auto *s = static_cast<Policy *>(p);
if(s->calls++ != o->tick) throw std::runtime_error("lost memory");
if(o->tick>=2 && o->alive[0]) throw std::runtime_error("surrendered player still alive");
return 0;''')
    path = survivor/'policy.cpp'
    path.write_text(path.read_text().replace('int pid;', 'int pid; int calls=0;'))
    report = research.run([slow, survivor, survivor, 'baseline/random'], league=directory)
    assert report['ok'], report
    store = LeagueStore(report['league_directory'])
    match = store.matches()['matches'][0]
    result, blob = store.replay(match['game_id'])
    assert result['ticks'] == 4 and result['elimination_ticks'][0] == 2
    assert result['forfeits'][0]['seat'] == 0 and result['ranks'][0] == 4
    assert result['capture_ticks'] == [None]*4 and result['kills'] == [0]*4
    assert replay_kills(result, blob) == [0]*4
    assert [m['calls'] for m in result['decision_timing']] == [3,4,4]
    trace = json.loads(gzip.decompress(blob))
    assert all(Action(*row[0]) == Action.wait() for row in trace['actions'][2:])
    frames = replay_frames(blob, result)
    import numpy as np
    mask = np.array(frames[1]['owner']) == 0
    for frame in frames[2:]:
        assert not frame['alive'][0]
        assert (np.array(frame['owner'])[mask] == -2).all()
        assert (np.array(frame['army'])[mask] == np.array(frames[2]['army'])[mask]).all()


def test_native_parallel_and_serial_results_and_ratings_match(setup, tmp_path):
    import shutil
    directory, _ = setup
    cpp = source(tmp_path/'counter')
    spec = research.snapshot_policy(LeagueStore(directory), cpp)
    copy = tmp_path/'parallel'
    shutil.copytree(directory, copy)
    selector = 'policy/'+spec['version']
    left = research.run([selector,'baseline/aggressive'],league=directory,games=6,workers=1)
    right = research.run([selector,'baseline/aggressive'],league=copy,games=6,workers=2)
    assert left['ok'] and right['ok'] and right['workers'] == 2
    first, second = LeagueStore(left['league_directory']), LeagueStore(right['league_directory'])
    assert first.leaderboard() == second.leaderboard()
    for match in first.matches()['matches']:
        a, blob_a = first.replay(match['game_id'])
        b, blob_b = second.replay(match['game_id'])
        assert blob_a == blob_b
        assert a['final_digest'] == b['final_digest']
        assert a['rating_changes'] == b['rating_changes']


def test_duplicate_policy_timeouts_forfeit_each_seat_once_and_keep_playing(setup, tmp_path):
    directory, _ = setup
    slow = source(tmp_path/'two_slow',
        'if(o->tick==2) for(;;) { asm volatile("" ::: "memory"); } return 0;')
    fast = source(tmp_path/'two_fast')
    report = research.run([slow,slow,fast,fast],league=directory,games=2,workers=2)
    assert report['ok'] and report['timing']['timeouts'] == 4
    store = LeagueStore(report['league_directory'])
    for match in store.matches()['matches']:
        result, blob = store.replay(match['game_id'])
        assert result['ticks'] == 4 and sum(result['final_alive']) == 2
        assert len({event['seat'] for event in result['forfeits']}) == 2
        assert all(event['tick'] == 2 for event in result['forfeits'])
        assert sorted(record['calls'] for record in result['decision_timing']) == [3,3,4,4]
        assert len(replay_frames(blob, result, compact=True)) == 5


def test_human_match_preserves_cpp_initialization_failure(setup, tmp_path, monkeypatch):
    directory, official = setup
    code = source(tmp_path/'broken_create')
    path = code/'policy.cpp'
    path.write_text(path.read_text().replace('return new Policy{pid};', 'throw std::runtime_error("create");'))
    spec = research.snapshot_policy(official,code)
    entrant = official.register(spec)
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
    ('std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;',False)])
def test_human_cpp_match_uses_same_native_deadline_and_replay(setup,tmp_path,monkeypatch,body,failed):
    directory, official = setup
    spec = research.snapshot_policy(official,source(tmp_path/'human_cpp',body))
    entrant = official.register(spec)
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


def test_commit_alias_idempotency_versions_and_owned_artifacts(setup,tmp_path,monkeypatch):
    directory, official = setup
    code = source(tmp_path/'source')
    monkeypatch.setattr(research,'PUBLICATION_GAMES',3)
    reports = []
    for i in range(4):  # Arena itself imposes no three-checkpoint research quota.
        report = research.commit(code,league=directory,name=f'gpt-6.1-nebula-v{i+1}',parameters={'version':i})
        assert report['ok'] and report['completed_games'] == 3
        assert report['name'] == f'gpt-6.1-nebula-v{i+1}'
        assert report['timing']['max_ns'] <= DECISION_NS
        reports.append(report)
    assert {mode:b['completed_games'] for mode,b in official.leaderboards().items()} == {'1v1':4,'ffa':8}
    assert research.commit(code,league=directory,parameters={'version':0}) == reports[0]
    with pytest.raises(ValueError,match='already uses alias'):
        research.commit(code,league=directory,name='renamed',parameters={'version':0})
    with pytest.raises(ValueError,match='different version'):
        research.commit(code,league=directory,name=reports[0]['name'],parameters={'version':8})
    assert {mode:b['completed_games'] for mode,b in official.leaderboards().items()} == {'1v1':4,'ffa':8}
    assert len(research.status(directory)['commits']) == 4
    assert research.run(['@'+reports[0]['name'],'baseline/random'],league=directory)['ok']
    assert {mode:b['completed_games'] for mode,b in official.leaderboards().items()} == {'1v1':4,'ffa':8}
    from generals_arena.league.service import process_lock
    from types import SimpleNamespace
    submission = directory/'commits'/reports[0]['version']
    with process_lock(SimpleNamespace(directory=submission), '.commit.lock'):
        with pytest.raises(RuntimeError, match='another process'):
            research.commit(code,league=directory,parameters={'version':0})
    import shutil
    shutil.rmtree(code)
    from generals_arena.registry import load_controller
    entries = [e for e in official.entrants().values() if e['kind']=='cpp']
    assert all(Path(e['artifact_paths']['bundle']).is_relative_to(directory) for e in entries)
    for entry in entries:
        controller = load_controller(entry, SimpleNamespace(map=MapConfig(15,15,2),visibility='local'))
        try:
            controller.reset(0,42)
            assert controller.act(observe(generate_map(MapConfig(15,15,2),1),0)) == Action.wait()
        finally:
            controller.close()


def test_commit_preflight_failure_and_interrupt_resume(setup,tmp_path,monkeypatch):
    directory, official = setup
    bad = source(tmp_path/'bad','return -1;')
    report = research.commit(bad,league=directory,name='bad')
    assert not report['ok'] and report['phase']=='preflight' and report['completed_games']==0
    assert {e['name'] for e in official.entrants().values()} == {'random', 'WAIT'}
    code = source(tmp_path/'good')
    monkeypatch.setattr(research,'PUBLICATION_GAMES',3)
    original = research.run_jobs
    def interrupted(store,**kwargs):
        if store.directory == directory:
            original(store,**kwargs,limit=1)
            raise KeyboardInterrupt()
        return original(store,**kwargs)
    monkeypatch.setattr(research,'run_jobs',interrupted)
    with pytest.raises(KeyboardInterrupt):
        research.commit(code,league=directory,name='resumable')
    assert official.leaderboard()['completed_games']==1
    monkeypatch.setattr(research,'run_jobs',original)
    resumed = research.commit(code,league=directory)
    assert resumed['ok'] and resumed['completed_games']==3
    assert {mode:b['completed_games'] for mode,b in official.leaderboards().items()} == {'1v1':1,'ffa':2}
    assert len(official.jobs())==1


def test_real_cli_compiles_run_and_commits_full_default_budget(setup,tmp_path):
    import subprocess, sys
    directory, official = setup
    path = source(tmp_path/'source')/'policy.cpp'
    base = [sys.executable,'-m','generals_arena']
    private = subprocess.run([*base,'run','--league',str(directory),'--player','baseline/random',
        '--player',str(path),'--size','15x17','--games','2','--output',str(tmp_path/'export')],capture_output=True,text=True)
    assert private.returncode==0,private.stderr
    assert json.loads(private.stdout)['completed_games']==2
    submitted = subprocess.run([*base,'commit',str(path),'--league',str(directory),
        '--name','testmodel-comet-v1'],capture_output=True,text=True)
    assert submitted.returncode==0,submitted.stderr
    report = json.loads(submitted.stdout)
    assert report['ok'] and report['completed_games']==report['publication_games']==1000
    assert report['name']=='testmodel-comet-v1' and report['timing']['calls']>0
    assert all(board['completed_games'] == 500 for board in official.leaderboards().values())
    assert report['publication_games_by_mode'] == {'1v1':500,'ffa':500}
    assert set(report['ratings']) == set(report['ranks']) == {'1v1','ffa'}
    assert all(v == {'budget':500,'done':500} for v in report['job']['modes'].values())
    assert json.loads(Path(report['report']).read_text())==report
    again = subprocess.run([*base,'commit',str(path),'--league',str(directory)],capture_output=True,text=True)
    assert again.returncode==0 and json.loads(again.stdout)==report
