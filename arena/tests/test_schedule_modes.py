"""Schedule modes share execution, raw results, scoring evidence and replay."""
import json
from math import comb
from pathlib import Path

import pytest

from generals import MapConfig
from generals_arena.cli import main
from generals_arena.matches.config import MatchConfig
from generals_arena.policies.entries import prepare_entrant
from generals_arena.competitions.evaluation import evaluate
from generals_arena.competitions.evaluation import expected_games
from generals_arena.competitions.evaluation import make_profile
from generals_arena.competitions.evaluation import validate_coverage
from generals_arena.matches.replay import verified_replay
from generals_arena.matches.runner import MatchRunner
from generals_arena.competitions.schedules import fixed_panel
from generals_arena.competitions.schedules import round_robin
from generals_arena.competitions.schedules import seat_rotations
from generals_arena.competitions.tournament import tournament
from policy_test_utils import make_bundle


def roster():
    return [{'name': name, 'kind': 'baseline', 'policy': name}
            for name in ('random', 'aggressive', 'defensive')]


def test_round_robin_coverage_and_instance_streams():
    plans = list(round_robin(['a','b','c','d'], MapConfig(25, 25,3,.5,.5), maps_per_group=2, seed=5))
    assert len(plans) == comb(4,3)*2*3
    assert len({p['game_id'] for p in plans}) == len(plans)
    for offset in range(0,len(plans),3):
        group = plans[offset:offset+3]
        assert len({p['map_seed'] for p in group}) == 1
        identities = [dict(zip(p['instance_ids'], zip(p['entrant_ids'],p['controller_seeds']))) for p in group]
        assert identities[0] == identities[1] == identities[2]
        assert len({p['entrant_ids'][0] for p in group}) == 3
    assert list(seat_rotations(['a','b'],[1,2],count=1)) == [dict(entrant_ids=['a','b'],instance_ids=[0,1],controller_seeds=[1,2])]


def test_tournament_uses_common_execution_and_verified_replays(tmp_path):
    rules = MatchConfig(MapConfig(25, 25,2,.1,.05),max_ticks=12)
    report = tournament(roster(),rules,tmp_path/'rr',maps_per_group=2,record_replays=True,batch_size=0)
    assert report['games'] == report['expected_games'] == 24
    assert report['mode'] == 'tournament'
    assert len(report['ranking']) == 4
    assert next(row for row in report['ranking'] if row['name']=='WAIT')['elo'] == 0
    for game in report['results']:
        assert len(list(verified_replay(Path(game['replay']).read_bytes(),game))) == game['ticks']+1
    assert json.loads((tmp_path/'rr'/'report.json').read_text())['run_id'] == report['run_id']
    chunked = tournament(roster(),rules,maps_per_group=2,batch_size=1)
    assert [{k:v for k,v in r.items() if k!='replay'} for r in report['results']] == chunked['results']
    assert report['ranking'] == chunked['ranking']


def test_training_evaluation_uses_same_manager_and_coverage_protocol(tmp_path):
    bundle = make_bundle(tmp_path/'policy')
    profile = make_profile(MapConfig(25, 25,2,.1,.05),max_ticks=8,maps_per_opponent=2,
                           opponents=[roster()[0]])
    report = evaluate(bundle,profile,batch_size=0)
    validate_coverage(report,profile)
    assert report['games'] == expected_games(profile) == 8
    candidate = prepare_entrant({'name':'opaque-policy','kind':'package','bundle':str(bundle),'device':'cpu'})
    opponent = prepare_entrant(roster()[0])
    plans = list(fixed_panel(profile,opponent,candidate))
    with MatchRunner({e['entrant_id']:e for e in (candidate,opponent)},max_ticks=8,batch_size=0) as runner:
        results, _ = runner.run(plans)
    assert [{k:v for k,v in r.items() if k not in ('outcome','rating_changes','rating_mode','ranks')}
            for r in report['results'] if r['opponent']=='random'] == [
                {k:v for k,v in r.items() if k!='ranks'} for r in results]
    with pytest.raises(RuntimeError,match='every scheduled'):
        validate_coverage(dict(report,results=results[:-1]),profile)


def test_common_admission_accepts_package_and_baseline_tournament(tmp_path):
    bundle = make_bundle(tmp_path/'policy',players=(3,))
    entries = roster()[:2] + [{'name':'opaque-policy','kind':'package','bundle':str(bundle)}]
    report = tournament(entries,MatchConfig(MapConfig(25, 25,3,.5,.5),max_ticks=8))
    assert report['games'] == 3
    assert len(report['ranking']) == 3
    with pytest.raises(ValueError,match='does not support'):
        tournament(entries,MatchConfig(MapConfig(25, 25,2,.1,.05),max_ticks=8),tmp_path/'invalid')
    assert not (tmp_path/'invalid'/'report.json').exists()


def test_cli_round_robin_and_training_evaluation(tmp_path,capsys):
    root=tmp_path
    (root/'roster.json').write_text(json.dumps(roster()[:2]))
    (root/'rules.json').write_text(json.dumps({'map':{'width':25,'height':25,'players':2,'mountain_density':.1,'city_density':.05},'max_ticks':3}))
    assert main(['tournament','--entrants',str(root/'roster.json'),'--config',str(root/'rules.json'),
                 '--output',str(root/'rr')]) == 0
    report=json.loads((root/'rr'/'report.json').read_text())
    assert report['games'] == 6
    profile=make_profile(MapConfig(25, 25,2,.1,.05),max_ticks=3,maps_per_opponent=1,opponents=roster()[:1])
    (root/'profile.json').write_text(json.dumps(profile))
    bundle=make_bundle(root/'bundle')
    assert main(['evaluate','--bundle',str(bundle),'--profile',str(root/'profile.json'),
                 '--output',str(root/'eval')]) == 0
    assert json.loads((root/'eval'/'report.json').read_text())['games'] == 4


def test_cpp_round_robin_uses_isolated_executor_without_league_store(tmp_path):
    from generals_arena.policies.native import build_bundle
    source=tmp_path/'wait.cpp'
    source.write_text('''#include "arena_policy.h"
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
''')
    bundle=tmp_path/'wait.zip'
    build_bundle(source,bundle)
    entries=[{'name':'cpp-wait','kind':'cpp','bundle':str(bundle)},roster()[0]]
    report=tournament(entries,MatchConfig(MapConfig(25, 25,2,.1,.05),max_ticks=3),
                      tmp_path/'cpp',record_replays=True)
    assert report['games']==6 and report['execution']=='mixed'
    assert set(report['execution_modes'])=={'isolated','policy_batch'}
    assert not list(tmp_path.rglob('league.sqlite3'))
    cpp_id = prepare_entrant(entries[0])['entrant_id']
    native_games = 0
    for game in report['results']:
        if cpp_id in game['entrant_ids']:
            assert len(game['decision_timing']) == 1
            native_games += 1
        assert len(list(verified_replay(Path(game['replay']).read_bytes(),game)))==4
    assert native_games == 4
    with pytest.raises(ValueError,match='CPU'):
        tournament(entries,MatchConfig(MapConfig(25, 25,2,.1,.05),max_ticks=3),backend='gpu',device='cuda:0')


def test_duel_reference_name_cannot_be_reassigned_to_another_policy():
    renamed = {'name':'WAIT','kind':'baseline','policy':'aggressive'}
    with pytest.raises(ValueError, match='reserved'):
        make_profile(MapConfig(25, 25,2,.1,.05), opponents=[renamed])
    with pytest.raises(ValueError, match='reserved'):
        tournament([renamed,roster()[0]], MatchConfig(MapConfig(25, 25,2,.1,.05),max_ticks=3))


def test_evaluation_cli_resolves_custom_artifacts_relative_to_entrant_file(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    from generals import Action
    spec_dir=tmp_path/'specs'
    spec_dir.mkdir()
    (spec_dir/'data.txt').write_text('policy-data')
    used=[]

    class Controller:
        def reset(self,player_id,seed): pass
        def act(self,obs): return Action.wait()
        def on_result(self,result): pass

    def factory(*, player_id, seed, data):
        assert Path(data).read_text()=='policy-data'
        used.append(data)
        return Controller()

    monkeypatch.setitem(sys.modules,'relative_artifact_policy',SimpleNamespace(create_controller=factory))
    spec={'name':'relative-policy','kind':'custom','version':'test-v1',
          'factory':'relative_artifact_policy:create_controller','artifacts':{'data':'data.txt'}}
    (spec_dir/'entrant.json').write_text(json.dumps(spec))
    profile=make_profile(MapConfig(25, 25,2,.1,.05),max_ticks=2,maps_per_opponent=1,opponents=roster()[:1])
    (tmp_path/'profile.json').write_text(json.dumps(profile))
    assert main(['evaluate','--entrant',str(spec_dir/'entrant.json'),'--profile',str(tmp_path/'profile.json'),
                 '--output',str(tmp_path/'result')])==0
    assert used and set(used)=={str((spec_dir/'data.txt').resolve())}


def test_tournament_applies_inference_device_default_and_honors_per_policy_override(tmp_path,monkeypatch):
    import generals_arena.competitions.tournament as modes
    bundle=make_bundle(tmp_path/'policy')
    actual=modes.MatchRunner
    devices=[]
    class Capture(actual):
        def __init__(self,entries,**options):
            devices.extend(e['factory_kwargs']['device'] for e in entries.values() if e['kind']=='package')
            super().__init__(entries,**options)
    monkeypatch.setattr(modes,'MatchRunner',Capture)
    policy={'name':'opaque-policy','kind':'package','bundle':str(bundle)}
    rules=MatchConfig(MapConfig(25, 25,2,.1,.05),max_ticks=2)
    modes.tournament([policy,roster()[0]],rules,device='cuda:0')
    modes.tournament([dict(policy,device='cpu'),roster()[0]],rules,device='cuda:0')
    assert devices==['cuda:0','cpu']


def test_private_and_candidate_planners_accept_only_config_and_preserve_instance_streams():
    from types import SimpleNamespace
    from generals_arena.competitions.schedules import candidate_panel, fixed_seats

    config = SimpleNamespace(
        player_counts=(3,),
        sample_map=lambda players, seed: MapConfig(15, 17, players, .1, .05),
        manifest=lambda: {'rules_id': 'test-rules'},
    )
    fixed = fixed_seats(config, ['a', 'b', 'a'], 7, 42, 'fixed',
                        {'mode': 'fixed', 'width': 19, 'height': 23})
    panel = candidate_panel(config, 'candidate', ['a', 'b'], 7, 42, 'panel')
    assert len(fixed) == len(panel) == 7
    assert fixed == fixed_seats(config, ['a', 'b', 'a'], 7, 42, 'fixed',
                                {'mode': 'fixed', 'width': 19, 'height': 23})
    assert panel == candidate_panel(config, 'candidate', ['a', 'b'], 7, 42, 'panel')
    assert {(p['map']['width'], p['map']['height']) for p in fixed} == {(19, 23)}
    assert {(p['map']['width'], p['map']['height']) for p in panel} == {(15, 17)}
    for plans in (fixed, panel):
        assert len({p['game_id'] for p in plans}) == 7
        assert all(p['rules_id'] == 'test-rules' for p in plans)
        for start in range(0, len(plans), 3):
            group = plans[start:start + 3]
            assert {p['rating_group_size'] for p in group} == {len(group)}
            streams = [dict(zip(p['instance_ids'], zip(p['entrant_ids'], p['controller_seeds'])))
                       for p in group]
            assert all(stream == streams[0] for stream in streams)
    assert all(plan['entrant_ids'].count('candidate') == 1 for plan in panel)
