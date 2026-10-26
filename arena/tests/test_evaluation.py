"""Generic batch execution preserves scalar matches, seed pairing and replays."""
from copy import deepcopy
import json

import numpy as np
import pytest

from generals import MapConfig
from generals_arena.competitions.evaluation import evaluate
from generals_arena.competitions.evaluation import make_profile
from generals_arena.competitions.evaluation import profile_id
from generals_arena.matches.replay import verified_replay
from policy_test_utils import make_bundle


def profile(*, maps=2, players=2, ticks=18):
    return make_profile(MapConfig(9, 9, players, .1, .05), max_ticks=ticks,
                        opponents=[{'name':'random','kind':'baseline','policy':'random'}],
                        maps_per_opponent=maps, seed=912)


def test_batch_scalar_and_chunking_have_identical_games(tmp_path):
    bundle = make_bundle(tmp_path/'bundle')
    settings = profile()
    batch = evaluate(bundle, settings, tmp_path/'batch.json', batch_size=0)
    scalar = evaluate(bundle, settings, batch_size=0, scalar=True)
    chunks = evaluate(bundle, settings, batch_size=1)
    assert batch['results'] == scalar['results'] == chunks['results']
    assert batch['elo'] == scalar['elo'] == chunks['elo']
    assert batch['games'] == batch['expected_games'] == 8
    assert batch['wins'] + batch['losses'] + batch['draws'] == 8
    assert set(batch['by_opponent']) == {'random', 'WAIT'}
    assert batch['status'] == 'completed' and not batch['errors']
    assert json.loads((tmp_path/'batch.json').read_text())['policy_id'] == batch['policy_id']
    maps = {}
    for game in batch['results']:
        maps.setdefault((game['opponent'], game['map_index']), []).append(game)
    assert all({row['candidate_seat'] for row in pair} == {0,1} for pair in maps.values())
    assert all(len({row['map_seed'] for row in pair}) == 1 for pair in maps.values())


def test_profile_identity_pins_rules_seeds_opponents_and_budget():
    settings = profile()
    original = profile_id(settings)
    assert original == profile_id(deepcopy(settings))
    for key, value in [('seed',914), ('maps_per_opponent',3), ('max_ticks',19), ('paired_seats',False)]:
        assert profile_id(dict(settings, **{key:value})) != original
    with pytest.raises(ValueError, match='fields'):
        profile_id(dict(settings, opponent_ratings={'random':100.}))
    with pytest.raises(ValueError, match='rating mode'):
        profile_id(dict(settings, rating_mode='ffa'))


def test_recorded_games_replay_under_authoritative_engine(tmp_path):
    bundle = make_bundle(tmp_path/'bundle')
    report = evaluate(bundle, profile(maps=1), tmp_path/'with_replays', record_replays=True)
    for game in report['results']:
        from pathlib import Path
        states = list(verified_replay(Path(game['replay']).read_bytes(), game))
        assert len(states) == game['ticks'] + 1


def test_fixed_schedule_uses_shared_gor_and_independent_maps(tmp_path):
    bundle = make_bundle(tmp_path/'bundle')
    settings = profile(maps=6, ticks=2)
    report = evaluate(bundle, settings, batch_size=0)
    assert report['elo'] == 0.  # Every game is a tick-limit draw.
    candidate = next(row for row in report['ranking'] if row['entrant_id'] != report['anchor'] and row['name'] == 'opaque-policy')
    assert candidate['ci_blocks'] == 12  # Six independent maps against each fixed policy.
    assert report['elo_ci']['lower'] < report['elo'] < report['elo_ci']['upper']
    assert report['rating_policy'] == 'gor-relative-codeforces-v9'
    assert report['rating_mode'] == '1v1' and report['reference'] == 'WAIT'
    assert next(row for row in report['ranking'] if row['name'] == 'WAIT')['elo'] == 0.
    assert 'opponent_ratings' not in report


def test_non_duel_uses_independent_scalar_controllers_and_all_seats(tmp_path):
    bundle = make_bundle(tmp_path/'bundle', players=(3,))
    report = evaluate(bundle, profile(maps=1,players=3,ticks=4), batch_size=0)
    assert report['games'] == report['expected_games'] == 3
    assert {row['candidate_seat'] for row in report['results']} == {0,1,2}
    assert report['rating_mode'] == 'ffa' and report['reference'] is None
    assert sum(report['raw_ratings'].values()) / len(report['raw_ratings']) == pytest.approx(1000.)


def test_incompatible_capability_and_bad_profile_do_not_produce_results(tmp_path):
    bundle = make_bundle(tmp_path/'bundle')
    with pytest.raises(ValueError, match='does not support'):
        evaluate(bundle, profile(players=3), tmp_path/'not_written.json')
    assert not (tmp_path/'not_written.json').exists()
    with pytest.raises(ValueError):
        profile_id(dict(profile(), rating_policy='invented'))


def test_cuda_vector_results_match_cpu_when_available(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    bundle = make_bundle(tmp_path/'bundle')
    cpu = evaluate(bundle, profile(maps=1,ticks=8), batch_size=0)
    gpu = evaluate(bundle, profile(maps=1,ticks=8), batch_size=0, backend='gpu',device='cuda:0')
    # Deployment IDs include the inference device; simulated game data do not.
    simulated = lambda rows: [{k:v for k,v in row.items() if k not in ('game_id','entrant_ids','rating_changes')} for row in rows]
    assert simulated(cpu['results']) == simulated(gpu['results'])
    assert cpu['elo'] == gpu['elo']


def test_eliminated_candidate_is_a_loss_when_multiplayer_game_hits_limit(tmp_path,monkeypatch):
    from generals.core.state import GameState,MapLayout
    shape=(9,9)
    owner=np.full(shape,-1,dtype=np.int16)
    structure=np.zeros(shape,dtype=np.uint8)
    army=np.zeros(shape,dtype=np.int64)
    for player,cell in enumerate((0,1,80)):
        owner.flat[cell],structure.flat[cell],army.flat[cell]=player,2,1
    army.flat[0]=20
    state=GameState(MapLayout(np.zeros(shape,dtype=np.uint8)),structure,owner,army,
                    np.ones(3,dtype=np.bool_),np.array([0,1,80],dtype=np.int32))
    monkeypatch.setattr('generals_arena.matches.batch.generate_map',lambda *args: state.copy())
    bundle=make_bundle(tmp_path/'bundle',players=(3,))
    settings=make_profile(MapConfig(9,9,3,0,0),max_ticks=1,maps_per_opponent=1,
        opponents=[{'name':'aggressive','kind':'baseline','policy':'aggressive'}])
    report=evaluate(bundle,settings,batch_size=0)
    eliminated=next(row for row in report['results'] if row['candidate_seat']==1)
    assert eliminated['winner_seat'] is None
    assert not eliminated['final_alive'][1] and sum(eliminated['final_alive'])==2
    assert eliminated['outcome']=='loss'


def test_neural_opponents_have_independent_streams_that_follow_seat_rotations(tmp_path, monkeypatch):
    import generals_arena.competitions.evaluation as evaluation
    from generals_arena.matches import batch as batch_execution
    candidate = make_bundle(tmp_path/'candidate', name='candidate', players=(3,))
    opponent = make_bundle(tmp_path/'opponent', name='opponent', players=(3,))
    settings = make_profile(MapConfig(9, 9, 3, 0, 0), maps_per_opponent=2, max_ticks=2,
        opponents=[{'name':'opponent', 'kind':'package', 'bundle':str(opponent)}])
    resets = []
    original = batch_execution._PolicyGroup.__init__

    def capture(self, entry, rules, seats, seeds, **kwargs):
        resets.append((entry['name'], tuple(seats), tuple(seeds)))
        original(self, entry, rules, seats, seeds, **kwargs)

    monkeypatch.setattr(batch_execution._PolicyGroup, '__init__', capture)
    evaluation.evaluate(candidate, settings, batch_size=0)
    assert len(resets) == 3  # Candidate and two distinct opponent instances.
    streams = [row[2] for row in resets]
    assert not set(streams[0]) & set(streams[1])
    assert not set(streams[0]) & set(streams[2])
    assert not set(streams[1]) & set(streams[2])
    for stream in streams:
        assert stream[0] == stream[1] == stream[2]  # Same instance, same map, rotated seats.
        assert stream[3] == stream[4] == stream[5] != stream[0]
    resets.clear()
    evaluation.evaluate(candidate, settings, batch_size=1)
    for slot in range(3):
        assert tuple(resets[i][2][0] for i in range(slot, len(resets), 3)) == streams[slot]


def test_profile_identity_follows_relocated_schedule_and_rating_sources(monkeypatch):
    import generals_arena.competitions.evaluation as evaluation
    settings = profile()
    before = profile_id(settings)
    original = evaluation.source_digest
    monkeypatch.setattr(evaluation, 'source_digest', lambda package:
                        'f'*64 if package == 'generals_arena.competitions' else original(package))
    assert profile_id(settings) != before


def test_profile_relative_opponent_paths_are_independent_of_cwd(tmp_path, monkeypatch):
    directory = tmp_path / 'profiles'
    make_bundle(directory / 'opponent', name='opponent')
    settings = make_profile(MapConfig(9, 9, 2, 0, 0), max_ticks=2, maps_per_opponent=1,
        opponents=[{'name': 'opponent', 'kind': 'package', 'bundle': 'opponent'}])
    path = directory / 'profile.json'
    path.write_text(json.dumps(settings))
    monkeypatch.chdir(directory)
    identity = profile_id(path)
    monkeypatch.chdir(tmp_path)
    report = evaluate({'name': 'candidate', 'kind': 'baseline', 'policy': 'random'}, path)
    assert report['profile_id'] == identity and report['games'] == 4
    assert report['profile']['opponents'][0]['bundle'] == str(directory / 'opponent')
