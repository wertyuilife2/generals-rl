"""Protocol-bound RL league admission, generic capabilities and native mixing."""
from dataclasses import asdict

import pytest
import torch

from generals import MapConfig
from generals_arena.matches.replay import replay_frames
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.jobs import run_jobs
from generals_arena.competitions.store import CompetitionStore
from generals_arena.policies.native import build_bundle
from rl_policy_test_utils import make_submission


def fixed_rules(**kwargs):
    return LeagueConfig(**{**dict(player_counts=(2,), min_side=9, max_side=9, side_cap=9,
                                  max_ticks=2, mountain_min=0, mountain_max=0,
                                  city_min=0, city_max=0), **kwargs})


@pytest.fixture
def rl_spec(tmp_path):
    torch.set_num_threads(1)
    bundle, _ = make_submission(tmp_path)
    return {'name': 'test-ViT-3', 'kind': 'package', 'bundle': str(bundle), 'device': 'cpu'}


def test_preflight_and_registration_do_not_schedule_until_add(tmp_path, rl_spec):
    store = CompetitionStore.create(tmp_path / 'league', fixed_rules(), reference='random')
    initial = set(store.entrants())
    prepared = store.validate_entrant_rules(rl_spec)
    assert set(store.entrants()) == initial and not store.jobs()
    entrant = store.register(rl_spec)
    assert entrant['entrant_id'] == prepared['entrant_id'] and not store.jobs()
    added, job = store.add(rl_spec, games=2)
    assert added['entrant_id'] == entrant['entrant_id']
    assert store.add(rl_spec, games=2)[1] == job
    assert run_jobs(store) == 2
    matches = store.matches()['matches']
    assert len(matches) == 2 and all(match['status'] == 'complete' for match in matches)
    assert {match['entrant_ids'].index(entrant['entrant_id']) for match in matches} == {0, 1}
    for match in matches:
        result, blob = store.replay(match['game_id'])
        assert replay_frames(blob, result)[-1]['tick'] == result['ticks']


def test_default_random_map_league_rejects_fixed_shape_rl_without_writes(tmp_path, rl_spec):
    store = CompetitionStore.create(tmp_path / 'league', reference='random')
    before = store.entrants()
    for register in (store.register, store.add, store.validate_entrant_rules):
        with pytest.raises(ValueError, match='complete league map range'):
            register(rl_spec)
        assert store.entrants() == before and not store.jobs()


def test_registered_rl_blocks_incompatible_resize_and_stale_registration(tmp_path, rl_spec):
    store = CompetitionStore.create(tmp_path / 'league', fixed_rules(), reference='random')
    store.register(rl_spec)
    previous = store.manifest
    with pytest.raises(ValueError, match='complete league map range'):
        store.resize_maps(10, 10, 10)
    assert CompetitionStore(store.directory).manifest == previous
    assert store.resize_maps(9, 9, 9) == {2: (9, 9)}
    other = CompetitionStore.create(tmp_path / 'other', fixed_rules(), reference='random')
    CompetitionStore(other.directory).resize_maps(10, 10, 10)
    with pytest.raises(ValueError, match='league changed'):
        other.register(rl_spec)
    assert not any(entry['kind'] == 'package' for entry in CompetitionStore(other.directory).entrants().values())


def test_capability_coverage_is_generic_and_covers_rectangular_combinations(monkeypatch):
    capabilities = {'maps': {'width': [7, 10], 'height': [7, 10]}, 'player_counts': [2, 4]}
    monkeypatch.setattr('generals_arena.policies.artifacts.validate_bundle',
                        lambda *args, **kwargs: {'manifest': {'rules': 'generals-legacy-v1', 'capabilities': capabilities}})
    entrant = {'kind': 'package', 'factory_kwargs': {'bundle': 'synthetic'}}
    # P2 uses side 7 and P4 side 10: no policy geometry is hard-coded here.
    CompetitionStore._validate_package_profile(entrant, fixed_rules(player_counts=(2, 4), min_side=7,
                                                         max_side=7, side_cap=10))
    capabilities['player_counts'] = [2]
    with pytest.raises(ValueError, match='player count 4'):
        CompetitionStore._validate_package_profile(entrant, fixed_rules(player_counts=(2, 4), min_side=7,
                                                             max_side=7, side_cap=10))
    capabilities.update(maps={'shapes': [[3, 3], [4, 4]]}, player_counts=[2])
    rules = fixed_rules(min_side=3, max_side=4, side_cap=4)
    with pytest.raises(ValueError, match='complete league map range'):
        CompetitionStore._validate_package_profile(entrant, rules)
    capabilities['maps']['shapes'].extend([[3, 4], [4, 3]])
    CompetitionStore._validate_package_profile(entrant, rules)
    capabilities['maps'] = {'width': [3, 4], 'height': [3, 3]}
    with pytest.raises(ValueError, match='complete league map range'):
        CompetitionStore._validate_package_profile(entrant, rules)


@pytest.mark.parametrize('workers', (1, 2))
def test_real_rl_and_cpp_share_existing_native_worker_and_verified_replay(tmp_path, rl_spec, workers):
    store = CompetitionStore.create(tmp_path / 'league', fixed_rules(), reference='random')
    source = tmp_path / 'native'
    source.mkdir()
    (source / 'policy.cpp').write_text('''#include "arena_policy.h"
extern "C" void *arena_create(int pid, std::uint64_t, const char *) { return new int(pid); }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *) { return 0; }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
''')
    bundle = tmp_path / 'wait.zip'
    build_bundle(source, bundle)
    cpp = store.register({'name': 'native-wait', 'kind': 'cpp', 'bundle': str(bundle)})
    rl, job = store.add(rl_spec, games=2)
    plans = []
    for seat in (0, 1):
        entries = [cpp['entrant_id'], cpp['entrant_id']]
        entries[seat] = rl['entrant_id']
        plans.append({'game_id': f'mixed-{seat}', 'entrant_ids': entries,
                      'instance_ids': [0, 1], 'controller_seeds': [73, 74],
                      'map': asdict(MapConfig(9, 9, 2, 0, 0)), 'map_seed': 73,
                      'rules_id': store.manifest['rules_id'], 'human': False,
                      'rating_policy': POLICY, 'rating_group_size': 2})
    store.insert_plans(plans, job)
    assert run_jobs(store, workers=workers) == 2
    assert store.jobs()[0]['status'] == 'complete'
    assert store.leaderboard()['completed_games'] == 2
    for plan in plans:
        result, blob = store.replay(plan['game_id'])
        assert result['entrant_ids'] == plan['entrant_ids']
        assert result['reason'] == 'tick_limit'
        assert len(result['decision_timing']) == 1  # The native seat retained its deadline metrics.
        assert result['decision_timing'][0]['timeouts'] == result['decision_timing'][0]['errors'] == 0
        assert replay_frames(blob, result)[-1]['tick'] == result['ticks']
