"""Calibrated GoR is order independent and joint finalists share real evidence."""
from copy import deepcopy
import json

import pytest

from generals import MapConfig
from generals_arena.competitions.evaluation import (evaluate, make_calibrated_profile,
                                                   profile_id, validate_joint_coverage)
from generals_arena.competitions.rating import probability
from generals_arena.competitions.rating_session import score_results
from generals_arena.competitions.tournament import tournament
from generals_arena.matches.config import MatchConfig
from test_match_scoring import match


def test_fit_recovers_empirical_probability_and_ignores_result_order():
    games = [match('learner', 'WAIT', i, winner='learner' if i < 80 else 'WAIT') for i in range(100)]
    first = score_results(['learner', 'WAIT'], deepcopy(games), mode='1v1', reference='WAIT', fit=True)
    second = score_results(['WAIT', 'learner'], list(reversed(games)), mode='1v1', reference='WAIT', fit=True)
    assert first['raw_ratings'] == second['raw_ratings']
    assert first['ranking'] == second['ranking']
    assert first['raw_ratings']['WAIT'] == 0
    assert probability('1v1', first['raw_ratings']['learner'], 0) == pytest.approx(80.5/101)
    assert first['win_matrix']['learner']['WAIT']['win_rate'] == .8
    assert all('rating_changes' not in row for row in games)


def test_frozen_reference_scores_do_not_drift_and_direct_matches_influence_finalists():
    games = [match(candidate, reference, i) for candidate in ('a', 'b') for reference in ('WAIT', 'expert') for i in range(20)]
    fixed = {'WAIT': 0., 'expert': 1200.}
    tied = score_results(['a', 'b', *fixed], deepcopy(games), mode='1v1', reference='WAIT', fixed_ratings=fixed)
    assert tied['raw_ratings']['a'] == pytest.approx(tied['raw_ratings']['b'])
    games += [match('a', 'b', i, winner='a') for i in range(20)]
    final = score_results(['a', 'b', *fixed], games, mode='1v1', reference='WAIT', fixed_ratings=fixed)
    assert {key: final['raw_ratings'][key] for key in fixed} == fixed
    assert final['raw_ratings']['a'] > final['raw_ratings']['b']
    assert final['win_matrix']['a']['b']['wins'] == final['win_matrix']['b']['a']['losses'] == 20
    assert all(row['elo_ci'] is not None for row in final['ranking'] if row['entrant_id'] in ('a', 'b'))


def test_unbeaten_policy_is_finite_and_disconnected_pool_is_rejected():
    games = [match('a', 'WAIT', i, winner='a') for i in range(20)]
    report = score_results(['a', 'WAIT'], games, mode='1v1', reference='WAIT', fit=True)
    assert 0 < report['raw_ratings']['a'] < 3299
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match='path'):
        score_results(['a', 'b', 'WAIT'], games, mode='1v1', reference='WAIT', fit=True)


def test_calibration_periodic_and_joint_evaluation_use_one_scale(tmp_path):
    roster = [{'name': name, 'kind': 'baseline', 'policy': name} for name in ('random', 'defensive')]
    rules = MatchConfig(MapConfig(25, 25, 2, 0, 0), max_ticks=3)
    calibration = tournament(roster, rules, fit_ratings=True, maps_per_group=2, batch_size=0)
    profile = make_calibrated_profile(calibration, roster, maps_per_opponent=1)
    candidates = [{'name': name, 'kind': 'baseline', 'policy': name} for name in ('aggressive', 'expansion')]
    periodic = evaluate(bundle=candidates[0], profile=profile, output=tmp_path/'periodic', batch_size=0)
    joint = evaluate(candidates, profile, tmp_path/'joint', batch_size=0)
    assert periodic['games'] == 6
    assert joint['games'] == 14  # 2*3 reference pairs plus one candidate pair, each rotated.
    assert all(row['games'] == 8 for row in joint['candidates'])
    reference_scores = {row['name']: row['raw_elo'] for row in calibration['ranking']}
    assert {row['name']: row['raw_elo'] for row in joint['ranking'] if row['name'] in reference_scores} == reference_scores
    assert periodic['elo'] == pytest.approx(0)
    assert (tmp_path/'joint/report.win_matrix.csv').is_file()
    assert joint['estimator'] == 'gor-batch-fit-v1'
    for bad in ('game', 'candidate'):
        broken = deepcopy(joint)
        if bad == 'game':
            broken['results'][-1] = broken['results'][0]
        else:
            broken['candidates'][0]['policy_id'] = 'unrequested'
        with pytest.raises(RuntimeError):
            validate_joint_coverage(broken, profile, [row['policy_id'] for row in joint['candidates']])
    changed = deepcopy(profile)
    changed['calibration']['references']['random']['identity'] = 'changed'
    with pytest.raises(ValueError, match='reference changed'):
        profile_id(changed)
    changed = deepcopy(profile)
    changed['max_ticks'] = 4
    with pytest.raises(ValueError, match='calibration requires'):
        profile_id(changed)
