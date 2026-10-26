"""All schedule modes replay current rating rules, with isolated pool origins."""
from copy import deepcopy
import json

import pytest

from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating_session import score_results


def match(left, right, index, *, winner=None, rotation=0, rotations=1):
    ids = [left, right] if rotation == 0 else [right, left]
    winning_seat = None if winner is None else ids.index(winner)
    return {'game_id': f'{left}-{right}-{index}-{rotation}', 'entrant_ids': ids,
            'winner_seat': winning_seat, 'final_alive': [winner is None or seat == winning_seat for seat in range(2)],
            'elimination_ticks': [None if winner is None or seat == winning_seat else 10 for seat in range(2)],
            'ticks': 10, 'reason': 'tick_limit' if winner is None else 'capture',
            'map_seed': index, 'map': {'width': 9, 'height': 9, 'players': 2},
            'rating_group_size': rotations}


def test_duels_update_wait_raw_rating_and_only_offset_the_display():
    rows = [match('candidate', 'WAIT', index, winner='candidate', rotation=rotation, rotations=2)
            for index in range(6) for rotation in range(2)]
    report = score_results(['candidate', 'WAIT'], rows, mode='1v1', reference='WAIT')
    ranking = {row['entrant_id']: row for row in report['ranking']}
    raw = report['raw_ratings']
    assert raw['candidate'] > 0 > raw['WAIT']
    assert ranking['WAIT']['elo'] == 0
    assert ranking['candidate']['elo'] == raw['candidate'] - raw['WAIT']
    assert report['rating_offset'] == raw['WAIT']
    assert report['rating_policy'] == POLICY and report['rating_model'] == 'egf-gor'
    assert ranking['candidate']['ci_blocks'] == 6
    assert ranking['candidate']['elo_ci'] is not None
    assert all(row['rating_changes']['WAIT']['delta'] < 0 for row in rows)
    json.dumps(report, allow_nan=False)


def test_schedule_is_repeatable_but_online_ratings_depend_on_game_order():
    rows = [match('a', 'b', index, winner='a' if index < 4 else 'b') for index in range(6)]
    actual = score_results(['a', 'b'], deepcopy(rows), mode='1v1', reference='b')
    repeat = score_results(['b', 'a'], deepcopy(rows), mode='1v1', reference='b')
    reverse = score_results(['a', 'b'], deepcopy(rows[::-1]), mode='1v1', reference='b')
    assert actual == repeat
    assert actual['raw_ratings'] != reverse['raw_ratings']


def test_unobserved_policy_gets_no_interval_and_unreferenced_duel_gets_no_origin():
    rows = [match('a', 'b', index, winner='a') for index in range(6)]
    report = score_results(['a', 'b', 'c'], rows, mode='1v1', reference='b')
    absent = next(row for row in report['ranking'] if row['entrant_id'] == 'c')
    assert absent['games'] == 0 and absent['elo_ci'] is None
    unreferenced = score_results(['a', 'b'], rows, mode='1v1')
    assert all(row['elo_ci'] is None for row in unreferenced['ranking'])


def test_ffa_preserves_mean_including_unplayed_policy_and_has_no_anchor():
    rows = [{'game_id': f'{index}-{rotation}', 'entrant_ids': ['a', 'b', 'c'],
             'winner_seat': 0, 'final_alive': [True, False, False], 'elimination_ticks': [None, 20, 10],
             'ticks': 20, 'reason': 'capture', 'map_seed': index,
             'map': {'width': 9, 'height': 9, 'players': 3}, 'rating_group_size': 3}
            for index in range(6) for rotation in range(3)]
    report = score_results(['a', 'b', 'c', 'unplayed'], rows, mode='ffa')
    assert sum(report['raw_ratings'].values()) == pytest.approx(4000.)
    assert report['raw_ratings']['unplayed'] == 1000.
    assert report['raw_ratings']['a'] > report['raw_ratings']['b'] > report['raw_ratings']['c']
    assert report['anchor'] is None and report['rating_offset'] == 0
    assert report['rating_mean'] == 1000. and report['rating_model'] == 'codeforces'
    assert all(row['elo'] == row['raw_elo'] and not row['anchor'] for row in report['ranking'])
    assert all(row['elo_ci'] is not None for row in report['ranking'] if row['games'])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('bad', ['failed', 'duplicate', 'outsider', 'mode'])
def test_failed_duplicate_unrelated_or_wrong_pool_matches_are_not_evidence(bad):
    row = match('a', 'b', 0, winner='a')
    rows = [{**row, 'reason': 'policy_error'}] if bad == 'failed' else [row, row] if bad == 'duplicate' else [match('a', 'c', 0)] if bad == 'outsider' else [row]
    with pytest.raises(ValueError):
        score_results(['a', 'b'], rows, mode='ffa' if bad == 'mode' else '1v1')


@pytest.mark.parametrize('ids,mode,reference', [(['a', 'a'], '1v1', None), (['a'], '1v1', None),
    (['a','b'], 'ffa', 'a'), (['a','b'], '1v1', 'absent'), (['a','b'], 'mixed', None)])
def test_roster_and_reference_must_belong_to_one_pool(ids, mode, reference):
    with pytest.raises(ValueError):
        score_results(ids, [], mode=mode, reference=reference)
