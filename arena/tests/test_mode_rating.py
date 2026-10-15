"""Independent numerical oracles and invariants for the GoR / CF policies."""
from collections import Counter
import math
import random

import pytest

from generals_arena.league.mode_rating import probability
from generals_arena.league.rating_policy import POLICY, FIXED_GOR_POLICY, score
from generals_arena.league.scheduler import choose_opponents
from generals_arena.league.uncertainty import RatingUncertainty


def result(ids, ticks, **extra):
    return dict(entrant_ids=ids, elimination_ticks=ticks, rating_policy=POLICY, **extra)


def test_gor_matches_formula_rates_wait_normally_and_keeps_late_updates():
    game = result(['a', 'wait'], [None, 1])
    change = score({}, game, 'wait', 32, {})
    assert change['a']['delta'] == pytest.approx((3300/200)**1.6 / 2)
    assert change['wait']['after'] == pytest.approx(-change['a']['after'])
    assert score({}, {**game,'rating_policy':FIXED_GOR_POLICY}, 'wait', 32, {})['wait']['after'] == 0
    r = 3300 * (1 - 9**(-1/7))
    assert probability('1v1', r, 0) == pytest.approx(.9)
    win = score({'a':r}, game, 'wait', 32, {'a':100000})['a']['delta']
    loss = score({'a':r}, result(['a','wait'], [1,None]), 'wait', 32, {'a':100000})['a']['delta']
    assert win > 0 > loss
    assert .9*win + .1*loss == pytest.approx(0, abs=1e-10)
    assert score({}, result(['a','wait'], [None,None]), 'wait', 32, {})['a']['delta'] == 0
    assert score({}, {**game,'rating_group_size':2}, 'wait', 32, {})['a']['delta'] == pytest.approx(change['a']['delta']/2)


def test_cf_three_equal_ratings_against_closed_form_performance():
    game = result(['a','b','wait'], [None,2,1])
    changes = score({}, game, 'wait', 32, {})
    # Three equal players: seed=2, rank=1,2,3. Solve the logistic seed
    # analytically instead of duplicating the implementation's binary search.
    raw = [200*math.log10(2/(math.sqrt(2*rank)-1)-1) for rank in (1,2,3)]
    center = sum(raw)/3
    for key, delta in zip(('a','b','wait'), raw):
        assert changes[key]['delta'] == pytest.approx(delta-center, abs=1e-9)
        assert changes[key]['before'] == 1000
    assert sum(c['after'] for c in changes.values()) == pytest.approx(3000)
    assert changes['a']['after'] > 1000 > changes['wait']['after']


def test_cf_ties_duplicate_strategies_self_play_and_seat_permutations():
    ratings = {'a':1200.,'b':1000.,'c':800.,'idle':1700.}
    game = result(['a','b','c'], [None,2,1])
    ordinary = score(ratings, game, 'b', 32, {})
    repeated = result(['a','b','b','b','c'], [None,2,2,2,1])
    assert score(ratings, repeated, 'b', 32, {}) == ordinary
    reverse = result(list(reversed(repeated['entrant_ids'])), list(reversed(repeated['elimination_ticks'])))
    assert score(ratings, reverse, 'b', 32, {}) == ordinary
    self_play = score(ratings, result(['a']*8, [None,7,6,5,4,3,2,1]), 'a', 32, {})
    assert self_play['a']['after'] == 1200 and self_play['a']['rated_games'] == 0
    tied = score({}, result(['a','b','c'], [None,None,None]), 'wait', 32, {})
    assert all(abs(c['delta']) < 1e-9 for c in tied.values())
    tied = score({}, result(['a','b','c'], [None,None,1]), 'wait', 32, {})
    assert tied['a']['delta'] == tied['b']['delta'] > 0 > tied['c']['delta']
    assert set(ordinary) == {'a','b','c'} and ratings['idle'] == 1700


def test_cf_long_history_new_players_keep_mean_without_moving_absentees():
    rng = random.Random(17)
    ratings, games = {}, Counter()
    for n in range(1000):
        roster = ['wait', *[f'bot-{i}' for i in range(2+n//100)]]
        ids = rng.sample(roster, min(len(roster), rng.randint(3,8)))
        ticks = [None, *range(1,len(ids))]
        rng.shuffle(ticks)
        old = dict(ratings)
        changes = score(ratings, result(ids,ticks), 'wait', 32, games)
        assert set(changes) == set(ids)
        for key,c in changes.items():
            ratings[key], games[key] = c['after'], c['rated_games']
        assert all(ratings[key] == old[key] for key in old if key not in ids)
        assert sum(ratings.values()) / len(ratings) == pytest.approx(1000, abs=1e-9)


def test_matchmaker_prefers_nearby_and_never_special_cases_anchor():
    ratings = {'target':1000.,'near':1000.,'far':2500.}
    counts = Counter()
    for seed in range(3000):
        a = choose_opponents('target', ['near','far'], 2, ratings, {}, 'near', random.Random(seed), seed)
        b = choose_opponents('target', ['near','far'], 2, ratings, {}, 'far', random.Random(seed), seed)
        assert a == b
        counts[a[0]] += 1
    assert .87 < counts['near']/3000 < .93


def test_mode_uncertainty_uses_gor_derivative_and_ffa_mean_constraint():
    model = RatingUncertainty()
    for seed in range(100):
        model.add(dict(game_id=str(seed), entrant_ids=['a','wait']))
    board = dict(anchor='wait', rating_model='egf-gor', ranking=[
        dict(entrant_id='wait',elo=0.), dict(entrant_id='a',elo=0.)])
    model.annotate(board)
    assert board['ranking'][1]['elo_ci']['half_width'] == pytest.approx(1.959963984540054*2/(7/3300*10))
    board.update(anchor=None, rating_model='codeforces')
    for row in board['ranking']:
        row['elo'] = 1000.
    model.annotate(board)
    width = 1.959963984540054/(math.log(10)/400*10)
    assert all(row['elo_ci']['half_width'] == pytest.approx(width) for row in board['ranking'])
    # An unplayed WAIT does not prevent FFA intervals; disconnected rated
    # components do, because their relative origins cannot be inferred.
    model.add(dict(game_id='other', entrant_ids=['b','c']))
    board['ranking'].extend([dict(entrant_id=k,elo=1000.) for k in ('b','c')])
    model.annotate(board)
    assert all(row['elo_ci'] is None for row in board['ranking'])


def test_gor_uncertainty_uses_raw_scores_before_translating_to_wait_zero():
    model = RatingUncertainty()
    for seed in range(100):
        model.add(dict(game_id=str(seed),entrant_ids=['a','wait']))
    raw = dict(anchor='wait',rating_model='egf-gor',ranking=[
        dict(entrant_id='wait',elo=-800.),dict(entrant_id='a',elo=300.)])
    relative = dict(anchor='wait',rating_model='egf-gor',ranking=[
        dict(entrant_id='wait',elo=0.,raw_elo=-800.),dict(entrant_id='a',elo=1100.,raw_elo=300.)])
    model.annotate(raw); model.annotate(relative)
    first,second = raw['ranking'][1]['elo_ci'],relative['ranking'][1]['elo_ci']
    assert first['half_width'] == pytest.approx(second['half_width'])
    assert second['lower'] == pytest.approx(first['lower']+800)
    assert relative['ranking'][0]['elo_ci']['half_width'] == 0
