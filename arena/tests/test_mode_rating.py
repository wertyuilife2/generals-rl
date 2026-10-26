"""Closed-form GoR/Codeforces oracles and long-sequence rating invariants."""
from collections import Counter
import math
import random

import pytest

from generals_arena.competitions.rating import GO_MAX
from generals_arena.competitions.rating import GO_MIN
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating import probability
from generals_arena.competitions.rating import score


def result(ids,ticks,**extra):
    return dict(entrant_ids=ids,elimination_ticks=ticks,rating_policy=POLICY,**extra)


def test_gor_rates_wait_normally_and_has_no_late_game_freeze():
    game=result(['a','wait'],[None,1])
    change=score({},game,{})
    assert change['a']['delta'] == pytest.approx((3300/200)**1.6/2)
    assert change['wait']['after'] == -change['a']['after']
    rating=3300*(1-9**(-1/7))
    assert probability('1v1',rating,0) == pytest.approx(.9)
    win=score({'a':rating},game,{'a':100000})['a']['delta']
    loss=score({'a':rating},result(['a','wait'],[1,None]),{'a':100000})['a']['delta']
    assert win>0>loss and .9*win+.1*loss == pytest.approx(0,abs=1e-10)
    assert score({},result(['a','wait'],[None,None]),{})['a']['delta'] == 0
    assert score({},dict(game,rating_group_size=2),{})['a']['delta'] == change['a']['delta']/2
    edge=score({'a':GO_MAX,'wait':GO_MIN},game,{})
    assert all(GO_MIN<=change['after']<=GO_MAX for change in edge.values())


def test_codeforces_three_equal_players_match_closed_form_performance():
    changes=score({},result(['a','b','wait'],[None,2,1]),{})
    raw=[200*math.log10(2/(math.sqrt(2*rank)-1)-1) for rank in (1,2,3)]
    correction=-math.fsum(raw)/3
    for key,delta in zip(('a','b','wait'),raw,strict=True):
        assert changes[key]['delta'] == pytest.approx(delta+correction,abs=1e-9)
        assert changes[key]['before'] == 1000
    assert sum(change['after'] for change in changes.values()) == pytest.approx(3000)


def test_duplicates_permutations_ties_and_self_play_do_not_add_rated_opponents():
    ratings={'a':1200.,'b':1000.,'c':800.,'idle':1700.}
    ordinary=score(ratings,result(['a','b','c'],[None,2,1]),{})
    duplicate=result(['a','b','b','b','c'],[None,2,2,2,1])
    assert score(ratings,duplicate,{}) == ordinary
    reverse=result(list(reversed(duplicate['entrant_ids'])),list(reversed(duplicate['elimination_ticks'])))
    assert score(ratings,reverse,{}) == ordinary
    self_play=score(ratings,result(['a']*8,[None,7,6,5,4,3,2,1]),{})
    assert self_play['a']['after']==1200 and self_play['a']['rated_games']==0
    tied=score({},result(['a','b','c'],[None,None,None]),{})
    assert all(abs(change['delta'])<1e-9 for change in tied.values())
    tied=score({},result(['a','b','c'],[None,None,1]),{})
    assert tied['a']['delta']==tied['b']['delta']>0>tied['c']['delta']
    assert ratings['idle']==1700 and set(ordinary)=={'a','b','c'}


def test_long_ffa_history_and_new_entrants_keep_mean_1000_and_absentees_fixed():
    rng=random.Random(17)
    ratings,games={},Counter()
    for index in range(1000):
        roster=['wait',*[f'bot-{i}' for i in range(2+index//100)]]
        ids=rng.sample(roster,min(len(roster),rng.randint(3,8)))
        ticks=[None,*range(1,len(ids))]
        rng.shuffle(ticks)
        previous=dict(ratings)
        changes=score(ratings,result(ids,ticks),games)
        for key,change in changes.items():
            ratings[key],games[key]=change['after'],change['rated_games']
        assert all(ratings[key]==previous[key] for key in previous if key not in ids)
        assert sum(ratings.values())/len(ratings) == pytest.approx(1000,abs=1e-9)
