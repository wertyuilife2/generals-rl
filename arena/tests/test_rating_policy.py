"""The public rating contract has one rule and no historical algorithm fallback."""
import pytest
from policy_test_utils import synthetic_placements

from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating import comparisons
from generals_arena.competitions.rating import initial_rating
from generals_arena.competitions.rating import ranks
from generals_arena.competitions.rating import rating_mode
from generals_arena.competitions.rating import score
from generals_arena.competitions.rating import validate_mode


def result(ids=('a','wait'),ticks=(None,1),**extra):
    return dict(dict(entrant_ids=list(ids),elimination_ticks=list(ticks),placements=synthetic_placements(ticks),rating_policy=POLICY),**extra)


def test_modes_follow_seat_count_and_rank_ties_follow_survival():
    assert rating_mode(result(('a','a'))) == '1v1'
    assert rating_mode(result(('a','a','b'),(None,2,1))) == 'ffa'
    assert initial_rating('1v1') == 0 and initial_rating('ffa') == 1000
    assert ranks(result(('a','b','c','d'),(None,8,8,1))) == [1,2.5,2.5,4]
    for mode in ('mixed','duel',''):
        with pytest.raises(ValueError):
            validate_mode(mode)
    with pytest.raises(ValueError):
        rating_mode(result(('a',),(None,)))


def test_only_current_recorded_policy_is_accepted():
    for policy in ('rank-balanced-opponents-v4','gor-codeforces-v8','invented'):
        with pytest.raises(ValueError,match='unsupported'):
            score({},result(rating_policy=policy),{})
    game=result()
    del game['rating_policy']
    assert score({},game,{}) == score({},result(),{})


def test_information_weights_are_distinct_from_actual_cf_update_weight():
    duel=list(comparisons(result(rating_group_size=2)))
    assert duel == [('a','wait',.5,1)]
    ffa=result(('a','b','b','c'),(None,2,2,1),rating_group_size=4)
    assert list(comparisons(ffa)) == [('a','b',.0625,1),('a','c',.0625,1),('b','c',.0625,1)]
    assert all(change['weight']==.25 for change in score({},ffa,{}).values())
    for size in (0,-1,True,1.5):
        with pytest.raises((ValueError,TypeError)):
            score({},result(rating_group_size=size),{})
