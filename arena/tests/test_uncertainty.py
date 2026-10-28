"""Raw-coordinate GoR and mean-constrained CF confidence intervals."""
import math

import pytest

from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating import slope
from generals_arena.competitions.uncertainty import RatingUncertainty


def report(mode='1v1',ids=('wait','a','b'),raw=None,offset=0,anchor='wait'):
    raw=raw or {key:0. if mode=='1v1' else 1000. for key in ids}
    return {'rating_mode':mode,'anchor':anchor if mode=='1v1' else None,
            'ranking':[{'entrant_id':key,'raw_elo':raw[key],'elo':raw[key]-offset} for key in ids]}


def add_maps(model,ids,seeds,rotations=1):
    for seed in seeds:
        for rotation in range(rotations):
            game=dict(game_id=f'{seed}-{rotation}-'+','.join(ids),entrant_ids=ids,map_seed=seed,map={'width':25,'height':25},
                      elimination_ticks=[None]*len(ids),placements=[(len(ids)+1)/2]*len(ids),rating_policy=POLICY,rating_group_size=rotations)
            model.add(game,'job')
            model.add(game,'job')


def test_gor_interval_scales_with_independent_map_count():
    model=RatingUncertainty('1v1')
    add_maps(model,['wait','a'],range(100))
    board=report()
    model.annotate(board)
    origin,a,disconnected=board['ranking']
    expected=1.959963984540054*2/(slope('1v1',0)*math.sqrt(100))
    assert origin['elo_ci']['half_width']==0
    assert a['elo_ci']['half_width'] == pytest.approx(expected)
    assert disconnected['elo_ci'] is None
    add_maps(model,['wait','a'],range(100,400))
    model.annotate(board)
    assert a['elo_ci']['half_width'] == pytest.approx(expected/2)


def test_rotations_and_duplicate_ffa_seats_do_not_multiply_information():
    plain,repeated=RatingUncertainty('ffa'),RatingUncertainty('ffa')
    add_maps(plain,['a','wait','b'],range(20))
    add_maps(repeated,['a','wait','wait','wait','b'],range(20),rotations=8)
    add_maps(repeated,['a']*8,range(100))
    left,right=report('ffa'),report('ffa')
    plain.annotate(left);repeated.annotate(right)
    assert left==right
    assert repeated.blocks['a']==20
    assert repeated.pairs['a','wait'] == pytest.approx(5)


def test_gor_uses_raw_coordinates_before_translating_display_to_wait_zero():
    model=RatingUncertainty('1v1')
    add_maps(model,['a','wait'],range(100))
    values={'wait':-800.,'a':300.}
    raw=report(ids=('wait','a'),raw=values)
    displayed=report(ids=('wait','a'),raw=values,offset=-800)
    model.annotate(raw);model.annotate(displayed)
    first,second=raw['ranking'][1]['elo_ci'],displayed['ranking'][1]['elo_ci']
    assert second['half_width']==first['half_width']
    assert second['lower'] == pytest.approx(first['lower']+800)
    assert displayed['ranking'][0]['elo_ci']['half_width']==0


def test_ffa_uses_the_sum_zero_subspace_and_rejects_disconnected_pool_offsets():
    model=RatingUncertainty('ffa')
    add_maps(model,['a','b','c'],range(100))
    board=report('ffa',ids=('wait','a','b','c'))
    model.annotate(board)
    edge=100*.25*.25*slope('ffa',1000)**2
    expected=1.959963984540054*math.sqrt(2/(9*edge))
    assert board['ranking'][0]['elo_ci'] is None
    assert all(row['elo_ci']['half_width']==pytest.approx(expected) for row in board['ranking'][1:])
    add_maps(model,['d','e','f'],range(100))
    board=report('ffa',ids=('a','b','c','d','e','f'))
    model.annotate(board)
    assert all(row['elo_ci'] is None for row in board['ranking'])


def test_gor_without_wait_sparse_samples_and_mixed_modes_do_not_invent_precision():
    model=RatingUncertainty('1v1')
    add_maps(model,['a','wait'],range(5))
    board=report()
    model.annotate(board)
    assert board['ranking'][1]['elo_ci'] is None
    board=report(ids=('a','b'),anchor=None)
    model.annotate(board)
    assert all(row['elo_ci'] is None for row in board['ranking'])
    with pytest.raises(ValueError,match='different rating'):
        model.add(dict(game_id='ffa',entrant_ids=['a','b','c']))
    with pytest.raises(ValueError,match='different rating'):
        model.annotate(report('ffa'))
