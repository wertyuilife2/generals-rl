"""Information is relative to WAIT and cannot be multiplied by duplicate seats."""
import math

import pytest

from generals_arena.league.rating import LOGISTIC_SLOPE
from generals_arena.league.uncertainty import RatingUncertainty
from generals_arena.league.rating_policy import POLICY


def report():
    return {'anchor':'wait', 'ranking':[{'entrant_id':key, 'elo':0.} for key in ('wait','a','b')]}


def add_maps(model, ids, seeds):
    for seed in seeds:
        model.add({'game_id':str(seed), 'entrant_ids':ids,
                   'map_seed':seed, 'map':{'width':15,'height':15}}, 'job')


def test_two_player_interval_and_inverse_square_root_sample_size():
    model = RatingUncertainty()
    add_maps(model, ['wait','a'], range(100))
    board = report()
    model.annotate(board)
    origin, a, disconnected = board['ranking']
    expected = 1.959963984540054 * 2 / (LOGISTIC_SLOPE * math.sqrt(100))
    assert origin['elo_ci']['half_width'] == 0
    assert a['elo_ci']['half_width'] == pytest.approx(expected)
    assert a['elo_ci']['lower'] == -a['elo_ci']['upper']
    assert disconnected['elo_ci'] is None
    add_maps(model, ['wait','a'], range(100,400))
    model.annotate(board)
    assert a['elo_ci']['half_width'] == pytest.approx(expected / 2)


def test_repeated_seats_rotations_and_self_play_do_not_shrink_intervals():
    plain, repeated = RatingUncertainty(), RatingUncertainty()
    add_maps(plain, ['wait','a'], range(20))
    for _ in range(8):
        add_maps(repeated, ['a']*7 + ['wait'], range(20))
    add_maps(repeated, ['a']*8, range(100))
    left, right = report(), report()
    plain.annotate(left)
    repeated.annotate(right)
    assert left == right


def test_indirect_reference_uncertainty_propagates_and_sparse_samples_are_hidden():
    model = RatingUncertainty()
    add_maps(model, ['wait','a'], range(100))
    add_maps(model, ['a','b'], range(100,200))
    board = report()
    model.annotate(board)
    _, a, b = board['ranking']
    assert b['elo_ci']['half_width'] == pytest.approx(a['elo_ci']['half_width'] * math.sqrt(2))
    sparse = RatingUncertainty()
    add_maps(sparse, ['wait','a'], range(5))
    sparse.annotate(board)
    assert board['ranking'][1]['elo_ci'] is None
    assert board['ranking'][2]['elo_ci'] is None


def test_current_model_uses_rotation_weights_but_not_pseudo_observations():
    model = RatingUncertainty()
    for seed in range(10):
        for rotation in range(2):
            result = dict(game_id=f'{seed}-{rotation}',map_seed=seed,map={},
                          entrant_ids=['a','wait'],elimination_ticks=[None,1],
                          rating_policy=POLICY,rating_group_size=2)
            model.add(result, 'job')
            model.add(result, 'job')  # Idempotent repeated reads.
    board = report()
    model.annotate(board)
    assert model.pairs['a','wait'] == 10
    assert board['ranking'][1]['ci_blocks'] == 10
    assert board['ranking'][1]['elo_ci']['half_width'] == pytest.approx(
        1.959963984540054*2/(LOGISTIC_SLOPE*math.sqrt(10)))
