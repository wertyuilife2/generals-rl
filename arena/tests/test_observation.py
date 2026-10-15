"""Arena's fast observation path must preserve the environment information boundary."""
from dataclasses import fields

import numpy as np
import pytest

from generals import MapConfig
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import Action, compute_stats
from generals_arena.league.observation import local_observation
from generals_baselines import make_controllers


def assert_same(actual, expected):
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype and not a.flags.writeable
        elif field.name == 'stats':
            for key in ('army', 'territory', 'cities'):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
                assert not getattr(a, key).flags.writeable
        else:
            assert a == b


@pytest.mark.parametrize('players', [2, 4, 8])
def test_observations_match_at_every_tick_and_remain_isolated(players):
    state = generate_map(MapConfig(width=15, height=17, players=players), players)
    engine = CoreEngine(state)
    controllers = make_controllers(['random', 'aggressive', 'expansion', 'defensive'][:players]
                                   if players < 8 else ['aggressive'] * 8, seed=17)
    stats = compute_stats(state)
    saved = []
    for _ in range(160):
        observations = []
        for seat in range(players):
            actual, expected = local_observation(state, seat, stats), observe(state, seat)
            assert_same(actual, expected)
            observations.append(actual)
            assert not np.shares_memory(actual.owner, state.owner)
            assert not np.shares_memory(actual.army, state.army)
        saved.append((observations[0], observe(state, 0)))
        if state.terminated:
            break
        actions = [c.act(obs) if state.alive[seat] else Action.wait()
                   for seat, (c, obs) in enumerate(zip(controllers, observations))]
        result = engine.step(actions)
        stats = result.stats
        for move in result.moves:
            controllers[move.player_id].on_result(move)
    for actual, expected in saved:
        assert_same(actual, expected)
    # Even deliberately re-enabling a board buffer cannot modify the engine or
    # another seat's observation. Public stats retain immutable backing bytes.
    actual = local_observation(state, 0, stats)
    original = state.owner.copy()
    actual.owner.flags.writeable = True
    actual.owner[:] = 123
    np.testing.assert_array_equal(state.owner, original)
    with pytest.raises(ValueError):
        actual.stats.army.flags.writeable = True


def test_exact_large_integer_totals_and_hidden_city_counts():
    state = generate_map(MapConfig(width=15, height=15, players=2), 32)
    state.army.ravel()[state.general_pos[0]] = 2**53 + 11
    stats = compute_stats(state)
    for seat in range(2):
        assert_same(local_observation(state, seat, stats), observe(state, seat))
        assert local_observation(state, seat, stats).stats.army[0] == 2**53 + 11


@pytest.mark.parametrize('shape', [(1, 2), (2, 1), (1, 65), (64, 1), (7, 9)])
def test_native_visibility_edges_and_noncontiguous_state(shape):
    from generals.core.state import GameState, MapLayout
    n = np.prod(shape)
    owner = np.full(shape, -1, np.int16)
    structure = np.zeros(shape, np.uint8)
    army = np.zeros(shape, np.int64)
    owner.flat[0], owner.flat[-1] = 0, 1
    structure.flat[0] = structure.flat[-1] = 2
    army.flat[0], army.flat[-1] = 2**53 + 11, 7
    state = GameState(MapLayout(np.zeros(shape, np.uint8)), structure, owner, army,
                      np.ones(2, bool), np.array([0, n - 1], np.int32))
    for name in ('structure', 'owner', 'army'):
        setattr(state, name, np.asfortranarray(getattr(state, name)))
    stats = compute_stats(state)
    for seat in range(2):
        assert_same(local_observation(state, seat, stats), observe(state, seat))
