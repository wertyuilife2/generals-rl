"""The public batch protocol preserves scalar actions, memory, RNG and feedback."""
import numpy as np
import pytest

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.state import Action, MoveResult
from generals_baselines import AI_KINDS, create_controller


def assert_planners_equal(left, right):
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    assert left.path == right.path
    assert left._pending == right._pending
    assert left.last_search_attempts == right.last_search_attempts
    for name in ('terrain', 'structure', 'owner', 'army', 'last_seen'):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize('kind', AI_KINDS)
@pytest.mark.parametrize('players', [2, 3, 8])
def test_public_batch_matches_scalar_memory_rng_and_receipts(kind, players):
    config = MapConfig(25, 27, players)
    count = 4
    seats = np.arange(count) % players
    seeds = [2**48 + i for i in range(count)]
    states = [generate_map(config, 22+i) for i in range(count)]
    singles = [create_controller(kind, int(p), s) for p, s in zip(seats, seeds)]
    batch = create_controller(kind)
    batch.reset_batch(seats.tolist(), seeds)
    try:
        with VecEnv.from_states(states, device='cpu', threads=1, episode_ticks=160) as env:
            read = env.observer(seats)
            for _ in range(160):
                raw = read()
                if not raw['active'].any():
                    break
                current = env.to_states()
                expected = [singles[i].act(observe(current[i], int(seats[i]))).encode()
                            if raw['active'][i] else 0 for i in range(count)]
                actual = batch.act_batch(raw)
                np.testing.assert_array_equal(actual, expected)
                for left, right in zip(batch._batch.controllers, singles):
                    assert_planners_equal(left, right)
                joint = np.zeros((count, players), np.int64)
                joint[np.arange(count), seats] = actual
                result = env.step(joint)
                rows = result.moves[np.arange(count), seats]
                active = raw['active'] & rows[:, 0].astype(bool)
                batch.on_result_batch(dict(moves=rows, actions=actual.copy(), active=active, player_ids=seats))
                for lane in np.flatnonzero(active):
                    row = rows[lane]
                    action = Action.decode(int(actual[lane]))
                    singles[lane].on_result(MoveResult(int(seats[lane]), action, bool(row[1]), '',
                        action.source, int(row[3]), int(row[4]), bool(row[5]),
                        None if row[6] < 0 else int(row[6]), bool(row[7])))
                for left, right in zip(batch._batch.controllers, singles):
                    assert_planners_equal(left, right)
    finally:
        batch.close()
        for controller in singles:
            controller.close()


def test_public_fog_inactive_lanes_and_reset():
    states = [generate_map(MapConfig(15, 17, 2), 22+i) for i in range(4)]
    seats, seeds = [0, 1, 0, 1], [11, 22, 33, 44]
    batch = create_controller('expansion')
    try:
        with VecEnv.from_states(states, threads=1) as env:
            raw = env.observe(seats)
            batch.reset_batch(seats, seeds)
            expected = batch.act_batch(raw).copy()
            batch.reset_batch(seats, seeds)
            raw['active'][::2] = False
            actual = batch.act_batch(raw)
            assert not actual[::2].any()
            np.testing.assert_array_equal(actual[1::2], expected[1::2])
            for lane, controller in enumerate(batch._batch.controllers):
                if lane % 2 == 0:
                    assert controller.terrain is None
                    continue
                hidden = ~raw['visible'][lane].reshape(17, 15)
                for name, value in (('terrain', 255), ('structure', 255), ('owner', -3), ('army', -1)):
                    assert np.all(getattr(controller, name)[hidden] == value)
            batch.reset_batch(seats, seeds)
            raw['active'][:] = True
            np.testing.assert_array_equal(batch.act_batch(raw), expected)
    finally:
        batch.close()
    with pytest.raises(RuntimeError, match='reset_batch'):
        batch.act_batch(raw)


def test_array_contract_rejects_malformed_or_cross_seat_observations():
    batch = create_controller('random')
    batch.reset_batch([0], [17])
    try:
        with VecEnv.from_config(MapConfig(15, 15, 2), threads=1) as env:
            raw = env.observe([0])
            for field, bad in (('owner', raw['state']['owner'].astype(np.int64)),
                               ('army', raw['state']['army'][:, ::-1]),
                               ('tick', raw['state']['tick'].astype(np.int32))):
                wrong = {**raw, 'state': {**raw['state'], field: bad}}
                with pytest.raises(ValueError, match=field):
                    batch.act_batch(wrong)
            with pytest.raises(ValueError, match='player_ids'):
                batch.act_batch({**raw, 'player_ids': np.array([1])})
            batch.act_batch(raw)
            with pytest.raises(ValueError, match='moves'):
                batch.on_result_batch(dict(moves=np.zeros((1, 8), np.int64)))
    finally:
        batch.close()
