"""Batch native decisions preserve the existing per-match information and RNG."""
from types import SimpleNamespace

import numpy as np
import pytest

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.observation import observe, visibility_mask
from generals.core.state import Action, MoveResult
from generals_baselines import AI_KINDS, BatchBaselineControllers, make_controllers
from generals_baselines.registry import create_controller


def _specs():
    return [SimpleNamespace(opponent=kind, learner_seat=seat, controller_seed=700 + i * 2 + seat)
            for i, kind in enumerate(AI_KINDS) for seat in (0, 1)]


def _single_controllers(specs, seat_seeds=None):
    result = []
    for index, spec in enumerate(specs):
        if seat_seeds is not None:
            result.append(create_controller(spec.opponent, 1 - spec.learner_seat, int(seat_seeds[index])))
            continue
        kinds = ["random", "random"]
        kinds[1 - spec.learner_seat] = spec.opponent
        pair = make_controllers(kinds, spec.controller_seed)
        result.append(pair[1 - spec.learner_seat])
        pair[spec.learner_seat].close()
    return result


def _assert_planner_equal(left, right):
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    assert left.path == right.path
    assert left._pending == right._pending
    assert left.last_search_attempts == right.last_search_attempts
    for name in ("terrain", "structure", "owner", "army", "last_seen"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize("shape", [(9, 9), (11, 13)])
@pytest.mark.parametrize("direct_seeds", [False, True])
def test_batch_matches_single_lane_actions_rng_memory_and_feedback(shape, direct_seeds):
    specs = _specs()
    states = [generate_map(MapConfig(shape[1], shape[0], 2), 22 + i) for i in range(len(specs))]
    seat_seeds = [2**48 + 100 + i for i in range(len(specs))] if direct_seeds else None
    singles = _single_controllers(specs, seat_seeds)
    baseline = BatchBaselineControllers(specs, height=shape[0], width=shape[1], seat_seeds=seat_seeds)
    learner = np.random.default_rng(971)
    active = np.ones(len(specs), dtype=bool)
    seen_half = False
    seen_feedback = False
    try:
        with VecEnv.from_states(states, device="cpu", threads=1, auto_reset=False,
                                episode_ticks=240) as env:
            for _ in range(240):
                if not active.any():
                    break
                current = env.to_states()
                expected = np.zeros(len(specs), dtype=np.int64)
                for lane in np.flatnonzero(active):
                    seat = 1 - specs[lane].learner_seat
                    expected[lane] = singles[lane].act(observe(current[lane], seat)).encode()
                actual = baseline.act(env.state, active)
                np.testing.assert_array_equal(actual, expected)
                seen_half |= bool(np.any((actual > 0) & ((actual - 1) % 2 == 1)))
                for left, right in zip(baseline.controllers, singles):
                    _assert_planner_equal(left, right)
                joint = np.zeros((len(specs), 2), dtype=np.int64)
                for lane in np.flatnonzero(active):
                    seat = specs[lane].learner_seat
                    obs = observe(current[lane], seat)
                    sources = np.flatnonzero(obs.owner == seat)
                    source = int(sources[learner.integers(len(sources))])
                    joint[lane, seat] = Action.move(source, int(learner.integers(4)),
                                                   int(learner.integers(2))).encode()
                    joint[lane, 1 - seat] = actual[lane]
                result = env.step(joint)
                env.check_errors()
                baseline.on_result(result.moves, joint, active)
                for lane in np.flatnonzero(active):
                    seat = 1 - specs[lane].learner_seat
                    row = result.moves[lane, seat]
                    if row[0]:
                        action = Action.decode(int(joint[lane, seat]))
                        singles[lane].on_result(MoveResult(
                            seat, action, bool(row[1]), "", action.source, int(row[3]),
                            int(row[4]), bool(row[5]), None if row[6] < 0 else int(row[6]), bool(row[7])))
                        seen_feedback |= bool(row[7])
                for left, right in zip(baseline.controllers, singles):
                    _assert_planner_equal(left, right)
                active &= ~result.done.any(axis=1)
        assert seen_half
        assert seen_feedback
    finally:
        baseline.close()
        for controller in singles:
            controller.close()


def test_fogged_batch_cannot_use_hidden_engine_arrays():
    shape = (9, 9)
    specs = _specs()
    states = [generate_map(MapConfig(*shape, 2), 913 + i) for i in range(len(specs))]
    with VecEnv.from_states(states, device="cpu", threads=1) as env, \
            BatchBaselineControllers(specs, height=9, width=9) as left, \
            BatchBaselineControllers(specs, height=9, width=9) as right:
        changed = {name: value.copy() for name, value in env.state.items()}
        for lane, spec in enumerate(specs):
            seat = 1 - spec.learner_seat
            hidden = ~visibility_mask(states[lane].owner, seat).ravel()
            changed["terrain"][lane, hidden] ^= 1
            changed["structure"][lane, hidden] = 2
            changed["army"][lane, hidden] = 991717
            # Changing hidden owners must not add owned cells, which legitimately
            # expand this player's vision; alternate enemies/empty remains hidden.
            changed["owner"][lane, hidden] = spec.learner_seat
        active = np.ones(len(specs), dtype=bool)
        np.testing.assert_array_equal(left.act(env.state, active), right.act(changed, active))
        for lane, (a, b) in enumerate(zip(left.controllers, right.controllers)):
            _assert_planner_equal(a, b)
            hidden = ~visibility_mask(states[lane].owner, a.player_id)
            assert np.all(a.terrain[hidden] == 255)
            assert np.all(a.structure[hidden] == 255)
            assert np.all(a.owner[hidden] == -3)
            assert np.all(a.army[hidden] == -1)


def test_batch_inactive_lifecycle_reset_closed_and_default_feedback():
    specs = _specs()
    states = [generate_map(MapConfig(9, 9, 2), 18 + i) for i in range(len(specs))]
    with VecEnv.from_states(states, device="cpu", threads=1) as env:
        batch = BatchBaselineControllers(specs, height=9, width=9)
        with BatchBaselineControllers(specs, height=9, width=9) as fresh:
            active = np.ones(len(specs), dtype=bool)
            expected = fresh.act(env.state, active).copy()
            active[::2] = False
            actual = batch.act(env.state, active).copy()
            np.testing.assert_array_equal(actual[active], expected[active])
            assert not actual[~active].any()
            for lane in np.flatnonzero(~active):
                assert batch.controllers[lane].terrain is None
                single = _single_controllers([specs[lane]])[0]
                assert batch.controllers[lane].rng.bit_generator.state == single.rng.bit_generator.state
                single.close()
            moves = np.zeros((len(specs), 2, 9), dtype=np.int64)
            for lane in np.flatnonzero(active):
                seat = batch.controllers[lane].player_id
                moves[lane, seat, 0] = 1
                moves[lane, seat, 1] = 0
            batch.on_result(moves, active=active)
            for lane in np.flatnonzero(active):
                assert batch.controllers[lane].path == []
                assert batch.controllers[lane]._pending is None
            batch.close()
            with pytest.raises(RuntimeError, match="closed"):
                batch.act(env.state, active)
            with pytest.raises(RuntimeError, match="closed"):
                batch.on_result(moves)
            batch.reset()
            active[:] = True
            np.testing.assert_array_equal(batch.act(env.state, active), expected)
        batch.close()


def test_batch_array_contract_is_checked_without_silent_copies():
    spec = SimpleNamespace(opponent="random", learner_seat=0, controller_seed=17)
    state = generate_map(MapConfig(9, 9, 2), 17)
    with VecEnv.from_states([state], device="cpu", threads=1) as env, \
            BatchBaselineControllers([spec], height=9, width=9) as batch:
        active = np.ones(1, dtype=bool)
        for field, invalid in (("owner", env.state["owner"].astype(np.int64)),
                               ("army", env.state["army"][:, ::-1]),
                               ("tick", env.state["tick"].astype(np.int32))):
            bad = dict(env.state, **{field: invalid})
            with pytest.raises(ValueError, match=field):
                batch.act(bad, active)
        with pytest.raises(ValueError, match="active"):
            batch.act(env.state, np.ones(1, dtype=np.int8))
        shaped = dict(env.state)
        for field in ("terrain", "structure", "owner", "army"):
            shaped[field] = shaped[field].reshape(1, 9, 9)
        flat_action = batch.act(env.state, active).copy()
        batch.reset()
        np.testing.assert_array_equal(batch.act(shaped, active), flat_action)
        with pytest.raises(ValueError, match="moves"):
            batch.on_result(np.zeros((1, 2, 8), dtype=np.int64))
