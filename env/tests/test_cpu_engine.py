"""Native single-game parity tests (require a C++/OpenMP toolchain)."""
import numpy as np
import pytest

from generals import MapConfig
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.recording import state_digest
from generals.core.state import Action, Direction, GameState, MapLayout, MoveMode
from generals.backends.cpu.engine import CPUEngine


def compare_tick(core, native, actions):
    expected, actual = core.step(actions), native.step(actions)
    assert expected.moves == actual.moves
    assert expected.tick == actual.tick
    assert expected.terminated == actual.terminated
    assert expected.winner_id == actual.winner_id
    assert expected.eliminated == actual.eliminated
    assert state_digest(core.state) == state_digest(native.state)
    for name in ('army', 'territory', 'cities'):
        np.testing.assert_array_equal(getattr(expected.stats, name), getattr(actual.stats, name))
    return actual


@pytest.mark.parametrize('players', [2, 4, 8])
def test_move_results_stats_and_states_for_valid_and_invalid_actions(players):
    state = generate_map(MapConfig(15, 17, players), 17)
    state.army.ravel()[state.general_pos] = 57
    core, native = CoreEngine(state), CPUEngine(state.copy())
    rng = np.random.default_rng(17)
    saved = []
    try:
        for _ in range(300):
            if state.terminated:
                break
            actions = []
            for pid in range(players):
                if not state.alive[pid] or rng.random() < .1:
                    actions.append(Action.wait())
                else:
                    source = int(rng.choice(np.flatnonzero(state.owner.ravel() == pid)))
                    if rng.random() < .3:
                        source = int(rng.choice([-1, -(2**62), state.layout.size, 2**80]))
                    actions.append(Action.move(source, Direction(int(rng.integers(4))), MoveMode(int(rng.integers(2)))))
            result = compare_tick(core, native, actions)
            saved.append((result.stats, result.stats.army.copy()))
        for stats, previous in saved:
            np.testing.assert_array_equal(stats.army, previous)
            with pytest.raises(ValueError):
                stats.army.flags.writeable = True
    finally:
        native.close()


def test_capture_elimination_order_inheritance_wait_and_large_integer_stats():
    state = GameState(MapLayout(np.zeros((1, 5), dtype=np.uint8)),
        np.array([[2, 2, 0, 0, 2]], dtype=np.uint8),
        np.array([[0, 1, 1, -1, 2]], dtype=np.int16),
        np.array([[10, 1, 5, 0, 2**53+11]], dtype=np.int64),
        np.ones(3, dtype=bool), np.array([0, 1, 4], dtype=np.int32))
    core, native = CoreEngine(state), CPUEngine(state.copy())
    try:
        first = compare_tick(core, native, [Action.move(0, Direction.RIGHT), Action.move(1, Direction.RIGHT), Action.wait()])
        assert first.eliminated == (1,)
        assert [move.player_id for move in first.moves] == [0, 2]
        compare_tick(core, native, [Action.wait(), Action.wait(), Action.move(4, Direction.LEFT, MoveMode.HALF)])
        compare_tick(core, native, [Action.wait(), Action.wait(), Action.move(3, Direction.LEFT)])
        compare_tick(core, native, [Action.wait(), Action.wait(), Action.move(2, Direction.LEFT)])
        final = compare_tick(core, native, [Action.wait(), Action.wait(), Action.move(1, Direction.LEFT)])
        assert final.terminated and final.winner_id == 2
    finally:
        native.close()


def test_malformed_batches_do_not_advance_and_native_overflow_is_not_ignored():
    state = generate_map(MapConfig(15, 15, 2), 3)
    engine = CPUEngine(state)
    try:
        for actions in ([], [Action.wait(), None], {0: Action.wait(), 1: Action.wait()}):
            with pytest.raises((TypeError, ValueError)):
                engine.step(actions)
            assert engine.state.tick == 0
        engine.env.state['army'][0, state.general_pos[0]] = np.iinfo(np.int64).max
        engine.env.state['tick'][0] = 1
        engine.env.invalidate_cache()
        with pytest.raises(OverflowError):
            engine.step([Action.wait(), Action.wait()])
    finally:
        engine.close()
