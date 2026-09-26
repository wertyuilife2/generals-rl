import unittest

import numpy as np

from generals_env.engine import CoreEngine
from generals_env.state import Action, Direction, EMPTY, GameState, MapLayout, Structure


def fixture():
    # Two generals face a shared ordinary cell on a one-row map.
    return GameState(
        MapLayout(np.zeros((1, 3), dtype=np.uint8)),
        np.array([[Structure.GENERAL, Structure.NONE, Structure.GENERAL]], dtype=np.uint8),
        np.array([[0, EMPTY, 1]], dtype=np.int16),
        np.array([[5, 0, 7]], dtype=np.int64),
        np.ones(2, dtype=bool), np.array([0, 2], dtype=np.int32),
    )


class TickOrderTests(unittest.TestCase):
    def test_later_player_attacks_result_of_previous_action(self):
        state = fixture()
        engine = CoreEngine(state, debug=True)
        engine.begin_tick()
        self.assertEqual(engine.next_player, 0)
        engine.apply_action(0, Action.move(0, Direction.RIGHT))
        self.assertEqual((state.owner[0, 1], state.army[0, 1]), (0, 4))
        self.assertEqual(engine.next_player, 1)
        move = engine.apply_action(1, Action.move(2, Direction.LEFT))
        self.assertEqual(move.moved, 6)
        self.assertEqual((state.owner[0, 1], state.army[0, 1]), (1, 2))
        result = engine.finish_tick()
        np.testing.assert_array_equal(result.stats.army, [1, 3])
        np.testing.assert_array_equal(result.stats.territory, [1, 2])

    def test_step_and_transaction_are_identical(self):
        first = fixture()
        second = first.copy()
        first.tick = second.tick = 1
        batch = CoreEngine(first, debug=True)
        slots = CoreEngine(second, debug=True)
        actions = [Action.move(0, Direction.RIGHT), Action.move(2, Direction.LEFT)]
        one = batch.step(actions)
        slots.begin_tick()
        self.assertEqual(second.army[0, 0], 6)  # Growth precedes the first action.
        self.assertEqual(second.army[0, 2], 8)
        while slots.next_player is not None:
            pid = slots.next_player
            slots.apply_action(pid, actions[pid])
        two = slots.finish_tick()
        self.assertEqual(one.moves, two.moves)
        np.testing.assert_array_equal(first.army, second.army)
        np.testing.assert_array_equal(first.owner, second.owner)
        np.testing.assert_array_equal(one.stats.army, two.stats.army)

    def test_phase_duplicate_and_out_of_order_errors(self):
        engine = CoreEngine(fixture())
        with self.assertRaises(RuntimeError):
            engine.apply_action(0, Action.wait())
        with self.assertRaises(RuntimeError):
            engine.finish_tick()
        engine.begin_tick()
        with self.assertRaises(RuntimeError):
            engine.begin_tick()
        with self.assertRaises(ValueError):
            engine.apply_action(1, Action.wait())
        with self.assertRaises(RuntimeError):
            engine.finish_tick()
        with self.assertRaises(TypeError):
            engine.apply_action(0, None)
        with self.assertRaises(TypeError):
            engine.apply_action(True, Action.wait())
        self.assertEqual(engine.next_player, 0)
        engine.apply_action(0, Action.wait())
        with self.assertRaises(ValueError):
            engine.apply_action(0, Action.wait())
        engine.apply_action(1, Action.wait())
        with self.assertRaises(RuntimeError):
            engine.apply_action(1, Action.wait())
        engine.finish_tick()

    def test_invalid_batch_cannot_start_a_tick(self):
        state = fixture()
        engine = CoreEngine(state)
        for actions in ([Action.wait()], [Action.wait(), None], {0: Action.wait(), 1: Action.wait()}):
            with self.subTest(actions=actions), self.assertRaises((TypeError, ValueError)):
                engine.step(actions)
            self.assertEqual(state.tick, 0)


if __name__ == "__main__":
    unittest.main()
