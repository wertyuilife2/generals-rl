"""Rule-level fixtures are deliberately tiny and independent of map generation."""

import unittest

import numpy as np

from generals_env.config import MapConfig, RunConfig
from generals_env.engine import CoreEngine
from generals_env.state import (
    Action, ActionKind, Direction, EMPTY, GameState, MapLayout, MoveMode,
    NEUTRAL, Structure, Terrain, compute_stats,
)


def make_state(height=3, width=4, generals=None, terrain=None):
    generals = generals if generals is not None else [0, height * width - 1]
    terrain = np.zeros((height, width), dtype=np.uint8) if terrain is None else terrain
    structure = np.zeros((height, width), dtype=np.uint8)
    owner = np.full((height, width), EMPTY, dtype=np.int16)
    army = np.zeros((height, width), dtype=np.int64)
    for pid, cell in enumerate(generals):
        structure.flat[cell] = Structure.GENERAL
        owner.flat[cell] = pid
        army.flat[cell] = 1
    return GameState(MapLayout(terrain), structure, owner, army,
                     np.ones(len(generals), dtype=bool), np.array(generals, dtype=np.int32))


class EconomyTests(unittest.TestCase):
    def test_exact_economic_cycles_and_no_double_growth(self):
        for tick in (1, 2, 25, 49, 50, 100):
            with self.subTest(tick=tick):
                state = make_state()
                state.owner.flat[1] = 0
                state.structure.flat[1] = Structure.CITY
                state.army.flat[1] = 7
                state.owner.flat[2] = 0  # An owned zero-army plain grows too.
                state.owner.flat[3] = NEUTRAL
                state.structure.flat[3] = Structure.CITY
                state.army.flat[3] = 17
                state.tick = tick - 1
                engine = CoreEngine(state, debug=True)
                result = engine.step([Action.wait(), Action.wait()])
                self.assertEqual(result.tick, tick)
                self.assertEqual(state.army.flat[0], 1 + (tick % 2 == 0))
                self.assertEqual(state.army.flat[1], 7 + (tick % 2 == 0))
                self.assertEqual(state.army.flat[2], int(tick % 50 == 0))
                self.assertEqual(state.army.flat[3], 17)
                self.assertEqual(state.army.flat[4], 0)

    def test_no_retroactive_growth_after_capture(self):
        state = make_state()
        state.tick = 49
        state.army.flat[0] = 3
        CoreEngine(state, debug=True).step([Action.move(0, Direction.RIGHT), Action.wait()])
        self.assertEqual(state.army.flat[1], 3)  # General 3 -> 4; three move.

    def test_growth_overflow_is_explicit_and_does_not_advance(self):
        state = make_state()
        state.tick = 1
        state.army.flat[0] = np.iinfo(np.int64).max
        engine = CoreEngine(state)
        with self.assertRaises(OverflowError):
            engine.begin_tick()
        self.assertEqual(state.tick, 1)
        self.assertEqual(state.army.flat[0], np.iinfo(np.int64).max)


class MovementTests(unittest.TestCase):
    def test_empty_capture_and_all_but_one(self):
        state = make_state()
        state.army.flat[0] = 10
        result = CoreEngine(state, debug=True).step([Action.move(0, Direction.RIGHT), Action.wait()])
        move = result.moves[0]
        self.assertTrue(move.valid and move.captured and move.target_owned)
        self.assertEqual((move.source, move.target, move.moved), (0, 1, 9))
        self.assertEqual((state.army.flat[0], state.army.flat[1]), (1, 9))
        self.assertEqual(state.owner.flat[1], 0)

    def test_half_rounds_down_and_friendly_merge(self):
        state = make_state()
        state.army.flat[0] = 9
        state.owner.flat[1] = 0
        state.army.flat[1] = 3
        result = CoreEngine(state).step([Action.move(0, Direction.RIGHT, MoveMode.HALF), Action.wait()])
        self.assertEqual((state.army.flat[0], state.army.flat[1]), (5, 7))
        self.assertFalse(result.moves[0].captured)
        self.assertTrue(result.moves[0].target_owned)

    def test_stronger_equal_and_weaker_attacks(self):
        for attackers, defense, remaining, captured in ((9, 5, 4, True), (5, 5, 0, False), (3, 5, 2, False)):
            with self.subTest(attackers=attackers):
                state = make_state()
                state.army.flat[0] = attackers + 1
                state.owner.flat[1] = 1
                state.army.flat[1] = defense
                move = CoreEngine(state, debug=True).step([Action.move(0, Direction.RIGHT), Action.wait()]).moves[0]
                self.assertTrue(move.valid)
                self.assertEqual(move.captured, captured)
                self.assertEqual(state.army.flat[1], remaining)
                self.assertEqual(state.owner.flat[1], 0 if captured else 1)

    def test_city_requires_strictly_more_than_garrison(self):
        state = make_state()
        state.army.flat[0] = 41
        state.owner.flat[1] = NEUTRAL
        state.structure.flat[1] = Structure.CITY
        state.army.flat[1] = 40
        engine = CoreEngine(state, debug=True)
        first = engine.step([Action.move(0, Direction.RIGHT), Action.wait()])
        self.assertFalse(first.moves[0].captured)
        self.assertEqual((state.owner.flat[1], state.army.flat[1]), (NEUTRAL, 0))
        second = engine.step([Action.move(0, Direction.RIGHT), Action.wait()])
        self.assertTrue(second.moves[0].captured)
        self.assertEqual((state.owner.flat[1], state.army.flat[1]), (0, 1))
        self.assertEqual(state.structure.flat[1], Structure.CITY)

    def test_invalid_actions_leave_state_unchanged_and_consume_slot(self):
        cases = (
            (Action.move(-1, Direction.RIGHT), "OUT_OF_BOUNDS"),
            (Action.move(12, Direction.LEFT), "OUT_OF_BOUNDS"),
            (Action.move(0, Direction.UP), "OUT_OF_BOUNDS"),
            (Action.move(3, Direction.RIGHT), "OUT_OF_BOUNDS"),
            (Action.move(11, Direction.LEFT), "NOT_OWNER"),
            (Action.move(0, Direction.RIGHT), "INSUFFICIENT_ARMY"),
        )
        for action, reason in cases:
            with self.subTest(reason=reason, action=action):
                state = make_state()
                before = state.copy()
                result = CoreEngine(state, debug=True).step([action, Action.wait()])
                self.assertEqual(result.moves[0].reason, reason)
                self.assertFalse(result.moves[0].valid)
                np.testing.assert_array_equal(state.owner, before.owner)
                np.testing.assert_array_equal(state.army, before.army)
                self.assertEqual(len(result.moves), 2)

    def test_mountains_cannot_be_entered(self):
        terrain = np.zeros((3, 4), dtype=np.uint8)
        terrain.flat[1] = Terrain.MOUNTAIN
        state = make_state(terrain=terrain)
        state.army.flat[0] = 9
        result = CoreEngine(state, debug=True).step([Action.move(0, Direction.RIGHT), Action.wait()])
        self.assertEqual(result.moves[0].reason, "MOUNTAIN")
        self.assertEqual(state.army.flat[0], 9)

    def test_merge_overflow_does_not_partially_move(self):
        state = make_state()
        state.army.flat[0] = 2
        state.owner.flat[1] = 0
        state.army.flat[1] = np.iinfo(np.int64).max
        engine = CoreEngine(state)
        engine.begin_tick()
        with self.assertRaises(OverflowError):
            engine.apply_action(0, Action.move(0, Direction.RIGHT))
        self.assertEqual(state.army.flat[0], 2)
        self.assertEqual(engine.next_player, 0)


class CaptureTests(unittest.TestCase):
    def test_general_capture_inheritance_and_immediate_termination(self):
        state = make_state(2, 3, generals=[0, 1])
        state.army.flat[0] = 10
        state.army.flat[1] = 2
        for cell, army in zip((2, 3, 4, 5), (0, 1, 5, 8)):
            state.owner.flat[cell] = 1
            state.army.flat[cell] = army
        engine = CoreEngine(state, debug=True)
        result = engine.step([Action.move(0, Direction.RIGHT), Action.move(1, Direction.LEFT)])
        self.assertEqual(result.eliminated, (1,))
        self.assertEqual(len(result.moves), 1)
        self.assertTrue(result.terminated)
        self.assertEqual(result.winner_id, 0)
        self.assertEqual(state.structure.flat[1], Structure.CITY)
        self.assertEqual(state.army.flat[1], 7)  # Directly captured general is not halved.
        np.testing.assert_array_equal(state.army.ravel()[2:], [1, 1, 2, 4])
        np.testing.assert_array_equal(state.owner, np.zeros((2, 3), dtype=np.int16))
        self.assertEqual(result.stats.army[0], 16)
        self.assertEqual(result.stats.territory[0], 6)
        self.assertEqual(result.stats.cities[0], 1)
        before = state.copy()
        with self.assertRaises(RuntimeError):
            engine.step([Action.wait(), Action.wait()])
        with self.assertRaises(RuntimeError):
            engine.begin_tick()
        self.assertEqual(state.tick, before.tick)
        np.testing.assert_array_equal(state.army, before.army)

    def test_elimination_continues_with_other_survivors(self):
        state = make_state(2, 3, generals=[0, 1, 5])
        state.army.flat[0] = 10
        state.army.flat[5] = 3
        state.owner.flat[2] = 1
        state.army.flat[2] = 5
        result = CoreEngine(state, debug=True).step([
            Action.move(0, Direction.RIGHT),
            Action.move(1, Direction.LEFT),
            Action.move(5, Direction.LEFT),
        ])
        self.assertFalse(result.terminated)
        self.assertIsNone(result.winner_id)
        self.assertEqual([move.player_id for move in result.moves], [0, 2])
        self.assertEqual(state.owner.flat[2], 0)
        self.assertEqual(state.army.flat[2], 2)
        self.assertEqual(state.owner.flat[4], 2)


class ContractTests(unittest.TestCase):
    def test_action_encoding_and_numpy_integer_inputs(self):
        self.assertEqual(Action.decode(0), Action.wait())
        for source in (0, 1, 99):
            for direction in Direction:
                for mode in MoveMode:
                    action = Action.move(np.int32(source), direction, mode)
                    self.assertEqual(Action.decode(np.int64(action.encode())), action)
        for kwargs in ({"direction": 4}, {"direction": -1}, {"mode": 2}, {"source": True}, {"source": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                Action(ActionKind.MOVE, **kwargs)
        with self.assertRaises(ValueError):
            Action.decode(-1)

    def test_config_validation(self):
        for players in range(2, 9):
            self.assertEqual(MapConfig(players=np.int64(players)).players, players)
        for kwargs in ({"players": 1}, {"players": 9}, {"width": 0}, {"width": 1, "height": 1}, {"mountain_density": float("nan")}, {"city_density": 0.21}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MapConfig(**kwargs)
        for kwargs in ({"seed": -1}, {"max_ticks": 0}, {"visibility": "secret"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RunConfig(**kwargs)

    def test_layout_is_defensive_and_truly_immutable(self):
        source = np.zeros((2, 3), dtype=np.uint8)
        layout = MapLayout(source)
        source[0, 0] = Terrain.MOUNTAIN
        self.assertEqual(layout.terrain[0, 0], Terrain.PLAIN)
        with self.assertRaises(ValueError):
            layout.terrain[0, 0] = Terrain.MOUNTAIN
        with self.assertRaises(ValueError):
            layout.terrain.setflags(write=True)
        self.assertEqual(layout.topology_id, MapLayout(np.zeros((2, 3), dtype=np.uint8)).topology_id)
        self.assertNotEqual(layout.topology_id, MapLayout(np.zeros((3, 2), dtype=np.uint8)).topology_id)

    def test_state_copy_and_invalid_values(self):
        state = make_state()
        copied = state.copy()
        self.assertIs(copied.layout, state.layout)
        copied.army.flat[0] = 8
        self.assertEqual(state.army.flat[0], 1)
        for name, value in (("army", -1), ("owner", -3), ("structure", 255)):
            broken = state.copy()
            getattr(broken, name).flat[0] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                broken.validate()

    def test_statistics_preserve_integer_precision_and_are_readonly(self):
        state = make_state()
        state.army.flat[0] = 2**53 + 1
        state.owner.flat[1] = 0
        state.army.flat[1] = 2
        stats = compute_stats(state)
        self.assertEqual(stats.army[0], 2**53 + 3)
        self.assertEqual(stats.territory[0], 2)
        with self.assertRaises(ValueError):
            stats.army[0] = 0


    def test_statistics_reject_per_player_overflow_without_changing_state(self):
        state = make_state()
        state.army.flat[0] = np.iinfo(np.int64).max
        state.owner.flat[1] = 0
        state.army.flat[1] = 1
        before = state.army.copy()
        with self.assertRaisesRegex(OverflowError, "total army exceeds int64"):
            compute_stats(state)
        np.testing.assert_array_equal(state.army, before)

    def test_large_statistics_are_limited_per_player_not_globally(self):
        state = make_state()
        maximum = np.iinfo(np.int64).max
        state.army.flat[0] = maximum - 2
        state.owner.flat[1] = 0
        state.army.flat[1] = 2
        state.army.flat[11] = maximum
        stats = compute_stats(state)
        np.testing.assert_array_equal(stats.army, [maximum, maximum])


if __name__ == "__main__":
    unittest.main()
