"""Behavior tests for information boundaries and queued player intent."""

from dataclasses import replace
import unittest

import numpy as np

from generals.controllers import HumanController
from generals.core.engine import CoreEngine
from generals.core.observation import observe
from generals.core.state import (
    Action, ActionKind, Direction, EMPTY, GameState, MapLayout, MoveMode,
    MoveResult, Structure, Terrain, UNKNOWN, UNKNOWN_OWNER,
)


def board(height=5, width=6, army=10):
    terrain = np.zeros((height, width), dtype=np.uint8)
    structure = np.zeros_like(terrain)
    owner = np.full((height, width), EMPTY, dtype=np.int16)
    armies = np.zeros((height, width), dtype=np.int64)
    structure[0, 0] = structure[-1, -1] = Structure.GENERAL
    owner[0, 0], owner[-1, -1] = 0, 1
    armies[0, 0], armies[-1, -1] = army, 1
    return GameState(MapLayout(terrain), structure, owner, armies, np.ones(2, dtype=bool), np.array([0, height * width - 1], dtype=np.int32))


class HumanControllerTests(unittest.TestCase):
    def setUp(self):
        self.state = board()
        self.obs = observe(self.state, 0)
        self.human = HumanController()
        self.human.select(0, self.obs)

    def test_queue_uses_planned_cursor_and_resets_half(self):
        self.human.toggle_half()
        self.human.enqueue(Direction.RIGHT, self.obs)
        self.human.enqueue(Direction.DOWN, self.obs)
        self.assertEqual([action.source for action in self.human.queue], [0, 1])
        self.assertEqual(self.human.cursor, 7)
        self.assertEqual(self.human.anchor, 0)
        self.assertEqual(self.human.queue[0].mode, MoveMode.HALF)
        self.assertEqual(self.human.queue[1].mode, MoveMode.ALL_BUT_ONE)
        self.assertFalse(self.human.half)

    def test_zero_army_order_executes_and_next_order_runs_on_next_tick(self):
        for count in (0, 1):
            for mode in MoveMode:
                with self.subTest(count=count, mode=mode):
                    state = board(army=count)
                    state.owner[0, 1], state.army[0, 1] = 0, 7
                    human = HumanController()
                    obs = observe(state, 0)
                    human.select(0, obs)
                    if mode == MoveMode.HALF:
                        human.toggle_half()
                    human.enqueue(Direction.RIGHT, obs)
                    human.enqueue(Direction.RIGHT, obs)
                    engine = CoreEngine(state, debug=True)
                    engine.begin_tick()
                    action = human.act(observe(state, 0))
                    self.assertEqual(action, Action.move(0, Direction.RIGHT, mode))
                    move = engine.apply_action(0, action)
                    human.on_result(move)
                    engine.apply_action(1, Action.wait())
                    engine.finish_tick()
                    self.assertTrue(move.valid)
                    self.assertEqual(move.moved, 0)
                    self.assertEqual((state.army[0, 0], state.army[0, 1]), (count, 7))
                    self.assertEqual((human.selected, human.anchor, human.cursor), (1, 1, 2))
                    self.assertEqual(list(human.queue), [Action.move(1, Direction.RIGHT)])
                    engine.begin_tick()
                    move = engine.apply_action(0, human.act(observe(state, 0)))
                    human.on_result(move)
                    engine.apply_action(1, Action.wait())
                    engine.finish_tick()
                    self.assertEqual(move.moved, 6)
                    self.assertEqual(state.owner[0, 2], 0)
                    self.assertEqual(state.army[0, 2], 6)
                    self.assertFalse(human.queue)

    def test_zero_army_attack_is_consumed_instead_of_waiting(self):
        state = board(army=1)
        obs = observe(state, 0)
        human = HumanController()
        human.select(0, obs)
        human.enqueue(Direction.RIGHT, obs)
        human.enqueue(Direction.RIGHT, obs)
        engine = CoreEngine(state, debug=True)
        engine.begin_tick()
        move = engine.apply_action(0, human.act(observe(state, 0)))
        human.on_result(move)
        engine.apply_action(1, Action.wait())
        engine.finish_tick()
        self.assertTrue(move.valid)
        self.assertEqual(move.moved, 0)
        self.assertFalse(move.target_owned)
        self.assertEqual(state.owner[0, 1], EMPTY)
        self.assertFalse(human.queue)
        self.assertEqual((human.selected, human.anchor, human.cursor), (0, 0, 0))
        engine.begin_tick()  # Growth must not revive a consumed order.
        self.assertEqual(human.act(observe(state, 0)), Action.wait())
        engine.apply_action(0, Action.wait())
        engine.apply_action(1, Action.wait())
        engine.finish_tick()
        self.assertEqual(state.owner[0, 1], EMPTY)

    def test_failed_attack_consumes_action_and_clears_downstream(self):
        self.human.enqueue(Direction.RIGHT, self.obs)
        self.human.enqueue(Direction.RIGHT, self.obs)
        action = self.human.act(self.obs)
        self.human.on_result(MoveResult(0, action, True, "MOVED", 0, 1, 9, target_owned=False))
        self.assertFalse(self.human.queue)
        self.assertEqual(self.human.anchor, 0)
        self.assertEqual(self.human.selected, 0)

    def test_undo_after_capture_uses_result_not_old_observation(self):
        self.human.enqueue(Direction.RIGHT, self.obs)
        self.human.enqueue(Direction.RIGHT, self.obs)
        action = self.human.act(self.obs)
        self.human.on_result(MoveResult(0, action, True, "MOVED", 0, 1, 9, target_owned=True))
        self.assertEqual(self.human.anchor, 1)
        self.assertEqual(self.human.cursor, 2)
        self.human.undo()
        self.assertEqual(self.human.cursor, 1)
        self.assertEqual(self.human.selected, 1)
        self.assertFalse(self.human.queue)

    def test_deselect_retains_queue_but_new_source_clears_it(self):
        self.human.enqueue(Direction.RIGHT, self.obs)
        self.human.deselect()
        self.assertIsNone(self.human.selected)
        self.assertEqual(len(self.human.queue), 1)
        self.assertEqual(self.human.act(self.obs), self.human.queue[0])
        self.human.select(0, self.obs)
        self.assertFalse(self.human.queue)

    def test_lost_source_clears_path(self):
        self.human.enqueue(Direction.RIGHT, self.obs)
        self.state.owner[0, 0] = 1
        self.assertEqual(self.human.act(observe(self.state, 0)), Action.wait())
        self.assertFalse(self.human.queue)
        self.assertIsNone(self.human.selected)

    def test_adjacent_friendly_click_moves_unless_forced(self):
        self.state.owner[0, 1], self.state.army[0, 1] = 0, 2
        obs = observe(self.state, 0)
        self.human.select(1, obs)
        self.assertEqual(len(self.human.queue), 1)
        self.assertEqual(self.human.anchor, 0)
        self.human.select(1, obs, force=True)
        self.assertFalse(self.human.queue)
        self.assertEqual(self.human.anchor, 1)

    def test_click_planned_cursor_toggles_half_and_anchor_queues_return(self):
        self.human.select(1, self.obs)
        self.assertEqual(self.human.cursor, 1)
        self.assertEqual(self.human.selected, 0)
        # The board highlights the path endpoint, even before it is owned.
        self.human.select(1, self.obs)
        self.assertTrue(self.human.half)
        self.assertEqual(len(self.human.queue), 1)
        # The old anchor is adjacent: clicking it now queues a return move.
        self.human.select(0, self.obs)
        self.assertEqual(len(self.human.queue), 2)
        self.assertEqual(self.human.queue[-1], Action.move(1, Direction.LEFT, MoveMode.HALF))
        self.assertEqual(self.human.cursor, 0)
        self.assertFalse(self.human.half)

    def test_gui_undo_and_clear_use_latest_ownership_after_enemy_turn(self):
        for command in ("undo", "clear_queue"):
            with self.subTest(command=command):
                state = board()
                state.owner[0, 1], state.army[0, 1] = 0, 1
                human = HumanController()
                human.select(1, observe(state, 0))
                human.enqueue(Direction.RIGHT, observe(state, 0))
                self.assertEqual(human.act(observe(state, 0)), Action.move(1, Direction.RIGHT))
                # A later player captures the selected outpost in this tick.
                state.owner[0, 1], state.army[0, 1] = 1, 2
                getattr(human, command)(observe(state, 0))
                self.assertFalse(human.queue)
                self.assertIsNone(human.selected)
                self.assertIsNone(human.anchor)
                self.assertIsNone(human.cursor)

    def test_bounds_and_mountains_are_rejected(self):
        self.human.enqueue(Direction.LEFT, self.obs)
        self.assertFalse(self.human.queue)
        terrain = self.state.layout.terrain.copy()
        terrain[0, 1] = Terrain.MOUNTAIN
        self.state.layout = MapLayout(terrain)
        self.human.enqueue(Direction.RIGHT, observe(self.state, 0))
        self.assertFalse(self.human.queue)

    def test_queue_limit_and_elimination(self):
        for i in range(70):
            self.human.enqueue(Direction.RIGHT if i % 2 == 0 else Direction.LEFT, self.obs)
        self.assertEqual(len(self.human.queue), 64)
        self.human.eliminate()
        self.human.select(0, self.obs)
        self.human.enqueue(Direction.RIGHT, self.obs)
        self.assertFalse(self.human.queue)
        self.assertIsNone(self.human.selected)


if __name__ == "__main__":
    unittest.main()
