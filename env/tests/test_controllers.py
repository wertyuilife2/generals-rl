"""Behavior tests for information boundaries and queued player intent."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from generals_env.controllers import (
    AggressiveAI, DefensiveAI, ExpansionAI, HumanController, RandomAI,
    make_controllers,
)
from generals_env.engine import CoreEngine
from generals_env.observation import observe
from generals_env.state import (
    Action, ActionKind, Direction, EMPTY, GameState, MapLayout, MoveMode,
    MoveResult, Structure, Terrain, UNKNOWN, UNKNOWN_OWNER,
)


AI_TYPES = (AggressiveAI, ExpansionAI, DefensiveAI, RandomAI)


def board(height=5, width=6, army=10):
    terrain = np.zeros((height, width), dtype=np.uint8)
    structure = np.zeros_like(terrain)
    owner = np.full((height, width), EMPTY, dtype=np.int16)
    armies = np.zeros((height, width), dtype=np.int64)
    structure[0, 0] = structure[-1, -1] = Structure.GENERAL
    owner[0, 0], owner[-1, -1] = 0, 1
    armies[0, 0], armies[-1, -1] = army, 1
    return GameState(MapLayout(terrain), structure, owner, armies, np.ones(2, dtype=bool), np.array([0, height * width - 1], dtype=np.int32))


class ControllerTests(unittest.TestCase):
    def test_enclosed_player_waits_without_recursion(self):
        state = board()
        terrain = state.layout.terrain.copy()
        terrain[0, 1] = terrain[1, 0] = Terrain.MOUNTAIN
        state.layout = MapLayout(terrain)
        for ai_type in AI_TYPES:
            with self.subTest(ai=ai_type.__name__):
                controller = ai_type(seed=7)
                for _ in range(5):
                    self.assertEqual(controller.act(observe(state, 0)), Action.wait())

    def test_replanning_has_fixed_attempt_budget(self):
        state = board(5, 6, army=100)
        for ai_type in AI_TYPES[:3]:
            with self.subTest(ai=ai_type.__name__):
                controller = ai_type(seed=7)
                with patch("generals_env.controllers.base.shortest_path", return_value=None) as search:
                    controller.act(observe(state, 0, "full"))
                    self.assertLessEqual(search.call_count, controller.max_path_attempts)

    def test_same_seed_reproduces_decisions(self):
        obs = observe(board(), 0)
        for ai_type in AI_TYPES:
            with self.subTest(ai=ai_type.__name__):
                first, second = ai_type(seed=99), ai_type(seed=99)
                self.assertEqual([first.act(obs) for _ in range(12)], [second.act(obs) for _ in range(12)])
                first.reset(0, 99)
                self.assertEqual(first.act(obs), ai_type(seed=99).act(obs))

    def test_hidden_map_changes_cannot_change_decisions(self):
        first = board(7, 7)
        second = first.copy()
        terrain = second.layout.terrain.copy()
        terrain[2:6, 2:6] = Terrain.MOUNTAIN
        second.layout = MapLayout(terrain)
        # Same visible board and public statistics, different hidden topology.
        obs1, obs2 = observe(first, 0), observe(second, 0)
        np.testing.assert_array_equal(obs1.terrain, obs2.terrain)
        for ai_type in AI_TYPES:
            left, right = ai_type(seed=11), ai_type(seed=11)
            self.assertEqual(left.act(obs1), right.act(obs2))
            self.assertEqual(left.terrain[3, 3], UNKNOWN)

    def test_memory_expires_and_visible_facts_override(self):
        state = board(5, 6)
        state.owner[0, 1] = 1
        state.army[0, 1] = 4
        ai = AggressiveAI()
        ai.act(observe(state, 0))
        self.assertEqual(ai.owner[0, 1], 1)
        # A synthetic observation is useful here: keep a historical sighting
        # while the cell is outside current vision, then expire it.
        obs = observe(state, 0)
        visible = obs.visible.copy()
        visible[0, 1] = False
        hidden = replace(obs, tick=51, visible=visible)
        ai.act(hidden)
        self.assertEqual(ai.owner[0, 1], UNKNOWN_OWNER)
        state.owner[0, 1], state.army[0, 1] = EMPTY, 0
        state.tick = 52
        ai.act(observe(state, 0))
        self.assertEqual(ai.owner[0, 1], EMPTY)

    def test_path_advance_requires_target_ownership(self):
        ai = AggressiveAI()
        ai.path = [0, 1, 2]
        action = Action.move(0, Direction.RIGHT)
        ai._pending = action
        ai.on_result(MoveResult(0, action, True, "MOVED", 0, 1, 2, target_owned=False))
        self.assertEqual(ai.path, [])
        ai.path, ai._pending = [0, 1, 2], action
        ai.on_result(MoveResult(0, action, True, "MOVED", 0, 1, 2, target_owned=True))
        self.assertEqual(ai.path, [1, 2])

    def test_initial_expansion_and_legal_actions(self):
        for ai_type in AI_TYPES:
            with self.subTest(ai=ai_type.__name__):
                state = board(5, 6, army=20)
                engine = CoreEngine(state, debug=True)
                ai = ai_type(seed=8)
                moved = False
                for _ in range(25):
                    if state.terminated:
                        break
                    engine.begin_tick()
                    action = ai.act(observe(state, 0))
                    result = engine.apply_action(0, action)
                    self.assertTrue(result.valid)
                    moved |= action.kind == ActionKind.MOVE
                    ai.on_result(result)
                    if engine.next_player is not None:
                        engine.apply_action(1, Action.wait())
                    engine.finish_tick()
                self.assertTrue(moved)
                self.assertGreater(int(np.count_nonzero(state.owner == 0)), 1)

    def test_terminated_observation_does_not_consume_rng(self):
        for ai_type in AI_TYPES:
            ai = ai_type(seed=12)
            before = ai.rng.bit_generator.state
            self.assertEqual(ai.act(replace(observe(board(), 0), terminated=True)), Action.wait())
            self.assertEqual(before, ai.rng.bit_generator.state)

    def test_factory_uses_reproducible_independent_seat_streams(self):
        first = make_controllers(["aggressive", "random"], 123)
        second = make_controllers(["aggressive", "random"], 123)
        self.assertEqual(first[0].rng.bit_generator.state, second[0].rng.bit_generator.state)
        self.assertNotEqual(first[0].rng.bit_generator.state, first[1].rng.bit_generator.state)


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
