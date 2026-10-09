"""Behavior tests for information boundaries and queued player intent."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from generals_baselines import (
    AggressiveAI, DefensiveAI, ExpansionAI, RandomAI,
    make_controllers,
)
from generals.core.engine import CoreEngine
from generals.core.observation import observe
from generals.core.state import (
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
                with patch("generals_baselines.base.shortest_known_path", return_value=None) as search:
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



if __name__ == "__main__":
    unittest.main()
