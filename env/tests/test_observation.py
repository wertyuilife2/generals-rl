"""Player snapshots expose precisely the permitted information."""

from dataclasses import replace
import unittest

import numpy as np

from generals_env.engine import CoreEngine
from generals_env.observation import observe, snapshot, visibility_mask, legal_action_mask
from generals_env.state import (
    Action, Direction, EMPTY, GameState, MapLayout, MoveMode, Structure,
    Terrain, UNKNOWN, UNKNOWN_OWNER,
)


def state_with_generals():
    terrain = np.zeros((5, 5), dtype=np.uint8)
    structure = np.zeros_like(terrain)
    owner = np.full((5, 5), EMPTY, dtype=np.int16)
    army = np.zeros((5, 5), dtype=np.int64)
    structure[2, 2] = structure[4, 4] = Structure.GENERAL
    owner[2, 2], owner[4, 4] = 0, 1
    army[2, 2], army[4, 4] = 3, 50
    return GameState(MapLayout(terrain), structure, owner, army,
                     np.array([True, True]), np.array([12, 24], dtype=np.int32))


class ObservationTests(unittest.TestCase):
    def test_local_vision_includes_diagonals_and_clips_edges(self):
        owner = np.full((5, 6), EMPTY, dtype=np.int16)
        owner[2, 2] = 0
        expected = np.zeros_like(owner, dtype=bool)
        expected[1:4, 1:4] = True
        np.testing.assert_array_equal(visibility_mask(owner, 0), expected)
        owner[0, 0] = 0
        expected[:2, :2] = True
        np.testing.assert_array_equal(visibility_mask(owner, 0), expected)
        owner[4, 5] = 1
        expected_enemy = np.zeros_like(owner, dtype=bool)
        expected_enemy[3:5, 4:6] = True
        np.testing.assert_array_equal(visibility_mask(owner, 1), expected_enemy)

    def test_hidden_fields_use_unknown_and_only_own_general_is_exposed(self):
        state = state_with_generals()
        obs = observe(state, 0)
        hidden = ~obs.visible
        self.assertTrue(np.all(obs.terrain[hidden] == UNKNOWN))
        self.assertTrue(np.all(obs.structure[hidden] == UNKNOWN))
        self.assertTrue(np.all(obs.owner[hidden] == UNKNOWN_OWNER))
        self.assertTrue(np.all(obs.army[hidden] == -1))
        self.assertEqual(obs.general_pos, 12)
        np.testing.assert_array_equal(obs.alive, [True, True])
        np.testing.assert_array_equal(obs.stats.army, [3, 50])
        np.testing.assert_array_equal(obs.stats.territory, [1, 1])

    def test_snapshots_are_readonly_and_do_not_alias_state(self):
        state = state_with_generals()
        obs = observe(state, 0)
        for name in ("structure", "owner", "army", "alive"):
            original, published = getattr(state, name), getattr(obs, name)
            self.assertFalse(np.shares_memory(original, published))
            self.assertFalse(published.flags.writeable)
            saved = published.copy()
            original.flat[0] = 1
            np.testing.assert_array_equal(published, saved)
            with self.assertRaises(ValueError):
                published.flat[0] = 1
        self.assertFalse(np.shares_memory(obs.terrain, state.layout.terrain))
        before = obs.stats.army.copy()
        state.army[2, 2] += 100
        np.testing.assert_array_equal(obs.stats.army, before)
        self.assertFalse(obs.stats.army.flags.writeable)

    def test_equal_public_information_masks_changed_hidden_world(self):
        first = state_with_generals()
        second = first.copy()
        # Relocate a hidden opponent and alter hidden terrain while keeping
        # all public totals and visible facts exactly equal.
        second.structure[4, 4], second.owner[4, 4], second.army[4, 4] = Structure.NONE, EMPTY, 0
        second.structure[4, 0], second.owner[4, 0], second.army[4, 0] = Structure.GENERAL, 1, 50
        second.general_pos[1] = 20
        terrain = second.layout.terrain.copy()
        terrain[0, 2] = Terrain.MOUNTAIN
        second.layout = MapLayout(terrain)
        first.validate()
        second.validate()
        left, right = observe(first, 0), observe(second, 0)
        for field in ("terrain", "structure", "owner", "army", "visible", "alive"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        for field in ("army", "territory", "cities"):
            np.testing.assert_array_equal(getattr(left.stats, field), getattr(right.stats, field))
        self.assertEqual(left.general_pos, right.general_pos)
        # FULL is an explicit privileged mode, so this change is visible there.
        self.assertFalse(np.array_equal(observe(first, 0, "full").owner, observe(second, 0, "full").owner))

    def test_hidden_city_counts_do_not_leak_through_public_totals(self):
        first = state_with_generals()
        first.owner[2, 1], first.army[2, 1], first.structure[2, 1] = 0, 4, Structure.CITY
        first.owner[2, 3], first.army[2, 3], first.structure[2, 3] = 1, 7, Structure.CITY
        first.owner[4, 3], first.army[4, 3] = 1, 8
        second = first.copy()
        # Only a hidden opponent tile changes from plain land into a city.
        # Public army, territory and alive, plus all visible cells, are equal.
        second.structure[4, 3] = Structure.CITY
        first.validate()
        second.validate()
        left, right = observe(first, 0), observe(second, 0)
        self.assertFalse(left.visible[4, 3])
        self.assertEqual(right.structure[2, 3], Structure.CITY)
        for field in ("terrain", "structure", "owner", "army", "visible", "alive"):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        for field in ("army", "territory", "cities"):
            np.testing.assert_array_equal(getattr(left.stats, field), getattr(right.stats, field))
        np.testing.assert_array_equal(left.stats.cities, [1, -1])
        np.testing.assert_array_equal(right.stats.cities, [1, -1])
        # Seeing one opponent city does not reveal or estimate their total.
        self.assertEqual(right.stats.cities[1], -1)
        np.testing.assert_array_equal(observe(second, 1).stats.cities, [-1, 2])
        np.testing.assert_array_equal(observe(first, 0, "full").stats.cities, [1, 1])
        np.testing.assert_array_equal(observe(second, 0, "full").stats.cities, [1, 2])
        np.testing.assert_array_equal(snapshot(second).stats.cities, [1, 2])
        with self.assertRaises(ValueError):
            right.stats.cities[1] = 2

    def test_spectator_snapshot_is_explicit_full_view(self):
        state = state_with_generals()
        view = snapshot(state)
        self.assertEqual(view.player_id, -1)
        self.assertEqual(view.general_pos, -1)
        self.assertEqual(view.visibility, "full")
        self.assertTrue(np.all(view.visible))
        np.testing.assert_array_equal(view.army, state.army)
        self.assertFalse(np.shares_memory(view.army, state.army))

    def test_legal_mask_matches_actual_engine_including_losing_attacks(self):
        state = state_with_generals()
        state.owner[2, 3], state.army[2, 3] = 1, 99
        terrain = state.layout.terrain.copy()
        terrain[1, 2] = Terrain.MOUNTAIN
        state.layout = MapLayout(terrain)
        state.owner[0, 0], state.army[0, 0] = 0, 1
        state.validate()
        obs = observe(state, 0)
        mask, wait = legal_action_mask(obs)
        self.assertTrue(wait)
        self.assertEqual(mask.shape, (25, 4, 2))
        self.assertFalse(mask[12, Direction.UP].any())
        self.assertTrue(mask[12, Direction.RIGHT].all())  # Cannot win, but legal.
        self.assertTrue(mask[0, Direction.RIGHT].all())  # Zero-army transfer is legal.
        self.assertTrue(mask[0, Direction.DOWN].all())
        self.assertFalse(mask[0, Direction.UP].any())
        self.assertFalse(mask[0, Direction.LEFT].any())
        for source in range(state.layout.size):
            for direction in Direction:
                for mode in MoveMode:
                    engine = CoreEngine(state.copy())
                    engine.begin_tick()  # tick 1: no growth changes legality.
                    result = engine.apply_action(0, Action.move(source, direction, mode))
                    self.assertEqual(bool(mask[source, direction, mode]), result.valid)

    def test_no_legal_actions_after_elimination_or_termination(self):
        obs = observe(state_with_generals(), 0)
        for closed in (replace(obs, terminated=True), replace(obs, alive=np.array([False, True])), snapshot(state_with_generals())):
            mask, wait = legal_action_mask(closed)
            self.assertFalse(mask.any())
            self.assertFalse(wait)

    def test_invalid_player_and_visibility_rejected(self):
        state = state_with_generals()
        for pid in (-1, 2):
            with self.assertRaises(ValueError):
                observe(state, pid)
        with self.assertRaises(TypeError):
            observe(state, True)
        with self.assertRaises(ValueError):
            observe(state, 0, "omniscient")


if __name__ == "__main__":
    unittest.main()
