"""Generated maps satisfy rule invariants over seats, seeds, and densities."""

from collections import deque
import unittest

import numpy as np

from generals_env.config import MapConfig
from generals_env.mapgen import _ensure_connected, generate_map
from generals_env.state import EMPTY, NEUTRAL, Structure, Terrain


def reachable_cells(terrain, source):
    """Independent flood fill, rather than testing generation with its BFS."""
    h, w = terrain.shape
    seen = {source}
    queue = deque([source])
    while queue:
        y, x = divmod(queue.popleft(), w)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            cell = ny * w + nx
            if 0 <= ny < h and 0 <= nx < w and terrain[ny, nx] != Terrain.MOUNTAIN and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return seen


class MapGenerationTests(unittest.TestCase):
    def test_all_player_counts_across_seeds_and_density_extremes(self):
        for players in range(2, 9):
            for seed in (0, 17, 2026):
                for mountain_density, city_density in ((0, 0), (0.2, 0.1), (0.4, 0.2)):
                    with self.subTest(players=players, seed=seed, mountains=mountain_density, cities=city_density):
                        config = MapConfig(width=9, height=7, players=players,
                                           mountain_density=mountain_density, city_density=city_density)
                        state = generate_map(config, seed)
                        state.validate()
                        positions = state.general_pos.tolist()
                        self.assertEqual(len(set(positions)), players)
                        self.assertEqual(set(np.flatnonzero(state.layout.terrain.ravel() != Terrain.MOUNTAIN)),
                                         reachable_cells(state.layout.terrain, positions[0]))
                        self.assertEqual(state.tick, 0)
                        self.assertFalse(state.terminated)
                        self.assertTrue(np.all(state.alive))
                        np.testing.assert_array_equal(state.owner.ravel()[positions], np.arange(players))
                        np.testing.assert_array_equal(state.army.ravel()[positions], np.ones(players))
                        np.testing.assert_array_equal(state.structure.ravel()[positions], np.full(players, Structure.GENERAL))
                        city_mask = state.structure == Structure.CITY
                        mountain_mask = state.layout.terrain == Terrain.MOUNTAIN
                        self.assertEqual(int(city_mask.sum()), int(63 * city_density))
                        self.assertLessEqual(int(mountain_mask.sum()), int(63 * mountain_density))
                        self.assertTrue(np.all(state.owner[city_mask] == NEUTRAL))
                        self.assertTrue(np.all(state.army[city_mask] == 40))
                        self.assertTrue(np.all(state.owner[mountain_mask] == EMPTY))
                        self.assertTrue(np.all(state.army[mountain_mask] == 0))

    def test_default_seed_has_no_isolated_cells(self):
        state = generate_map(MapConfig(), seed=42)
        reachable = reachable_cells(state.layout.terrain, int(state.general_pos[0]))
        # Both cells were isolated under the previous generals-only repair.
        self.assertIn(10 * 25 + 6, reachable)
        self.assertIn(14 * 25 + 22, reachable)
        self.assertEqual(len(reachable), int((state.layout.terrain != Terrain.MOUNTAIN).sum()))

    def test_preset_maps_are_fully_connected(self):
        for size, players in ((15, 2), (25, 4), (35, 8)):
            for density in (0, 0.2, 0.4):
                for seed in (0, 42, 2026):
                    with self.subTest(size=size, players=players, density=density, seed=seed):
                        state = generate_map(MapConfig(size, size, players, density, 0.2), seed)
                        reachable = reachable_cells(state.layout.terrain, int(state.general_pos[0]))
                        self.assertEqual(len(reachable), int((state.layout.terrain == Terrain.PLAIN).sum()))
                        self.assertEqual(int((state.structure == Structure.CITY).sum()), int(size * size * 0.2))
                        self.assertLessEqual(int((state.layout.terrain == Terrain.MOUNTAIN).sum()),
                                             int(size * size * density))
                        state.validate()

    def test_already_connected_terrain_is_unchanged(self):
        terrain = np.zeros((5, 7), dtype=np.uint8)
        terrain[1:4, 3] = Terrain.MOUNTAIN
        before = terrain.copy()
        _ensure_connected(terrain, 0)
        np.testing.assert_array_equal(terrain, before)

    def test_small_terrain_patterns_preserve_open_cells_and_connect_everything(self):
        # Exhaust all 3x3 layouts with at least one passable cell, including
        # enclosed centers, diagonal-only contacts, and disconnected corners.
        for bits in range((1 << 9) - 1):
            with self.subTest(bits=bits):
                original = np.array([(bits >> cell) & 1 for cell in range(9)], dtype=np.uint8).reshape(3, 3)
                source = int(np.flatnonzero(original.ravel() == Terrain.PLAIN)[0])
                first, second = original.copy(), original.copy()
                _ensure_connected(first, source)
                _ensure_connected(second, source)
                np.testing.assert_array_equal(first, second)
                self.assertTrue(np.all(first[original == Terrain.PLAIN] == Terrain.PLAIN))
                self.assertTrue(np.isin(first, [Terrain.PLAIN, Terrain.MOUNTAIN]).all())
                self.assertEqual(len(reachable_cells(first, source)), int((first == Terrain.PLAIN).sum()))
                repaired = first.copy()
                _ensure_connected(first, source)
                np.testing.assert_array_equal(first, repaired)

    def test_single_row_and_column_connect_through_thick_walls(self):
        for shape in ((1, 7), (7, 1)):
            with self.subTest(shape=shape):
                terrain = np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.uint8).reshape(shape)
                _ensure_connected(terrain, 3)
                np.testing.assert_array_equal(terrain, np.zeros(shape, dtype=np.uint8))

    def test_connection_prefers_fewer_mountains_over_shorter_distance(self):
        terrain = np.ones((5, 7), dtype=np.uint8)
        terrain[:, 0] = Terrain.PLAIN
        terrain[-1, :] = Terrain.PLAIN
        terrain[0, -1] = Terrain.PLAIN
        before = int((terrain == Terrain.MOUNTAIN).sum())
        _ensure_connected(terrain, 0)
        # The bottom detour needs three mountain removals; the direct top
        # route would remove five. Existing open routes must be reused.
        self.assertEqual(before - int((terrain == Terrain.MOUNTAIN).sum()), 3)
        self.assertEqual(len(reachable_cells(terrain, 0)), int((terrain == Terrain.PLAIN).sum()))

    def test_seed_reproducibility_and_independent_mutable_arrays(self):
        config = MapConfig(width=15, height=15, players=8)
        first, second = generate_map(config, 982), generate_map(config, 982)
        self.assertEqual(first.layout.topology_id, second.layout.topology_id)
        np.testing.assert_array_equal(first.layout.terrain, second.layout.terrain)
        for name in ("structure", "owner", "army", "alive", "general_pos"):
            left, right = getattr(first, name), getattr(second, name)
            np.testing.assert_array_equal(left, right)
            self.assertFalse(np.shares_memory(left, right))
        first.army.ravel()[first.general_pos[0]] = 100
        self.assertEqual(second.army.ravel()[second.general_pos[0]], 1)
        with self.assertRaises(ValueError):
            first.layout.terrain[0, 0] = Terrain.MOUNTAIN

    def test_crowded_single_row_terminates_with_unique_generals(self):
        config = MapConfig(width=8, height=1, players=8, mountain_density=0, city_density=0)
        state = generate_map(config, 3)
        self.assertEqual(set(state.general_pos.tolist()), set(range(8)))
        state.validate()

    def test_invalid_configuration_rejected_before_generation(self):
        for kwargs in (
            {"players": 1}, {"players": 9}, {"width": 0},
            {"mountain_density": 0.5}, {"city_density": -0.1},
            {"width": 2, "height": 2, "players": 8},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MapConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
