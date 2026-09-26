"""Generated maps satisfy rule invariants over seats, seeds, and densities."""

from collections import deque
import unittest

import numpy as np

from generals_env.config import MapConfig
from generals_env.mapgen import generate_map
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
                        self.assertTrue(set(positions).issubset(reachable_cells(state.layout.terrain, positions[0])))
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
