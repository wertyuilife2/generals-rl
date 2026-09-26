"""Distance semantics, knowledge boundaries, and bounded cache behavior."""

import unittest
from unittest.mock import patch

import numpy as np

from generals_env.pathfinding import (
    DistanceCache, all_pairs_distances, neighbors, prepare_map,
    shortest_path, single_source_distances,
)
from generals_env.state import MapLayout, Terrain, UNKNOWN


class PathfindingTests(unittest.TestCase):
    def test_neighbors_do_not_wrap_and_exclude_mountains(self):
        terrain = np.zeros((2, 3), dtype=np.uint8)
        terrain[0, 1] = Terrain.MOUNTAIN
        graph = neighbors(terrain)
        np.testing.assert_array_equal(graph[0], [-1, -1, 3, -1])
        np.testing.assert_array_equal(graph[1], [-1, -1, -1, -1])
        np.testing.assert_array_equal(graph[2], [-1, -1, 5, -1])
        np.testing.assert_array_equal(graph[3], [0, 4, -1, -1])
        self.assertFalse(graph.flags.writeable)

    def test_shortest_route_self_blocked_and_unreachable(self):
        terrain = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        self.assertEqual(shortest_path(terrain, 0, 2), [0, 3, 6, 7, 8, 5, 2])
        self.assertEqual(shortest_path(terrain, 0, 0), [0])
        self.assertIsNone(shortest_path(terrain, 1, 1))
        self.assertIsNone(shortest_path(terrain, 0, 1))
        self.assertTrue(np.all(single_source_distances(terrain, 1) == -1))
        terrain[2, 1] = Terrain.MOUNTAIN
        self.assertIsNone(shortest_path(terrain, 0, 2))
        self.assertEqual(single_source_distances(terrain, 0)[2], -1)

    def test_unknown_cells_are_explicit_optimism(self):
        terrain = np.array([[0, UNKNOWN, 0]], dtype=np.uint8)
        self.assertEqual(shortest_path(terrain, 0, 2), [0, 1, 2])
        self.assertIsNone(shortest_path(terrain, 0, 2, unknown_passable=False))
        self.assertIsNone(shortest_path(terrain, 1, 1, unknown_passable=False))
        self.assertEqual(shortest_path(terrain, 1, 1, unknown_passable=True), [1])

    def test_distances_on_rectangular_open_map_are_manhattan(self):
        terrain = np.zeros((3, 7), dtype=np.uint8)
        source = 10  # [1, 3]
        expected = np.array([abs(y - 1) + abs(x - 3) for y in range(3) for x in range(7)], dtype=np.int32)
        np.testing.assert_array_equal(single_source_distances(terrain, source), expected)

    def test_cache_lru_capacity_clear_and_topology_changes(self):
        terrain = np.zeros((2, 3), dtype=np.uint8)
        cache = DistanceCache(capacity=2)
        with patch("generals_env.pathfinding.single_source_distances", wraps=single_source_distances) as search:
            original = cache.distances(terrain, 0)
            cache.distances(terrain, 1)
            self.assertIs(cache.distances(terrain.copy(), 0), original)
            self.assertEqual(search.call_count, 2)
            cache.distances(terrain, 2)  # Evicts source 1, source 0 was touched.
            self.assertEqual(len(cache), 2)
            cache.distances(terrain, 1)
            self.assertEqual(search.call_count, 4)
            terrain[0, 1] = Terrain.MOUNTAIN
            changed = cache.distances(terrain, 0)
            self.assertEqual(changed[2], 4)
            self.assertEqual(original[2], 2)
            self.assertLessEqual(len(cache), 2)
            cache.clear()
            self.assertEqual(len(cache), 0)

    def test_cache_keys_include_shape_and_unknown_policy(self):
        cache = DistanceCache()
        first = cache.distances(np.zeros((2, 3), dtype=np.uint8), 0)
        second = cache.distances(np.zeros((3, 2), dtype=np.uint8), 0)
        self.assertEqual(first[2], 2)
        self.assertEqual(second[2], 1)
        terrain = np.array([[0, UNKNOWN, 0]], dtype=np.uint8)
        self.assertEqual(cache.distances(terrain, 0, True)[2], 2)
        self.assertEqual(cache.distances(terrain, 0, False)[2], -1)
        self.assertEqual(len(cache), 4)

    def test_invalid_sources_never_alias_a_cache_hit(self):
        terrain = np.zeros((2, 3), dtype=np.uint8)
        cache = DistanceCache()
        cache.distances(terrain, 0)
        cache.distances(terrain, 1)
        for source in (0.5, True):
            with self.subTest(source=source), self.assertRaises(TypeError):
                cache.distances(terrain, source)
        for source in (-1, 6):
            with self.subTest(source=source), self.assertRaises(ValueError):
                single_source_distances(terrain, source)
        with self.assertRaises(TypeError):
            shortest_path(terrain, 0, True)
        with self.assertRaises(ValueError):
            shortest_path(terrain, 0, 6)
        with self.assertRaises(ValueError):
            DistanceCache(0)

    def test_explicit_all_pairs_and_memory_budget(self):
        terrain = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        required = terrain.size ** 2 * 4
        with self.assertRaises(ValueError):
            all_pairs_distances(terrain, max_bytes=required - 1)
        distances = all_pairs_distances(terrain, max_bytes=required)
        self.assertEqual(distances.shape, (6, 6))
        self.assertEqual(distances.dtype, np.int32)
        np.testing.assert_array_equal(distances, distances.T)
        self.assertTrue(np.all(distances[1] == -1))
        self.assertEqual(distances[0, 2], 4)
        self.assertEqual(distances[0, 0], 0)

    def test_preprocessing_is_bound_to_immutable_layout(self):
        layout = MapLayout(np.zeros((2, 3), dtype=np.uint8))
        prepared = prepare_map(layout)
        self.assertEqual(prepared.topology_id, layout.topology_id)
        np.testing.assert_array_equal(prepared.neighbors, neighbors(layout))
        changed = MapLayout(np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8))
        self.assertNotEqual(prepare_map(changed).topology_id, prepared.topology_id)


if __name__ == "__main__":
    unittest.main()
