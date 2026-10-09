"""Optimized searches must retain the original planner's exact tie order."""
import numpy as np
import pytest

from generals.core.pathfinding import shortest_path
from generals.core.state import Terrain, UNKNOWN
from generals_baselines.planning import nearest_cells, shortest_known_path


@pytest.mark.parametrize('shape', [(1, 1), (1, 15), (15, 1), (15, 20), (35, 35)])
def test_partial_nearest_selection_matches_stable_sort(shape):
    rng = np.random.default_rng(42)
    size = int(np.prod(shape))
    ys, xs = np.divmod(np.arange(size), shape[1])
    for _ in range(30):
        source = int(rng.integers(size))
        sy, sx = divmod(source, shape[1])
        distance = abs(ys - sy) + abs(xs - sx)
        cells = np.flatnonzero(rng.random(size) > rng.random())
        expected = cells[np.argsort(distance[cells], kind='stable')[:16]]
        np.testing.assert_array_equal(nearest_cells(cells, distance, size), expected)


@pytest.mark.parametrize('shape', [(1, 1), (1, 15), (15, 1), (15, 20), (35, 35)])
def test_cached_geometry_bfs_matches_reference_including_unknown_and_unreachable(shape):
    rng = np.random.default_rng(15)
    for _ in range(100):
        terrain = rng.choice(np.array([0, Terrain.MOUNTAIN, UNKNOWN], dtype=np.uint8),
                             size=shape, p=[.4, .3, .3])
        source, target = map(int, rng.integers(terrain.size, size=2))
        assert shortest_known_path(terrain == Terrain.MOUNTAIN, source, target) == shortest_path(
            terrain, source, target, unknown_passable=True)
