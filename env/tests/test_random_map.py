"""Standard random maps: geometry, reproducibility, constraints and engine use."""
from collections import deque
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock

import numpy as np

from generals import Action, MapConfig, MapGenerationConfig, make_engine
from generals.core.city_placement import Planner, paired_cities
from generals.core.mapgen import adjacent, generate_map
from generals.core.random_map import (Grid, MapGenerationError, _build,
                                     _spawns, sample_map_config)
from generals.core.recording import state_digest, state_from_dict, state_to_dict
from generals.core.state import Structure, Terrain


class RandomMapTests(unittest.TestCase):
    def test_official_size_ranges_and_nonuniform_distribution(self):
        rules = MapGenerationConfig()
        expected = [(18, 23), (19, 25), (20, 26), (21, 28), (22, 29), (24, 31), (25, 33)]
        for players, bounds in zip(range(2, 9), expected):
            self.assertEqual(rules.side_range(players), bounds)
        rng = np.random.default_rng(91)
        sizes = np.array([rules.sample_size(2, rng) for _ in range(10000)])
        self.assertLess(float(sizes.mean()), 20)
        self.assertLess(abs(float(np.corrcoef(sizes.T)[0, 1])), .05)
        self.assertEqual(replace(rules, width=0).side_range(2), (16, 21))
        self.assertEqual(replace(rules, height=1).side_range(2, 'height'), (22, 27))

    def test_default_maps_satisfy_independently_checked_constraints(self):
        army_values = set()
        for players in range(2, 9):
            for seed in range(100):
                config = sample_map_config(players, seed)
                state = generate_map(config, seed)
                with self.subTest(players=players, seed=seed):
                    state.validate()
                    self.assertEqual(state.layout.terrain.shape, (config.height, config.width))
                    generals = list(map(int, state.general_pos))
                    cities = set(map(int, np.flatnonzero(state.structure.ravel() == Structure.CITY)))
                    army_values.update(map(int, state.army.ravel()[list(cities)]))
                    width = config.width
                    distance = lambda a, b: abs(a // width - b // width) + abs(a % width - b % width)
                    for index, general in enumerate(generals):
                        for other in generals[index + 1:]:
                            self.assertGreater(distance(general, other), config.generation.spawn_spacing(players))
                        if len(cities) >= players:
                            self.assertTrue(any(distance(general, c) <= 6 for c in cities))
                    # Independently flood only empty land; cities and other
                    # generals can be reached but cannot be walked through.
                    for source in generals[:2]:
                        seen, queue = {source}, deque([source])
                        while queue:
                            for nxt in adjacent(queue.popleft(), width, config.height):
                                if nxt not in seen and nxt not in cities and state.layout.terrain.flat[nxt] != Terrain.MOUNTAIN:
                                    seen.add(nxt)
                                    if nxt not in generals:
                                        queue.append(nxt)
                        self.assertTrue(set(generals) <= seen)
        self.assertEqual(army_values, set(range(40, 51)))

    def test_pair_fairness_tracks_paths_after_opening_cities(self):
        rng = np.random.default_rng(37)
        mountains = (rng.random(20 * 20) < .25).tolist()
        generals = [21, 378]
        for cell in generals:
            mountains[cell] = False
        planner = Planner(Grid(20, 20, mountains), generals, .5, rng)
        groups = planner.place(10)
        self.assertIsNotNone(groups)
        # Full BFS recomputation is independent of incremental relaxation.
        for general, actual in zip(generals, planner.distances):
            self.assertEqual(actual, planner.grid.distances(general, planner.cities))
        total = 0
        for cells, neutral in groups:
            if neutral:
                continue
            self.assertEqual(len(cells), 2)
            a, b = (planner.info(cell) for cell in cells)
            self.assertLess(a[1] * b[1], 0)
            if a[1] > b[1]:
                a, b = b, a
            total += 1.5 * (a[1] + b[1]) + a[2] - b[2]
        self.assertLessEqual(abs(total), 1 + 19 * .5 ** 3)

    def test_spawn_fairness_rejects_unequal_expansion_space(self):
        grid = Grid(20, 20, [False] * 400)
        options = MapGenerationConfig(spawn_attempts=1)
        # Both positions satisfy spacing; the corner has much less nearby
        # land than the center. Disabling fairness permits the same draw.
        rng = Mock(integers=Mock(side_effect=[0, 210]))
        self.assertIsNone(_spawns(grid, 2, {}, options, rng))
        rng = Mock(integers=Mock(side_effect=[0, 210]))
        self.assertEqual(_spawns(grid, 2, {}, replace(options, spawn_fairness=0), rng), [0, 210])

    def test_city_placement_matches_official_reference_cases(self):
        # Outputs captured from planTwoTeamFairCityPlacement in the official
        # v31.4.3 bundle, supplying this deterministic random stream. The
        # production implementation does not depend on Node or that bundle.
        class Random:
            def __init__(self, seed):
                self.seed = seed

            def integers(self, count):
                self.seed = (1664525 * self.seed + 1013904223) % 2**32
                return int(count * self.seed / 2**32)

        expected = [
            [([101, 315], False), ([157], True), ([240], True), ([143], True),
             ([334, 62], False), ([221], True), ([310], True), ([272], True)],
            [([107, 291], False), ([224, 366], False), ([28, 256], False),
             ([384], True), ([229], True), ([9, 278], False)],
            [([86, 258], False), ([215, 89], False), ([49, 196], False),
             ([183, 218], False), ([6, 393], False)],
        ]
        for seed, fairness in enumerate((.1, .5, 1)):
            mountains = (np.random.default_rng(seed).random(400) < .25).tolist()
            mountains[21] = mountains[378] = False
            actual = paired_cities(Grid(20, 20, mountains), [21, 378], 10, fairness, Random(seed))
            self.assertEqual(actual, expected[seed])

    def test_json_roundtrip_cache_isolation_and_worker_reproduction(self):
        seed = 1729
        config = sample_map_config(2, seed)
        expected = state_digest(generate_map(config, seed))
        config = MapConfig(**json.loads(json.dumps(asdict(config))))
        state = generate_map(config, seed)
        state.army[:] = 999
        self.assertEqual(state_digest(generate_map(config, seed)), expected)
        _build.cache_clear()  # A spawned worker starts without the parent cache.
        self.assertEqual(state_digest(generate_map(config, seed)), expected)
        self.assertEqual(sample_map_config(2, seed), config)

    def test_custom_parameters_and_impossible_maps(self):
        rules = MapGenerationConfig(city_army_min=71, city_army_max=71)
        config = sample_map_config(4, 71, rules, width=25, height=20)
        state = generate_map(config, 71)
        self.assertEqual(state.layout.terrain.shape, (20, 25))
        self.assertTrue(np.all(state.army[state.structure == Structure.CITY] == 71))
        empty = MapGenerationConfig(city_density=0, mountain_density=0, min_general_spacing=0)
        state = generate_map(sample_map_config(2, 0, empty, width=3, height=3), 0)
        self.assertFalse(np.any(state.structure == Structure.CITY))
        self.assertFalse(np.any(state.layout.terrain))
        with self.assertRaisesRegex(MapGenerationError, 'spacing'):
            sample_map_config(2, 0, width=3, height=3)
        impossible = replace(rules, city_attempts=1, max_attempts=2, close_city_distance=0)
        with self.assertRaisesRegex(MapGenerationError, '2 attempts'):
            sample_map_config(2, 0, impossible)

    def test_generation_parameters_are_validated(self):
        for values in ({'width': 1.1}, {'city_density': float('nan')},
                       {'spawn_fairness': -1}, {'max_attempts': 0},
                       {'city_army_min': 60, 'city_army_max': 50}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MapGenerationConfig(**values)
        for values in ({'width': True}, {'max_attempts': 1.5}, {'require_close_city': 1}):
            with self.subTest(values=values), self.assertRaises(TypeError):
                MapGenerationConfig(**values)

    def test_city_armies_above_forty_work_in_native_engine_and_recordings(self):
        config = sample_map_config(2, 42, MapGenerationConfig(city_army_min=50, city_army_max=50))
        state = generate_map(config, 42)
        engine = make_engine(state)
        try:
            for _ in range(51):
                engine.step([Action.wait(), Action.wait()])
            self.assertTrue(np.all(state.army[state.structure == Structure.CITY] == 50))
            self.assertEqual(state_digest(state_from_dict(state_to_dict(state))), state_digest(state))
        finally:
            engine.close()

    def test_fixed_training_maps_retain_historical_seed_results(self):
        fixtures = [
            ((9, 7, 4), '2b5fcbb78dbfd6346a3d9b3b9dafd3d3b6d65d4ad4ed4d42d290bb32d1aa15c3'),
            ((25, 25, 4), 'b8fc54b92a58c14efc71292b4050744bac8d7394dbd3518e83f0f6242270e9d5'),
            ((15, 15, 2), '29da05d8ce815e5c5a1ae57545a6cb01d96f5c023edf620a632479219e58bf6b'),
            ((15, 15, 8), '52586a64b78a5bd2e7feab25d989dc8df2a25a2ccf1be969a30080e96a6c0592'),
        ]
        for dimensions, expected in fixtures:
            self.assertEqual(state_digest(generate_map(MapConfig(*dimensions), 42)), expected)


if __name__ == '__main__':
    unittest.main()
