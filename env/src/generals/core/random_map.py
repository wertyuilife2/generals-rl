"""Seeded standard-map generation shared by Arena and local environments.

Rules follow the ordinary 2–8 player generator in the public generals.io
v31.4.3 client. RNG streams are local/reproducible, not JavaScript Math.random.
Failed candidates are rejected, never repaired by carving mountains. An
accepted attempt is recorded so a worker can reconstruct it directly.
"""
from collections import deque
from functools import lru_cache
from math import ceil, floor, log, sqrt

import numpy as np

from generals.protocol.config import MapConfig
from generals.protocol.mapgen import MapGenerationConfig
from .state import GameState, MapLayout, Structure, Terrain, EMPTY, NEUTRAL


class MapGenerationError(ValueError):
    """The requested map constraints cannot be satisfied within the budget."""


class Grid:
    def __init__(self, width, height, mountains):
        self.width, self.height = width, height
        self.size = width * height
        self.mountains = mountains
        self.neighbors = _neighbors(width, height)

    def distance(self, left, right):
        return abs(left // self.width - right // self.width) + abs(left % self.width - right % self.width)

    def distances(self, source, cities=()):
        """BFS through land/cities; a mountain is a reachable terminal tile."""
        distance = [self.size + 1] * self.size
        distance[source] = 0
        queue = deque([source])
        while queue:
            cell = queue.popleft()
            depth = distance[cell] + 1
            for neighbor in self.neighbors[cell]:
                if depth < distance[neighbor]:
                    distance[neighbor] = depth
                    if not self.mountains[neighbor] or neighbor in cities:
                        queue.append(neighbor)
        return distance

    def expand_city(self, distances, cities, cell):
        """Incrementally relax distance maps after a mountain becomes a city."""
        best = distances[cell]
        for neighbor in self.neighbors[cell]:
            if not self.mountains[neighbor] or neighbor in cities:
                best = min(best, distances[neighbor] + 1)
        if best > self.size:
            return
        distances[cell] = best
        queue = deque([cell])
        while queue:
            source = queue.popleft()
            depth = distances[source] + 1
            for neighbor in self.neighbors[source]:
                if depth < distances[neighbor]:
                    distances[neighbor] = depth
                    if not self.mountains[neighbor] or neighbor in cities:
                        queue.append(neighbor)

    def connected_generals(self, generals, cities):
        """Reach every general without crossing a city or another general."""
        targets = set(generals)
        for source in generals[:2]:
            seen = {source}
            queue = deque([source])
            while queue:
                for neighbor in self.neighbors[queue.popleft()]:
                    if neighbor not in seen and not self.mountains[neighbor] and neighbor not in cities:
                        seen.add(neighbor)
                        if neighbor not in targets:
                            queue.append(neighbor)
            if not targets <= seen:
                return False
        return True


@lru_cache(maxsize=128)
def _neighbors(width, height):
    return tuple(tuple(n for n in (cell - width if cell >= width else -1,
                                  cell + width if cell < width * (height - 1) else -1,
                                  cell - 1 if cell % width else -1,
                                  cell + 1 if cell % width < width - 1 else -1) if n >= 0)
                 for cell in range(width * height))


def _army(options, rng):
    return floor(options.city_army_min + float(rng.random()) *
                 (options.city_army_max - options.city_army_min) + .5)


def _spawn_score(grid, source, cities):
    counts = [0] * 15
    seen = {source}
    queue = deque([(source, 0)])
    while queue:
        cell, depth = queue.popleft()
        counts[depth] += 1
        if depth < 14:
            for neighbor in grid.neighbors[cell]:
                if neighbor not in seen and not grid.mountains[neighbor] and neighbor not in cities:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
    counts[1] = min(3, counts[1])
    weight, score = floor(1.18 ** 15), 0.
    for depth in range(1, 15):
        score += weight * counts[depth]
        weight /= 1.18 if depth < 7 else 1.1
        if depth < 5:
            weight /= 1.6
    return score


def _spawns(grid, players, cities, options, rng):
    valid = [cell for cell in range(grid.size) if not grid.mountains[cell] and cell not in cities
             and any(not grid.mountains[n] for n in grid.neighbors[cell])]
    if len(valid) < players:
        return None
    spacing = options.spawn_spacing(players)
    if players == 2:
        failures, scores = 0, {}
        for _ in range(options.spawn_attempts):
            left, right = (valid[int(rng.integers(len(valid)))] for _ in range(2))
            if grid.distance(left, right) <= spacing:
                continue
            fairness = options.spawn_fairness - .005 * failures
            threshold = 1 - (1 - fairness) ** 4 - .02
            if threshold > 0:
                for cell in (left, right):
                    if cell not in scores:
                        scores[cell] = _spawn_score(grid, cell, cities)
                if min(scores[left], scores[right]) / max(1, scores[left], scores[right]) < threshold:
                    failures += 1
                    continue
            return [left, right]
        return None
    order = rng.permutation(players)
    positions = []
    for _ in order:
        for _ in range(options.spawn_attempts):
            cell = valid[int(rng.integers(len(valid)))]
            if all(grid.distance(cell, other) > spacing for other in positions):
                positions.append(cell)
                break
        else:
            return None
    result = [0] * players
    for seat, cell in zip(order, positions):
        result[int(seat)] = cell
    return result


def _ffa_cities(grid, generals, desired, options, rng):
    """Multi-source placement with balanced nearby counts and shared strengths."""
    players = len(generals)
    mountain_count = sum(grid.mountains)
    if not mountain_count:
        return None
    target = max(1, floor(desired + .5))
    minimum = target - players // 2
    near = max(6, max(grid.width, grid.height) / (2 + log(players)))
    base = .1 * target / mountain_count
    increase = 10 * players * base * sqrt(options.city_fairness)
    decrease = increase / (players - 1)
    thresholds = [1 - base] * players
    counts, cities = [0] * players, {}
    strengths = [_army(options, rng) for _ in range(target)]
    for iteration in range(1, 51):
        if len(cities) >= minimum and (options.city_fairness < .5 or len(set(counts)) == 1):
            return cities if any(counts) else None
        if iteration > 25 and len(cities) < minimum and players <= 4:
            return None
        if len(cities) >= minimum:
            for seat in range(players):
                if counts[seat] == max(counts):
                    thresholds[seat] = 10
        seen = set(generals)
        queue = deque((g, 0, seat) for seat, g in enumerate(generals))
        while queue:
            cell, depth, seat = queue.popleft()
            if iteration > 1 and depth >= near:
                continue
            if grid.mountains[cell]:
                if cell in cities:
                    # Placed cities become land for subsequent BFS passes.
                    pass
                else:
                    place = (0 < depth < near and thresholds[seat] < 1 and rng.random() >= thresholds[seat])
                    place |= depth >= near and iteration == 1 and rng.random() >= 1 - target / mountain_count
                    if place:
                        if depth < near:
                            while len(strengths) <= counts[seat]:
                                strengths.append(_army(options, rng))
                            cities[cell] = strengths[counts[seat]]
                            counts[seat] += 1
                            thresholds = [value + increase if i == seat else value - decrease
                                          for i, value in enumerate(thresholds)]
                        else:
                            cities[cell] = _army(options, rng)
                        if len(cities) >= minimum:
                            break
                    continue
            for neighbor in grid.neighbors[cell]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1, seat))
    return None


def _best_effort_cities(grid, generals, desired, options, rng):
    candidates = list(map(int, rng.permutation(np.flatnonzero(grid.mountains))))
    target = min(max(1, floor(desired + .5)), len(candidates))
    cities = {}
    if options.require_close_city and target >= len(generals):
        for general in generals:
            cell = next((c for c in candidates if c not in cities and
                         grid.distance(c, general) <= options.close_city_distance), None)
            if cell is not None:
                cities[cell] = _army(options, rng)
    for cell in candidates:
        if len(cities) >= target:
            break
        if cell not in cities:
            cities[cell] = _army(options, rng)
    return cities or None


def _candidate(width, height, players, options, rng):
    size = width * height
    desired = (5 + players * (2 + float(rng.random()))) * 2 * options.city_density
    cities = {}
    legacy_cities = options.city_fairness == 0
    if legacy_cities:
        count = ceil(desired)
        if count + players > size:
            return None
        cities = {int(cell): _army(options, rng) for cell in rng.choice(size, count, replace=False)}
    density = options.mountain_density
    strength = (.2 + .08 * float(rng.random())) * (2 * density if density < .5 else density + .5)
    attempts = ceil(size * min(.5, strength) + (0 if legacy_cities else desired))
    mountains = [False] * size
    available = [cell for cell in range(size) if cell not in cities]
    if attempts and not available:
        return None
    for index in rng.integers(len(available), size=attempts):
        mountains[available[int(index)]] = True
    grid = Grid(width, height, mountains)
    generals = _spawns(grid, players, cities, options, rng)
    if generals is None:
        return None
    if not legacy_cities and desired > 0:
        from .city_placement import paired_cities
        for _ in range(options.city_attempts):
            if players == 2:
                groups = paired_cities(grid, generals, desired, options.city_fairness, rng)
                cities = {} if groups is None else {
                    cell: army for indexes, _ in groups
                    for army in [_army(options, rng)] for cell in indexes}
            else:
                cities = (_ffa_cities(grid, generals, desired, options, rng) or
                          _best_effort_cities(grid, generals, desired, options, rng))
            if cities and _valid(grid, generals, cities, options):
                break
        else:
            return None
    elif not _valid(grid, generals, cities, options):
        return None
    terrain = np.array(mountains, dtype=np.uint8).reshape(height, width)
    structure = np.zeros((height, width), dtype=np.uint8)
    owner = np.full((height, width), EMPTY, dtype=np.int16)
    army = np.zeros((height, width), dtype=np.int64)
    for cell, strength in cities.items():
        terrain.flat[cell] = Terrain.PLAIN
        structure.flat[cell], owner.flat[cell], army.flat[cell] = Structure.CITY, NEUTRAL, strength
    for seat, cell in enumerate(generals):
        structure.flat[cell], owner.flat[cell], army.flat[cell] = Structure.GENERAL, seat, 1
    state = GameState(MapLayout(terrain), structure, owner, army,
                      np.ones(players, dtype=bool), np.asarray(generals, dtype=np.int32))
    state.validate()
    return state


def _valid(grid, generals, cities, options):
    if options.require_close_city and len(cities) >= len(generals):
        if any(not any(grid.distance(g, city) <= options.close_city_distance for city in cities)
               for g in generals):
            return False
    return grid.connected_generals(generals, cities)


def _rng(seed, attempt):
    return np.random.default_rng(np.random.SeedSequence([seed, 714, attempt]))


@lru_cache(maxsize=32)
def _build(width, height, players, options, seed, attempt):
    rng = _rng(seed, attempt)
    options.sample_size(players, rng)  # Reserve the same draws for fixed/auto sizes.
    return _candidate(width, height, players, options, rng)


def sample_map_config(players, seed=42, options=None, *, width=None, height=None):
    """Resolve a valid random map, including rejection/resampling of dimensions."""
    options = MapGenerationConfig() if options is None else options
    if not isinstance(options, MapGenerationConfig):
        raise TypeError('options must be MapGenerationConfig')
    MapConfig(players=players)  # Validate the seat count before any generation.
    if (width is None) != (height is None):
        raise ValueError('provide both width and height, or neither')
    for attempt in range(options.max_attempts):
        sampled = options.sample_size(players, _rng(seed, attempt))
        w, h = sampled if width is None else (width, height)
        config = MapConfig(w, h, players, generation=options, generation_attempt=attempt)
        _check_size(config)
        if _build(w, h, players, options, seed, attempt) is not None:
            return config
    raise MapGenerationError(f'could not generate a valid {players}-player map in {options.max_attempts} attempts')


def _check_size(config):
    if config.width + config.height - 2 <= config.generation.spawn_spacing(config.players):
        raise MapGenerationError('map dimensions cannot satisfy the requested general spacing')


def generate_configured_map(config, seed):
    _check_size(config)
    options = config.generation
    for attempt in range(config.generation_attempt, config.generation_attempt + options.max_attempts):
        state = _build(config.width, config.height, config.players, options, seed, attempt)
        if state is not None:
            return state.copy()  # Cached initial states must never become live state.
    raise MapGenerationError(f'could not generate a valid map in {options.max_attempts} attempts')
