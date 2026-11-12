"""Seeded standard 1v1/FFA random maps for the pinned official rules."""
from functools import lru_cache
import math
import numpy as np
from generals.protocol import MapConfig
from generals.core.state import GameState, MapLayout, EMPTY, NEUTRAL
from generals.core.cities import place_cities


def adjacent(cell, width, height):
    y, x = divmod(int(cell), width)
    if y: yield cell - width
    if y + 1 < height: yield cell + width
    if x: yield cell - 1
    if x + 1 < width: yield cell + 1


@lru_cache(maxsize=64)
def _neighbors(width, height):
    return tuple(tuple(adjacent(c, width, height)) for c in range(width * height))


def random_shape(players, rng):
    base = 16 + players + max(0, players - 6) * .5
    return tuple(math.floor(base + rng.random() ** 2 * (4 + players / 2) + .5) for _ in range(2))


def _distance(a, b, width):
    return abs(a // width - b // width) + abs(a % width - b % width)


def _spawns(mountains, p, width, neighbors, rng):
    if p == 2:
        from .mapgen_native import spawns
        return spawns(mountains, width, rng)
    candidates = np.flatnonzero(~mountains)
    spacing = max(9, 9 * (1.8 - .1 * p))
    positions = [-1] * p
    seats = list(range(p))
    rng.shuffle(seats)
    for seat in seats:
        for _ in range(1000):
            cell = int(candidates[int(rng.integers(len(candidates)))])
            if all(other < 0 or _distance(cell, other, width) > spacing for other in positions):
                positions[seat] = cell
                break
        else:
            return None
    return positions


def _valid(mountains, generals, cities, neighbors, width):
    from .mapgen_native import valid
    return valid(mountains, generals, cities, width)


def generate_map(config: MapConfig, seed=42) -> GameState:
    rng = np.random.default_rng(seed)
    p = config.players
    for _ in range(1000):
        random_w, random_h = random_shape(p, rng)
        w, h = config.width or random_w, config.height or random_h
        if w + h - 2 <= max(9, 9 * (1.8 - .1 * p)):
            raise ValueError('map is too small for official general spacing; use an explicit layout for custom scenarios')
        n, neighbors = w * h, _neighbors(w, h)
        if p > 2:
            neighbors = tuple(tuple(rng.permutation(row)) for row in neighbors)
        desired = (5 + p * (2 + rng.random())) * 2 * config.city_density
        density = config.mountain_density
        draws = min(n * (.2 + .08 * rng.random()) * (2 * density if density < .5 else density + .5), .5 * n)
        mountains = np.zeros(n, dtype=bool)
        mountains[rng.integers(n, size=math.ceil(draws + desired))] = True
        generals = _spawns(mountains, p, w, neighbors, rng)
        if generals is None:
            continue
        for _ in range(8):
            groups = place_cities(mountains, generals, neighbors, rng, desired, w, h) if desired else []
            if groups is None:
                continue
            cities = {c: army for cells, army in groups for c in cells}
            terrain = mountains.copy()
            terrain[list(cities)] = False
            if not _valid(terrain, generals, cities, neighbors, w):
                continue
            shape = (h, w)
            structure = np.zeros(n, dtype=np.uint8)
            owner = np.full(n, EMPTY, dtype=np.int16)
            army = np.zeros(n, dtype=np.int64)
            for c, count in cities.items():
                structure[c], owner[c], army[c] = 1, NEUTRAL, count
            for pid, c in enumerate(generals):
                structure[c], owner[c], army[c] = 2, pid, 1
            state = GameState(MapLayout(terrain.astype(np.uint8).reshape(shape)), structure.reshape(shape),
                              owner.reshape(shape), army.reshape(shape), np.ones(p, dtype=bool),
                              np.asarray(generals, dtype=np.int32), afk_turn=71 if p == 2 else 121)
            state.validate()
            return state
    raise ValueError('cannot generate an official map with these dimensions and densities')
