"""Seeded standard 1v1/FFA random maps for the pinned official rules."""
from collections import deque
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


def _spawn_score(source, mountains, neighbors):
    counts = [0] * 15
    seen, queue = {source}, deque([(source, 0)])
    while queue:
        cell, depth = queue.popleft()
        counts[depth] += 1
        if depth < 14:
            for other in neighbors[cell]:
                if other not in seen and not mountains[other]:
                    seen.add(other)
                    queue.append((other, depth + 1))
    counts[1] = min(3, counts[1])
    weight, score = int(1.18 ** 15), 0.
    for depth in range(1, 15):
        score += weight * counts[depth]
        weight /= 1.18 if depth < 7 else 1.1
        if depth < 5:
            weight /= 1.6
    return score


def _spawns(mountains, p, width, neighbors, rng):
    candidates = np.flatnonzero(~mountains)
    spacing = max(9, 9 * (1.8 - .1 * p))
    positions = [-1] * p
    if p == 2:
        scores = {}
        fairness_retries = 0
        for _ in range(1000):
            positions = [int(candidates[int(rng.integers(len(candidates)))]) for _ in range(2)]
            if _distance(*positions, width) <= spacing:
                continue
            required = 1 - (1 - (.5 - .005 * fairness_retries)) ** 4 - .02
            if required <= 0:
                return positions
            for c in positions:
                if c not in scores:
                    scores[c] = _spawn_score(c, mountains, neighbors)
            values = [scores[c] for c in positions]
            if min(values) / max(1, max(values)) >= required:
                return positions
            fairness_retries += 1
        return None
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
    if len(cities) >= len(generals) and any(not any(_distance(g, c, width) <= 6 for c in cities) for g in generals):
        return False
    # Other generals are obstacles except when they are the endpoint. Preserve
    # official accessibility without carving isolated plain pockets.
    blocked = set(cities) | set(generals)
    for source in generals[1:]:
        for goal in generals[:2]:
            seen, queue = {source}, deque([source])
            while queue:
                cell = queue.popleft()
                if cell == goal:
                    break
                for other in neighbors[cell]:
                    if other not in seen and not mountains[other] and (other == goal or other not in blocked):
                        seen.add(other)
                        queue.append(other)
            else:
                return False
    return True


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
