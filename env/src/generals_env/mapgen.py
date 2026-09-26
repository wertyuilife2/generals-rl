"""Seeded CPU map generation; no GUI or controller dependency."""
from collections import deque
import numpy as np
from .config import MapConfig
from .state import GameState, MapLayout, Structure, Terrain, EMPTY, NEUTRAL


def adjacent(cell: int, width: int, height: int):
    y, x = divmod(int(cell), width)
    if y: yield cell - width
    if x + 1 < width: yield cell + 1
    if y + 1 < height: yield cell + width
    if x: yield cell - 1


def _connect(terrain, source, target):
    """0–1 BFS: carve the fewest mountains along a connecting route."""
    h, w = terrain.shape
    flat = terrain.ravel()
    dist = [flat.size + 1] * flat.size
    parent = [-1] * flat.size
    dist[source] = 0
    q = deque([source])
    while q:
        cell = q.popleft()
        for nxt in adjacent(cell, w, h):
            cost = int(flat[nxt] == Terrain.MOUNTAIN)
            if dist[cell] + cost < dist[nxt]:
                dist[nxt] = dist[cell] + cost
                parent[nxt] = cell
                (q.append if cost else q.appendleft)(nxt)
    cell = target
    while cell != source:
        flat[cell] = Terrain.PLAIN
        cell = parent[cell]
        if cell < 0:
            raise RuntimeError("cannot connect map")


def generate_map(config: MapConfig, seed: int = 42) -> GameState:
    """Create tick-zero state. Map RNG is independent of all player RNGs."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    h, w, p = config.height, config.width, config.players
    n = h * w
    mountains = int(n * config.mountain_density)
    cities = int(n * config.city_density)
    if n - mountains - cities < p:
        raise ValueError("not enough plain cells for generals")
    order = rng.permutation(n)
    terrain = np.zeros((h, w), dtype=np.uint8)
    structure = np.zeros((h, w), dtype=np.uint8)
    owner = np.full((h, w), EMPTY, dtype=np.int16)
    army = np.zeros((h, w), dtype=np.int64)
    terrain.ravel()[order[:mountains]] = Terrain.MOUNTAIN
    city_pos = order[mountains:mountains + cities]
    structure.ravel()[city_pos] = Structure.CITY
    owner.ravel()[city_pos] = NEUTRAL
    army.ravel()[city_pos] = 40
    candidates = order[mountains + cities:]
    positions = []
    for distance in range(min(w, h) // 3, -1, -1):
        for candidate in candidates:
            cell = int(candidate)
            if cell in positions:
                continue
            y, x = divmod(cell, w)
            if all((y - other // w) ** 2 + (x - other % w) ** 2 >= distance ** 2
                   for other in positions):
                positions.append(cell)
                if len(positions) == p:
                    break
        if len(positions) == p:
            break
    if len(positions) != p:
        raise ValueError("cannot place all generals")
    for target in positions[1:]:
        _connect(terrain, positions[0], target)
    for pid, cell in enumerate(positions):
        structure.ravel()[cell] = Structure.GENERAL
        owner.ravel()[cell] = pid
        army.ravel()[cell] = 1
    state = GameState(MapLayout(terrain), structure, owner, army,
                      np.ones(p, dtype=bool), np.asarray(positions, dtype=np.int32))
    state.validate()
    return state
