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


def _ensure_connected(terrain: np.ndarray, source: int) -> None:
    """Connect all passable cells in-place with O(H*W) time and scratch space.

    A single multi-source 0–1 BFS minimizes mountain cost from the original
    source component. Shared parent chains are carved at most once; this does
    not promise the globally minimum number of removed mountains.
    """
    h, w = terrain.shape
    # Binary PLAIN=0 / MOUNTAIN=1 costs; Python scalars keep BFS loops cheap.
    cells = terrain.ravel().tolist()
    size = len(cells)
    connected = [False] * size
    connected[source] = True
    component = [source]
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for nxt in adjacent(cell, w, h):
            if not cells[nxt] and not connected[nxt]:
                connected[nxt] = True
                component.append(nxt)
                queue.append(nxt)
    if len(component) == cells.count(0):
        return

    distance = [size + 1] * size
    parent = [-1] * size
    settled = [False] * size
    for cell in component:
        distance[cell] = 0
    queue = deque(component)
    while queue:
        cell = queue.popleft()
        if settled[cell]:
            continue
        settled[cell] = True
        for nxt in adjacent(cell, w, h):
            cost = cells[nxt]
            candidate = distance[cell] + cost
            if candidate < distance[nxt]:
                distance[nxt] = candidate
                parent[nxt] = cell
                (queue.append if cost else queue.appendleft)(nxt)

    carved = []
    for target, cost in enumerate(cells):
        if cost:
            continue  # Only original passable cells need to be connected.
        cell = target
        while not connected[cell]:
            connected[cell] = True
            if cells[cell]:
                carved.append(cell)
            cell = parent[cell]
            if cell < 0:
                raise RuntimeError("cannot connect map")
    terrain.ravel()[carved] = Terrain.PLAIN


def generate_map(config: MapConfig, seed: int = 42) -> GameState:
    """Create a fully connected tick-zero map with an independent seeded RNG."""
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
    _ensure_connected(terrain, positions[0])
    for pid, cell in enumerate(positions):
        structure.ravel()[cell] = Structure.GENERAL
        owner.ravel()[cell] = pid
        army.ravel()[cell] = 1
    state = GameState(MapLayout(terrain), structure, owner, army,
                      np.ones(p, dtype=bool), np.asarray(positions, dtype=np.int32))
    state.validate()
    return state
