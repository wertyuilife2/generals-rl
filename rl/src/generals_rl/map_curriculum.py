"""RL-only initial-map curriculum; the simulator's official rules stay fixed.

Each seed has an easy layout and an unmodified official target. Intermediate
layouts interpolate their distance and obstacle counts. The last bank always
contains the official generator's exact output, including city fairness.
"""
from collections import deque
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from generals.core.mapgen import generate_map
from generals.core.state import GameState, MapLayout
from generals.protocol import EMPTY, NEUTRAL


@dataclass(frozen=True)
class MapCurriculumConfig:
    """Schedule thresholds count global board ticks, not per-seat samples.

    Both seats produce learner transitions in dual self-play, but advance the
    physical game only once. Keep the historical *_transitions config keys.
    """
    warmup_transitions: int
    end_transitions: int
    levels: int = 5
    easy_mountain_fraction: float = .1
    easy_cities_per_player: int = 2
    easy_spawn_reference_size: int | None = None

    def __post_init__(self):
        for name in ('warmup_transitions', 'end_transitions', 'levels', 'easy_cities_per_player'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'map_curriculum.{name} must be an integer')
        if not 0 <= self.warmup_transitions < self.end_transitions:
            raise ValueError('map curriculum requires 0 <= warmup_transitions < end_transitions')
        if self.levels < 2 or self.easy_cities_per_player < 0:
            raise ValueError('map curriculum requires >=2 levels and nonnegative city count')
        if not 0 <= self.easy_mountain_fraction < 1:
            raise ValueError('easy_mountain_fraction must be in [0, 1)')
        if self.easy_spawn_reference_size is not None and (
                isinstance(self.easy_spawn_reference_size, bool)
                or not isinstance(self.easy_spawn_reference_size, Integral)
                or self.easy_spawn_reference_size < 3):
            raise ValueError('easy_spawn_reference_size must be an integer >=3 or None')

    def position(self, transitions):
        """Absolute global board ticks make continuation reproducible."""
        if transitions < 0:
            raise ValueError('map curriculum transitions must be nonnegative')
        return float(np.clip((transitions - self.warmup_transitions)
                             / (self.end_transitions - self.warmup_transitions), 0, 1))


def _adjacent(cell, width, height):
    # Preserve the successful easy generator's tie-breaking and random stream.
    y, x = divmod(int(cell), width)
    if y: yield cell - width
    if x + 1 < width: yield cell + 1
    if y + 1 < height: yield cell + width
    if x: yield cell - 1


def _connect(terrain, source):
    """Legacy linear-time connectivity repair, used only for training layouts."""
    h, w = terrain.shape
    cells = terrain.ravel().tolist()
    size = len(cells)
    connected = [False] * size
    connected[source] = True
    component, queue = [source], deque([source])
    while queue:
        for other in _adjacent(queue.popleft(), w, h):
            if not cells[other] and not connected[other]:
                connected[other] = True
                component.append(other)
                queue.append(other)
    if len(component) == cells.count(0):
        return
    distance, parent, settled = [size + 1] * size, [-1] * size, [False] * size
    for cell in component:
        distance[cell] = 0
    queue = deque(component)
    while queue:
        cell = queue.popleft()
        if settled[cell]:
            continue
        settled[cell] = True
        for other in _adjacent(cell, w, h):
            cost = cells[other]
            candidate = distance[cell] + cost
            if candidate < distance[other]:
                distance[other], parent[other] = candidate, cell
                (queue.append if cost else queue.appendleft)(other)
    carved = []
    for cell, cost in enumerate(cells):
        if cost:
            continue
        while not connected[cell]:
            connected[cell] = True
            if cells[cell]:
                carved.append(cell)
            cell = parent[cell]
    terrain.ravel()[carved] = 0


def _state(terrain, cities, generals, rng):
    h, w = terrain.shape
    structure = np.zeros((h, w), dtype=np.uint8)
    owner = np.full((h, w), EMPTY, dtype=np.int16)
    army = np.zeros((h, w), dtype=np.int64)
    structure.ravel()[cities], owner.ravel()[cities] = 1, NEUTRAL
    army.ravel()[cities] = np.floor(40 + 10 * rng.random(len(cities)) + .5).astype(np.int64)
    for pid, cell in enumerate(generals):
        structure.ravel()[cell], owner.ravel()[cell], army.ravel()[cell] = 2, pid, 1
    return GameState(MapLayout(terrain), structure, owner, army, np.ones(len(generals), dtype=bool),
                     np.asarray(generals, dtype=np.int32), afk_turn=71)


def easy_map(config, seed, settings):
    """Exact successful 9x9 distribution; relative terrain and per-player cities
    also support larger canvases without scaling cities by map area.
    """
    h, w, p = config.height, config.width, config.players
    if p != 2 or h is None or w is None:
        raise ValueError('easy maps require a concrete shape and two players')
    n = h * w
    mountains = int(n * settings.easy_mountain_fraction)
    city_count = p * settings.easy_cities_per_player
    if n - mountains - city_count < p:
        raise ValueError('easy map has insufficient plain cells')
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    order = rng.permutation(n)
    generals = []
    reference = settings.easy_spawn_reference_size
    if reference is not None and (w > reference or h > reference):
        if min(w, h) < reference:
            raise ValueError('easy spawn reference must fit within both map sides')
        # Translate a reference pair, preserving its distance distribution on
        # a larger board. Reserve both cells before placing terrain/cities.
        sample = easy_map(replace(config, width=reference, height=reference), seed, settings)
        ys, xs = sample.general_pos // reference, sample.general_pos % reference
        xs, ys = xs - xs.min(), ys - ys.min()
        x = int(rng.integers(w - int(xs.max())))
        y = int(rng.integers(h - int(ys.max())))
        generals = ((ys + y) * w + xs + x).tolist()
        order = order[~np.isin(order, generals)]
    terrain = np.zeros((h, w), dtype=np.uint8)
    terrain.ravel()[order[:mountains]] = 1
    cities = order[mountains:mountains + city_count]
    candidates = order[mountains + city_count:]
    if not generals:
        for spacing in range(min(w, h) // 3, -1, -1):
            for candidate in candidates:
                cell = int(candidate)
                if cell not in generals and all((cell // w - other // w) ** 2
                        + (cell % w - other % w) ** 2 >= spacing ** 2 for other in generals):
                    generals.append(cell)
                    if len(generals) == p:
                        break
            if len(generals) == p:
                break
    _connect(terrain, generals[0])
    return _state(terrain, cities, generals, rng)


def map_metrics(state):
    """Generation-time diagnostics; no path searches enter env.step()."""
    h, w = state.owner.shape
    a, b = map(int, state.general_pos)
    blocked = state.layout.terrain.astype(bool) | (state.structure == 1)
    distance = np.full(h * w, -1, dtype=np.int32)
    distance[a] = 0
    queue = deque([a])
    while queue and distance[b] < 0:
        cell = queue.popleft()
        for other in _adjacent(cell, w, h):
            if not blocked.ravel()[other] and distance[other] < 0:
                distance[other] = distance[cell] + 1
                queue.append(other)
    return dict(manhattan=abs(a // w - b // w) + abs(a % w - b % w),
                mountain_fraction=float(state.layout.terrain.mean()),
                cities=int((state.structure == 1).sum()), open_path=int(distance[b]))


def curriculum_maps(config, seed, settings):
    """Yield increasing levels, calibrated against this shape's official map.

    We do not extrapolate 9x9's almost-diagonal spacing to larger maps. Sampling
    an official target per seed also preserves its count/distance variability.
    """
    target = generate_map(config, seed)
    h, w = target.owner.shape
    easy = easy_map(replace(config, width=w, height=h), seed, settings)
    yield easy
    endpoints = [map_metrics(state) for state in (easy, target)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x435552]))
    for alpha in np.linspace(0, 1, settings.levels)[1:-1]:
        desired = {key: int(np.floor((1 - alpha) * endpoints[0][key]
                                     + alpha * endpoints[1][key] + .5))
                   for key in ('manhattan', 'cities')}
        mountains = int(np.floor(((1 - alpha) * endpoints[0]['mountain_fraction']
                                  + alpha * endpoints[1]['mountain_fraction']) * h * w + .5))
        for _ in range(1000):
            order = rng.permutation(h * w)
            terrain = np.zeros((h, w), dtype=np.uint8)
            terrain.ravel()[order[:mountains]] = 1
            cities = order[mountains:mountains + desired['cities']]
            candidates = order[mountains + desired['cities']:]
            ys, xs = candidates // w, candidates % w
            pairs = np.argwhere((np.abs(ys[:, None] - ys) + np.abs(xs[:, None] - xs))
                                == desired['manhattan'])
            if not len(pairs):
                continue
            generals = candidates[pairs[int(rng.integers(len(pairs)))]]
            _connect(terrain, int(generals[0]))
            state = _state(terrain, cities, generals, rng)
            if map_metrics(state)['open_path'] >= 0:
                yield state
                break
        else:
            raise ValueError('cannot construct an intermediate curriculum map')
    yield target
