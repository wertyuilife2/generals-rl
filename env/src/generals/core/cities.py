"""Fair city placement for standard random maps.

Cities replace mountain candidates. In 1v1, paired cities share their initial
army, and path-distance deficits are balanced as each new city opens paths.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass(eq=False)
class _Group:
    cells: list
    neutral: bool = False


class _DuelCities:
    def __init__(self, mountains, generals, neighbors, rng, fairness=.5):
        self.mountains, self.generals, self.neighbors, self.rng = mountains, generals, neighbors, rng
        self.remaining = list(map(int, np.flatnonzero(mountains)))
        self.positions = {cell: i for i, cell in enumerate(self.remaining)}
        self.cities, self.groups = set(), []
        self.far = math.floor(5 + 25 * fairness)
        self.tolerance = 1 + 19 * (1 - fairness) ** 3
        self.unreachable = len(mountains) + 1
        self.rebuild()

    def take(self, cell):
        index = self.positions.pop(cell, None)
        if index is None:
            return False
        last = self.remaining.pop()
        if index < len(self.remaining):
            self.remaining[index] = last
            self.positions[last] = index
        return True

    def put_back(self, cell):
        if cell not in self.positions:
            self.positions[cell] = len(self.remaining)
            self.remaining.append(cell)

    def blocked(self, cell):
        return self.mountains[cell] and cell not in self.cities

    def relax(self, distances, cell):
        queue = deque([cell])
        while queue:
            source = queue.popleft()
            distance = distances[source] + 1
            for target in self.neighbors[source]:
                if distance < distances[target]:
                    distances[target] = distance
                    if not self.blocked(target):
                        queue.append(target)

    def rebuild(self):
        self.distances = []
        for general in self.generals:
            distance = np.full(len(self.mountains), self.unreachable, dtype=np.int32)
            distance[general] = 0
            self.relax(distance, general)
            self.distances.append(distance)
        self.score()

    def open(self, cell):
        self.cities.add(cell)
        for distance in self.distances:
            distance[cell] = min(distance[cell], min((distance[x] + 1 for x in self.neighbors[cell]
                                                       if not self.blocked(x)), default=self.unreachable))
            if distance[cell] < self.unreachable:
                self.relax(distance, cell)
        removed = False
        for group in list(reversed(self.groups)):
            if group.neutral:
                info = self.info(group.cells[0])
                if info is not None and not info[2]:
                    self.cities.remove(group.cells[0])
                    self.put_back(group.cells[0])
                    self.groups.remove(group)
                    removed = True
        if removed:
            self.rebuild()
        self.score()

    def info(self, cell, reachable=True):
        a, b = (int(d[cell]) for d in self.distances)
        if reachable and (a >= self.unreachable or b >= self.unreachable):
            return None
        return (a - b, a + b, a == b or min(a, b) >= self.far)

    @staticmethod
    def pair_deficit(a, b):
        return a[1] - b[1] if a[0] <= b[0] else b[1] - a[1]

    def score(self):
        self.deficit = 0.
        for group in self.groups:
            if group.neutral:
                continue
            infos = [self.info(c, reachable=False) for c in group.cells]
            self.deficit += 1.5 * sum(i[0] for i in infos)
            if len(infos) == 2:
                self.deficit += self.pair_deficit(*infos)

    def partners(self, info):
        result = []
        for cell in self.remaining:
            other = self.info(cell)
            if other is None or other[2] or info[0] * other[0] >= 0:
                continue
            partial = self.deficit + 1.5 * other[0]
            pair = self.pair_deficit(info, other)
            if max(abs(partial), abs(pair), abs(partial + pair)) < self.tolerance:
                result.append(cell)
        return result

    def remove(self, group):
        if group in self.groups:
            self.groups.remove(group)
            for cell in group.cells:
                self.cities.remove(cell)
                self.put_back(cell)
            self.rebuild()

    def repair(self):
        for _ in range(max(4 * len(self.groups), 2 * len(self.remaining), 1)):
            changed = False
            for group in list(self.groups):
                if group.neutral or len(group.cells) != 2:
                    continue
                a, b = map(self.info, group.cells)
                if a is None or b is None:
                    continue
                bad = 0 if a[2] else 1 if b[2] or (a[0] < 0) == (b[0] < 0) else None
                if bad is None:
                    continue
                orphan = group.cells.pop(bad)
                self.groups.append(_Group([orphan], True))
                self.score()
                info = self.info(group.cells[0])
                if info is None or info[2]:
                    return False
                options = self.partners(info)
                if not options:
                    return False
                cell = options[int(self.rng.integers(len(options)))]
                self.take(cell)
                group.cells.append(cell)
                self.open(cell)
                if abs(self.deficit) > self.tolerance:
                    return False
                changed = True
                break
            if not changed:
                return True
        return False

    def plan(self, count):
        for _ in range(max(2 * len(self.remaining), 8 * count)):
            if sum(len(g.cells) for g in self.groups) >= count and abs(self.deficit) <= self.tolerance:
                break
            candidates, success = self.remaining.copy(), False
            while candidates:
                i = int(self.rng.integers(len(candidates)))
                cell = candidates[i]
                candidates[i] = candidates[-1]
                candidates.pop()
                if not self.take(cell):
                    continue
                group = _Group([cell])
                self.groups.append(group)
                self.open(cell)
                info = self.info(cell)
                if info is not None:
                    group.neutral = info[2]
                    self.score()
                    if group.neutral:
                        success = True
                        break
                    options = self.partners(info)
                    if options:
                        partner = options[int(self.rng.integers(len(options)))]
                        self.take(partner)
                        group.cells.append(partner)
                        self.open(partner)
                        if self.repair() and abs(self.deficit) <= self.tolerance:
                            success = True
                            break
                self.remove(group)
            if not success:
                break
        if not self.groups or abs(self.deficit) > self.tolerance:
            return None
        return [(g.cells, int(math.floor(40 + 10 * self.rng.random() + .5))) for g in self.groups]


def place_cities(mountains, generals, neighbors, rng, count, width, height):
    count = max(1, math.floor(count + .5))
    if len(generals) == 2:
        return _DuelCities(mountains, generals, neighbors, rng).plan(count)
    p, available = len(generals), int(mountains.sum())
    if not available:
        return None
    strengths = [math.floor(40 + 10 * rng.random() + .5) for _ in range(count)]
    radius = max(6, max(height, width) / (2 + math.log(p)))
    target, probability = count - p // 2, count / available
    immediate = probability * .1
    own_shift = 10 * p * immediate * math.sqrt(.5)
    other_shift = own_shift / (p - 1)
    thresholds = [1 - immediate] * p
    near = [0] * p
    cities, groups = set(), []
    for attempt in range(1, 52):
        if len(cities) >= target and len(set(near)) == 1:
            if sum(near):
                return groups
            break
        if attempt > 50 or (attempt > 25 and len(cities) < target and p <= 4):
            break
        if len(cities) >= target:
            largest = max(near)
            thresholds = [10 if near[k] == largest else thresholds[k] for k in range(p)]
        visited = set(map(int, generals))
        queue = deque((int(cell), 0, k) for k, cell in enumerate(generals))
        while queue:
            cell, distance, pid = queue.popleft()
            if attempt > 1 and distance >= radius:
                continue
            if mountains[cell] and cell not in cities:
                close = 0 < distance < radius
                chance = thresholds[pid] if close else 1 - probability
                if ((close and chance < 1) or (distance >= radius and attempt == 1)) and rng.random() >= chance:
                    value = strengths[near[pid]] if close else math.floor(40 + 10 * rng.random() + .5)
                    cities.add(cell)
                    groups.append(([cell], value))
                    if close:
                        near[pid] += 1
                        thresholds = [x + own_shift if k == pid else x - other_shift for k, x in enumerate(thresholds)]
                    if len(cities) >= target:
                        break
            else:
                for other in neighbors[cell]:
                    if other not in visited:
                        visited.add(other)
                        queue.append((other, distance + 1, pid))
    # Official FFA fallback: prioritize one nearby candidate per spawn, then fill.
    remaining = [int(c) for c in np.flatnonzero(mountains) if c not in cities]
    rng.shuffle(remaining)
    chosen = []
    for general in generals if min(count, len(remaining)) >= p else ():
        cell = next((c for c in remaining if c not in chosen and
                     abs(c // width - general // width) + abs(c % width - general % width) <= 6), None)
        if cell is not None:
            chosen.append(cell)
    chosen += [c for c in remaining if c not in chosen][:max(0, min(count, len(remaining)) - len(chosen))]
    groups += [([cell], math.floor(40 + 10 * rng.random() + .5)) for cell in chosen]
    return groups or None
