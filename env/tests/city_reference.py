"""Test oracle for official city fixtures; production uses the native generator."""
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
