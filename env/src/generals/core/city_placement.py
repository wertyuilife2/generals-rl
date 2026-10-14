"""Distance-balanced city pairs for two opposing spawns.

Opening a city can shorten other city paths. Recompute/relax those distances
and repair affected pairs before accepting placement. Shared army values are
assigned to each accepted pair by the caller.
"""
from dataclasses import dataclass
from math import floor


@dataclass(eq=False)
class Group:
    cells: list
    neutral: bool = False


class Planner:
    def __init__(self, grid, generals, fairness, rng):
        self.grid, self.generals, self.rng = grid, generals, rng
        self.tolerance = 1 + 19 * (1 - fairness) ** 3
        self.neutral_distance = floor(5 + 25 * fairness)
        self.cities, self.groups = set(), []
        self.remaining = [i for i, mountain in enumerate(grid.mountains) if mountain]
        self.positions = {cell: index for index, cell in enumerate(self.remaining)}
        self.recompute()

    def remove_candidate(self, cell):
        index = self.positions.pop(cell, None)
        if index is None:
            return False
        last = self.remaining.pop()
        if index < len(self.remaining):
            self.remaining[index] = last
            self.positions[last] = index
        return True

    def restore_candidate(self, cell):
        if cell not in self.positions:
            self.positions[cell] = len(self.remaining)
            self.remaining.append(cell)

    def recompute(self):
        self.distances = [self.grid.distances(g, self.cities) for g in self.generals]

    def info(self, cell):
        left, right = self.distances[0][cell], self.distances[1][cell]
        if max(left, right) > self.grid.size:
            return None
        return (cell, left - right, left + right,
                left == right or min(left, right) >= self.neutral_distance)

    @staticmethod
    def pair_difference(left, right):
        return left[2] - right[2] if left[1] <= right[1] else right[2] - left[2]

    def deficit(self):
        total = 0.
        for group in self.groups:
            if not group.neutral:
                infos = [info for cell in group.cells if (info := self.info(cell)) is not None]
                total += 1.5 * sum(info[1] for info in infos)
                if len(infos) == 2:
                    total += self.pair_difference(*infos)
        return total

    def open_city(self, cell):
        self.cities.add(cell)
        for distances in self.distances:
            self.grid.expand_city(distances, self.cities, cell)
        removed = False
        for group in list(reversed(self.groups)):
            info = self.info(group.cells[0])
            if group.neutral and info is not None and not info[3]:
                self.cities.remove(group.cells[0])
                self.restore_candidate(group.cells[0])
                self.groups.remove(group)
                removed = True
        if removed:
            self.recompute()

    def discard(self, group):
        if group in self.groups:
            for cell in group.cells:
                self.cities.remove(cell)
                self.restore_candidate(cell)
            self.groups.remove(group)
            self.recompute()

    def partners(self, origin):
        result, deficit = [], self.deficit()
        for cell in self.remaining:
            info = self.info(cell)
            if info is None or info[3] or origin[1] * info[1] >= 0:
                continue
            distance_deficit = deficit + 1.5 * info[1]
            pair_deficit = self.pair_difference(origin, info)
            if max(abs(distance_deficit), abs(pair_deficit),
                   abs(distance_deficit + pair_deficit)) < self.tolerance:
                result.append(cell)
        return result

    def pair(self, group, origin):
        candidates = self.partners(origin)
        if not candidates:
            return False
        cell = candidates[int(self.rng.integers(len(candidates)))]
        self.remove_candidate(cell)
        group.cells.append(cell)
        self.open_city(cell)
        return True

    def repair_pairs(self):
        budget = max(4 * len(self.groups), 2 * len(self.remaining), 1)
        for _ in range(budget):
            for group in self.groups:
                if group.neutral or len(group.cells) != 2:
                    continue
                left, right = (self.info(cell) for cell in group.cells)
                if left is None or right is None:
                    continue
                invalid = left if left[3] else right if right[3] or (left[1] < 0) == (right[1] < 0) else None
                if invalid is None:
                    continue
                retained = group.cells[1] if invalid[0] == group.cells[0] else group.cells[0]
                self.groups.append(Group([invalid[0]], True))
                group.cells = [retained]
                origin = self.info(retained)
                if origin is None or origin[3] or not self.pair(group, origin):
                    return False
                if abs(self.deficit()) > self.tolerance:
                    return False
                break
            else:
                return True
        return False

    def place(self, desired):
        target = max(1, floor(desired + .5))
        budget = max(2 * len(self.remaining), 8 * target)
        for _ in range(budget):
            if not self.remaining or (len(self.cities) >= target and abs(self.deficit()) <= self.tolerance):
                break
            candidates = self.remaining.copy()
            placed = False
            while candidates:
                index = int(self.rng.integers(len(candidates)))
                cell = candidates[index]
                candidates[index] = candidates[-1]
                candidates.pop()
                if not self.remove_candidate(cell):
                    continue
                group = Group([cell])
                self.groups.append(group)
                self.open_city(cell)
                info = self.info(cell)
                if info is not None:
                    group.neutral = info[3]
                    if group.neutral or (self.pair(group, info) and self.repair_pairs() and
                                         abs(self.deficit()) <= self.tolerance):
                        placed = True
                        break
                self.discard(group)
            if not placed:
                break
        if not self.cities or abs(self.deficit()) > self.tolerance:
            return None
        return [(group.cells, group.neutral) for group in self.groups]


def paired_cities(grid, generals, desired, fairness, rng):
    return Planner(grid, generals, fairness, rng).place(desired)
