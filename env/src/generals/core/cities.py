"""Fair city placement for standard random maps.

Cities replace mountain candidates. In 1v1, paired cities share their initial
army, and path-distance deficits are balanced as each new city opens paths.
"""
from collections import deque
import math
import numpy as np


def place_cities(mountains, generals, neighbors, rng, count, width, height):
    count = max(1, math.floor(count + .5))
    if len(generals) == 2:
        from .mapgen_native import duel_cities
        return duel_cities(mountains, generals, width, height, rng, count)
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
