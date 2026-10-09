"""Small geometry caches and bounded searches using only the planner's memory."""
from collections import deque
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def geometry(height, width):
    """Immutable geometry, never terrain or observations shared between seats."""
    graph = tuple(tuple(n for n in (
        cell - width if cell >= width else -1,
        cell + 1 if cell % width < width - 1 else -1,
        cell + width if cell < (height - 1) * width else -1,
        cell - 1 if cell % width else -1,
    ) if n >= 0) for cell in range(height * width))
    ys, xs = np.divmod(np.arange(height * width), width)
    ys.flags.writeable = xs.flags.writeable = False
    return graph, ys, xs


def nearest_cells(cells, distance, size, count=16):
    """Select by (Manhattan distance, cell index) without sorting the map."""
    keys = distance[cells] * size + cells
    if len(cells) > count:
        selected = np.argpartition(keys, count - 1)[:count]
        selected = selected[np.argsort(keys[selected])]
    else:
        selected = np.argsort(keys)
    return cells[selected]


def shortest_known_path(blocked, source, target):
    """BFS with the engine's up/right/down/left tie order.

    Only blocked destinations and discovered parents are needed. Reuse the
    geometry instead of rebuilding an N x 4 array and N Python lists per search.
    """
    graph = geometry(*blocked.shape)[0]
    blocked = blocked.ravel().tolist()
    if blocked[source] or blocked[target]:
        return None
    parent = [-1] * len(graph)
    parent[source] = source
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        if cell == target:
            path = [target]
            while path[-1] != source:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for neighbor in graph[cell]:
            if parent[neighbor] < 0 and not blocked[neighbor]:
                parent[neighbor] = cell
                queue.append(neighbor)
    return None
