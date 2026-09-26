"""Four-neighbor terrain distances. Pass only terrain the caller may know."""
from collections import OrderedDict, deque
from dataclasses import dataclass
from numbers import Integral
import hashlib
import numpy as np
from .state import MapLayout, Terrain, UNKNOWN


def _terrain(value):
    arr = value.terrain if isinstance(value, MapLayout) else np.asarray(value)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("terrain must be a nonempty 2D array")
    return arr


def topology_key(terrain, unknown_passable=True):
    arr = _terrain(terrain)
    return (arr.shape, bool(unknown_passable), hashlib.sha256(arr.tobytes()).digest())


def neighbors(terrain, unknown_passable=True):
    arr = _terrain(terrain)
    h, w = arr.shape
    ids = np.arange(arr.size, dtype=np.int32).reshape(h, w)
    result = np.full((arr.size, 4), -1, dtype=np.int32)
    result[ids[1:].ravel(), 0] = ids[:-1].ravel()
    result[ids[:, :-1].ravel(), 1] = ids[:, 1:].ravel()
    result[ids[:-1].ravel(), 2] = ids[1:].ravel()
    result[ids[:, 1:].ravel(), 3] = ids[:, :-1].ravel()
    blocked = arr.ravel() == Terrain.MOUNTAIN
    if not unknown_passable:
        blocked |= arr.ravel() == UNKNOWN
    valid = result >= 0
    result[valid & blocked[np.maximum(result, 0)]] = -1
    result[blocked] = -1
    result.flags.writeable = False
    return result


def _search(terrain, source, target=None, unknown_passable=True):
    arr = _terrain(terrain)
    n = arr.size
    if not isinstance(source, Integral) or isinstance(source, (bool, np.bool_)):
        raise TypeError("source must be an integer")
    if not 0 <= source < n:
        raise ValueError("source outside map")
    if target is not None:
        if not isinstance(target, Integral) or isinstance(target, (bool, np.bool_)):
            raise TypeError("target must be an integer")
        if not 0 <= target < n:
            raise ValueError("target outside map")
    parent = [-1] * n
    distance = [-1] * n
    flat = arr.ravel()
    if flat[source] == Terrain.MOUNTAIN or (flat[source] == UNKNOWN and not unknown_passable):
        return distance, parent
    graph = neighbors(arr, unknown_passable).tolist()
    distance[source] = 0
    q = deque([int(source)])
    while q:
        cell = q.popleft()
        if cell == target:
            break
        for nxt in graph[cell]:
            if nxt >= 0 and distance[nxt] < 0:
                parent[nxt] = cell
                distance[nxt] = distance[cell] + 1
                q.append(nxt)
    return distance, parent


def single_source_distances(terrain, source, unknown_passable=True):
    distances, _ = _search(terrain, source, unknown_passable=unknown_passable)
    result = np.asarray(distances, dtype=np.int32)
    result.flags.writeable = False
    return result


def shortest_path(terrain, source, target, unknown_passable=True):
    distances, parent = _search(terrain, source, target, unknown_passable)
    if distances[target] < 0:
        return None
    result = [int(target)]
    while result[-1] != source:
        result.append(parent[result[-1]])
    result.reverse()
    return result


class DistanceCache:
    """Bounded CPU cache; keys include the complete supplied knowledge map."""
    def __init__(self, capacity=64):
        if capacity < 1: raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._values = OrderedDict()

    def distances(self, terrain, source, unknown_passable=True):
        if not isinstance(source, Integral) or isinstance(source, (bool, np.bool_)):
            raise TypeError("source must be an integer")
        key = (topology_key(terrain, unknown_passable), int(source))
        if key not in self._values:
            self._values[key] = single_source_distances(terrain, source, unknown_passable)
            if len(self._values) > self.capacity:
                self._values.popitem(last=False)
        self._values.move_to_end(key)
        return self._values[key]

    def clear(self): self._values.clear()
    def __len__(self): return len(self._values)


@dataclass(frozen=True)
class StaticFeatures:
    topology_id: str
    neighbors: np.ndarray


def prepare_map(layout: MapLayout):
    """Privileged full-map preprocessing; do not give this to LOCAL policies."""
    return StaticFeatures(layout.topology_id, neighbors(layout, unknown_passable=False))


def all_pairs_distances(terrain, max_bytes=128 * 1024 * 1024):
    """Explicit quadratic-memory preprocessing, never enabled by default."""
    arr = _terrain(terrain)
    required = arr.size ** 2 * np.dtype(np.int32).itemsize
    if required > max_bytes:
        raise ValueError(f"distance matrix needs {required} bytes, exceeding budget")
    return np.stack([single_source_distances(arr, i, False) for i in range(arr.size)])
