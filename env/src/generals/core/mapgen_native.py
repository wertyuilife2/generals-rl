"""Array-owning bridge to C++ map kernels, using NumPy's unchanged RNG stream."""
import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np

from generals.backends._build import build_shared


@lru_cache(maxsize=1)
def _library():
    lib, _ = build_shared(Path(__file__).with_suffix('.cpp'), name='mapgen', include_dirs=[np.get_include()])
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    signatures = {
        'spawns': [pointer, integer, integer, pointer, pointer],
        'cities': [pointer, pointer, integer, integer, pointer, integer, pointer, pointer, pointer],
        'valid': [pointer, pointer, integer, pointer, integer, integer, integer],
        'connect': [pointer, integer, integer, integer],
        'distance': [pointer, pointer, integer, integer, integer, integer],
        'pair': [pointer, integer, integer, integer, pointer, pointer],
    }
    for name, args in signatures.items():
        fn = getattr(lib, 'generals_map_'+name)
        fn.argtypes, fn.restype = args, None if name == 'connect' else integer
    return lib


def spawns(mountains, width, rng):
    mountains = np.ascontiguousarray(mountains, dtype=np.uint8)
    out = np.empty(2, dtype=np.int32)
    with rng.bit_generator.lock:
        ok = _library().generals_map_spawns(mountains.ctypes.data, width, mountains.size//width,
                                            rng.bit_generator.ctypes.bit_generator, out.ctypes.data)
    return out.tolist() if ok else None


def duel_cities(mountains, generals, width, height, rng, count):
    mountains = np.ascontiguousarray(mountains, dtype=np.uint8)
    generals = np.ascontiguousarray(generals, dtype=np.int32)
    cells, armies, groups = (np.empty(mountains.size, dtype=np.int32) for _ in range(3))
    with rng.bit_generator.lock:
        size = _library().generals_map_cities(mountains.ctypes.data, generals.ctypes.data, width, height,
            rng.bit_generator.ctypes.bit_generator, count, cells.ctypes.data, armies.ctypes.data, groups.ctypes.data)
    if size < 0:
        return None
    result = []
    for i in range(size):
        if not result or groups[i] != groups[i-1]:
            result.append(([], int(armies[i])))
        result[-1][0].append(int(cells[i]))
    return result


def valid(mountains, generals, cities, width):
    mountains = np.ascontiguousarray(mountains, dtype=np.uint8)
    generals = np.ascontiguousarray(generals, dtype=np.int32)
    cities = np.ascontiguousarray(list(cities), dtype=np.int32)
    return bool(_library().generals_map_valid(mountains.ctypes.data, generals.ctypes.data, len(generals),
                cities.ctypes.data, len(cities), width, mountains.size//width))


def connect(terrain, source):
    if not terrain.flags.writeable:
        raise ValueError('connect requires writable terrain')
    values = np.ascontiguousarray(terrain, dtype=np.uint8)
    height, width = values.shape
    _library().generals_map_connect(values.ctypes.data, width, height, int(source))
    if values is not terrain:
        terrain[:] = values


def distance(state):
    height, width = state.owner.shape
    return _library().generals_map_distance(state.layout.terrain.ctypes.data, state.structure.ctypes.data,
                                             width, height, *map(int, state.general_pos))


def select_pair(candidates, width, distance, rng):
    candidates = np.ascontiguousarray(candidates, dtype=np.int64)
    result = np.empty(2, dtype=np.int64)
    with rng.bit_generator.lock:
        found = _library().generals_map_pair(candidates.ctypes.data, len(candidates), width, distance,
                                             rng.bit_generator.ctypes.bit_generator, result.ctypes.data)
    return result if found else None
