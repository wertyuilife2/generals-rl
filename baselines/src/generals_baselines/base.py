"""Thin Python lifecycle/array binding for the C++ observation-only policies."""
from functools import lru_cache
import ctypes
from pathlib import Path
import weakref

import numpy as np

from generals.backends._build import build_shared
from generals.core.state import Action


class _Observation(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in
                ('terrain', 'structure', 'owner', 'army', 'visible', 'alive')] + [
        ('height', ctypes.c_int), ('width', ctypes.c_int), ('players', ctypes.c_int),
        ('player', ctypes.c_int), ('tick', ctypes.c_int64), ('terminated', ctypes.c_int)]


class _Random(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ('state', 'next64', 'next32', 'next_double')]


class _Options(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in
                ('max_sources', 'max_attempts', 'lifetime', 'memory_ttl')] + [
        ('temperature', ctypes.c_double), ('weights', ctypes.c_double * 3)]


@lru_cache(maxsize=1)
def _library():
    lib, metadata = build_shared(Path(__file__).with_name('policy.cpp'), name='baselines')
    signatures = {
        'baseline_error': ([], ctypes.c_char_p),
        'baseline_create': ([ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
        'baseline_destroy': ([ctypes.c_void_p], None),
        'baseline_act': ([ctypes.c_void_p, ctypes.POINTER(_Observation), ctypes.POINTER(_Options), ctypes.POINTER(_Random)], ctypes.c_int64),
        'baseline_result': ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, *([ctypes.c_int]*4)], None),
        'baseline_info': ([ctypes.c_void_p, ctypes.c_int], ctypes.c_int64),
        'baseline_copy': ([ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
        'baseline_set_path': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
        'baseline_set_pending': ([ctypes.c_void_p, ctypes.c_int64], None),
        'baseline_batch_create': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
        'baseline_batch_destroy': ([ctypes.c_void_p], None),
        'baseline_batch_act': ([ctypes.c_void_p, *([ctypes.c_void_p] * 9)], ctypes.c_int),
        'baseline_batch_result': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib, metadata


class BaseAI:
    """Native policy; owns its C++ memory/path and independent NumPy RNG state."""
    kind = 0
    weights = {'merge': .5, 'explore': .8, 'attack': 1.5}
    temperature = .5
    path_lifetime = 20
    max_sources = 6
    max_path_attempts = 12
    memory_ttl = 50

    def __init__(self, player_id=0, seed=0):
        self.reset(player_id, seed)

    def reset(self, player_id, seed):
        self.close()
        self.player_id = int(player_id)
        self.rng = np.random.default_rng(seed)
        self._lib, self.native_metadata = _library()
        self._handle = self._lib.baseline_create(self.kind, self.player_id)
        if not self._handle:
            raise ValueError(self._lib.baseline_error().decode())
        self._finalizer = weakref.finalize(self, self._lib.baseline_destroy, self._handle)
        # Keep the bit generator and its ctypes interface alive for native calls.
        bits = self.rng.bit_generator.ctypes
        self._bits = bits
        self._random = _Random(bits.state, *[ctypes.cast(function, ctypes.c_void_p) for function in
                                          (bits.next_uint64, bits.next_uint32, bits.next_double)])

    def close(self):
        finalizer = getattr(self, '_finalizer', None)
        if finalizer is not None:
            finalizer()
        self._handle = None

    def act(self, observation):
        if not self._handle:
            raise RuntimeError('controller is closed; reset it before acting')
        obs = observation
        if not 2 <= len(obs.alive) <= 8 or not 0 <= self.player_id < len(obs.alive):
            raise ValueError('invalid observation player count')
        arrays = []
        for name, dtype in [('terrain', np.uint8), ('structure', np.uint8), ('owner', np.int16),
                            ('army', np.int64), ('visible', np.bool_), ('alive', np.bool_)]:
            value = getattr(obs, name)
            shape = (len(obs.alive),) if name == 'alive' else (obs.height, obs.width)
            if value.dtype != dtype or value.shape != shape:
                raise ValueError(f'invalid observation array: {name}')
            arrays.append(np.ascontiguousarray(value))
        if obs.height < 1 or obs.width < 1 or obs.size > 1024*1024:
            raise ValueError('unsupported observation dimensions')
        if min(self.max_sources, self.max_path_attempts, self.path_lifetime, self.memory_ttl) < 0:
            raise ValueError('planning limits cannot be negative')
        native = _Observation(*(value.ctypes.data for value in arrays), obs.height, obs.width,
                              len(obs.alive), obs.player_id, obs.tick, bool(obs.terminated))
        options = _Options(self.max_sources, self.max_path_attempts, self.path_lifetime,
                           self.memory_ttl, self.temperature, (ctypes.c_double*3)(*self.weights.values()))
        with self.rng.bit_generator.lock:
            encoded = self._lib.baseline_act(self._handle, ctypes.byref(native), ctypes.byref(options), ctypes.byref(self._random))
        if encoded < 0:
            raise RuntimeError(self._lib.baseline_error().decode())
        return Action.decode(encoded)

    def on_result(self, result):
        if not self._handle:
            return
        if result.action.source < 0 and int(result.action.kind) != 0:
            return
        code = result.action.encode()
        if code > np.iinfo(np.int64).max:
            return
        self._lib.baseline_result(self._handle, result.player_id, code, result.valid,
                                  result.target_owned, result.source, result.target)

    def _info(self, field):
        return int(self._lib.baseline_info(self._handle, field)) if self._handle else 0

    def _memory(self, field, dtype):
        height, width = self._info(0), self._info(1)
        if not height:
            return None
        values = np.empty(height*width, np.int64)
        self._lib.baseline_copy(self._handle, field, values.ctypes.data)
        return values.astype(dtype).reshape(height, width)

    # Diagnostic snapshots never alias the native planner or another seat.
    terrain = property(lambda self: self._memory(0, np.uint8))
    structure = property(lambda self: self._memory(1, np.uint8))
    owner = property(lambda self: self._memory(2, np.int16))
    army = property(lambda self: self._memory(3, np.int64))
    last_seen = property(lambda self: self._memory(4, np.int64))
    last_search_attempts = property(lambda self: self._info(3))

    @property
    def path(self):
        if not self._handle:
            return []
        values = np.empty(self._info(4), np.int64)
        self._lib.baseline_copy(self._handle, 6, values.ctypes.data)
        return values.tolist()

    @path.setter
    def path(self, cells):
        if not self._handle:
            raise RuntimeError('controller is closed; reset it before editing a path')
        values = np.asarray(cells, dtype=np.int64)
        limit = self._info(0)*self._info(1) or 1024*1024
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= limit):
            raise ValueError('path cells must be in bounds')
        if self._lib.baseline_set_path(self._handle, values.ctypes.data, len(values)):
            raise RuntimeError(self._lib.baseline_error().decode())

    @property
    def _pending(self):
        code = self._info(2)
        return Action.decode(code) if code else None

    @_pending.setter
    def _pending(self, action):
        if not self._handle:
            raise RuntimeError('controller is closed; reset it before editing an action')
        self._lib.baseline_set_pending(self._handle, 0 if action is None else action.encode())
