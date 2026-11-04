"""Native, synchronous, batched env.step(actions) with NumPy-owned storage."""
import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from generals.protocol import STATE_DTYPES, state_shape
from generals.protocol.config import _integer
from generals.protocol.vector import BatchStep
from generals.backends._build import build_shared, openmp_flags
from generals.backends._abi import Batch as _Batch


def _pointers(arrays):
    return _Batch(*(arrays[name].ctypes.data for name in STATE_DTYPES))


class _Outputs(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "stats", "moves", "counters", "errors", "done", "current_stats",
        "growth_cells", "growth_count", "cache_valid", "owned_structures")]


class _Observation(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        'terrain', 'structure', 'owner', 'army', 'visible', 'alive', 'fog_obstacles',
        'has_kill', 'warn_afk', 'neutralize_in', 'stats', 'general_pos', 'active',
        'tick', 'terminated', 'winner_id')]


class _BoundCPU(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ('state', 'initial', 'out')] + [
        (name, ctypes.c_int) for name in ('count', 'height', 'width', 'players', 'threads', 'auto_reset')
    ] + [('horizon', ctypes.c_int64), ('observations', ctypes.c_void_p)]


class _ObservationCache:
    """The native lease owns observer caches independently of live state arrays."""
    def __init__(self, lib, state, batch, height, width, players):
        self.lib = lib
        self.handle = lib.generals_observation_cache_create(ctypes.byref(state), batch, height, width, players)
        if not self.handle:
            raise MemoryError('could not allocate native observation cache')

    def __del__(self):
        if getattr(self, 'handle', None):
            self.lib.generals_observation_cache_destroy(self.handle)
            self.handle = None


@lru_cache(maxsize=1)
def _library():
    lib, metadata = build_shared(Path(__file__).with_name("step.cpp"),
                                 name="cpu-step", flags=openmp_flags())
    lib.generals_step.argtypes = [ctypes.POINTER(_Batch), ctypes.POINTER(_Batch),
        ctypes.POINTER(_Outputs), ctypes.c_void_p, *([ctypes.c_int] * 6), ctypes.c_int64]
    lib.generals_step.restype = None
    lib.generals_observe.argtypes = [ctypes.POINTER(_Batch), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(_Observation), *([ctypes.c_int] * 4)]
    lib.generals_observe.restype = None
    lib.generals_depart.argtypes = [ctypes.POINTER(_Batch), ctypes.POINTER(_Outputs),
        *([ctypes.c_int] * 5), ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]
    lib.generals_depart.restype = None
    lib.generals_visibility.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *([ctypes.c_int] * 3)]
    lib.generals_legal.argtypes = [*([ctypes.c_void_p] * 4), *([ctypes.c_int] * 3)]
    lib.omp_get_num_procs.argtypes = []
    lib.omp_get_num_procs.restype = ctypes.c_int
    lib.generals_observation_cache_create.argtypes = [ctypes.POINTER(_Batch), *([ctypes.c_int]*4)]
    lib.generals_observation_cache_create.restype = ctypes.c_void_p
    lib.generals_observation_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.generals_observation_cache_destroy.restype = None
    return lib, metadata


class CPUEnv:
    def __init__(self, initial, *, width, height, threads=0,
                 auto_reset=False, episode_ticks=0, seed=42, _validated=False, _take_initial=False):
        self._closed = False
        self.seed = _integer("seed", seed, 0)
        self.width, self.height = _integer("width", width, 1), _integer("height", height, 1)
        n = self.width * self.height
        if n > 4096 or set(initial) != set(STATE_DTYPES):
            raise ValueError("CPUEnv needs the complete shared state protocol and at most 4096 cells")
        if (not isinstance(initial["army"], np.ndarray) or initial["army"].ndim != 2
                or not isinstance(initial["alive"], np.ndarray) or initial["alive"].ndim != 2):
            raise ValueError("army and alive must be two-dimensional NumPy arrays")
        self.batch_size, size = initial["army"].shape
        self.players = initial["alive"].shape[1]
        if not self.batch_size or size != n or not 2 <= self.players <= 8:
            raise ValueError("invalid batch dimensions or player count")
        requested = _integer("threads", threads, 0)
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        self.threads = min(requested or available, self.batch_size)
        self.auto_reset = bool(auto_reset)
        self.episode_ticks = _integer("episode_ticks", episode_ticks, 0)
        if self.episode_ticks > np.iinfo(np.int64).max:
            raise ValueError("episode_ticks must fit int64")
        self.initial = {}
        for name, dtype in STATE_DTYPES.items():
            shape = state_shape(name, self.batch_size, n, self.players)
            value = initial[name]
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(dtype):
                raise ValueError(f"{name} must be {dtype} {shape}")
            self.initial[name] = value if _take_initial else np.array(value, copy=True, order="C")
        self.state = {name: value.copy() for name, value in self.initial.items()}
        # Reuse the independent CPU state's invariant checks only at construction.
        if not _validated:
            self.to_states(validate=True)
        self._initial_stats = np.zeros((self.batch_size, self.players, 3), dtype=np.int64)
        for lane in range(self.batch_size):
            for pid in range(self.players):
                cells = self.state["owner"][lane] == pid
                total = sum(map(int, self.state["army"][lane, cells]))
                if total > np.iinfo(np.int64).max:
                    raise OverflowError("player's total army exceeds int64 statistics capacity")
                self._initial_stats[lane, pid] = [total, cells.sum(),
                    (cells & (self.state["structure"][lane] == 1)).sum()]
        self.stats = self._initial_stats.copy()
        self.last_moves = np.zeros((self.batch_size, self.players, 9), np.int64)
        self.counters = np.zeros((self.batch_size, 9), np.int64)
        self.errors = np.zeros(self.batch_size, np.int32)
        self.done = np.zeros((self.batch_size, 2), bool)
        self._current_stats = self.stats.copy()
        self._growth_cells = np.empty((self.batch_size, n), np.uint16)
        self._growth_count = np.zeros(self.batch_size, np.int32)
        self._cache_valid = np.zeros(self.batch_size, bool)
        self._owned_structures = np.empty((self.batch_size, self.players), np.int32)
        self._initial_rng = np.array([int(np.random.SeedSequence([seed, lane, 1]).generate_state(
            1, dtype=np.uint32)[0]) or 1 for lane in range(self.batch_size)], np.uint32)
        self.rng_state = self._initial_rng.copy()
        self._array_refs = tuple(self.state.values()) + tuple(self.initial.values())
        self._array_ids = tuple(map(id, self._array_refs))
        self._state, self._initial = _pointers(self.state), _pointers(self.initial)
        values = [self.stats, self.last_moves, self.counters, self.errors, self.done,
                  self._current_stats, self._growth_cells, self._growth_count, self._cache_valid, self._owned_structures]
        self._out = _Outputs(*(value.ctypes.data for value in values))
        self._lib, self.metadata = _library()
        # OpenMP binding may narrow only the calling thread's OS affinity to
        # one CPU. Ask the runtime for its original available CPU set in auto mode.
        if requested == 0:
            self.threads = min(int(self._lib.omp_get_num_procs()), self.batch_size)
        self._result = BatchStep(self.last_moves, self.stats, self.done, self.errors)

    @classmethod
    def from_states(cls, states, **kwargs):
        from generals.core.batch import pack_states
        arrays, width, height = pack_states(states)
        return cls(arrays, width=width, height=height, _validated=True, _take_initial=True, **kwargs)

    @classmethod
    def from_config(cls, config, batch_size=1, *, seed=42, **kwargs):
        from generals.core.batch import generate_states
        return cls.from_states(generate_states(config, batch_size, seed=seed), seed=seed, **kwargs)

    def step(self, actions):
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        if not isinstance(actions, np.ndarray) or actions.dtype != np.int64:
            raise TypeError("actions must be an int64 NumPy array")
        if actions.shape != (self.batch_size, self.players):
            raise ValueError("actions must have shape [batch, players]")
        if tuple(map(id, self.state.values())) + tuple(map(id, self.initial.values())) != self._array_ids:
            raise ValueError("edit arrays in place; replacing their storage is unsupported")
        actions = np.ascontiguousarray(actions)
        self._lib.generals_step(ctypes.byref(self._state), ctypes.byref(self._initial), ctypes.byref(self._out),
            actions.ctypes.data, self.batch_size, self.height, self.width, self.players, self.threads,
            self.auto_reset, self.episode_ticks)
        return self._result

    def invalidate_cache(self):
        """Call after directly editing state arrays; direct pointer replacement is unsupported."""
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        self._cache_valid.fill(False)

    def native_binding(self):
        """Lease the CPU kernel ABI to an exclusive synchronous executor.

        The lease holds all buffers and ctypes structures alive. Callers must
        finish execution before closing/resetting this environment or editing
        its storage. The public step/observer APIs keep their existing checks.
        """
        from generals.protocol.native import ABI_VERSION, HOST_BUFFERS, BackendABI, BackendBinding, address
        if self._closed:
            raise RuntimeError('CPU environment is closed')
        if self.auto_reset:
            raise ValueError('native match execution requires auto_reset=False')
        if tuple(map(id, self.state.values())) + tuple(map(id, self.initial.values())) != self._array_ids:
            raise ValueError('edit arrays in place; replacing their storage is unsupported')
        observations = _ObservationCache(self._lib, self._state, self.batch_size, self.height, self.width, self.players)
        context = _BoundCPU(ctypes.addressof(self._state), ctypes.addressof(self._initial),
            ctypes.addressof(self._out), self.batch_size, self.height, self.width, self.players,
            self.threads, 0, self.episode_ticks, observations.handle)
        abi = BackendABI(ABI_VERSION, ctypes.addressof(context), address(self._lib.generals_bound_observe),
            address(self._lib.generals_bound_step), self.state['alive'].ctypes.data, self.done.ctypes.data,
            self.state['tick'].ctypes.data, self.last_moves.ctypes.data, self.errors.ctypes.data,
            address(self._lib.generals_bound_depart), address(self._lib.generals_bound_snapshot),
            self.stats.ctypes.data, address(self._lib.generals_bound_legal), HOST_BUFFERS, 0, None)
        return BackendBinding(abi, (self, context, observations, self._state, self._initial, self._out,
            *self._array_refs, self.stats, self.last_moves, self.done, self.errors,
            self._current_stats, self._growth_cells, self._growth_count, self._cache_valid, self._owned_structures))

    def observer(self, player_ids, *, full=False):
        """Bind seats; produce independent public snapshots in one native call."""
        ids = np.asarray(player_ids)
        if ids.ndim == 0:
            ids = np.full(self.batch_size, ids)
        if (ids.shape != (self.batch_size,) or ids.dtype.kind not in 'iu'
                or np.any(ids < (-1 if full else 0)) or np.any(ids >= self.players)):
            raise ValueError('one valid player id is required for every environment')
        ids = np.array(ids, dtype=np.int64, order='C', copy=True)
        native_ids = -ids-2 if full else ids

        def observe():
            if self._closed:
                raise RuntimeError('CPU environment is closed')
            boards = {key: np.empty_like(self.state[key]) for key in
                      ('terrain', 'structure', 'owner', 'army', 'alive', 'tick', 'terminated')}
            boards.update(fog_obstacles=np.empty_like(self.state['terrain'], dtype=bool),
                has_kill=np.empty_like(self.state['alive']), warn_afk=np.empty_like(self.state['alive']),
                neutralize_in=np.empty(self.state['alive'].shape, dtype=np.float64),
                winner_id=np.empty_like(self.state['winner']))
            raw = dict(state=boards, visible=np.empty_like(self.state['terrain'], dtype=bool),
                stats=np.empty_like(self.stats), general_pos=np.empty(self.batch_size, np.int32),
                active=np.empty(self.batch_size, bool), player_ids=ids.copy(),
                width=self.width, height=self.height, visibility='full' if full else 'local')
            arrays = {**boards, **{k: raw[k] for k in ('visible', 'stats', 'general_pos', 'active')}}
            output = _Observation(*(arrays[key].ctypes.data for key, _ in _Observation._fields_))
            self._lib.generals_observe(ctypes.byref(self._state), self.stats.ctypes.data,
                self.done.ctypes.data, native_ids.ctypes.data, ctypes.byref(output),
                self.batch_size, self.height, self.width, self.players)
            return raw
        return observe

    def surrender(self, player_id, mask=None):
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        self.depart(player_id, mask)

    def depart(self, player_id, mask=None, *, neutralize=False):
        player_id = _integer('player_id', player_id, 0)
        if player_id >= self.players:
            raise ValueError('invalid player id')
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != bool
                                 or mask.shape != (self.batch_size,)):
            raise ValueError('surrender mask must be bool [batch] on the environment device')
        mask = np.ascontiguousarray(mask) if mask is not None else None
        self._lib.generals_depart(ctypes.byref(self._state), ctypes.byref(self._out),
            self.batch_size, self.height, self.width, self.players, player_id,
            None if mask is None else mask.ctypes.data, neutralize, self.episode_ticks)

    def reset(self, mask=None):
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != bool
                                 or mask.shape != (self.batch_size,)):
            raise ValueError("reset mask must be bool [batch]")
        selected = slice(None) if mask is None else mask
        for name, value in self.initial.items():
            self.state[name][selected] = value[selected]
        self.stats[selected] = self._initial_stats[selected]
        self.rng_state[selected] = self._initial_rng[selected]
        for value in (self.counters, self.last_moves, self.errors, self.done, self._cache_valid):
            value[selected] = 0
        return self.state

    def check_errors(self):
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        bad = np.flatnonzero(self.errors)
        if len(bad):
            raise OverflowError(f"CPU lane {bad[0]}: numerical error {self.errors[bad[0]]}")

    def synchronize(self):
        # step is synchronous, including the OpenMP join.
        if self._closed:
            raise RuntimeError("CPU environment is closed")

    def to_states(self, validate=False):
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        if hasattr(self, "errors"):
            self.check_errors()
        from generals.core.batch import unpack_states
        return unpack_states(self.state, width=self.width, height=self.height, validate=validate)

    def close(self):
        self._closed = True
        self.state.clear()
        self.initial.clear()
        self._array_refs = ()
        for name in ("stats", "last_moves", "counters", "errors", "done", "_initial_stats",
                     "_current_stats", "_growth_cells", "_growth_count", "_cache_valid", "_owned_structures",
                     "_initial_rng", "rng_state", "_result"):
            setattr(self, name, None)
