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


@lru_cache(maxsize=1)
def _library():
    lib, metadata = build_shared(Path(__file__).with_name("step.cpp"),
                                 name="cpu-step", flags=openmp_flags())
    lib.generals_step.argtypes = [ctypes.POINTER(_Batch), ctypes.POINTER(_Batch),
        ctypes.POINTER(_Outputs), ctypes.c_void_p, *([ctypes.c_int] * 6), ctypes.c_int64]
    lib.generals_step.restype = None
    lib.omp_get_num_procs.argtypes = []
    lib.omp_get_num_procs.restype = ctypes.c_int
    return lib, metadata


class CPUEnv:
    def __init__(self, initial, *, width, height, threads=0,
                 auto_reset=False, episode_ticks=0, seed=42):
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
            self.initial[name] = np.array(value, copy=True, order="C")
        self.state = {name: value.copy() for name, value in self.initial.items()}
        # Reuse the independent CPU state's invariant checks only at construction.
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
        return cls(arrays, width=width, height=height, **kwargs)

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

    def surrender(self, player_id, mask=None):
        if self._closed:
            raise RuntimeError("CPU environment is closed")
        from generals.core.lifecycle import surrender_batch
        surrender_batch(self, player_id, mask)

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
