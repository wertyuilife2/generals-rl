"""Native batch binding for the public reset_batch/act_batch/result protocol."""
import ctypes
import threading
import weakref

import numpy as np

from .base import _Options, _Random, _library


def _array(value, dtype, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype != dtype
            or value.shape != shape or not value.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous {np.dtype(dtype).name} array of shape {shape}")
    return value


class PolicyBatch:
    """Private planner instances; only masked env observations enter act()."""
    def __init__(self, prototype, seats, seeds):
        if len(seats) != len(seeds) or not len(seats):
            raise ValueError('one seat and seed are required per lane')
        if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer))
               or not 0 <= s < 8 for s in seats):
            raise ValueError('invalid player ids')
        if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer))
               or not 0 <= s < 2**64 for s in seeds):
            raise ValueError('seeds must be uint64')
        self.seats = np.asarray(seats, np.int64)
        self.count, self.controllers = len(seats), []
        self._lock, self._handle, self._closed = threading.RLock(), None, False
        self._actions = np.zeros(self.count, np.int64)
        self._lib, _ = _library()
        try:
            for seat, seed in zip(seats, seeds, strict=True):
                controller = type(prototype)(player_id=int(seat), seed=int(seed))
                self.controllers.append(controller)
                for name in ('max_sources', 'max_path_attempts', 'path_lifetime', 'memory_ttl',
                             'temperature', 'weights'):
                    setattr(controller, name, getattr(prototype, name))
        except BaseException:
            self.close()
            raise

    def _bind(self, height, width, players):
        if self._closed:
            raise RuntimeError('batch is closed; reset it before use')
        shape = height, width, players
        if self._handle:
            if shape != self.shape:
                raise ValueError('batch geometry changed; reset_batch first')
            return
        options = (_Options * self.count)(*[
            _Options(c.max_sources, c.max_path_attempts, c.path_lifetime,
                     c.memory_ttl, c.temperature, (ctypes.c_double * 3)(*c.weights.values()))
            for c in self.controllers])
        handles = (ctypes.c_void_p * self.count)(*[c._handle for c in self.controllers])
        random = (_Random * self.count)(*[c._random for c in self.controllers])
        self._handle = self._lib.baseline_batch_create(handles, options, random, self.count, *shape)
        if not self._handle:
            raise ValueError(self._lib.baseline_error().decode())
        self._finalizer = weakref.finalize(self, self._lib.baseline_batch_destroy, self._handle)
        self.shape = shape

    def act(self, raw):
        state = raw['state']
        height, width, players = raw['height'], raw['width'], state['alive'].shape[1]
        if not np.array_equal(raw['player_ids'], self.seats):
            raise ValueError('observation player_ids differ from reset_batch')
        arrays = [_array(state[name], dtype, (self.count, height * width), name)
                  for name, dtype in (('terrain', np.uint8), ('structure', np.uint8),
                                      ('owner', np.int16), ('army', np.int64))]
        arrays.extend([
            _array(raw['visible'], np.bool_, (self.count, height * width), 'visible'),
            _array(state['alive'], np.bool_, (self.count, players), 'alive'),
            _array(state['tick'], np.int64, (self.count,), 'tick'),
            _array(state['terminated'], np.bool_, (self.count,), 'terminated'),
            _array(raw['active'], np.bool_, (self.count,), 'active')])
        with self._lock:
            self._bind(height, width, players)
            status = self._lib.baseline_batch_act(
                self._handle, *(a.ctypes.data for a in arrays), self._actions.ctypes.data)
            if status:
                raise RuntimeError(self._lib.baseline_error().decode())
            return self._actions

    def on_result(self, raw):
        moves = _array(raw['moves'], np.int64, (self.count, 9), 'moves')
        actions = _array(raw['actions'], np.int64, (self.count,), 'actions')
        active = _array(raw['active'], np.bool_, (self.count,), 'active')
        with self._lock:
            if not self._handle:
                raise RuntimeError('batch must act before receiving results')
            self._lib.baseline_batch_result(self._handle, moves.ctypes.data,
                                            actions.ctypes.data, active.ctypes.data)

    def close(self):
        with self._lock:
            finalizer = getattr(self, '_finalizer', None)
            if finalizer is not None:
                finalizer()
            self._handle, self._closed = None, True
            for controller in self.controllers:
                controller.close()
            self.controllers = []
