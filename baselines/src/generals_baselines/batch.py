"""One native call per tick for independent observation-only baseline seats.

The adapter accepts contiguous host arrays, applies local fog in C++, and returns
encoded actions. It shares the existing planners and NumPy random streams; no
Observation, Action, MoveResult, multiprocessing, or Torch is needed per tick.
"""
import ctypes
import threading
import weakref

import numpy as np

from .base import _Options, _Random, _library
from .registry import create_controller, make_controllers


def _array(value, dtype, shape, name):
    if (not isinstance(value, np.ndarray) or value.dtype != dtype
            or value.shape not in shape or not value.flags.c_contiguous):
        raise ValueError(f"{name} must be a contiguous {np.dtype(dtype).name} array of shape {shape}")
    return value


class BatchBaselineControllers:
    """Own one baseline per match, with the original make_controllers seeds.

    ``specs`` supplies opponent, learner_seat and controller_seed attributes.
    ``act`` returns a borrowed [N] int64 buffer, overwritten by the next call.
    Host state maps use [N,H*W] or [N,H,W]; alive uses [N,P], and tick,
    terminated and active use [N]. ``on_result`` accepts native [N,P,9] move
    rows and either enemy [N] actions or joint [N,P] actions. Calls on this
    owner are serialized, including all independent bit-generator locks.
    Optional ``seat_seeds`` are final per-controller seeds and are never spawned
    again; otherwise the original match-root ``controller_seed`` behavior applies.
    """

    def __init__(self, specs, *, height, width, players=2, threads=1, seat_seeds=None):
        if isinstance(threads, bool) or threads != 1:
            raise ValueError("batch baselines currently use one native thread")
        if (isinstance(height, bool) or isinstance(width, bool)
                or not isinstance(height, (int, np.integer))
                or not isinstance(width, (int, np.integer))
                or min(height, width) < 1 or height * width > 1024 * 1024):
            raise ValueError("invalid baseline batch dimensions")
        if players != 2:
            raise ValueError("evaluation baseline batches require two-player matches")
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("provide at least one baseline spec")
        self.seat_seeds = None if seat_seeds is None else tuple(seat_seeds)
        if self.seat_seeds is not None and (len(self.seat_seeds) != len(self.specs) or any(
                isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
                or not 0 <= seed < 2**64 for seed in self.seat_seeds)):
            raise ValueError("seat_seeds must contain one uint64 seed per baseline")
        self.height, self.width, self.players = int(height), int(width), int(players)
        self.count = len(self.specs)
        self._lock = threading.RLock()
        self._lib, self.native_metadata = _library()
        self._actions = np.zeros(self.count, dtype=np.int64)
        self._handle = None
        self.controllers = []
        self.reset()

    def reset(self):
        """Restart every planner and stream using its original match seed."""
        with self._lock:
            self.close()
            controllers = []
            try:
                for index, spec in enumerate(self.specs):
                    if spec.learner_seat not in (0, 1):
                        raise ValueError("learner_seat must be 0 or 1")
                    seat = 1 - spec.learner_seat
                    if self.seat_seeds is not None:
                        controllers.append(create_controller(spec.opponent, seat, int(self.seat_seeds[index])))
                    else:
                        kinds = ["random", "random"]
                        kinds[seat] = spec.opponent
                        pair = make_controllers(kinds, seed=spec.controller_seed)
                        controllers.append(pair[seat])
                        pair[1 - seat].close()
                options = (_Options * self.count)(*[
                    _Options(c.max_sources, c.max_path_attempts, c.path_lifetime,
                             c.memory_ttl, c.temperature,
                             (ctypes.c_double * 3)(*c.weights.values())) for c in controllers])
                handles = (ctypes.c_void_p * self.count)(*[c._handle for c in controllers])
                random = (_Random * self.count)(*[c._random for c in controllers])
                self._handle = self._lib.baseline_batch_create(
                    handles, options, random, self.count, self.height, self.width, self.players)
                if not self._handle:
                    raise RuntimeError(self._lib.baseline_error().decode())
                self._finalizer = weakref.finalize(self, self._lib.baseline_batch_destroy, self._handle)
                self.controllers = controllers
                self._rng_locks = tuple(c.rng.bit_generator.lock for c in controllers)
                self._actions.fill(0)
            except BaseException:
                for controller in controllers:
                    controller.close()
                raise
        return self

    def act(self, state, active):
        """Apply local fog, planner memory and decisions entirely in C++."""
        map_shapes = ((self.count, self.height * self.width),
                      (self.count, self.height, self.width))
        arrays = [_array(state[name], dtype, map_shapes, name) for name, dtype in (
            ("terrain", np.uint8), ("structure", np.uint8),
            ("owner", np.int16), ("army", np.int64))]
        arrays.extend([_array(state["alive"], np.bool_, ((self.count, self.players),), "alive"),
                       _array(state["tick"], np.int64, ((self.count,),), "tick"),
                       _array(state["terminated"], np.bool_, ((self.count,),), "terminated"),
                       _array(active, np.bool_, ((self.count,),), "active")])
        with self._lock:
            self._check_open()
            acquired = []
            try:
                for lock in self._rng_locks:
                    lock.acquire()
                    acquired.append(lock)
                status = self._lib.baseline_batch_act(
                    self._handle, *(a.ctypes.data for a in arrays), self._actions.ctypes.data)
            finally:
                for lock in reversed(acquired):
                    lock.release()
            if status:
                raise RuntimeError(self._lib.baseline_error().decode())
            return self._actions

    def on_result(self, moves, actions=None, active=None):
        """Advance the original pending-path lifecycle using native move rows."""
        moves = _array(moves, np.int64, ((self.count, self.players, 9),), "moves")
        actions = self._actions if actions is None else actions
        actions = _array(actions, np.int64, ((self.count,), (self.count, self.players)), "actions")
        if active is not None:
            active = _array(active, np.bool_, ((self.count,),), "active")
        with self._lock:
            self._check_open()
            self._lib.baseline_batch_result(self._handle, moves.ctypes.data, actions.ctypes.data,
                                           1 if actions.ndim == 1 else self.players,
                                           None if active is None else active.ctypes.data)

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("baseline batch is closed; reset it before use")

    def close(self):
        with self._lock:
            finalizer = getattr(self, "_finalizer", None)
            if finalizer is not None:
                finalizer()
            self._handle = None
            for controller in self.controllers:
                controller.close()
            self.controllers = []

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
