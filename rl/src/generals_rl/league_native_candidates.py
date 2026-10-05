"""Isolated original-AggressiveAI candidate backend; original RNG stays NumPy."""
import ctypes
from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np

from generals.controllers.base import BaseAI, Candidate, softmax_choice


class NativeCandidate(ctypes.Structure):
    _fields_ = [("source", ctypes.c_int), ("target", ctypes.c_int),
                ("strategy", ctypes.c_int), ("score", ctypes.c_double)]


def load_library():
    source = Path(__file__).with_name("candidates.cpp")
    digest = sha256(source.read_bytes()).hexdigest()[:16]
    target = Path("/tmp") / f"generals_aggressive_candidates_{digest}.so"
    if not target.exists():
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off",
                        str(source), "-o", str(target)], check=True, capture_output=True)
    library = ctypes.CDLL(str(target))
    pointer = ctypes.c_void_p
    library.prepare_sources.argtypes = [pointer] * 4 + [ctypes.c_int] + [pointer] * 3
    library.prepare_sources.restype = ctypes.c_int
    library.build_candidates.argtypes = ([pointer] * 6 + [ctypes.c_int, pointer, ctypes.c_int]
                                         + [pointer] * 3)
    library.build_candidates.restype = ctypes.c_int
    return library


class NativeCandidateBackend:
    """Two small native calls per replan, not a full batch/GPU controller.

    Original source-choice softmax, PCG64, act/path/memory/on_result all remain
    original. This backend only supports unchanged default 9x9 AggressiveAI;
    other settings fall back to the original method.
    """
    def __init__(self):
        self.library = load_library()
        self.original = BaseAI._candidates
        self.native_calls = self.fallback_calls = 0

    def candidates(self, controller, obs):
        if (obs.width, obs.height) != (9, 9) or len(obs.alive) != 2 or controller.player_id not in (0, 1) \
                or tuple(controller.weights.items()) != (("merge", 0.5), ("explore", 0.8), ("attack", 1.5)) \
                or controller.max_sources != 6:
            self.fallback_calls += 1
            return self.original(controller, obs)
        names = (obs.owner, obs.army, controller.owner, controller.army,
                 controller.terrain, controller.structure)
        dtypes = (np.int16, np.int64, np.int16, np.int64, np.uint8, np.uint8)
        if any(a.shape != (9, 9) or a.dtype != dtype or not a.flags.c_contiguous
               for a, dtype in zip(names, dtypes)):
            raise ValueError("native candidate arrays must follow original contiguous 9x9 dtype contract")
        owner, army, memory_owner, memory_army, terrain, structure = names
        sources = np.empty(6, dtype=np.int32)
        frontiers, threats = np.empty(81, dtype=np.int64), np.empty(81, dtype=np.int64)
        count = self.library.prepare_sources(owner.ctypes.data, army.ctypes.data,
                                            memory_owner.ctypes.data, terrain.ctypes.data,
                                            controller.player_id, sources.ctypes.data,
                                            frontiers.ctypes.data, threats.ctypes.data)
        if count > 1:
            chosen = softmax_choice(controller.rng, [np.log1p(float(army.ravel()[s]))
                                                     for s in sources[:count]], controller.temperature)
            order = sources[:count].tolist()
            order.insert(0, order.pop(chosen))
            sources[:count] = order
        output = (NativeCandidate * 72)()
        found = self.library.build_candidates(owner.ctypes.data, army.ctypes.data,
                                             memory_owner.ctypes.data, memory_army.ctypes.data,
                                             terrain.ctypes.data, structure.ctypes.data,
                                             controller.player_id, sources.ctypes.data, count,
                                             frontiers.ctypes.data, threats.ctypes.data, output)
        self.native_calls += 1
        strategies = ("merge", "explore", "attack")
        return [Candidate(row.source, row.target, strategies[row.strategy], row.score)
                for row in output[:found]]

    def method(self):
        backend = self
        def selected(controller, obs):
            return backend.candidates(controller, obs)
        return selected
