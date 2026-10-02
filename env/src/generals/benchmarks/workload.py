"""Prepared pre-tick random decisions. Preparation is never simulation timing."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

WORKLOAD = "pre_tick_owned_mobile_step_v2"


def array_digest(arrays):
    result = sha256()
    for name, value in sorted(arrays.items()):
        result.update(json.dumps([name, value.dtype.str, list(value.shape)]).encode())
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def action_digest(ticks):
    result = sha256()
    for actions in ticks:
        result.update(actions.astype("<i8", copy=False).tobytes(order="C"))
    return result.hexdigest()


def host(value):
    return value if isinstance(value, np.ndarray) else value.cpu().numpy()


def snapshot(env):
    raw = getattr(env, "unwrapped", env)
    return {**{name: host(value) for name, value in raw.state.items()},
            **{name: host(getattr(raw, name)) for name in
               ("stats", "counters", "last_moves", "done", "errors", "rng_state")}}


@lru_cache(maxsize=1)
def native_tools():
    """Private temporary build; reuse the production ABI and actual rule source."""
    from generals.backends.cpu.env import _library, _Batch, _Outputs
    _, metadata = _library()
    source = Path(__file__).with_name("native.cpp")
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="generals-step-tools-") as directory:
        target = Path(directory) / "tools.so"
        result = subprocess.run([compiler, *metadata["compiler_flags"], str(source), "-o", str(target)],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"step tooling compilation failed:\n{result.stderr}")
        lib = ctypes.CDLL(str(target))
    lib.generals_step.argtypes = [ctypes.POINTER(_Batch), ctypes.POINTER(_Batch),
        ctypes.POINTER(_Outputs), ctypes.c_void_p, *([ctypes.c_int] * 6), ctypes.c_int64]
    lib.generals_step.restype = None
    lib.generals_sample_actions.argtypes = [ctypes.POINTER(_Batch), ctypes.c_void_p,
                                           ctypes.c_void_p, *([ctypes.c_int] * 5)]
    lib.generals_sample_actions.restype = None
    lib.generals_profile_begin.argtypes = [ctypes.c_int] * 3
    lib.generals_profile_begin.restype = ctypes.POINTER(ctypes.c_double)
    return lib


@dataclass
class PreparedWorkload:
    initial: dict
    history: np.ndarray  # contiguous [tick, environment, player], one host copy
    fingerprints: dict
    tick_kinds: list[str]
    steps: int


def prepare_workload(case, steps, warmup):
    from generals.protocol.config import _integer
    steps, warmup = _integer("steps", steps, 1), _integer("warmup", warmup, 0)
    from generals.core.batch import generate_states, pack_states
    from generals.backends.cpu import CPUEnv
    initial, width, height = pack_states(generate_states(case.map_config, case.batch_size, seed=case.seed))
    history = np.empty((max(steps, warmup), case.batch_size, case.map_config.players), np.int64)
    rng = np.array([int(np.random.SeedSequence([case.seed, lane, 2]).generate_state(
        1, dtype=np.uint32)[0]) or 1 for lane in range(case.batch_size)], np.uint32)
    lib = native_tools()
    env = CPUEnv(initial, width=width, height=height, threads=case.threads,
                 auto_reset=True, episode_ticks=case.episode_ticks, seed=case.seed)
    kinds, expected = [], None
    try:
        for tick, actions in enumerate(history):
            next_tick = env.state["tick"] + 1
            lib.generals_sample_actions(ctypes.byref(env._state), rng.ctypes.data,
                actions.ctypes.data, case.batch_size, height, width, env.players, env.threads)
            env.step(actions)
            env.check_errors()
            if tick < steps:
                kinds.append("reset" if env.done.any() else "cold_start" if tick == 0 else
                             "territory_growth" if np.any(next_tick % 50 == 0) else
                             "structure_growth" if np.any(next_tick % 2 == 0) else "move_only")
            if tick + 1 == steps:
                expected = array_digest(snapshot(env))
    finally:
        env.close()
    return PreparedWorkload(initial, history, {
        "initial_state_sha256": array_digest(initial),
        "action_history_sha256": action_digest(history[:steps]),
        "expected_final_state_sha256": expected}, kinds, steps)
