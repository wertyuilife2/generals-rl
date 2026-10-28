"""Lazy CUDA C++ loader with a small, torch-ABI-independent launch interface.

Compilation is cached outside the source tree and excluded from benchmarks.
Only tensor pointers cross ctypes; launch uses the caller's current CUDA stream.
No state, action, or diagnostic tensor is copied to the host by ``launch``.
"""
import ctypes
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

import torch

from generals.protocol.config import _integer
from generals.backends._abi import Batch
from generals.backends._build import source_bytes
from generals.protocol import STATE_DTYPES


_POINTERS = (
    "actions", "counters", "stats", "moves", "errors", "done",
)


class _Arguments(ctypes.Structure):
    _fields_ = ([("state", Batch), ("initial", Batch)]
                + [(name, ctypes.c_void_p) for name in _POINTERS]
                + [("batch", ctypes.c_int32), ("horizon", ctypes.c_int64),
                   ("auto_reset", ctypes.c_int32)]
                + [(name, ctypes.c_void_p) for name in (
                    "cache_stats", "growth_cells", "growth_count", "cache_valid", "owned_structures")])


def compiler():
    """Prefer this Python environment's toolkit over an unrelated system CUDA."""
    candidates = [os.environ.get("GENERALS_NVCC"), str(Path(sys.prefix) / "bin" / "nvcc"),
                  shutil.which("nvcc")]
    return next((str(Path(path).resolve()) for path in candidates
                 if path and Path(path).is_file() and os.access(path, os.X_OK)), None)


@lru_cache(maxsize=None)
def _library(capability, nvcc, group_threads, envs_per_block, shape, packed_metadata):
    if sys.platform != "linux":
        raise RuntimeError("the CUDA C++ loader currently supports Linux")
    source = Path(__file__).with_name("kernel.cu")
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True,
                             text=True, timeout=30).stdout.strip()
    toolkit = Path(nvcc).parent.parent
    libraries = [toolkit / "lib", toolkit / "lib64", toolkit / "targets" / "x86_64-linux" / "lib"]
    flags = ["-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", "-lineinfo",
             "--cudart=shared", f"-arch=sm_{capability[0]}{capability[1]}"]
    flags += [f"-DGENERALS_PACKED_METADATA={int(packed_metadata)}",
              f"-DGENERALS_GROUP_THREADS={group_threads}", f"-DGENERALS_ENVS_PER_BLOCK={envs_per_block}",
              f"-DGENERALS_HEIGHT={shape[0]}", f"-DGENERALS_WIDTH={shape[1]}",
              f"-DGENERALS_PLAYERS={shape[2]}"]
    for path in libraries:
        if path.is_dir():
            flags += [f"-L{path}", f"-Xlinker=-rpath,{path}"]
    digest = sha256(source_bytes(source) + json.dumps([nvcc, version, flags]).encode()).hexdigest()
    root = Path(tempfile.gettempdir()) / f"generals-cuda-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"CUDA cache must be a private directory owned by this user: {root}")
    target = root / f"{digest}.so"
    if not target.exists():
        # Atomic publication also allows two independent benchmark processes to
        # request the same compilation without loading a partially written .so.
        with tempfile.TemporaryDirectory(prefix="build-", dir=root) as scratch:
            output = Path(scratch) / "kernel.so"
            result = subprocess.run([nvcc, *flags, str(source), "-o", str(output)],
                                    capture_output=True, text=True, timeout=240)
            if result.returncode:
                raise RuntimeError(f"CUDA kernel compilation failed:\n{result.stdout}\n{result.stderr}")
            os.replace(output, target)
    lib = ctypes.CDLL(str(target))
    lib.generals_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.generals_launch.restype = ctypes.c_int
    lib.generals_error.argtypes = [ctypes.c_int]
    lib.generals_error.restype = ctypes.c_char_p
    metadata = {"cuda_source_sha256": sha256(source.read_bytes()).hexdigest(),
                "cuda_build_sha256": digest, "compiler": nvcc, "compiler_version": version,
                "compiler_flags": flags}
    return lib, metadata


def _check(lib, status):
    if status:
        raise RuntimeError(f"CUDA simulation: {lib.generals_error(status).decode()}")


class StepKernel:
    def __init__(self, env, *, group_threads, envs_per_block, packed_metadata=None):
        group_threads = _integer("group_threads", group_threads, 1)
        envs_per_block = _integer("envs_per_block", envs_per_block, 1)
        if (group_threads not in (4, 8, 16, 32, 64, 128, 256, 512, 1024) or envs_per_block < 1
                or group_threads * envs_per_block % 32
                or group_threads * envs_per_block > 1024):
            raise ValueError("invalid CUDA environment group/block dimensions")
        if group_threads > 32 and envs_per_block != 1:
            raise ValueError("groups larger than one warp require one environment per block")
        nvcc = compiler()
        if nvcc is None:
            raise RuntimeError("CUDA step requires nvcc in the Python environment")
        if packed_metadata is not None and not isinstance(packed_metadata, bool):
            raise TypeError("packed_metadata must be bool or None")
        if packed_metadata is None:
            packed_metadata = env.batch_size * env.height * env.width >= 20_000_000
        self.env = env
        self.group_threads, self.envs_per_block = group_threads, envs_per_block
        self.packed_metadata = packed_metadata
        self.lib, self.build = _library(torch.cuda.get_device_capability(env.device), nvcc,
                                       group_threads, envs_per_block,
                                       (env.height, env.width, env.players),
                                       packed_metadata)
        self.alias_stats = not env.stats.is_inference()
        self.cache_stats = env.stats if self.alias_stats else torch.empty_like(env.stats)
        capacity = env.state["army"].numel()
        if packed_metadata:
            capacity = ((capacity + 1) & ~1) + 2 * capacity
        self.growth_cells = torch.empty(capacity, dtype=torch.uint16, device=env.device)
        self.growth_count = torch.empty(env.batch_size, dtype=torch.int32, device=env.device)
        self.cache_valid = torch.zeros(env.batch_size, dtype=torch.bool, device=env.device)
        self.owned_structures = torch.empty((env.batch_size, env.players), dtype=torch.int32, device=env.device)
        self.dirty = False
        arrays = {"actions": env._wait_actions, "counters": env.counters, "stats": env.stats,
                  "moves": env.last_moves, "errors": env.errors,
                  "done": env.done, "cache_stats": self.cache_stats,
                  "growth_cells": self.growth_cells, "growth_count": self.growth_count,
                  "cache_valid": self.cache_valid, "owned_structures": self.owned_structures}
        self.args = _Arguments()
        self.args.state = Batch(*(env.state[name].data_ptr() for name in STATE_DTYPES))
        self.args.initial = Batch(*(env.initial[name].data_ptr() for name in STATE_DTYPES))
        for name, value in arrays.items():
            setattr(self.args, name, value.data_ptr())
        self.args.batch = env.batch_size
        self._args_ptr = ctypes.byref(self.args)
        self._launch = self.lib.generals_launch
        # Retain storage and fail clearly if callers replace exposed tensors.
        self._state_refs = tuple(env.state.values()) + tuple(env.initial.values())
        self._identities = tuple(id(value) for value in self._state_refs)
        self._storage = tuple((value.data_ptr(), value.shape, value.dtype) for value in self._state_refs)
        self._versioned = tuple(value for value in self._state_refs if not value.is_inference())
        if self.alias_stats:
            self._versioned += (env.stats,)
        self._versions = self._state_versions()
        from generals.backends.gpu.host import bind
        module, self.host_build = bind(self)
        self.native_step = module.step

    def _state_versions(self):
        return tuple(value._version for value in self._versioned)

    def prepare(self):
        env = self.env
        if tuple(map(id, env.state.values())) + tuple(map(id, env.initial.values())) != self._identities:
            raise ValueError("edit state tensors in place; replacing their storage is unsupported")
        versions = self._state_versions()
        if self.dirty or versions != self._versions:
            if any((v.data_ptr(), v.shape, v.dtype) != layout for v, layout in zip(self._state_refs, self._storage)):
                raise ValueError("state tensors must retain their storage, shape and dtype")
            self.cache_valid.zero_()
            self.dirty = False
            self._versions = versions
        self.args.auto_reset, self.args.horizon = env.auto_reset, env.episode_ticks

    def metadata(self):
        resources = (ctypes.c_int * 4)()
        with torch.cuda.device(self.env.device):
            _check(self.lib, self._launch(self._args_ptr, None, resources))
        threads = self.group_threads * self.envs_per_block
        props = torch.cuda.get_device_properties(self.env.device)
        return {**self.build, "host_build": self.host_build, "threads_per_environment": self.group_threads,
                "environments_per_block": self.envs_per_block, "threads_per_block": threads,
                "aliases_output_statistics": self.alias_stats,
                "packed_metadata": self.packed_metadata,
                "registers_per_thread": resources[0], "local_memory_bytes_per_thread": resources[1],
                "shared_memory_bytes_per_block": resources[2], "resident_blocks_per_sm": resources[3],
                "theoretical_occupancy": resources[3] * threads / props.max_threads_per_multi_processor}
