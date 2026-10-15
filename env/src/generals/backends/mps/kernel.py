"""Shape-specialized Metal kernels; compilation is outside the timed hot path."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import torch

from generals.protocol.config import _integer
from .env import field_major, pack_tensors


@lru_cache(maxsize=32)
def library(width, height, players, batch, layouts):
    source = Path(__file__).with_name("kernel.metal").read_text()
    preamble = f"#define W {width}\n#define H {height}\n#define P {players}\n#define B {batch}\n"
    for prefix, offsets in layouts:
        preamble += "".join(f"#define OFF_{prefix}_{name.upper()} {offset}\n" for name, offset in offsets)
    code = preamble + source
    return torch.mps.compile_shader(code), sha256(code.encode()).hexdigest()


class StepKernel:
    def __init__(self, env, *, threads_per_group=128):
        self.env = env
        self.cache = field_major(torch.zeros_like(env.stats))
        self.growth = torch.empty((env.width * env.height, env.batch_size), dtype=torch.uint16, device=env.device)
        # Growth count, cache validity, owned count, per-player structures.
        self.aux = torch.zeros((3 + env.players, env.batch_size), dtype=torch.int32, device=env.device)
        self.owned = torch.empty((env.batch_size, env.width * env.height), dtype=torch.uint16, device=env.device)
        self.storage, views, offsets = pack_tensors({
            "totals": self.cache, "growth": self.growth, "aux": self.aux, "owned": self.owned})
        self.cache, self.growth, self.aux, self.owned = (views[k] for k in ("totals", "growth", "aux", "owned"))
        self.lib, self.digest = library(env.width, env.height, env.players, env.batch_size,
            (("STATE", env._state_offsets), ("INITIAL", env._initial_offsets),
             ("OUTPUT", env._output_offsets), ("CACHE", offsets)))
        self.function = self.lib.step
        self.configure_group(threads_per_group)
        self.refs = tuple(env.state.values()) + tuple(env.initial.values())
        self.identities = tuple(map(id, self.refs))
        self.layout = tuple((v.data_ptr(), v.shape, v.stride(), v.dtype) for v in self.refs)
        self.versioned = tuple(v for v in (env._state_storage, env._initial_storage) if not v.is_inference())
        self.versions = tuple(v._version for v in self.versioned)
        self.dirty = False
        self.args = (env._state_storage, env._initial_storage, env._output_storage, self.storage)

    def configure_group(self, threads_per_group):
        value = _integer("threads_per_group", threads_per_group, 1)
        if value > self.function.max_threads_per_threadgroup:
            raise ValueError("threads_per_group exceeds the Metal kernel limit")
        self.threads_per_group = value

    def launch(self, actions):
        env = self.env
        if tuple(map(id, env.state.values())) + tuple(map(id, env.initial.values())) != self.identities:
            raise ValueError("edit state tensors in place; replacing their storage is unsupported")
        versions = tuple(v._version for v in self.versioned)
        dirty = self.dirty or versions != self.versions
        if dirty:
            if any((v.data_ptr(), v.shape, v.stride(), v.dtype) != layout for v, layout in zip(self.refs, self.layout)):
                raise ValueError("state tensors must retain their storage, shape, stride and dtype")
        flags = int(env.auto_reset) | (int(dirty) << 1)
        self.function(*self.args, actions, env.episode_ticks, flags,
                      threads=env.batch_size, group_size=self.threads_per_group)
        self.versions, self.dirty = versions, False

    def metadata(self):
        return {"source_sha256": self.digest, "threads_per_group": self.threads_per_group,
                "thread_execution_width": self.function.thread_execution_width,
                "integer_bits": 64}
