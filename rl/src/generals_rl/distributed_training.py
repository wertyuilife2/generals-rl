"""Small torchrun lifecycle and collective helpers for the training entry point.

Environment counts stay per rank. All ranks call the control/timing helpers at
the same update boundary; only rank zero creates files or chooses whether to
stop. This module does not own the rollout or PPO update loop.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class DistributedConfig:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device | str = "cpu"

    def __post_init__(self):
        _nonnegative_integer(self.rank, "rank")
        _nonnegative_integer(self.local_rank, "local_rank")
        _nonnegative_integer(self.world_size, "world_size")
        if self.world_size < 1 or self.rank >= self.world_size or self.local_rank >= self.world_size:
            raise ValueError("rank and local_rank must be smaller than positive world_size")
        object.__setattr__(self, "device", torch.device(self.device))

    @classmethod
    def from_env(cls, device, environ=None):
        environ = os.environ if environ is None else environ
        try:
            world_size = int(environ.get("WORLD_SIZE", "1"))
            rank = int(environ.get("RANK", "0"))
            local_rank = int(environ.get("LOCAL_RANK", "0"))
        except (TypeError, ValueError) as error:
            raise ValueError("WORLD_SIZE, RANK and LOCAL_RANK must be integers") from error
        device = torch.device(device)
        if world_size > 1 and device.type == "cuda":
            device = torch.device("cuda", local_rank)
        return cls(rank=rank, local_rank=local_rank, world_size=world_size, device=device)

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def multi_gpu_cfg(self):
        if not self.is_distributed:
            return None
        return {"global_rank": self.rank, "local_rank": self.local_rank, "world_size": self.world_size}

    def rank_seed(self, base_seed):
        """Separate rank streams while retaining rank zero's usual seed."""
        _nonnegative_integer(base_seed, "base_seed")
        return (base_seed + self.rank * 1_000_003) % (2 ** 32)

    def global_transitions(self, num_envs, ticks):
        """Count new learner transitions from equal per-rank environment shards."""
        return (_nonnegative_integer(num_envs, "num_envs")
                * _nonnegative_integer(ticks, "ticks") * self.world_size)

    def _verify_group(self):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("initialize the distributed process group before using training collectives")
        if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
            raise ValueError("existing process group rank/world_size does not match DistributedConfig")

    def _collective_device(self):
        self._verify_group()
        return self.device if dist.get_backend() == "nccl" else torch.device("cpu")

    def broadcast_control(self, control=None):
        """Broadcast rank zero's small control dictionary to every rank."""
        if not self.is_distributed:
            return control
        device = self._collective_device()
        values = [control if self.is_main else None]
        dist.broadcast_object_list(values, src=0, device=device)
        return values[0]

    def max_timings(self, *seconds):
        """Report each phase's slowest rank rather than summing worker times."""
        if not self.is_distributed:
            return tuple(float(value) for value in seconds)
        values = torch.tensor(seconds, dtype=torch.float64, device=self._collective_device())
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        return tuple(values.tolist())

    def sum_counts(self, *counts):
        counts = tuple(_nonnegative_integer(value, "count") for value in counts)
        if not self.is_distributed:
            return counts
        values = torch.tensor(counts, dtype=torch.int64, device=self._collective_device())
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return tuple(values.tolist())

    def all_true(self, condition):
        """Agree on rank-local health checks before rank zero chooses control."""
        if not self.is_distributed:
            return bool(condition)
        value = torch.tensor(int(bool(condition)), dtype=torch.int32, device=self._collective_device())
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def wait_on_cpu(self):
        """Wait for rank zero's background work without a spinning NCCL kernel.

        All ranks call this before returning to training collectives after
        synchronous evaluation, including initialization and final queue drains.
        """
        if not self.is_distributed:
            return
        self._verify_group()
        # Draining a queue can outlast the training collective timeout. Use a
        # separate CPU group even when the learner itself runs with Gloo.
        group = dist.new_group(backend="gloo", timeout=timedelta(days=1))
        try:
            dist.barrier(group=group)
        finally:
            dist.destroy_process_group(group)


@contextmanager
def distributed_process_group(config, *, backend=None, init_method=None, timeout_seconds=1800):
    """Initialize once and clean up only a group created by this context.

    Select the local CUDA device before NCCL initialization. Borrowed groups
    remain alive for their owner. Cleanup has no barrier, so one failed worker
    does not require its peers to reach another collective during unwinding.
    """
    if config.is_distributed and config.device.type == "cuda":
        torch.cuda.set_device(config.device)
    if dist.is_available() and dist.is_initialized():
        config._verify_group()
        if backend is not None and dist.get_backend() != backend:
            raise ValueError("existing process group backend does not match requested backend")
        yield config
        return
    if not config.is_distributed:
        yield config
        return
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    dist.init_process_group(backend=backend or ("nccl" if config.device.type == "cuda" else "gloo"),
                            init_method=init_method or "env://", rank=config.rank,
                            world_size=config.world_size, timeout=timedelta(seconds=timeout_seconds))
    try:
        yield config
    finally:
        exception_in_flight = sys.exc_info()[0] is not None
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                if not exception_in_flight:
                    raise
