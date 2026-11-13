"""A shared CPU budget for bulk matches, including independent Arena commands."""
from contextlib import contextmanager
import fcntl
from functools import wraps, lru_cache
import os
from pathlib import Path
import tempfile
import threading
import weakref


def cpu_capacity():
    available = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    # Keep two CPUs available for the live game/server and normal desktop work.
    return max(1, available - 2)


_pools = weakref.WeakSet()


class _Slots:
    def __init__(self, directory, capacity):
        self.directory = directory
        self.files = [None] * capacity
        self.offset = os.getpid() % capacity
        self.busy = set()
        self.changed = threading.Condition()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _pools.add(self)

    def __del__(self):
        for stream in self.files:
            if stream is not None:
                stream.close()

    @contextmanager
    def lease(self):
        # Threads share descriptors, so reserve locally before exposing the lease.
        # Independent commands still exclude one another with flock.
        with self.changed:
            while True:
                for step in range(len(self.files)):
                    index = (self.offset + step) % len(self.files)
                    if index in self.busy:
                        continue
                    stream = self.files[index]
                    if stream is None:
                        stream = self.files[index] = open(self.directory / f'{index}.lock', 'a')
                    try:
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    self.busy.add(index)
                    break
                else:
                    # Local releases notify immediately; poll only for releases
                    # in another process, whose condition variable is separate.
                    self.changed.wait(None if len(self.busy) == len(self.files) else .005)
                    continue
                break
        try:
            yield
        finally:
            with self.changed:
                fcntl.flock(stream, fcntl.LOCK_UN)
                self.busy.remove(index)
                self.changed.notify()


_pool_lock = threading.Lock()


@lru_cache(maxsize=32)
def _slots(directory, capacity):
    return _Slots(directory, capacity)


def _after_fork():
    global _pool_lock
    # Never reuse inherited condition locks or flock descriptions in the child.
    _pool_lock = threading.Lock()
    for pool in list(_pools):
        for stream in pool.files:
            if stream is not None:
                stream.close()
    _slots.cache_clear()


os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def cpu_slot(*, directory=None, capacity=None):
    """Lease a CPU using reusable, lazily opened cross-process lock files."""
    capacity = cpu_capacity() if capacity is None else capacity
    if type(capacity) is not int or capacity < 1:
        raise ValueError('CPU capacity must be a positive integer')
    directory = Path(directory) if directory is not None else Path(tempfile.gettempdir()) / f'generals-arena-cpu-{os.getuid()}'
    with _pool_lock:
        slots = _slots(directory.absolute(), capacity)
    with slots.lease():
        yield


def cpu_bounded(function):
    @wraps(function)
    def run(*args, **kwargs):
        # CUDA compute has its own resource budget; do not hold a CPU lease for it.
        if kwargs.get('backend', 'cpu') != 'cpu':
            return function(*args, **kwargs)
        with cpu_slot():
            return function(*args, **kwargs)
    return run
