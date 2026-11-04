"""A shared CPU budget for bulk matches, including independent Arena commands."""
from contextlib import contextmanager
import fcntl
from functools import wraps
import os
from pathlib import Path
import tempfile
import threading


def cpu_capacity():
    available = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    # Keep two CPUs available for the live game/server and normal desktop work.
    return max(1, available - 2)


@contextmanager
def cpu_slot(*, directory=None, capacity=None):
    """Lease one CPU while computing; the OS releases it on process death.

    Wait outside SQLite transactions and process lifecycle locks. A separate
    open file description per caller also excludes other threads in this process.
    """
    capacity = cpu_capacity() if capacity is None else capacity
    if type(capacity) is not int or capacity < 1:
        raise ValueError('CPU capacity must be a positive integer')
    directory = Path(directory) if directory is not None else Path(tempfile.gettempdir()) / f'generals-arena-cpu-{os.getuid()}'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = [open(directory / f'{index}.lock', 'a') for index in range(capacity)]
    selected = None
    offset = (os.getpid() + threading.get_native_id()) % capacity
    try:
        while selected is None:
            for index in range(capacity):
                stream = files[(index+offset) % capacity]
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                selected = stream
                break
            if selected is None:
                threading.Event().wait(.05)
        yield
    finally:
        if selected is not None:
            fcntl.flock(selected, fcntl.LOCK_UN)
        for stream in files:
            stream.close()


def cpu_bounded(function):
    @wraps(function)
    def run(*args, **kwargs):
        # CUDA compute has its own resource budget; do not hold a CPU lease for it.
        if kwargs.get('backend', 'cpu') != 'cpu':
            return function(*args, **kwargs)
        with cpu_slot():
            return function(*args, **kwargs)
    return run
