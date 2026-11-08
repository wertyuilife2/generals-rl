"""Bulk Arena commands share capacity without holding unrelated locks."""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from generals_arena.matches.resources import cpu_slot


def _hold_slot(directory, connection):
    with cpu_slot(directory=directory, capacity=1):
        connection.send('acquired')
        connection.recv()


def test_capacity_is_shared_across_processes_and_released_on_exit(tmp_path):
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_hold_slot, args=(tmp_path, child))
    process.start()
    child.close()
    acquired = threading.Event()
    def waiting():
        with cpu_slot(directory=tmp_path, capacity=1):
            acquired.set()
    try:
        assert parent.poll(5) and parent.recv() == 'acquired'
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiting)
            try:
                assert not acquired.wait(.15)
            finally:
                process.terminate()
                process.join(5)
            future.result(timeout=5)
            assert acquired.is_set()
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()


def test_exception_releases_a_slot_for_another_thread(tmp_path):
    with pytest.raises(RuntimeError, match='failed work'):
        with cpu_slot(directory=tmp_path, capacity=1):
            raise RuntimeError('failed work')
    with ThreadPoolExecutor(max_workers=1) as pool:
        def acquire():
            with cpu_slot(directory=tmp_path, capacity=1):
                return True
        assert pool.submit(acquire).result(timeout=3)
