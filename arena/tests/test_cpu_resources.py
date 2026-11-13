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


def test_slots_reuse_descriptors_and_open_only_needed_files(tmp_path, monkeypatch):
    from generals_arena.matches import resources
    import builtins
    opened = []
    original = builtins.open
    def tracked(*args, **kwargs):
        stream = original(*args, **kwargs)
        opened.append(stream)
        return stream
    monkeypatch.setattr(resources, 'open', tracked, raising=False)
    for _ in range(20):
        with cpu_slot(directory=tmp_path, capacity=4):
            pass
    assert len(opened) == 1
    with cpu_slot(directory=tmp_path, capacity=4):
        with cpu_slot(directory=tmp_path, capacity=4):
            assert len(opened) == 2
    assert all(not stream.closed for stream in opened)


def test_threads_respect_shared_capacity_and_release_wakes_waiter(tmp_path):
    acquired = threading.Event()
    release = threading.Event()
    def waiting():
        with cpu_slot(directory=tmp_path, capacity=1):
            acquired.set()
            assert release.wait(3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with cpu_slot(directory=tmp_path, capacity=1):
            future = pool.submit(waiting)
            assert not acquired.wait(.05)
        try:
            assert acquired.wait(2)
        finally:
            release.set()
        future.result(timeout=3)


def test_forked_child_opens_independent_lock_descriptors(tmp_path):
    context = multiprocessing.get_context('fork')
    parent, child = context.Pipe()
    process = context.Process(target=_hold_slot, args=(tmp_path, child))
    try:
        with cpu_slot(directory=tmp_path, capacity=1):
            process.start()
            child.close()
            assert not parent.poll(.1)
        assert parent.poll(3) and parent.recv() == 'acquired'
        parent.send('release')
        process.join(3)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join(3)
        parent.close()
