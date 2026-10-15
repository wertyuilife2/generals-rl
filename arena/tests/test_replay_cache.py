import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import Mock, patch

import pytest

from generals_arena.replay_cache import ReplayResponses


def test_encoded_replay_cache_invalidates_result_and_trace():
    store = Mock()
    store.replay.return_value = ({'ticks': 1, 'rating_changes': {'bot': 2}}, b'trace')
    cache = ReplayResponses(store)
    with patch('generals_arena.replay_cache.replay_frames', return_value=[{'tick': 0}, {'tick': 1}]) as render:
        first = cache.get('game')
        assert cache.get('game') == first
        assert render.call_count == 1
        assert len(json.loads(gzip.decompress(first))['frames']) == 2
        store.replay.return_value = ({'ticks': 1, 'rating_changes': {'bot': 3}}, b'trace')
        assert json.loads(gzip.decompress(cache.get('game')))['result']['rating_changes']['bot'] == 3
        # A changed/corrupt trace must be checked again, even with the same ID.
        store.replay.return_value = (store.replay.return_value[0], b'corrupt')
        render.side_effect = ValueError('replay diverged')
        for _ in range(2):
            with pytest.raises(ValueError, match='diverged'):
                cache.get('game')
        assert render.call_count == 4


def test_encoded_replay_cache_is_bounded_and_evicts_old_entries():
    store = Mock()
    store.replay.return_value = ({'ticks': 0}, b'trace')
    cache = ReplayResponses(store)
    with patch('generals_arena.replay_cache.replay_frames', return_value=[{'tick': 0}]) as render:
        cache.max_bytes = len(cache.get('a'))
        cache.get('b')
        cache.get('b')
        assert render.call_count == 2
        cache.get('a')
        assert render.call_count == 3
        assert cache.size <= cache.max_bytes
    oversized = ReplayResponses(store, max_bytes=1)
    with patch('generals_arena.replay_cache.replay_frames', return_value=[{'tick': 0}]) as render:
        assert oversized.get('a') == oversized.get('a')
        assert render.call_count == 2 and oversized.size == 0


def test_concurrent_opens_share_replay_verification():
    store, barrier, entered, release = Mock(), Barrier(4), Event(), Event()
    def read(_):
        barrier.wait(timeout=5)
        return {'ticks': 0}, b'trace'
    def render(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return [{'tick': 0}]
    store.replay.side_effect = read
    cache = ReplayResponses(store)
    with patch('generals_arena.replay_cache.replay_frames', side_effect=render) as renderer:
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(cache.get, 'game') for _ in range(4)]
            try:
                assert entered.wait(5)
            finally:
                release.set()
            assert len({task.result(timeout=5) for task in tasks}) == 1
        assert renderer.call_count == 1
