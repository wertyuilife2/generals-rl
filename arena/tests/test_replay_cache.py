import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import Mock, patch

import pytest

from generals_arena.web.replays import ReplayResponses


def test_encoded_replay_cache_invalidates_result_and_trace():
    store = Mock()
    store.replay.return_value = ({'ticks': 1, 'rating_changes': {'bot': 2}}, b'trace')
    cache = ReplayResponses(store)
    with patch('generals_arena.web.replays.replay_frames', return_value=[{'tick': 0}, {'tick': 1}]) as render:
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
    with patch('generals_arena.web.replays.replay_frames', return_value=[{'tick': 0}]) as render:
        cache.max_bytes = len(cache.get('a'))
        cache.get('b')
        cache.get('b')
        assert render.call_count == 2
        cache.get('a')
        assert render.call_count == 3
        assert cache.size <= cache.max_bytes
    oversized = ReplayResponses(store, max_bytes=1)
    with patch('generals_arena.web.replays.replay_frames', return_value=[{'tick': 0}]) as render:
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
    with patch('generals_arena.web.replays.replay_frames', side_effect=render) as renderer:
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(cache.get, 'game') for _ in range(4)]
            try:
                assert entered.wait(5)
            finally:
                release.set()
            assert len({task.result(timeout=5) for task in tasks}) == 1
        assert renderer.call_count == 1


def test_http_replay_uses_same_payload_with_and_without_gzip(tmp_path):
    import threading
    from urllib.request import Request, urlopen
    from test_league import league, add
    from generals_arena.competitions.jobs import run_jobs
    from generals_arena.web.server import make_server

    store = league(tmp_path)
    add(store, games=1)
    run_jobs(store)
    game_id = store.matches()['matches'][0]['game_id']
    server = make_server(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}/api/replay/{game_id}'
        with urlopen(url) as response:
            assert response.headers.get('Content-Encoding') is None
            plain = response.read()
        with urlopen(Request(url, headers={'Accept-Encoding': 'gzip'})) as response:
            assert response.headers['Content-Encoding'] == 'gzip'
            assert gzip.decompress(response.read()) == plain
        assert json.loads(plain)['frames']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
