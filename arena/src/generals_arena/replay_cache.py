"""Bounded cache of verified, encoded replays for the local web server."""
from collections import OrderedDict
from concurrent.futures import Future
import gzip
from hashlib import sha256
import json
from threading import Lock

from .execution import replay_frames


class ReplayResponses:
    def __init__(self, store, max_bytes=32 * 1024 * 1024):
        self.store = store
        self.max_bytes = max_bytes
        self.size = 0
        self.cache = OrderedDict()
        self.pending = {}
        self.lock = Lock()

    def get(self, game_id):
        """Return gzip JSON; changed outcomes/traces invalidate cached data.

        Keep compressed bytes rather than thousands of Python frame objects.
        Concurrent requests for the same replay share one verification pass.
        The caller must check visibility on every request, even cache hits.
        """
        result, blob = self.store.replay(game_id)
        metadata = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        key = (game_id, sha256(metadata).digest(), sha256(blob).digest())
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                return cached
            pending = self.pending.get(key)
            build = pending is None
            if build:
                pending = self.pending[key] = Future()
        if not build:
            return pending.result()
        try:
            frames = replay_frames(blob, result, compact=True)
            payload = b'{"result":' + metadata + b',"frames":' + json.dumps(
                frames, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode() + b'}'
            compressed = gzip.compress(payload, compresslevel=3, mtime=0)
            with self.lock:
                if len(compressed) <= self.max_bytes:
                    while self.cache and self.size + len(compressed) > self.max_bytes:
                        _, old = self.cache.popitem(last=False)
                        self.size -= len(old)
                    self.cache[key] = compressed
                    self.size += len(compressed)
            pending.set_result(compressed)
            return compressed
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self.lock:
                del self.pending[key]
