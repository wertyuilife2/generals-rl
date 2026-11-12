"""Bounded deterministic prefetch of fresh CPU map chunks; no CUDA or disk cache."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from time import perf_counter

import numpy as np

from .map_bank import build_bank


class MapStream:
    def __init__(self, config, size, seed, *, width, height, curriculum, workers, chunk_size, reuse):
        self.config, self.size, self.seed, self.reuse = config, size, seed, reuse
        self.options = dict(width=width, height=height, curriculum=curriculum)
        self.spans = [(start, min(chunk_size, size-start)) for start in range(0, size, chunk_size)]
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))
        self.pending, self.submitted, self.draws = deque(), 0, 0
        self.wait_seconds, self.introduced, self.replacements = 0., 0, 0
        for _ in range(workers):
            self._submit()

    def _submit(self):
        start, count = self.spans[self.submitted % len(self.spans)]
        seed = int(np.random.SeedSequence([self.seed, 0x5354524D, self.submitted]).generate_state(1, dtype=np.uint64)[0])
        self.pending.append((start, count, self.pool.submit(build_bank, self.config, count, seed, **self.options)))
        self.submitted += 1

    def consume(self, count):
        self.draws += count
        while self.draws >= self.pending[0][1]*self.reuse:
            start, size, future = self.pending.popleft()
            began = perf_counter()
            bank = future.result()  # Deliberate deterministic backpressure.
            self.wait_seconds += perf_counter()-began
            self.draws -= size*self.reuse
            self.introduced += size
            self.replacements += 1
            self._submit()
            yield start, size, bank

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
