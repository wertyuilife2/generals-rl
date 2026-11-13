"""Bounded worker resources; mutable policy memory is reset for every match."""
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from generals.core.mapgen import generate_map


@lru_cache(maxsize=256)
def initial_map(config, seed):
    state = generate_map(config, seed)
    # The backend packs independent storage. Cached maps are never live states.
    for value in vars(state).values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return state


class RuntimeCache(OrderedDict):
    """Retain resettable controllers, evicting only between execution leases."""
    def __init__(self):
        super().__init__()
        self.resources = OrderedDict()

    def trim(self, capacity=32):
        while len(self) > capacity:
            _, controller = self.popitem(last=False)
            close = getattr(controller, 'close', None)
            if close is not None:
                close()

    def close(self):
        self.trim(0)
        self.resources.clear()
