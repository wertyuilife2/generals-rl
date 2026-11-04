"""Compatibility names for the native CPU engine; no Python rule implementation."""
from generals.backends.cpu.engine import CPUEngine


class CoreEngine(CPUEngine):
    """Legacy mutable-state API, delegated exclusively to the C++ backend.

    Legacy callers may edit arrays or scalar fields between calls. Invalidate
    cached totals at that public boundary; Arena leases the backend directly.
    """
    def step(self, actions):
        self.invalidate_cache()
        return super().step(actions)

    def surrender(self, pid):
        self.invalidate_cache()
        return super().surrender(pid)


def surrender(state, pid):
    engine = CPUEngine(state)
    try:
        engine.surrender(pid)
    finally:
        engine.close()


def neutralize(state, pid):
    engine = CPUEngine(state)
    try:
        engine.neutralize(pid)
    finally:
        engine.close()
