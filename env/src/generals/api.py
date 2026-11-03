"""Create simulation environments without importing unused execution backends."""
from .protocol import MapConfig


def make_engine(state, *, backend='cpu'):
    """Single-game step interface; CPU shares the native VecEnv implementation.

    The reference engine remains available for independent rule verification.
    """
    if backend == 'cpu':
        from .backends.cpu.engine import CPUEngine
        return CPUEngine(state)
    if backend == 'reference':
        from .core.engine import CoreEngine
        return CoreEngine(state)
    raise ValueError('backend must be cpu or reference')


def make_env(config=None, *, mode="vector", device="cpu", **options):
    """Create a VecEnv for step simulation.

    Vector options are passed to the selected backend's from_config constructor
    (e.g. batch_size, seed, auto_reset, episode_ticks and CPU-only threads).
    Only vector mode is supported; interactive play is provided by Arena WebUI.
    """
    config = MapConfig() if config is None else config
    if not isinstance(config, MapConfig):
        raise TypeError("config must be MapConfig")
    if mode != "vector":
        raise ValueError("mode must be vector; interactive play is available in Arena WebUI")
    from .vector import VecEnv
    return VecEnv.from_config(config, device=device, **options)
