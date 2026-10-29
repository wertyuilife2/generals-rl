"""Common batch interface over the existing native CPU and CUDA backends.

Device selection and method binding happen once, at construction. step and
synchronize are the backend's own bound methods: there is no forwarding call,
implicit transfer or additional completion barrier in the measured hot path.
"""
from collections.abc import Callable
from typing import Any

from .protocol.vector import BatchStep


def _backend(device, backend=None):
    name = str(device)
    if backend not in (None, "cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu")
    if name == "cpu":
        if backend not in (None, "cpu"):
            raise ValueError("GPU backend requires a CUDA device")
        from .backends.cpu import CPUEnv
        return CPUEnv, {}
    if name == "cuda" or (name.startswith("cuda:") and name[5:].isdigit()):
        if backend == "cpu":
            raise ValueError("CPU backend requires device=cpu")
        from .backends.gpu import GPUEnv
        return GPUEnv, {"device": device}
    raise ValueError("device must be cpu, cuda, or cuda:N")


class VecEnv:
    """Identical batch contract with NumPy on CPU or torch tensors on CUDA.

    Common state attributes are live backend views. Use unwrapped for
    backend-specific capabilities and configuration; replacing storage remains
    unsupported. Outputs are borrowed buffers, as in the original backends.
    """
    step: Callable[[Any], BatchStep]
    reset: Callable[..., Any]
    surrender: Callable[..., None]
    synchronize: Callable[[], None]
    check_errors: Callable[[], None]
    invalidate_cache: Callable[[], None]
    to_states: Callable[..., list]
    close: Callable[[], None]

    def __init__(self, initial, *, width, height, device="cpu", backend=None, **options):
        backend, placement = _backend(device, backend)
        self._bind(backend(initial, width=width, height=height, **placement, **options))

    def _bind(self, backend):
        self.unwrapped = backend
        self.device = getattr(backend, "device", "cpu")
        for name in ("step", "reset", "surrender", "synchronize", "check_errors", "invalidate_cache",
                     "to_states", "close"):
            setattr(self, name, getattr(backend, name))

    @classmethod
    def from_config(cls, config, batch_size=1, *, device="cpu", backend=None, **options):
        backend, placement = _backend(device, backend)
        instance = cls.__new__(cls)
        instance._bind(backend.from_config(config, batch_size=batch_size, **placement, **options))
        return instance

    @classmethod
    def from_states(cls, states, *, device="cpu", backend=None, **options):
        backend, placement = _backend(device, backend)
        instance = cls.__new__(cls)
        instance._bind(backend.from_states(states, **placement, **options))
        return instance

    @property
    def num_envs(self):
        return self.unwrapped.batch_size

    @property
    def state(self):
        return self.unwrapped.state

    @property
    def stats(self):
        return self.unwrapped.stats

    @property
    def done(self):
        return self.unwrapped.done

    @property
    def batch_size(self):
        return self.unwrapped.batch_size

    @property
    def players(self):
        return self.unwrapped.players

    @property
    def width(self):
        return self.unwrapped.width

    @property
    def height(self):
        return self.unwrapped.height

    def observer(self, player_ids):
        """Bind a public on-device observation batch to fixed lane seats."""
        from .core.batch_observation import batch_observer
        return batch_observer(self, player_ids)

    def observe(self, player_ids):
        return self.observer(player_ids)()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
