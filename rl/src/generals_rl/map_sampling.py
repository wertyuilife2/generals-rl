"""RL-owned map banks with independent sampling and native selected-lane reset."""
import numpy as np
import torch

from generals.core.mapgen import generate_map
from generals.protocol import STATE_DTYPES


def _replace_initial(backend, rows=None, arrays=None, stats=None, rng=None):
    """The sole adapter to private reset caches; never replace bound storage.

    A call without replacements validates compatibility before bank generation.
    The environment's public reset then clears diagnostics and invalidates its
    native growth/statistics cache for the replaced initial maps.
    """
    required = ("initial", "_initial_stats", "_initial_rng")
    if arrays is None:
        expected = {"terrain", "structure", "owner", "army", "alive", "general_pos", "tick", "terminated", "winner"}
        if set(STATE_DTYPES) != expected or set(backend.state) != expected:
            raise RuntimeError("map sampling does not support this backend state schema")
        if any(not hasattr(backend, name) for name in required):
            raise RuntimeError("map sampling needs backend initial/_initial_stats/_initial_rng reset caches")
        initial, initial_stats, initial_rng = backend.initial, backend._initial_stats, backend._initial_rng
        tensor = isinstance(backend.state["army"], torch.Tensor)
        value_type = torch.Tensor if tensor else np.ndarray
        if (not isinstance(initial, dict) or set(initial) != set(STATE_DTYPES)
                or not isinstance(initial_stats, value_type) or not isinstance(initial_rng, value_type)):
            raise RuntimeError("backend reset cache structure is incompatible with map sampling")
        compatible = (initial_stats.shape == (backend.batch_size, backend.players, 3)
                      and initial_rng.shape == (backend.batch_size,))
        if tensor:
            compatible &= initial_stats.dtype == torch.int64 and initial_rng.dtype == torch.uint32
            compatible &= initial_stats.device == initial_rng.device == backend.state["army"].device
            compatible &= all(isinstance(value, torch.Tensor)
                              and value.shape == backend.state[name].shape and value.dtype == backend.state[name].dtype
                              and value.device == backend.state[name].device
                              for name, value in initial.items())
        else:
            compatible &= initial_stats.dtype == np.int64 and initial_rng.dtype == np.uint32
            compatible &= initial_stats.flags.writeable and initial_rng.flags.writeable
            compatible &= all(isinstance(value, np.ndarray) and value.flags.writeable
                              and value.shape == backend.state[name].shape and value.dtype == backend.state[name].dtype
                              for name, value in initial.items())
        if not compatible:
            raise RuntimeError("backend reset cache shape/dtype is incompatible with map sampling")
        return
    if isinstance(rows, torch.Tensor):
        for name, value in arrays.items():
            backend.initial[name].index_copy_(0, rows, value)
        backend._initial_stats.index_copy_(0, rows, stats)
        # uint32 indexed operations are unsupported in some torch builds.
        backend._initial_rng.view(torch.int32).index_copy_(0, rows, rng)
    else:
        for name, value in arrays.items():
            backend.initial[name][rows] = value
        backend._initial_stats[rows] = stats
        backend._initial_rng[rows] = rng


def _build_bank(config, size, seed):
    cells, players = config.width * config.height, config.players
    arrays = {}
    for name, dtype in STATE_DTYPES.items():
        shape = (size, cells) if name in ("terrain", "structure", "owner", "army") else (
            (size, players) if name in ("alive", "general_pos") else (size,))
        arrays[name] = np.empty(shape, dtype=dtype)
    seeds = np.random.SeedSequence([seed, 0x4D4150]).generate_state(size, dtype=np.uint64)
    for index, map_seed in enumerate(seeds):
        state = generate_map(config, int(map_seed))
        for name in ("structure", "owner", "army", "alive", "general_pos"):
            arrays[name][index] = getattr(state, name).ravel()
        arrays["terrain"][index] = state.layout.terrain.ravel()
    arrays["tick"].fill(0)
    arrays["terminated"].fill(False)
    arrays["winner"].fill(-1)
    stats = np.empty((size, players, 3), dtype=np.int64)
    for pid in range(players):
        owned = arrays["owner"] == pid
        stats[:, pid, 0] = np.where(owned, arrays["army"], 0).sum(axis=1)
        stats[:, pid, 1] = owned.sum(axis=1)
        stats[:, pid, 2] = (owned & (arrays["structure"] == 1)).sum(axis=1)
    rng = np.random.SeedSequence([seed, 0x524E47]).generate_state(size, dtype=np.uint32)
    rng[rng == 0] = 1
    return arrays, stats, rng


class MapSampler:
    """Uniform replacement from a pre-generated bank, independently of learner RNG.

    Native initial/current storage is owned by the backend. Bank selections are
    copied only into requested reset lanes, so active games keep their maps.
    """
    def __init__(self, env, config, size, *, seed):
        self.backend = getattr(env, "unwrapped", env)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("map bank size must be a positive integer")
        if self.backend.auto_reset:
            raise ValueError("map sampling requires auto_reset=False")
        if (self.backend.width, self.backend.height, self.backend.players) != (config.width, config.height, config.players):
            raise ValueError("map bank and backend dimensions must match")
        _replace_initial(self.backend)
        self.size = size
        self.tensor = isinstance(self.backend.state["army"], torch.Tensor)
        self.arrays, self.stats, self.rng = _build_bank(config, size, seed)
        if self.tensor:
            self.device = self.backend.state["army"].device
            self.arrays = {name: torch.from_numpy(value).to(self.device, copy=True)
                           for name, value in self.arrays.items()}
            self.stats = torch.from_numpy(self.stats).to(self.device, copy=True)
            self.rng = torch.from_numpy(self.rng.view(np.int32)).to(self.device, copy=True)
            stream_seed = int(np.random.SeedSequence([seed, 0x53414D]).generate_state(1, dtype=np.uint64)[0])
            self.generator = torch.Generator(device=self.device).manual_seed(stream_seed)
        else:
            self.generator = np.random.default_rng(np.random.SeedSequence([seed, 0x53414D]))

    def reset(self, mask=None):
        backend = self.backend
        if self.tensor:
            if mask is not None and (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != (backend.batch_size,) or mask.device != self.device):
                raise ValueError("map sampling reset mask must be bool [batch] on backend device")
            rows = torch.arange(backend.batch_size, device=self.device) if mask is None else torch.where(mask)[0]
            if not rows.numel():
                return backend.state
            selected = torch.randint(self.size, (rows.numel(),), generator=self.generator, device=self.device)
            arrays = {name: value.index_select(0, selected) for name, value in self.arrays.items()}
            stats, rng = self.stats.index_select(0, selected), self.rng.index_select(0, selected)
        else:
            if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != bool
                                     or mask.shape != (backend.batch_size,)):
                raise ValueError("map sampling reset mask must be bool [batch]")
            rows = np.arange(backend.batch_size) if mask is None else np.flatnonzero(mask)
            if not len(rows):
                return backend.state
            selected = self.generator.integers(self.size, size=len(rows))
            arrays = {name: value[selected] for name, value in self.arrays.items()}
            stats, rng = self.stats[selected], self.rng[selected]
        _replace_initial(backend, rows, arrays, stats, rng)
        return backend.reset(mask)
