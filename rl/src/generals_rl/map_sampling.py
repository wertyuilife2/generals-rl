"""RL-owned map banks with independent sampling and native selected-lane reset."""
import numpy as np
import torch

from generals.protocol import STATE_DTYPES
from .map_bank import build_bank as _build_bank, load_bank, pad_state


def _replace_initial(backend, rows=None, arrays=None, stats=None, rng=None):
    """The sole adapter to private reset caches; never replace bound storage.

    A call without replacements validates compatibility before bank generation.
    The environment's public reset then clears diagnostics and invalidates its
    native growth/statistics cache for the replaced initial maps.
    """
    required = ("initial", "_initial_stats", "_initial_rng")
    if arrays is None:
        expected = set(STATE_DTYPES)
        if set(backend.state) != expected:
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


class MapSampler:
    """Uniform replacement from a pre-generated bank, independently of learner RNG.

    Native initial/current storage is owned by the backend. Bank selections are
    copied only into requested reset lanes, so active games keep their maps.
    """
    def __init__(self, env, config, size, *, seed, curriculum=None, d4=False,
                 cache_dir=None, generation_workers=1, refresh_reuse=0, refresh_chunk_size=1024,
                 prefetch_chunks=8):
        self.backend = getattr(env, "unwrapped", env)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("map bank size must be a positive integer")
        if type(prefetch_chunks) is not int or prefetch_chunks < 1:
            raise ValueError('map prefetch chunks must be a positive integer')
        if self.backend.auto_reset:
            raise ValueError("map sampling requires auto_reset=False")
        if self.backend.players != config.players:
            raise ValueError("map bank and backend player counts must match")
        _replace_initial(self.backend)
        self.size = size
        self.curriculum = curriculum
        self.position = 0.
        self.tensor = isinstance(self.backend.state["army"], torch.Tensor)
        bank, self.bank_info = load_bank(config, size, seed, width=self.backend.width,
            height=self.backend.height, curriculum=curriculum, cache_dir=cache_dir,
            workers=min(generation_workers, max(1, (size+255)//256)), builder=_build_bank)
        self.arrays, self.stats, self.rng, self.bank_valid = bank
        self.stream = None
        self.sampled_games = 0
        self.level_ids = (np.zeros(self.backend.batch_size, dtype=np.int64)
                          if curriculum is not None else None)
        self.valid = np.ones((self.backend.batch_size, self.backend.width * self.backend.height), dtype=bool)
        if self.tensor:
            self.device = self.backend.state["army"].device
            self.arrays = {name: torch.from_numpy(value).to(self.device, copy=True)
                           for name, value in self.arrays.items()}
            self.bank_valid = torch.from_numpy(self.bank_valid).to(self.device)
            self.valid = torch.from_numpy(self.valid).to(self.device)
            if self.level_ids is not None:
                self.level_ids = torch.from_numpy(self.level_ids).to(self.device)
            self.stats = torch.from_numpy(self.stats).to(self.device, copy=True)
            self.rng = torch.from_numpy(self.rng.view(np.int32)).to(self.device, copy=True)
            stream_seed = int(np.random.SeedSequence([seed, 0x53414D]).generate_state(1, dtype=np.uint64)[0])
            self.generator = torch.Generator(device=self.device).manual_seed(stream_seed)
        else:
            self.generator = np.random.default_rng(np.random.SeedSequence([seed, 0x53414D]))
        self.symmetry = None
        if d4:
            if self.backend.width != self.backend.height or config.width is None or config.height is None:
                raise ValueError('D4 map augmentation requires fixed square maps')
            from .map_symmetry import MapSymmetry
            self.symmetry = MapSymmetry(self.backend.width, self.device if self.tensor else None)
            symmetry_seed = np.random.SeedSequence([seed, 0x4434])
            self.symmetry_generator = (torch.Generator(device=self.device).manual_seed(
                int(symmetry_seed.generate_state(1, dtype=np.uint64)[0])) if self.tensor
                else np.random.default_rng(symmetry_seed))
        if refresh_reuse:
            from .map_stream import MapStream
            self.stream = MapStream(config, size, seed, width=self.backend.width, height=self.backend.height,
                curriculum=curriculum, workers=generation_workers, chunk_size=refresh_chunk_size,
                reuse=refresh_reuse, prefetch_chunks=prefetch_chunks)

    def _refresh(self, count):
        initial_draw = self.sampled_games == 0
        self.sampled_games += count
        if self.stream is None or initial_draw:
            return
        levels = self.curriculum.levels if self.curriculum is not None else 1
        for start, size, (arrays, stats, rng, valid) in self.stream.consume(count):
            rows = np.concatenate([np.arange(level*self.size+start, level*self.size+start+size) for level in range(levels)])
            if self.tensor:
                rows = torch.as_tensor(rows, device=self.device)
                values = {**arrays, 'stats': stats, 'rng': rng.view(np.int32), 'valid': valid}
                targets = {**self.arrays, 'stats': self.stats, 'rng': self.rng, 'valid': self.bank_valid}
                for name, value in values.items():
                    # Owned pageable CPU arrays; copies and consumers share the
                    # current stream, so no host synchronization per field is needed.
                    targets[name].index_copy_(0, rows, torch.from_numpy(value).to(self.device, non_blocking=True))
            else:
                for name, value in arrays.items():
                    self.arrays[name][rows] = value
                self.stats[rows], self.rng[rows], self.bank_valid[rows] = stats, rng, valid

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def set_transitions(self, transitions):
        """Only future resets observe this schedule; live games are untouched."""
        self.position = self.curriculum.position(transitions)

    def metrics(self):
        stream = self.stream
        result = dict(map_bank_cache_hit=float(self.bank_info['cache_hit']),
            map_bank_load_seconds=self.bank_info['load_seconds'], map_sampled_games=self.sampled_games,
            map_introduced_base_maps=self.size+(stream.introduced if stream else 0),
            map_refresh_wait_seconds=stream.wait_seconds if stream else 0.)
        if self.curriculum is None:
            return result
        if self.tensor:
            mean = self.level_ids.float().mean().item()
            final = (self.level_ids == self.curriculum.levels - 1).float().mean().item()
        else:
            mean = float(self.level_ids.mean())
            final = float((self.level_ids == self.curriculum.levels - 1).mean())
        return dict(result, map_position=self.position, map_level_mean=mean, map_final_fraction=final)

    def reset(self, mask=None, *, _rows=None):
        # Task-internal callers may reuse their already compacted mask indices.
        backend = self.backend
        if self.tensor:
            if mask is not None and (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != (backend.batch_size,) or mask.device != self.device):
                raise ValueError("map sampling reset mask must be bool [batch] on backend device")
            rows = (_rows if _rows is not None else
                    torch.arange(backend.batch_size, device=self.device) if mask is None else torch.where(mask)[0])
            if not rows.numel():
                return backend.state
            self._refresh(len(rows))
            selected = torch.randint(self.size, (rows.numel(),), generator=self.generator, device=self.device)
            if self.curriculum is not None:
                location = self.position * (self.curriculum.levels - 1)
                level = int(location)
                if location > level:
                    selected += self.size * (level + (torch.rand(rows.numel(), generator=self.generator,
                                             device=self.device) < location - level).long())
                elif level:
                    selected += self.size * level
            arrays = {name: value.index_select(0, selected) for name, value in self.arrays.items()}
            stats, rng = self.stats.index_select(0, selected), self.rng.index_select(0, selected)
        else:
            if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != bool
                                     or mask.shape != (backend.batch_size,)):
                raise ValueError("map sampling reset mask must be bool [batch]")
            rows = _rows if _rows is not None else np.arange(backend.batch_size) if mask is None else np.flatnonzero(mask)
            if not len(rows):
                return backend.state
            self._refresh(len(rows))
            selected = self.generator.integers(self.size, size=len(rows))
            if self.curriculum is not None:
                location = self.position * (self.curriculum.levels - 1)
                level = int(location)
                if location > level:
                    selected += self.size * (level + (self.generator.random(len(rows)) < location - level))
                elif level:
                    selected += self.size * level
            arrays = {name: value[selected] for name, value in self.arrays.items()}
            stats, rng = self.stats[selected], self.rng[selected]
        if self.curriculum is not None:
            self.level_ids[rows] = selected // self.size
        valid = self.bank_valid[selected]
        if self.symmetry is not None:
            transforms = (torch.randint(8, (len(rows),), generator=self.symmetry_generator, device=self.device)
                          if self.tensor else self.symmetry_generator.integers(8, size=len(rows)))
            arrays, valid = self.symmetry.apply(arrays, valid, transforms)
        self.valid[rows] = valid
        _replace_initial(backend, rows, arrays, stats, rng)
        return backend.reset(mask)
