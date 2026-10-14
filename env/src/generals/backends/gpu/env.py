"""Independent device-resident batched environment with explicit host boundaries.

The CUDA rules live in kernel.cu. No CPU engine, policy, runner, or
GUI is imported here. CPU map loading/export is confined to the interop module.
"""
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from generals.protocol import STATE_DTYPES, STATS_FIELDS, COUNTER_FIELDS, MOVE_FIELDS
from generals.protocol.config import _integer
from generals.protocol.vector import BatchStep


class GPUEnv:
    def __init__(self, initial: Mapping[str, torch.Tensor], *, width: int, height: int,
                 device="cuda", seed=42, kernel_backend=None, auto_reset=False, episode_ticks=0):
        self._closed = False
        self.width = _integer("width", width, 1)
        self.height = _integer("height", height, 1)
        self.seed = _integer("seed", seed, 0)
        self.auto_reset = bool(auto_reset)
        self.episode_ticks = _integer("episode_ticks", episode_ticks, 0)
        if self.episode_ticks > np.iinfo(np.int64).max:
            raise ValueError("episode_ticks must fit int64")
        if kernel_backend not in (None, "cuda"):
            raise ValueError("kernel_backend must be cuda")
        self.kernel_backend = "cuda"
        requested = torch.device(device)
        if requested.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("GPUEnv requires an available CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if requested.index is None else requested.index)
        if set(initial) != set(STATE_DTYPES):
            raise ValueError(f"initial state must have fields {tuple(STATE_DTYPES)}")
        if not isinstance(initial["army"], torch.Tensor) or initial["army"].ndim != 2:
            raise ValueError("army must be a [batch, height*width] tensor")
        self.batch_size, size = initial["army"].shape
        if not self.batch_size or size != self.width * self.height:
            raise ValueError("initial batch is empty or does not match the map dimensions")
        if initial["alive"].ndim != 2:
            raise ValueError("alive must have shape [batch, players]")
        self.players = initial["alive"].shape[1]
        if not 2 <= self.players <= 8:
            raise ValueError("players must be between 2 and 8")
        # A single program owns a board; cap the vector at a documented supported
        # size rather than risk impractical register spills for unbounded inputs.
        if size > 4096:
            raise ValueError("GPUEnv currently supports up to 4096 cells per map")
        self.initial = {}
        for name, dtype in STATE_DTYPES.items():
            tensor = initial[name]
            shape = ((self.batch_size, size) if name in ("terrain", "structure", "owner", "army")
                     else (self.batch_size, self.players) if name in ("alive", "general_pos")
                     else (self.batch_size,))
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != getattr(torch, dtype):
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            self.initial[name] = tensor.to(self.device).contiguous().clone()
        self._validate_initial()
        self.state = {name: value.clone() for name, value in self.initial.items()}
        self._initial_stats = self._compute_stats()
        self.stats = self._initial_stats.clone()
        seeds = np.asarray([int(np.random.SeedSequence([self.seed, lane, 1]).generate_state(
            1, dtype=np.uint32)[0]) or 1 for lane in range(self.batch_size)], dtype=np.uint32)
        self._initial_rng = torch.from_numpy(seeds).to(self.device)
        self.rng_state = self._initial_rng.clone()
        self.counters = torch.zeros((self.batch_size, len(COUNTER_FIELDS)), dtype=torch.int64, device=self.device)
        self.last_moves = torch.zeros((self.batch_size, self.players, len(MOVE_FIELDS)), dtype=torch.int64, device=self.device)
        self.errors = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.done = torch.zeros((self.batch_size, 2), dtype=torch.bool, device=self.device)
        self._wait_actions = torch.zeros((self.batch_size, self.players), dtype=torch.int64, device=self.device)
        self._step_kernel = None
        self._step_result = BatchStep(self.last_moves, self.stats, self.done, self.errors)

    def _validate_initial(self):
        s = self.initial
        n = self.width * self.height
        terrain, structure, owner, army = (s[key] for key in ("terrain", "structure", "owner", "army"))
        bad = ((terrain > 1) | (structure > 2) | (owner < -2) | (owner >= self.players) | (army < 0)).any()
        bad |= ((terrain == 1) & ((structure != 0) | (owner != -1) | (army != 0))).any()
        bad |= ((owner == -1) & ((structure != 0) | (army != 0))).any()
        bad |= ((owner == -2) & (structure != 1)).any()
        bad |= ((s["general_pos"] < 0) | (s["general_pos"] >= n)).any()
        sorted_pos = s["general_pos"].sort(dim=1).values
        bad |= (sorted_pos[:, 1:] == sorted_pos[:, :-1]).any()
        live_count = s["alive"].sum(dim=1)
        bad |= ((live_count < 1) | (s["tick"] < 0) | (s["terminated"] != (live_count == 1))).any()
        winner = torch.where(live_count == 1, s["alive"].to(torch.int32).argmax(dim=1), -1)
        bad |= (s["winner"] != winner).any()
        indices = torch.arange(n, device=self.device)[None, :]
        expected_general = torch.zeros_like(structure, dtype=torch.bool)
        for pid in range(self.players):
            owned = owner == pid
            expected = (indices == s["general_pos"][:, pid:pid + 1]) & s["alive"][:, pid:pid + 1]
            expected_general |= expected
            bad |= (expected & (~owned | (structure != 2))).any()
            bad |= (owned & ~s["alive"][:, pid:pid + 1]).any()
        bad |= ((structure == 2) != expected_general).any()
        if bool(bad):
            raise ValueError("initial GPU state violates the shared game-state invariants")

    def _compute_stats(self):
        """GPU arithmetic assumes army/tick values and totals fit int64."""
        state = self.state
        rows = []
        for pid in range(self.players):
            owned = state["owner"] == pid
            total = torch.where(owned, state["army"], 0).sum(dim=1)
            rows.append(torch.stack((total, owned.sum(dim=1),
                                     (owned & (state["structure"] == 1)).sum(dim=1)), dim=1))
        return torch.stack(rows, dim=1)

    @classmethod
    def from_states(cls, states, *, device="cuda", seed=42, kernel_backend=None, **kwargs):
        from generals.core.batch import pack_states
        arrays, width, height = pack_states(states)
        return cls({name: torch.from_numpy(value) for name, value in arrays.items()},
                   width=width, height=height, device=device, seed=seed,
                   kernel_backend=kernel_backend, **kwargs)

    @classmethod
    def from_config(cls, config, batch_size=1, *, device="cuda", seed=42, kernel_backend=None, **kwargs):
        from generals.core.batch import generate_states
        return cls.from_states(generate_states(config, batch_size, seed=seed),
                               device=device, seed=seed, kernel_backend=kernel_backend, **kwargs)

    def to_states(self):
        from generals.core.batch import unpack_states
        self._require_open()
        self.check_errors()
        return unpack_states({name: value.cpu().numpy() for name, value in self.state.items()},
                             width=self.width, height=self.height)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("GPU environment is closed")

    def configure_step(self, *, group_threads=None, envs_per_block=None, packed_metadata=None):
        """Configure and compile the CUDA tick before starting a measurement."""
        self._require_open()
        if group_threads is None:
            group_threads = 32 if self.batch_size <= 2048 or self.batch_size >= 32768 else 8
        if envs_per_block is None:
            envs_per_block = 1 if group_threads > 32 else 4 if self.batch_size <= 2048 else 16 if group_threads == 8 else 8
        from generals.backends.gpu.kernel import StepKernel
        with torch.cuda.device(self.device):
            self._step_kernel = StepKernel(self, group_threads=group_threads,
                                           envs_per_block=envs_per_block,
                                           packed_metadata=packed_metadata)

    def invalidate_cache(self):
        """Call after raw external CUDA writes or edits to inference-mode tensors.

        Normal in-place PyTorch edits to owner, army and structure are detected
        automatically. Replacing state tensor objects/storage is unsupported.
        """
        if self._step_kernel is not None:
            self._step_kernel.dirty = True

    def step_metadata(self):
        if self._step_kernel is not None:
            return {"step_implementation": "cuda", "kernel": self._step_kernel.metadata()}
        return {"step_implementation": "cuda"}

    def step(self, actions):
        """Advance each nonterminal lane once with encoded actions [B,P].

        Eliminated players' slots are ignored. Terminated lanes stay frozen until
        reset. Army values, player totals and tick counters must fit int64; arithmetic
        does not check overflow. The errors field remains for protocol compatibility.
        """
        kernel = self._step_kernel
        if kernel is None:
            self.configure_step()
            kernel = self._step_kernel
        return kernel.native_step(kernel, actions)

    def reset(self, mask=None):
        """Restore initial state/RNG and clear diagnostics, for selected lanes."""
        self._require_open()
        if mask is None:
            for name, value in self.initial.items():
                self.state[name].copy_(value)
            self.rng_state.copy_(self._initial_rng)
            self.stats.copy_(self._initial_stats)
            self.counters.zero_()
            self.last_moves.zero_()
            self.errors.zero_()
            self.done.zero_()
        else:
            if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                    or mask.shape != (self.batch_size,) or mask.device != self.device):
                raise ValueError("reset mask must be bool [batch] on the environment device")
            for name, value in self.initial.items():
                current = self.state[name]
                current.copy_(torch.where(mask.reshape((self.batch_size,) + (1,) * (value.ndim - 1)), value, current))
            # uint32 torch.where is not supported by every PyTorch build.
            self.rng_state.view(torch.int32).copy_(torch.where(mask, self._initial_rng.view(torch.int32),
                                                              self.rng_state.view(torch.int32)))
            self.stats.copy_(torch.where(mask[:, None, None], self._initial_stats, self.stats))
            self.counters.masked_fill_(mask[:, None], 0)
            self.last_moves.masked_fill_(mask[:, None, None], 0)
            self.errors.masked_fill_(mask, 0)
            self.done.masked_fill_(mask[:, None], False)
        self.invalidate_cache()
        return self.state

    def observe(self, player_id, mode="local"):
        """Independent batched observation tensors, constructed on the GPU."""
        self._require_open()
        player_id = _integer("player_id", player_id, 0)
        if player_id >= self.players or mode not in ("local", "full"):
            raise ValueError("invalid player_id or observation mode")
        state = self.state
        shape = (self.batch_size, self.height, self.width)
        owned = (state["owner"] == player_id).reshape(shape)
        visible = (torch.ones_like(owned) if mode == "full" else
                   F.max_pool2d(owned[:, None].float(), 3, stride=1, padding=1)[:, 0].bool())
        stats = self._compute_stats()
        cities = stats[:, :, 2].clone()
        if mode == "local":
            cities.fill_(-1)
            cities[:, player_id] = stats[:, player_id, 2]
        result = {name: torch.where(visible, state[name].reshape(shape), unknown)
                  for name, unknown in (("terrain", 255), ("structure", 255), ("owner", -3), ("army", -1))}
        result.update(visible=visible, alive=state["alive"].clone(),
                      general_pos=state["general_pos"][:, player_id].clone(), tick=state["tick"].clone(),
                      terminated=state["terminated"].clone(), winner=state["winner"].clone(),
                      stats_army=stats[:, :, 0].clone(), stats_territory=stats[:, :, 1].clone(), stats_cities=cities)
        return result

    def legal_action_mask(self, player_id):
        """Return bool [B,N,4,2] moves and [B] WAIT, including zero-army moves."""
        self._require_open()
        player_id = _integer("player_id", player_id, 0)
        if player_id >= self.players:
            raise ValueError("player_id outside player slots")
        state = self.state
        terrain = state["terrain"].reshape(self.batch_size, self.height, self.width)
        padded = F.pad(terrain, (1, 1, 1, 1), value=1)
        targets = torch.stack((padded[:, :-2, 1:-1], padded[:, 1:-1, 2:],
                               padded[:, 2:, 1:-1], padded[:, 1:-1, :-2]), dim=-1)
        active = state["alive"][:, player_id] & ~state["terminated"] & (self.errors == 0)
        source = ((state["owner"] == player_id) & active[:, None]).reshape(self.batch_size, self.height, self.width)
        legal = (source[..., None] & (targets != 1)).reshape(self.batch_size, -1, 4)
        return legal[..., None].expand(-1, -1, -1, 2).clone(), active.clone()

    def synchronize(self):
        self._require_open()
        torch.cuda.synchronize(self.device)

    def check_errors(self):
        self._require_open()
        codes = self.errors.cpu().numpy()
        bad = np.flatnonzero(codes)
        if bad.size:
            lane = int(bad[0])
            raise RuntimeError(f"GPU lane {lane}: reserved device error code {int(codes[lane])}")

    def close(self):
        # PyTorch's allocator releases storage when outstanding tensor references
        # and stream work permit. Do not globally empty another caller's cache.
        if not self._closed:
            # Custom launches use raw pointers. Wait before releasing their
            # owning tensors, including work launched on a non-default stream.
            self.synchronize()
            self._closed = True
            self._step_kernel = None
            self._step_result = None
            self.state.clear()
            self.initial.clear()
            for name in ("stats", "_initial_stats", "rng_state", "_initial_rng",
                         "counters", "last_moves", "errors", "done", "_wait_actions"):
                setattr(self, name, None)
