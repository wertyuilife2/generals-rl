"""Stateful, vectorized public-observation policy inference without PPO/RSL."""
from collections.abc import Mapping
import json
from numbers import Integral
from pathlib import Path

import torch

from generals.protocol.actions import Action

from .actions import make_action_space
from .actor import build_actor
from .config import FeatureConfig
from .distributions import action_probabilities, greedy_actions
from .knowledge import KnowledgeState
from .observations import ObservationHistory, make_observation_encoder
from .precision import matmul_precision, validate_precision
from .state import GameBatch


class PolicyRuntime:
    """One actor with independent lane history, memory and counter-based RNG.

    Tensor batches stay on the actor device. Stable lane_ids identify states
    when callers filter or reorder a batch. Scalar controllers use lane zero.
    """
    def __init__(self, *, weights, config_path, device="cpu", player_id=0, seed=0):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.features = FeatureConfig(**self.config["features"])
        inference = self.config["inference"]
        if (inference.get("mode") not in ("greedy", "stochastic") or inference.get("precision") != "from_config"
                or inference.get("execution") != "eager"):
            raise ValueError("policy requires greedy/stochastic mode, from_config precision and eager execution")
        self.mode = inference["mode"]
        self.precision = (inference.get("matmul_precision"), inference.get("cudnn_allow_tf32"),
                          inference.get("cudnn_benchmark"))
        validate_precision(*self.precision)
        network = self.config["network"]
        self.factorized = network.get("wait_move_factorization", network.get("family") == "scaled")
        self.amp_bfloat16 = network.get("amp_bfloat16", False)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("policy supports CPU or CUDA inference")
        self.encoder = make_observation_encoder(self.features)
        self.action_space = make_action_space(self.features)
        with torch.random.fork_rng(devices=[]):
            self.actor = build_actor(network, self.config["features"]).to(self.device)
        saved = torch.load(weights, weights_only=True, map_location="cpu")
        if not isinstance(saved, Mapping) or not isinstance(saved.get("actor_state_dict"), Mapping):
            raise ValueError("weights must contain actor_state_dict")
        state = saved["actor_state_dict"]
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval().requires_grad_(False)
        self.capacity = 0
        self.history = ObservationHistory(self.features.history_length)
        self.knowledge = None
        self.seed = seed
        self.reset(player_id, seed)

    def validate_rules(self, map_config, visibility):
        self._check_geometry(map_config.width, map_config.height, map_config.players)
        if visibility != "local":
            raise ValueError("policy requires local visibility")

    def _check_geometry(self, width, height, players):
        if players != self.features.players:
            raise ValueError('observation player count does not match policy')
        if self.features.random_maps:
            if any(side is not None and not 18 <= side <= 23 for side in (width, height)):
                raise ValueError('this policy supports official 18–23-cell 1v1 maps')
        elif (width, height) != (self.features.width, self.features.height):
            raise ValueError('observation shape does not match policy')

    def _board_actions(self, encoded, widths):
        value = (encoded - 1).clamp_min(0)
        source = value // 8
        board_source = (source // self.features.width) * widths + source % self.features.width
        return torch.where(encoded > 0, 1 + board_source * 8 + value % 8, 0)

    def _grow(self, capacity):
        if capacity <= self.capacity:
            return
        old = self.capacity
        frame_dim = (12 + self.features.players) * self.features.width * self.features.height + 5 * self.features.players + 3
        frames = torch.zeros(capacity, self.features.history_length, frame_dim, device=self.device)
        if old:
            frames[:old] = self.history.frames
        self.history.frames = frames
        for name, dtype in (("player_ids", torch.long), ("_initialized", torch.bool),
                            ("_counter", torch.long),
                            ("_seed_low", torch.long), ("_seed_high", torch.long)):
            value = torch.zeros(capacity, dtype=dtype, device=self.device)
            if old:
                value[:old] = getattr(self, name)
            setattr(self, name, value)
        if self.features.persistent_knowledge:
            memory = KnowledgeState(capacity, self.features.width, self.features.height, self.features.players,
                                    self.device, age_scale=self.features.knowledge_age_scale)
            if old:
                for name in memory._fields:
                    getattr(memory, name)[:old] = getattr(self.knowledge, name)
            self.knowledge = memory
        self.capacity = capacity

    def reset(self, player_id, seed):
        self.player_id = player_id
        self.reset_batch([player_id], [seed], lane_ids=[0])

    @torch.inference_mode()
    def reset_batch(self, player_ids, seeds, *, lane_ids=None):
        # Resets are the explicit host boundary; tick inference is tensor-only.
        players = player_ids.tolist() if isinstance(player_ids, torch.Tensor) else list(player_ids)
        seeds = seeds.tolist() if isinstance(seeds, torch.Tensor) else list(seeds)
        lanes = (list(range(len(players))) if lane_ids is None else
                 lane_ids.tolist() if isinstance(lane_ids, torch.Tensor) else list(lane_ids))
        if len(players) != len(seeds) or len(lanes) != len(players) or len(set(lanes)) != len(lanes):
            raise ValueError("reset_batch requires matching independent lanes, players and seeds")
        if any(isinstance(lane, bool) or not isinstance(lane, Integral) or lane < 0 for lane in lanes):
            raise ValueError("lane_ids must be nonnegative integers")
        if any(isinstance(player, bool) or not isinstance(player, Integral) or not 0 <= player < self.features.players for player in players):
            raise ValueError("invalid player seat")
        if any(isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64 for seed in seeds):
            raise ValueError("seeds must be uint64 integers")
        lanes, players, seeds = [int(value) for value in lanes], [int(value) for value in players], [int(value) for value in seeds]
        if not lanes:
            return
        self._grow(max(lanes) + 1)
        ids = torch.tensor(lanes, dtype=torch.long, device=self.device)
        self.player_ids[ids] = torch.tensor(players, dtype=torch.long, device=self.device)
        self._seed_low[ids] = torch.tensor([seed & 0xffffffff for seed in seeds], device=self.device)
        self._seed_high[ids] = torch.tensor([seed >> 32 for seed in seeds], device=self.device)
        self._counter[ids] = 0
        self._initialized[ids] = False
        self.history.frames[ids] = 0
        if self.knowledge is not None:
            mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
            mask[ids] = True
            self.knowledge.reset(mask)

    def _rows(self, count, lane_ids):
        if lane_ids is None:
            if count > self.capacity:
                raise ValueError("reset_batch must initialize all inference lanes")
            return None if count == self.capacity else torch.arange(count, device=self.device)
        rows = torch.as_tensor(lane_ids, dtype=torch.long, device=self.device)
        if rows.shape != (count,):
            raise ValueError("lane_ids must match the observation batch")
        return rows

    @torch.inference_mode()
    def _inputs(self, batch, rows=None, active=None):
        take = slice(None) if rows is None else rows
        players = self.player_ids[take]
        if active is None:
            active = batch.state["alive"].gather(1, players[:, None]).squeeze(1) & ~batch.state["terminated"]
        visible = batch.visible if batch.visible is not None else self.encoder.visibility(batch, players)
        frame = self.encoder.encode(batch, players, visible=visible)
        old, initialized = self.history.frames[take], self._initialized[take]
        shifted = torch.cat((old[:, 1:], frame[:, None]), 1)
        frames = torch.where(initialized[:, None, None], shifted,
                             frame[:, None].expand(-1, self.features.history_length, -1))
        frames = torch.where(active[:, None, None], frames, old)
        if rows is None:
            self.history.frames = frames
            self._initialized |= active
        else:
            self.history.frames[rows] = frames
            self._initialized[rows] |= active
        inputs = {"policy": frames.flatten(1), "single_obs": frame,
                  "action_mask": self.action_space.mask(batch, players)}
        if self.knowledge is not None:
            self.knowledge.update(batch, players, mask=active, lane_ids=rows, visible=visible)
            inputs["knowledge"] = self.knowledge.encode(batch.state["tick"], players, lane_ids=rows).flatten(1)
        return inputs

    def _uniforms(self, rows, active):
        take = slice(None) if rows is None else rows
        counter = self._counter[take]
        # Counter32-v1: all products fit signed int64 before the 32-bit mask.
        x = (self._seed_low[take] + (counter & 0xffffffff) * 1664525 + 1013904223) & 0xffffffff
        x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xffffffff
        x = x ^ self._seed_high[take]
        x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xffffffff
        x = x ^ (x >> 16)
        self._counter[take] = counter + active.long()
        return (x.double() + 0.5) / 4294967296.0

    def _act_batch(self, batch, rows=None, active=None):
        take = slice(None) if rows is None else rows
        players = self.player_ids[take]
        alive = batch.state["alive"].gather(1, players[:, None]).squeeze(1) & ~batch.state["terminated"]
        active = alive if active is None else alive & active
        inputs = self._inputs(batch, rows, active)
        with torch.inference_mode():
            with matmul_precision(*self.precision), torch.autocast(
                    self.device.type, dtype=torch.bfloat16,
                    enabled=self.amp_bfloat16 and self.device.type == "cuda"):
                logits = self.actor(inputs["policy"], inputs.get("knowledge")).float()
            with torch.autocast(self.device.type, enabled=False):
                probabilities = action_probabilities(logits, inputs["action_mask"], factorized=self.factorized,
                                                       validate_mask=False)
                if self.mode == "greedy":
                    selected = greedy_actions(probabilities, factorized=self.factorized).flatten()
                else:
                    cdf = probabilities.double().cumsum(-1)
                    cdf = cdf / cdf[:, -1:]
                    selected = torch.searchsorted(cdf, self._uniforms(rows, active)[:, None], right=True).flatten()
        return torch.where(active, selected, torch.zeros_like(selected))

    @torch.inference_mode()
    def act_batch(self, observations, *, lane_ids=None):
        if isinstance(observations, Mapping):
            if observations.get("visibility", "local") != "local":
                raise ValueError("policy requires local visibility")
            batch = GameBatch.from_raw(observations, self.device)
            self._check_geometry(batch.width, batch.height, batch.players)
            real_width = batch.width
            batch = batch.padded(self.features.width, self.features.height)
            if not batch.num_envs:
                return torch.empty(0, dtype=torch.long, device=self.device)
            rows = self._rows(batch.num_envs, lane_ids)
            if "player_ids" in observations:
                players = torch.as_tensor(observations["player_ids"], device=self.device)
                expected = self.player_ids if rows is None else self.player_ids[rows]
                if self.device.type == "cuda":
                    torch._assert_async((players == expected).all(), "observation belongs to another player")
                elif not torch.equal(players, expected):
                    raise ValueError("observation belongs to another player")
            active = observations.get("active")
            active = None if active is None else torch.as_tensor(active, dtype=torch.bool, device=self.device)
            return self._board_actions(self._act_batch(batch, rows, active), real_width)
        observations = list(observations)
        if not observations:
            return []
        for observation in observations:
            if observation.visibility != "local":
                raise ValueError("observation does not match the local policy contract")
            self._check_geometry(observation.width, observation.height, len(observation.alive))
        rows = self._rows(len(observations), lane_ids)
        expected = self.player_ids if rows is None else self.player_ids[rows]
        if expected.tolist() != [observation.player_id for observation in observations]:
            raise ValueError("observation belongs to another player")
        batch = GameBatch.from_observations(observations, self.device, width=self.features.width, height=self.features.height)
        widths = torch.tensor([obs.width for obs in observations], device=self.device)
        encoded = self._board_actions(self._act_batch(batch, rows), widths)
        return [Action.decode(value) for value in encoded.tolist()]

    def act(self, observation):
        return self.act_batch([observation], lane_ids=[0])[0]

    def on_result(self, result):
        pass

    def on_result_batch(self, result):
        pass


def create_controller(*, weights, config_path, device="cpu", player_id=0, seed=0):
    return PolicyRuntime(weights=weights, config_path=config_path, device=device, player_id=player_id, seed=seed)
