"""Persistent, player-local facts, separate from short observation histories.

Only visible valid cells update this memory. General identity can change
after a mutual capture; all general markers mean last observed, not permanent; ownership and city structure mean *last observed*, not a
prediction of the hidden board. Each lane/seat owns its own memory so resetting
or rotating a learner cannot transfer another player's information.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .state import GameBatch


KNOWLEDGE_SCHEMA = "official_last_seen_knowledge_v2"
KNOWLEDGE_BASE_CHANNELS = (
    "explored", "mountain", "last_seen_city", "last_seen_general",
    "last_seen_enemy_general", "last_seen_non_general", "last_seen_empty",
    "last_seen_neutral",
)


def knowledge_channels(players: int) -> tuple[str, ...]:
    """Stable channel order; relative owner 0 is the observing player."""
    return (KNOWLEDGE_BASE_CHANNELS
            + tuple(f"last_seen_owner_{offset}" for offset in range(players))
            + ("last_seen_age",))


class KnowledgeState:
    """Compact memory for ``[num_envs, players, cells]``.

    ``update(batch, player_ids)`` observes one seat per lane. Call it separately
    for other seats when they make decisions. ``encode(tick, player_ids)``
    returns float32 ``[N, C, H, W]`` planes, without temporal stacking. All
    channels are binary except age, which is elapsed ticks / ``age_scale``,
    capped at 1. Unknown cells have age 0 and explored 0.

    General identity and structure are last-seen facts. Observing a converted
    city clears its old general identity. A general trade can change ownership
    in fog, so the memory never presents an old identity as a permanent fact.
    """

    semantics = KNOWLEDGE_SCHEMA
    _fields = ("terrain", "structure", "general_owner", "owner", "last_seen_tick")

    def __init__(self, num_envs: int, width: int, height: int, players: int,
                 device="cpu", *, age_scale: int = 2000):
        for name, value in (("num_envs", num_envs), ("width", width),
                            ("height", height), ("players", players),
                            ("age_scale", age_scale)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.num_envs, self.width, self.height, self.players = num_envs, width, height, players
        self.device = torch.device(device)
        self.age_scale = age_scale
        self.channels = knowledge_channels(players)
        shape = (num_envs, players, width * height)
        # -1 terrain/structure/tick means unobserved; owner -1/-2 are legal
        # empty/neutral values, so its unknown sentinel is -3.
        self.terrain = torch.full(shape, -1, dtype=torch.int8, device=self.device)
        self.device = self.terrain.device  # Resolve an unspecified CUDA index.
        self.structure = torch.full_like(self.terrain, -1)
        self.general_owner = torch.full(shape, -1, dtype=torch.int16, device=self.device)
        self.owner = torch.full_like(self.general_owner, -3)
        self.last_seen_tick = torch.full(shape, -1, dtype=torch.int64, device=self.device)

    def _ids(self, player_ids, count=None):
        ids = torch.as_tensor(player_ids, device=self.device)
        if ids.shape != (self.num_envs if count is None else count,) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("player_ids must contain one integer seat per lane")
        return ids.long()

    def _select(self, field, player_ids, lane_ids=None):
        # gather rejects invalid negative/out-of-range seats without a CUDA
        # host synchronization for a separate bounds check.
        if lane_ids is not None:
            field = field[lane_ids]
        indices = player_ids[:, None, None].expand(-1, 1, self.width * self.height)
        return field.gather(1, indices).squeeze(1)

    @torch.no_grad()
    def update(self, batch: GameBatch, player_ids, mask=None, lane_ids=None, *, visible=None):
        """Observe legal local visibility, optionally only in selected lanes."""
        if (batch.width, batch.height, batch.players) != (self.width, self.height, self.players) or (
                lane_ids is None and batch.num_envs != self.num_envs):
            raise ValueError("batch geometry does not match knowledge memory")
        if batch.device != self.device:
            raise ValueError("batch and knowledge memory must share a device")
        ids = self._ids(player_ids, batch.num_envs)
        old = {name: self._select(getattr(self, name), ids, lane_ids) for name in self._fields}
        state = batch.state
        if visible is None:
            owned = (state["owner"] == ids[:, None]).float()
            visible = F.max_pool2d(owned.reshape(batch.num_envs, 1, self.height, self.width),
                                   kernel_size=3, stride=1, padding=1).flatten(1).bool()
        visible = (visible | state["terminated"][:, None]) & batch.valid_cells
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
            if mask.shape != (batch.num_envs,) or mask.dtype != torch.bool:
                raise ValueError("update mask must contain one boolean per lane")
            visible = visible & mask[:, None]
        rows = torch.arange(batch.num_envs, device=self.device) if lane_ids is None else lane_ids
        for name in ("terrain", "structure", "owner"):
            field = getattr(self, name)
            field[rows, ids] = torch.where(visible, state[name].to(field.dtype), old[name])
        observed_general = visible & (state["structure"] == 2)
        self.general_owner[rows, ids] = torch.where(
            visible, torch.where(observed_general, state["owner"].to(self.general_owner.dtype), -1), old["general_owner"])
        self.last_seen_tick[rows, ids] = torch.where(
            visible, state["tick"][:, None], old["last_seen_tick"])

    @torch.no_grad()
    def encode(self, tick, player_ids, lane_ids=None):
        """Materialize independent non-stacked planes for the selected seats."""
        count = self.num_envs if lane_ids is None else len(lane_ids)
        ids = self._ids(player_ids, count)
        now = torch.as_tensor(tick, device=self.device)
        if now.shape != (count,) or now.dtype not in (torch.int32, torch.int64):
            raise ValueError("tick must contain one integer clock per lane")
        terrain, structure, general_owner, owner, last_seen = (
            self._select(getattr(self, name), ids, lane_ids) for name in self._fields)
        explored = last_seen >= 0
        general = general_owner >= 0
        enemy_general = general & (general_owner != ids[:, None])
        relative_ids = (ids[:, None] + torch.arange(self.players, device=self.device)) % self.players
        owners = owner[:, None] == relative_ids[:, :, None]
        age = (now[:, None] - last_seen).clamp(min=0, max=self.age_scale).float() / self.age_scale
        age *= explored
        binary = torch.stack((explored, terrain == 1, structure == 1, general,
                              enemy_general, explored & ~general,
                              owner == -1, owner == -2), 1)
        planes = torch.cat((binary.float(), owners.float(), age[:, None]), 1)
        return planes.reshape(count, len(self.channels), self.height, self.width)

    @torch.no_grad()
    def reset(self, mask=None):
        """Clear every seat in selected episode lanes; preserve other lanes."""
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
            if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
                raise ValueError("reset mask must contain one boolean per lane")
        for name in self._fields:
            field = getattr(self, name)
            value = -3 if name == "owner" else -1
            if mask is None:
                field.fill_(value)
            else:
                field[mask] = value

    def clone(self):
        """Independent memory for an alternate search branch or simulation."""
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        for name in self._fields:
            setattr(result, name, getattr(self, name).clone())
        return result
