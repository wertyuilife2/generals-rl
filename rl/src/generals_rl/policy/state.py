"""Tensor views constructed exclusively from a public Arena observation."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GameBatch:
    state: dict[str, torch.Tensor]
    stats: torch.Tensor
    width: int
    height: int
    visible: torch.Tensor | None = None
    valid: torch.Tensor | None = None

    @property
    def valid_cells(self):
        return torch.ones_like(self.state['owner'], dtype=torch.bool) if self.valid is None else self.valid

    def fog_obstacles(self, player_ids, visible):
        return self.state['fog_obstacles'] & ~visible & self.valid_cells

    def padded(self, width, height):
        """RL-owned fixed canvas; Arena observations retain their real geometry."""
        if self.width > width or self.height > height:
            raise ValueError('observation exceeds policy canvas')
        if (self.width, self.height) == (width, height):
            return self
        padding = (0, width - self.width, 0, height - self.height)
        def pad(value, fill=0):
            return F.pad(value.reshape(self.num_envs, self.height, self.width), padding, value=fill).flatten(1)
        fills = dict(terrain=1, structure=0, owner=-1, army=0, fog_obstacles=0)
        state = {name: pad(value, fills[name]) if name in fills else value for name, value in self.state.items()}
        return type(self)(state, self.stats, width, height,
                          None if self.visible is None else pad(self.visible), pad(self.valid_cells))

    @classmethod
    def from_raw(cls, observation, device):
        state = {name: torch.as_tensor(value, device=device) for name, value in observation["state"].items()}
        state["army"] = state["army"].clamp_min(0)
        return cls(state, torch.as_tensor(observation["stats"], device=device),
                   observation["width"], observation["height"],
                   torch.as_tensor(observation["visible"], device=device))

    @classmethod
    def from_observations(cls, observations, device, *, width=None, height=None):
        batches = [cls.from_observation(observation, device) for observation in observations]
        if width is not None:
            batches = [batch.padded(width, height) for batch in batches]
        first = batches[0]
        return cls({name: torch.cat([batch.state[name] for batch in batches]) for name in first.state},
                   torch.cat([batch.stats for batch in batches]), first.width, first.height,
                   torch.cat([batch.visible for batch in batches]), torch.cat([batch.valid_cells for batch in batches]))

    @classmethod
    def from_observation(cls, observation, device):
        state = {name: torch.tensor(getattr(observation, name).copy(), device=device).reshape(1, -1)
                 for name in ("terrain", "structure", "owner", "army", "alive", "fog_obstacles", "has_kill", "neutralize_in")}
        # Unknown armies are -1 in the public observation. Mask before log1p.
        state["army"] = state["army"].clamp_min(0)
        state["tick"] = torch.tensor([observation.tick], device=device, dtype=torch.long)
        state["terminated"] = torch.tensor([observation.terminated], device=device, dtype=torch.bool)
        stats = torch.stack([torch.tensor(getattr(observation.stats, name).copy(), device=device)
                             for name in ("army", "territory", "cities")], dim=-1)[None]
        return cls(state, stats, observation.width, observation.height,
                   torch.tensor(observation.visible.copy(), device=device).reshape(1, -1))

    @property
    def num_envs(self):
        return self.state["army"].shape[0]

    @property
    def players(self):
        return self.state["alive"].shape[1]

    @property
    def device(self):
        return self.stats.device
