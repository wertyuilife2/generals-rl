"""Tensor views constructed exclusively from a public Arena observation."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GameBatch:
    state: dict[str, torch.Tensor]
    stats: torch.Tensor
    width: int
    height: int
    visible: torch.Tensor | None = None

    @classmethod
    def from_raw(cls, observation, device):
        state = {name: torch.as_tensor(value, device=device) for name, value in observation["state"].items()}
        state["army"] = state["army"].clamp_min(0)
        return cls(state, torch.as_tensor(observation["stats"], device=device),
                   observation["width"], observation["height"],
                   torch.as_tensor(observation["visible"], device=device))

    @classmethod
    def from_observations(cls, observations, device):
        batches = [cls.from_observation(observation, device) for observation in observations]
        first = batches[0]
        return cls({name: torch.cat([batch.state[name] for batch in batches]) for name in first.state},
                   torch.cat([batch.stats for batch in batches]), first.width, first.height,
                   torch.cat([batch.visible for batch in batches]))

    @classmethod
    def from_observation(cls, observation, device):
        state = {name: torch.tensor(getattr(observation, name).copy(), device=device).reshape(1, -1)
                 for name in ("terrain", "structure", "owner", "army", "alive")}
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
