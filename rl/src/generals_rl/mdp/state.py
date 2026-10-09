"""Algorithm-neutral tensor views of the existing simulator protocol."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GameBatch:
    # Borrowed views: retain only explicit clones across a simulator step.
    state: dict[str, torch.Tensor]
    stats: torch.Tensor
    width: int
    height: int

    @classmethod
    def from_backend(cls, env, device):
        return cls({k: torch.as_tensor(v, device=device) for k, v in env.state.items()},
                   torch.as_tensor(env.stats, device=device), env.width, env.height)

    @classmethod
    def from_observation(cls, observation, device):
        """Build inference features using only a player's permitted information.

        Hidden army sentinels become zero before log1p to avoid -inf * 0.
        No hidden map, enemy general positions, or private totals are supplied.
        """
        obs = observation
        state = {name: torch.tensor(getattr(obs, name).copy(), device=device).reshape(1, -1)
                 for name in ("terrain", "structure", "owner", "army", "alive")}
        state["army"] = state["army"].clamp_min(0)
        state["tick"] = torch.tensor([obs.tick], device=device, dtype=torch.long)
        state["terminated"] = torch.tensor([obs.terminated], device=device, dtype=torch.bool)
        stats = torch.stack([torch.tensor(getattr(obs.stats, name).copy(), device=device)
                             for name in ("army", "territory", "cities")], dim=-1)[None]
        return cls(state, stats, obs.width, obs.height)

    @property
    def num_envs(self):
        return self.state["army"].shape[0]

    @property
    def players(self):
        return self.state["alive"].shape[1]

    @property
    def device(self):
        return self.stats.device

    def learner_stats(self, player_ids):
        return self.stats[torch.arange(self.num_envs, device=self.device), player_ids].clone()
