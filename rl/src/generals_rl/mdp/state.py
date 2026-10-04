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
