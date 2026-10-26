"""Training tensor views extend the public policy observation batch."""
import torch
from ..policy.state import GameBatch as PublicBatch


class GameBatch(PublicBatch):
    @classmethod
    def from_backend(cls, env, device):
        return cls({k: torch.as_tensor(v, device=device) for k, v in env.state.items()},
                   torch.as_tensor(env.stats, device=device), env.width, env.height)

    def learner_stats(self, player_ids):
        return self.stats[torch.arange(self.num_envs, device=self.device), player_ids].clone()
