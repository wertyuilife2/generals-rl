"""Training tensor views extend the public policy observation batch."""
import torch
from ..policy.state import GameBatch as PublicBatch


class GameBatch(PublicBatch):
    @classmethod
    def from_backend(cls, env, device, valid=None):
        return cls({k: torch.as_tensor(v, device=device) for k, v in env.state.items()},
                   torch.as_tensor(env.stats, device=device), env.width, env.height,
                   valid=None if valid is None else torch.as_tensor(valid, device=device))

    def fog_obstacles(self, player_ids, visible):
        from generals.core.batch_observation import public_obstacles
        return public_obstacles(self.state, player_ids) & ~visible & self.valid_cells

    def learner_stats(self, player_ids):
        return self.stats[torch.arange(self.num_envs, device=self.device), player_ids].clone()
