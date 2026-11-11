"""Training tensor views extend the public policy observation batch."""
import torch
from ..policy.state import GameBatch as PublicBatch


class GameBatch(PublicBatch):
    def select(self, rows):
        """A compact batch for reset lanes, retaining training visibility rules."""
        return type(self)({key: value.index_select(0, rows) for key, value in self.state.items()},
                          self.stats.index_select(0, rows), self.width, self.height,
                          None if self.visible is None else self.visible.index_select(0, rows),
                          self.valid_cells.index_select(0, rows))

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
