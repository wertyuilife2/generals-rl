"""Shared discrete action vocabulary for PPO, inference, and future search."""
import torch

from .state import GameBatch


class EncodedActionSpace:
    """WAIT=0; MOVE=1 + source*8 + direction*2 + mode (engine encoding)."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.size = 1 + width * height * 8
        self._tables = {}

    def destinations(self, device):
        key = str(device)
        if key not in self._tables:
            cells = torch.arange(self.width * self.height, device=device)
            y, x = cells // self.width, cells % self.width
            targets = torch.stack((cells - self.width, cells + 1,
                                   cells + self.width, cells - 1), -1)
            in_bounds = torch.stack((y > 0, x < self.width - 1,
                                     y < self.height - 1, x > 0), -1)
            self._tables[key] = targets.clamp(0, len(cells) - 1), in_bounds
        return self._tables[key]

    def mask(self, batch: GameBatch, player_ids: torch.Tensor) -> torch.Tensor:
        targets, in_bounds = self.destinations(batch.device)
        state = batch.state
        owned = (state["owner"] == player_ids[:, None]) & (state['army'] > 1) & batch.valid_cells
        active = state["alive"].gather(1, player_ids[:, None]).squeeze(1) & ~state["terminated"]
        moves = (owned[:, :, None] & in_bounds[None] &
                 (state["terrain"][:, targets] != 1) & batch.valid_cells[:, targets] & active[:, None, None])
        moves = moves[..., None].expand(-1, -1, -1, 2).reshape(batch.num_envs, -1)
        # A terminal or eliminated player's observation still needs a valid
        # distribution. WAIT is its absorbing action.
        return torch.cat((torch.ones_like(active[:, None]), moves), -1)

    def validate(self, actions: torch.Tensor, mask: torch.Tensor):
        if actions.shape not in ((len(mask),), (len(mask), 1)):
            raise ValueError("actions must be [num_envs] or [num_envs, 1]")
        if actions.dtype not in (torch.int32, torch.int64):
            raise TypeError("encoded actions must be integer tensors")
        encoded = actions.reshape(-1).long()
        if bool(((encoded < 0) | (encoded >= self.size)).any()):
            raise ValueError("encoded action is out of range")
        if not bool(mask.gather(1, encoded[:, None]).all()):
            raise ValueError("action was masked at the joint decision boundary")
        return encoded


def make_action_space(config):
    """Use the same configured action boundary for training and deployment."""
    return EncodedActionSpace(config.width, config.height)
