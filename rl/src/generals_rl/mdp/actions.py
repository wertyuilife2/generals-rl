"""Shared discrete action vocabulary for PPO, inference, and future search."""
import torch

from .state import GameBatch


class EncodedActionSpace:
    """WAIT=0; MOVE=1 + source*8 + direction*2 + mode (engine encoding)."""

    def __init__(self, width: int, height: int, *, productive_actions: bool = False):
        self.width, self.height = width, height
        self.size = 1 + width * height * 8
        self.productive_actions = productive_actions
        if productive_actions:
            self.semantics = "productive_half_post_growth_two_canonical_all_v1"
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
        owned = state["owner"] == player_ids[:, None]
        active = state["alive"].gather(1, player_ids[:, None]).squeeze(1) & ~state["terminated"]
        if self.productive_actions:
            # Growth precedes both players' moves. Own source strength and the
            # clock are public; do not anticipate the other player's action.
            next_tick = state["tick"][:, None] + 1
            grows = (next_tick % 2 == 0) & ((state["structure"] != 0) | (next_tick % 50 == 0))
            # Both ALL_BUT_ONE and HALF move positive armies exactly at >=2.
            # Avoid adding to int64 army values at the representable limit.
            productive = (state["army"] > 1) | ((state["army"] == 1) & grows)
            owned = owned & productive
        moves = (owned[:, :, None] & in_bounds[None] &
                 (state["terrain"][:, targets] != 1) & active[:, None, None])
        moves = moves[..., None].expand(-1, -1, -1, 2).clone()
        if self.productive_actions:
            # Own source=2 after public growth: HALF and ALL move the same army.
            equivalent = (((state["army"] == 2) & ~grows) |
                          ((state["army"] == 1) & grows))
            moves[..., 1] &= ~equivalent[..., None]
        moves = moves.reshape(batch.num_envs, -1)
        # A terminal or eliminated player's observation still needs a valid
        # distribution. WAIT is its absorbing action. The default matches core;
        # the productive variant removes predictable zero moves and equivalent
        # HALF aliases, with ALL as their canonical representative.
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
    return EncodedActionSpace(config.width, config.height,
                              productive_actions=config.productive_actions)


class ActionMetrics:
    """Per-lane native action outcomes; rates use joint decision ticks.

    A backtrack reverses two positive moves on adjacent ticks. WAIT, rejected
    moves, zero-army moves, and episode resets break that adjacency.
    """

    names = ("wait_rate", "zero_moved_rate", "backtrack_rate", "moved_army")

    def __init__(self, num_envs, device):
        self.previous_move = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)

    def reset(self, mask=None):
        if mask is None:
            self.previous_move.fill_(-1)
        else:
            self.previous_move[mask] = -1

    def update(self, moves):
        executed = moves[:, 0].bool() & moves[:, 1].bool()
        moved = executed & (moves[:, 8] == 2)
        positive = moved & (moves[:, 4] > 0)
        reverse = (positive & (moves[:, 2] == self.previous_move[:, 1])
                   & (moves[:, 3] == self.previous_move[:, 0]))
        self.previous_move.copy_(torch.where(positive[:, None], moves[:, 2:4], -1))
        return {
            "wait_rate": (executed & (moves[:, 8] == 1)).float(),
            "zero_moved_rate": (moved & (moves[:, 4] == 0)).float(),
            "backtrack_rate": reverse.float(),
            "moved_army": torch.where(positive, moves[:, 4], 0).float(),
        }
