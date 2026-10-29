"""Training-only action outcome metrics."""
import torch

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
