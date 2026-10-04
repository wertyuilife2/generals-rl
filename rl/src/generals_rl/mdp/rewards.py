"""Replaceable reward terms; contains no algorithm or simulator transitions."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardConfig:
    territory: float = 0.05
    cities: float = 0.2
    army: float = 0.001
    win: float = 1.0
    loss: float = -1.0
    tick: float = -0.001


@dataclass(frozen=True)
class RewardResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


class ShapedReward:
    def __init__(self, config=None):
        self.config = RewardConfig() if config is None else config

    def evaluate(self, before_metrics, after_metrics, terminated, truncated,
                 winner, learner_alive, player_ids):
        cfg = self.config
        delta = after_metrics.float() - before_metrics.float()
        terms = {
            "territory": cfg.territory * delta[:, 1],
            "cities": cfg.cities * delta[:, 2],
            "army": cfg.army * delta[:, 0],
            "terminal": (terminated & (winner == player_ids)).float() * cfg.win
                        + (~learner_alive).float() * cfg.loss,
            "tick": torch.full_like(delta[:, 0], cfg.tick),
        }
        # A timeout is a truncated trajectory, never an artificial defeat.
        return RewardResult(sum(terms.values()), terms)
