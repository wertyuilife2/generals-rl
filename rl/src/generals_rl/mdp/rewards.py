"""Replaceable reward terms; contains no algorithm or simulator transitions."""
from dataclasses import dataclass
import math
from numbers import Real

import torch


@dataclass(frozen=True)
class RewardConfig:
    territory: float = 0.05
    cities: float = 0.2
    army: float = 0.001
    win: float = 1.0
    loss: float = -1.0
    tick: float = -0.001
    potential_gamma: float | None = None

    def __post_init__(self):
        for name in ("territory", "cities", "army", "win", "loss", "tick", "potential_gamma"):
            value = getattr(self, name)
            if name == "potential_gamma" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"reward {name} must be a finite real number")
        if self.potential_gamma is not None and not 0 <= self.potential_gamma <= 1:
            raise ValueError("reward potential_gamma must be in [0, 1] or None")


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
        if cfg.potential_gamma is None:
            delta = after_metrics.float() - before_metrics.float()
        else:
            # True terminals are absorbing with Phi=0. A time limit retains
            # Phi(final state), which cancels against the critic bootstrap.
            after = after_metrics.float().masked_fill((terminated | ~learner_alive).unsqueeze(1), 0)
            delta = cfg.potential_gamma * after - before_metrics.float()
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
