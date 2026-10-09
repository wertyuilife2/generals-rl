"""Zero-sum Elo with wins=1, draws=0.5, losses=0."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EloConfig:
    initial: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0

    def __post_init__(self):
        for key in ("initial", "k_factor", "scale"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"elo.{key} must be finite")
        if self.k_factor <= 0 or self.scale <= 0:
            raise ValueError("Elo k_factor and scale must be positive")


def expected_score(left, right, scale=400.0):
    exponent = max(-300.0, min(300.0, (right - left) / scale))
    return 1.0 / (1.0 + 10.0 ** exponent)


def update(left, right, score, config=EloConfig()):
    if score not in (0.0, 0.5, 1.0):
        raise ValueError("Elo score must be 0, 0.5, or 1")
    delta = config.k_factor * (score - expected_score(left, right, config.scale))
    return left + delta, right - delta
