"""Zero-sum Elo; multiplayer uses averaged pair comparisons at pre-match ratings."""
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


def pair_score(winner_seat, left_seat, right_seat):
    """The winner beats everyone; all other pairs tie, including timeouts."""
    if winner_seat == left_seat:
        return 1.0
    if winner_seat == right_seat:
        return 0.0
    return 0.5


def update_multiplayer(ratings, winner_seat, config=EloConfig()):
    """Update all participants simultaneously, normalizing K by opponents.

    Each comparison uses ratings from before the match, making the result
    independent of seat iteration order. For two players this is ordinary Elo.
    No ordering among non-winners is inferred from eliminations or army counts.
    """
    count = len(ratings)
    if not 2 <= count <= 8:
        raise ValueError("provide ratings for 2–8 players")
    if winner_seat is not None and (type(winner_seat) is not int or not 0 <= winner_seat < count):
        raise ValueError("invalid winner seat")
    if not all(math.isfinite(value) for value in ratings):
        raise ValueError("ratings must be finite")
    deltas = [0.0] * count
    for left in range(count):
        for right in range(left + 1, count):
            score = pair_score(winner_seat, left, right)
            delta = config.k_factor / (count - 1) * (
                score - expected_score(ratings[left], ratings[right], config.scale))
            deltas[left] += delta
            deltas[right] -= delta
    return [rating + delta for rating, delta in zip(ratings, deltas)]
