"""Attack-oriented baseline."""

from .base import BaseAI


class AggressiveAI(BaseAI):
    weights = {"merge": 0.5, "explore": 0.8, "attack": 1.5}
    temperature = 0.5
    path_lifetime = 20
