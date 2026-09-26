"""Expansion-oriented baseline with opportunistic city capture."""

from ..state import EMPTY, MoveMode
from .base import BaseAI


class ExpansionAI(BaseAI):
    weights = {"expand": 1.5, "capture": 1.2, "gather": 0.5}
    temperature = 0.6
    path_lifetime = 25

    def _mode(self, strategy, source, target, obs):
        if strategy == "expand" and obs.owner.ravel()[target] == EMPTY and obs.army.ravel()[source] > 6:
            return MoveMode.HALF
        return MoveMode.ALL_BUT_ONE
