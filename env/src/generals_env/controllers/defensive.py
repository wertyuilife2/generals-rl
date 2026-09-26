"""Defensive baseline prioritizing support of threatened holdings."""

from ..state import EMPTY, MoveMode
from .base import BaseAI


class DefensiveAI(BaseAI):
    weights = {"fortify": 1.5, "gather": 1.0, "expand": 0.3}
    temperature = 0.7
    path_lifetime = 15

    def _mode(self, strategy, source, target, obs):
        if strategy == "expand" and obs.owner.ravel()[target] == EMPTY:
            return MoveMode.HALF
        if strategy == "fortify" and obs.army.ravel()[source] > 10:
            return MoveMode.HALF
        return MoveMode.ALL_BUT_ONE
