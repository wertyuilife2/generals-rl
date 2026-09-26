"""Seeded local random policy, lightly biased toward useful moves."""

import numpy as np

from ..state import EMPTY, Action, Direction, MoveMode, Structure
from .base import BaseAI, movable_army


class RandomAI(BaseAI):
    def act(self, observation):
        obs = observation
        self._pending = None
        if obs.player_id != self.player_id or obs.terminated or not obs.alive[self.player_id]:
            return Action.wait()
        self._remember(obs)
        sources = np.flatnonzero((obs.owner.ravel() == self.player_id) & (obs.army.ravel() > 1))
        candidates = []
        for source in self.rng.permutation(sources):
            for direction, target in self._neighbors(int(source), obs):
                mode = MoveMode.HALF if obs.army.ravel()[source] > 6 and self.rng.random() < 0.3 else MoveMode.ALL_BUT_ONE
                score = float(self.rng.random()) * 5.0
                owner = int(obs.owner.ravel()[target])
                moved = movable_army(obs.army.ravel()[source], mode)
                if owner == EMPTY:
                    score += 2.0
                elif owner != self.player_id:
                    score += 1.5 if moved > obs.army.ravel()[target] else -3.0
                    if obs.structure.ravel()[target] == Structure.GENERAL and moved > obs.army.ravel()[target]:
                        score += 4.0
                elif obs.army.ravel()[target] > obs.army.ravel()[source]:
                    score += 0.5
                candidates.append((score, Action.move(int(source), Direction(direction), mode)))
        return max(candidates, key=lambda pair: pair[0])[1] if candidates else Action.wait()
