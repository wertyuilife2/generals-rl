"""Seeded local random policy, lightly biased toward useful moves."""

import numpy as np

from generals.core.state import EMPTY, Action, Direction, MoveMode, Structure, Terrain
from generals_baselines.base import BaseAI, grid_neighbors


class RandomAI(BaseAI):
    def act(self, observation):
        obs = observation
        self._pending = None
        if obs.player_id != self.player_id or obs.terminated or not obs.alive[self.player_id]:
            return Action.wait()
        self._remember(obs)
        owner, army, structure = obs.owner.ravel(), obs.army.ravel(), obs.structure.ravel()
        sources = self.rng.permutation(np.flatnonzero((owner == self.player_id) & (army > 1)))
        neighbors = grid_neighbors(obs.height, obs.width)[sources]
        rows, directions = np.nonzero((neighbors >= 0) & (self.terrain.ravel()[neighbors] != Terrain.MOUNTAIN))
        if not rows.size:
            return Action.wait()
        targets, sources = neighbors[rows, directions], sources[rows]
        # Consume precisely the reference policy's random stream: an optional
        # mode draw, then a score draw, for each candidate in source/direction order.
        mode_draw = army[sources] > 6
        score_indices = np.cumsum(mode_draw.astype(np.intp) + 1) - 1
        draws = self.rng.random(int(score_indices[-1]) + 1)
        half = mode_draw & (draws[np.maximum(score_indices - 1, 0)] < .3)
        moved = np.where(half, army[sources] // 2, army[sources] - 1)
        scores = draws[score_indices] * 5.0
        empty, hostile = owner[targets] == EMPTY, (owner[targets] != self.player_id) & (owner[targets] != EMPTY)
        scores += np.where(empty, 2.0, np.where(hostile, np.where(moved > army[targets], 1.5, -3.0),
                            np.where(army[targets] > army[sources], .5, 0.0)))
        scores += (hostile & (structure[targets] == Structure.GENERAL) & (moved > army[targets])) * 4.0
        best = int(np.argmax(scores))
        return Action.move(int(sources[best]), Direction(int(directions[best])),
                           MoveMode.HALF if half[best] else MoveMode.ALL_BUT_ONE)
