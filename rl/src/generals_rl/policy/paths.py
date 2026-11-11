"""Cached terrain distances from player-local knowledge, never hidden state.

Unknown public mountain/city hints are blocked until terrain is observed.
Known cities are traversable; armies and ownership do not affect edge costs.
Distances use four-neighbour unit edges, -1 for unreachable/blocked cells and
-2 when the target general is unknown or ambiguous. Only two feature planes
are emitted; they are snapshots, not recomputed during PPO minibatch gathers.
"""
from collections import deque
import math

import torch


class TerrainDistances:
    def __init__(self, games, players, width, height, device):
        self.width, self.height, self.players = width, height, players
        self.cells = width * height
        if self.cells > 32766:
            raise ValueError("terrain-distance canvas exceeds int16 distance capacity")
        shape = (games, players)
        # Independent per-target keys let CUDA programs run without write races.
        self.blocked = torch.ones((*shape, 2, self.cells), dtype=torch.bool, device=device)
        self.goals = torch.full((*shape, 2), -2, dtype=torch.int32, device=device)
        self.distances = torch.full((*shape, 2, self.cells), -2, dtype=torch.int16, device=device)

    @torch.no_grad()
    def update(self, terrain, general_owner, fog, valid, player_ids, lane_ids=None, active=None):
        count = len(player_ids)
        rows = torch.arange(count, device=terrain.device) if lane_ids is None else lane_ids
        active = torch.ones(count, dtype=torch.bool, device=terrain.device) if active is None else active
        if terrain.is_cuda:
            from .paths_cuda import update_distances
            update_distances(self, terrain, general_owner, fog, valid, player_ids, rows, active)
            return
        # CPU inference/reference only; CUDA never copies to host or loops here.
        for i, (row, seat) in enumerate(zip(rows.tolist(), player_ids.tolist())):
            if not active[i]:
                continue
            known = terrain[row, seat]
            blocked = ~valid[i] | (known == 1) | ((known < 0) & fog[i])
            general = general_owner[row, seat]
            goals = []
            for marker in (general == seat, (general >= 0) & (general != seat)):
                found = (marker & ~blocked).nonzero().flatten()
                goals.append(int(found[0]) if len(found) == 1 else -1)
            for index, goal in enumerate(goals):
                changed = not torch.equal(self.blocked[row, seat, index], blocked)
                if changed or self.goals[row, seat, index] != goal:
                    self.distances[row, seat, index] = self._bfs(blocked, goal)
                self.blocked[row, seat, index] = blocked
            self.goals[row, seat] = torch.tensor(goals, dtype=torch.int32)

    def _bfs(self, blocked, goal):
        result = torch.full((self.cells,), -2 if goal < 0 else -1, dtype=torch.int16)
        if goal < 0:
            return result
        result[goal] = 0
        pending = deque([goal])
        while pending:
            cell = pending.popleft()
            y, x = divmod(cell, self.width)
            neighbours = ((cell-self.width, y > 0), (cell+self.width, y+1 < self.height),
                          (cell-1, x > 0), (cell+1, x+1 < self.width))
            for other, inside in neighbours:
                if inside and not blocked[other] and result[other] < 0:
                    result[other] = result[cell] + 1
                    pending.append(other)
        return result

    def features(self, player_ids, lane_ids=None):
        rows = torch.arange(len(player_ids), device=self.distances.device) if lane_ids is None else lane_ids
        value = self.distances[rows, player_ids].float()
        # Valid paths can be much longer than width+height; use canvas cell count.
        return torch.where(value >= 0, torch.log1p(value.clamp_min(0)) / math.log1p(self.cells), value)

    def reset(self, mask=None):
        target = slice(None) if mask is None else mask
        self.blocked[target] = True
        self.goals[target] = -2  # Invalidates the cache even for an unknown goal.
        self.distances[target] = -2

    def copy_prefix(self, other):
        for name in ("blocked", "goals", "distances"):
            getattr(self, name)[:len(other.goals)] = getattr(other, name)

    def clone(self):
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        for name in ("blocked", "goals", "distances"):
            setattr(result, name, getattr(self, name).clone())
        return result
