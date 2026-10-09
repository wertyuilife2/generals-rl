"""Bounded, observation-only heuristic planning.

The planners deliberately receive no engine or map layout.  Their terrain map
contains only facts seen by that player; unexplored cells are optimistic paths
that are rechecked as the army approaches them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from generals.core.grid import adjacent, direction_between
from generals.core.observation import Observation
from generals.core.pathfinding import shortest_path
from generals.core.state import (
    EMPTY, UNKNOWN, UNKNOWN_OWNER, Action, Direction,
    MoveMode, MoveResult, Structure, Terrain,
)


@lru_cache(maxsize=32)
def grid_neighbors(height: int, width: int) -> np.ndarray:
    """Geometry only: no hidden terrain or shared mutable policy state."""
    cells = np.arange(height * width).reshape(height, width)
    graph = np.full((height * width, 4), -1, dtype=np.intp)
    graph[cells[1:].ravel(), 0] = cells[:-1].ravel()
    graph[cells[:, :-1].ravel(), 1] = cells[:, 1:].ravel()
    graph[cells[:-1].ravel(), 2] = cells[1:].ravel()
    graph[cells[:, 1:].ravel(), 3] = cells[:, :-1].ravel()
    graph.flags.writeable = False
    return graph


def movable_army(army: int, mode: MoveMode = MoveMode.ALL_BUT_ONE) -> int:
    return max(0, int(army) // 2 if mode == MoveMode.HALF else int(army) - 1)


def softmax_choice(rng: np.random.Generator, scores: list[float], temperature: float) -> int:
    """Stable softmax; callers give finite scores and at least one candidate."""
    values = np.asarray(scores, dtype=np.float64)
    values = (values - values.max()) / max(temperature, 1e-6)
    weights = np.exp(np.clip(values, -700.0, 0.0))
    return int(rng.choice(len(scores), p=weights / weights.sum()))


@dataclass(frozen=True)
class Candidate:
    source: int
    target: int
    strategy: str
    score: float


class BaseAI:
    """A small shared planner with bounded retries and result-driven paths."""

    weights = {"merge": 0.5, "explore": 0.8, "attack": 1.5}
    temperature = 0.5
    path_lifetime = 20
    max_sources = 6
    max_path_attempts = 12
    memory_ttl = 50

    def __init__(self, player_id: int = 0, seed: int = 0):
        self.reset(player_id, seed)

    def reset(self, player_id: int, seed: int) -> None:
        self.player_id = int(player_id)
        self.rng = np.random.default_rng(seed)
        self.terrain: np.ndarray | None = None
        self.structure: np.ndarray | None = None
        self.owner: np.ndarray | None = None
        self.army: np.ndarray | None = None
        self.last_seen: np.ndarray | None = None
        self.path: list[int] = []
        self.path_strategy = ""
        self.path_tick = 0
        self._pending: Action | None = None

    def _remember(self, obs: Observation) -> None:
        if self.terrain is None or self.terrain.shape != obs.terrain.shape:
            self.terrain = np.full(obs.terrain.shape, UNKNOWN, dtype=np.uint8)
            self.structure = np.full(obs.terrain.shape, UNKNOWN, dtype=np.uint8)
            self.owner = np.full(obs.terrain.shape, UNKNOWN_OWNER, dtype=np.int16)
            self.army = np.full(obs.terrain.shape, -1, dtype=np.int64)
            self.last_seen = np.full(obs.terrain.shape, -1, dtype=np.int64)
            self.path = []
        visible = obs.visible
        self.terrain[visible] = obs.terrain[visible]
        self.structure[visible] = obs.structure[visible]
        self.owner[visible] = obs.owner[visible]
        self.army[visible] = obs.army[visible]
        self.last_seen[visible] = obs.tick

        # A remembered general is a location, never a claim about current army.
        hidden = ~visible
        stale = hidden & (obs.tick - self.last_seen > self.memory_ttl)
        ordinary = self.structure != Structure.GENERAL
        lost_own = hidden & (self.owner == self.player_id)
        forget = (stale & ordinary) | lost_own
        self.owner[forget] = UNKNOWN_OWNER
        self.army[hidden] = -1
        for pid, alive in enumerate(obs.alive):
            if not alive:
                defeated = hidden & (self.owner == pid)
                self.owner[defeated] = UNKNOWN_OWNER
                self.structure[defeated & (self.structure == Structure.GENERAL)] = Structure.CITY

    def _neighbors(self, cell: int, obs: Observation):
        terrain = self.terrain.ravel()
        for direction in range(4):
            target = adjacent(cell, direction, obs.height, obs.width)
            if target is not None and terrain[target] != Terrain.MOUNTAIN:
                yield direction, target

    def _mode(self, strategy: str, source: int, target: int, obs: Observation) -> MoveMode:
        return MoveMode.ALL_BUT_ONE

    def _safe_action(self, source: int, target: int, strategy: str, obs: Observation) -> Action | None:
        direction = direction_between(source, target, obs.height, obs.width)
        owner, army = obs.owner.ravel(), obs.army.ravel()
        if direction is None or owner[source] != self.player_id or army[source] < 2:
            return None
        if obs.terrain.ravel()[target] == Terrain.MOUNTAIN:
            return None
        mode = self._mode(strategy, source, target, obs)
        moved = movable_army(army[source], mode)
        # Attacks remain legal in the engine; heuristic AIs avoid knowingly
        # spending a stack on an attack that cannot take its next square.
        if owner[target] not in (self.player_id, EMPTY, UNKNOWN_OWNER) and moved <= army[target]:
            return None
        return Action.move(source, Direction(direction), mode)

    def _follow_path(self, obs: Observation) -> Action | None:
        if len(self.path) < 2 or obs.tick - self.path_tick >= self.path_lifetime:
            self.path = []
            return None
        destination_owner = int(self.owner.ravel()[self.path[-1]])
        strategy = self.path_strategy
        target = self.path[-1]
        invalid_target = (
            (strategy == "attack" and (destination_owner < 0 or destination_owner == self.player_id))
            or (strategy == "capture" and (destination_owner == self.player_id or self.structure.ravel()[target] != Structure.CITY))
            or (strategy in ("explore", "expand") and destination_owner not in (EMPTY, UNKNOWN_OWNER))
            or (strategy in ("merge", "gather", "fortify") and destination_owner != self.player_id)
        )
        if invalid_target:
            self.path = []
            return None
        action = self._safe_action(self.path[0], self.path[1], self.path_strategy, obs)
        if action is None:
            self.path = []
        return action

    def _candidates(self, obs: Observation) -> list[Candidate]:
        owner, army = obs.owner.ravel(), obs.army.ravel()
        memory_owner, structures = self.owner.ravel(), self.structure.ravel()
        terrain, remembered_army = self.terrain.ravel(), self.army.ravel()
        sources = np.flatnonzero((owner == self.player_id) & (army > 1))
        if not sources.size:
            return []
        owned = np.flatnonzero(owner == self.player_id)
        graph = grid_neighbors(obs.height, obs.width)
        nearby = graph[owned]
        valid = (nearby >= 0) & (terrain[nearby] != Terrain.MOUNTAIN)
        frontiers = np.zeros(obs.size, dtype=np.int64)
        threats = np.zeros(obs.size, dtype=np.int64)
        frontiers[owned] = np.sum(valid & (memory_owner[nearby] != self.player_id), axis=1)
        hostile = valid & (owner[nearby] >= 0) & (owner[nearby] != self.player_id)
        threats[owned] = np.sum(np.where(hostile, np.maximum(army[nearby], 0), 0), axis=1)
        source_scores = army[sources].astype(float) * (1.0 + .25 * (frontiers[sources] != 0))
        sources = sources[np.lexsort((sources, -source_scores))[:self.max_sources]].tolist()
        if len(sources) > 1:
            chosen = softmax_choice(self.rng, [np.log1p(float(army[s])) for s in sources], self.temperature)
            sources.insert(0, sources.pop(chosen))

        attacks = np.flatnonzero((memory_owner >= 0) & (memory_owner != self.player_id))
        cities = np.flatnonzero((structures == Structure.CITY) & (memory_owner != self.player_id))
        expansion = np.flatnonzero(((memory_owner == EMPTY) | (memory_owner == UNKNOWN_OWNER)) & (terrain != Terrain.MOUNTAIN))
        ys, xs = np.divmod(np.arange(obs.size), obs.width)
        result: list[Candidate] = []
        for source in sources:
            sy, sx = divmod(source, obs.width)
            distance = np.abs(ys - sy) + np.abs(xs - sx)
            for strategy in self.weights:
                if strategy in ("attack", "capture"):
                    pool = attacks if strategy == "attack" else cities
                    defending = np.maximum(remembered_army[pool], 0)
                    keep = int(army[source]) - distance[pool] > defending
                    targets, defending = pool[keep], defending[keep]
                    values = np.where(structures[targets] == Structure.GENERAL, 50.0,
                                      np.where(structures[targets] == Structure.CITY, 18.0, 8.0))
                    values = values / (distance[targets] + 1) + np.minimum((army[source] - defending).astype(float), 100.0) * .01
                elif strategy in ("explore", "expand"):
                    # expansion is ascending, so stable sorting also preserves
                    # the reference planner's cell-index tie break.
                    nearest = expansion[np.argsort(distance[expansion], kind="stable")[:16]]
                    targets = nearest[distance[nearest] < int(army[source])]
                    neighbors = graph[targets]
                    unknown = np.sum((neighbors >= 0) & (terrain[neighbors] == UNKNOWN), axis=1)
                    values = (5.0 + .4 * unknown) / (distance[targets] + 1)
                else:
                    if structures[source] == Structure.GENERAL and threats[source] > int(army[source]) // 2:
                        continue
                    keep = owned != source
                    if strategy == "fortify":
                        keep &= threats[owned] != 0
                    else:
                        keep &= (army[owned] > army[source]) | ((army[owned] == army[source]) & (owned < source))
                    targets = owned[keep]
                    threat = threats[targets]
                    values = 3.0 + np.minimum(army[targets].astype(float), 100) * .06 + frontiers[targets]
                    values += np.minimum(threat, 100) * .15
                    values += ((threat != 0) & (structures[targets] == Structure.GENERAL)) * 20.0
                    values /= distance[targets] + 1
                # Only the four best candidates of a strategy are consumed.
                # Construct objects after ranking, preserving RNG and tie order.
                best = np.lexsort((targets, -values))[:4]
                result.extend(Candidate(source, int(targets[i]), strategy, float(values[i])) for i in best)
        return result

    def _plan(self, candidate: Candidate, obs: Observation) -> list[int] | None:
        terrain = self.terrain.copy()
        # Do not plot a route through a known garrison that this stack cannot
        # beat, even if its destination is an attractive square beyond it.
        hostile = (self.owner != self.player_id) & (self.owner != EMPTY) & (self.owner != UNKNOWN_OWNER)
        too_strong = hostile & (self.army >= int(obs.army.ravel()[candidate.source]) - 1)
        terrain[too_strong] = Terrain.MOUNTAIN
        path = shortest_path(terrain, candidate.source, candidate.target, unknown_passable=True)
        if not path or len(path) < 2:
            return None
        # Account for leave-one costs and visible garrisons along the route.
        # Future enemy moves are unknowable, so every step is rechecked later.
        strength = int(obs.army.ravel()[candidate.source])
        for source, target in zip(path, path[1:]):
            mode = self._mode(candidate.strategy, source, target, obs)
            strength = movable_army(strength, mode)
            if self.owner.ravel()[target] == self.player_id:
                strength += max(0, int(obs.army.ravel()[target]))
            else:
                strength -= max(0, int(self.army.ravel()[target]))
            if strength <= 0:
                return None
        return path

    def act(self, observation: Observation) -> Action:
        obs = observation
        self._pending = None
        if obs.player_id != self.player_id or obs.terminated or not obs.alive[self.player_id]:
            self.path = []
            return Action.wait()
        self._remember(obs)
        action = self._follow_path(obs)
        if action is not None:
            self._pending = action
            return action
        candidates = self._candidates(obs)
        for _ in range(min(self.max_path_attempts, len(candidates))):
            strategies = [s for s in self.weights if any(c.strategy == s for c in candidates)]
            if not strategies:
                break
            strategy = strategies[softmax_choice(self.rng, [self.weights[s] for s in strategies], self.temperature)]
            group = [c for c in candidates if c.strategy == strategy]
            candidate = max(group, key=lambda c: c.score)
            candidates.remove(candidate)
            path = self._plan(candidate, obs)
            if path is None:
                continue
            self.path = path
            self.path_strategy = strategy
            self.path_tick = obs.tick
            action = self._follow_path(obs)
            if action is not None:
                self._pending = action
                return action
        # One-step fallback is bounded and prevents an unreachable distant
        # objective from suppressing useful expansion next to the player.
        fallback: list[tuple[float, Action]] = []
        for source in np.flatnonzero((obs.owner.ravel() == self.player_id) & (obs.army.ravel() > 1)):
            for _, target in self._neighbors(int(source), obs):
                if obs.owner.ravel()[target] == self.player_id:
                    continue
                action = self._safe_action(int(source), target, "expand", obs)
                if action is not None:
                    value = 20.0 if obs.structure.ravel()[target] == Structure.GENERAL else 1.0
                    fallback.append((value, action))
        if fallback:
            self._pending = max(fallback, key=lambda pair: pair[0])[1]
            return self._pending
        return Action.wait()

    def on_result(self, result: MoveResult) -> None:
        if result.player_id != self.player_id or self._pending is None:
            return
        if result.action != self._pending:
            return
        if result.valid and result.target_owned and len(self.path) >= 2 and self.path[:2] == [result.source, result.target]:
            self.path.pop(0)
            if len(self.path) < 2:
                self.path = []
        else:
            self.path = []
        self._pending = None
