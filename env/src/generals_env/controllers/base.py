"""Controller protocol and bounded, observation-only heuristic planning.

The planners deliberately receive no engine or map layout.  Their terrain map
contains only facts seen by that player; unexplored cells are optimistic paths
that are rechecked as the army approaches them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..observation import Observation
from ..pathfinding import shortest_path
from ..state import (
    EMPTY, UNKNOWN, UNKNOWN_OWNER, Action, Direction,
    MoveMode, MoveResult, Structure, Terrain,
)


@runtime_checkable
class Controller(Protocol):
    def reset(self, player_id: int, seed: int) -> None: ...
    def act(self, observation: Observation) -> Action: ...
    def on_result(self, result: MoveResult) -> None: ...


def adjacent(cell: int, direction: int, height: int, width: int) -> int | None:
    """Return a four-neighbor without allowing row wraparound."""
    if not 0 <= cell < height * width or not 0 <= direction < 4:
        return None
    y, x = divmod(int(cell), width)
    dy, dx = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(direction)]
    y, x = y + dy, x + dx
    return y * width + x if 0 <= y < height and 0 <= x < width else None


def direction_between(source: int, target: int, height: int, width: int) -> int | None:
    for direction in range(4):
        if adjacent(source, direction, height, width) == target:
            return direction
    return None


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
        terrain = self.terrain.ravel()
        sources = [int(i) for i in np.flatnonzero((owner == self.player_id) & (army > 1))]
        if not sources:
            return []
        frontiers: dict[int, int] = {}
        threats: dict[int, int] = {}
        for cell in np.flatnonzero(owner == self.player_id):
            neighbors = list(self._neighbors(int(cell), obs))
            frontiers[int(cell)] = sum(memory_owner[t] != self.player_id for _, t in neighbors)
            threats[int(cell)] = sum(max(0, int(army[t])) for _, t in neighbors if owner[t] >= 0 and owner[t] != self.player_id)
        source_scores = [float(army[s]) * (1.0 + 0.25 * bool(frontiers[s])) for s in sources]
        # The high-temperature source selection still favors stronger stacks;
        # a stable sorted tail gives a hard limit on the amount of planning.
        ordered = sorted(zip(sources, source_scores), key=lambda pair: (-pair[1], pair[0]))
        sources = [s for s, _ in ordered[: self.max_sources]]
        if len(sources) > 1:
            chosen = softmax_choice(self.rng, [np.log1p(float(army[s])) for s in sources], self.temperature)
            sources.insert(0, sources.pop(chosen))

        attacks = np.flatnonzero((memory_owner >= 0) & (memory_owner != self.player_id))
        cities = np.flatnonzero((structures == Structure.CITY) & (memory_owner != self.player_id))
        expansion = np.flatnonzero(np.isin(memory_owner, [EMPTY, UNKNOWN_OWNER]) & (terrain != Terrain.MOUNTAIN))
        owned = np.flatnonzero(owner == self.player_id)
        result: list[Candidate] = []

        for source in sources:
            sy, sx = divmod(source, obs.width)
            distance = lambda target: abs(int(target) // obs.width - sy) + abs(int(target) % obs.width - sx)
            local: dict[str, list[Candidate]] = {strategy: [] for strategy in self.weights}
            for strategy in self.weights:
                if strategy in ("attack", "capture"):
                    pool = attacks if strategy == "attack" else cities
                    for raw_target in pool:
                        target = int(raw_target)
                        d = distance(target)
                        defending = max(0, int(self.army.ravel()[target]))
                        if int(army[source]) - d <= defending:
                            continue
                        value = 50.0 if structures[target] == Structure.GENERAL else (18.0 if structures[target] == Structure.CITY else 8.0)
                        score = value / (d + 1) + min(float(army[source] - defending), 100.0) * 0.01
                        local[strategy].append(Candidate(source, target, strategy, score))
                elif strategy in ("explore", "expand"):
                    # Nearest candidates bound ranking work and encourage
                    # taking frontier cells before marching through owned land.
                    nearest = sorted(expansion, key=lambda t: (distance(t), int(t)))[:16]
                    for raw_target in nearest:
                        target = int(raw_target)
                        d = distance(target)
                        if d >= int(army[source]):
                            continue
                        unknown_neighbors = sum(terrain[t] == UNKNOWN for _, t in self._neighbors(target, obs))
                        score = (5.0 + 0.4 * unknown_neighbors) / (d + 1)
                        local[strategy].append(Candidate(source, target, strategy, score))
                else:  # merge / gather / fortify
                    for raw_target in owned:
                        target = int(raw_target)
                        if target == source:
                            continue
                        d = distance(target)
                        threat = threats[target]
                        # Monotone merging prevents stacks from oscillating
                        # between two similarly useful friendly squares.
                        if strategy != "fortify" and (army[target], -target) <= (army[source], -source):
                            continue
                        if strategy == "fortify" and threat == 0:
                            continue
                        if structures[source] == Structure.GENERAL and threats[source] > int(army[source]) // 2:
                            continue
                        value = 3.0 + min(float(army[target]), 100) * 0.06 + frontiers[target]
                        if threat:
                            value += min(threat, 100) * 0.15
                            if structures[target] == Structure.GENERAL:
                                value += 20.0
                        local[strategy].append(Candidate(source, target, strategy, value / (d + 1)))
                local[strategy].sort(key=lambda c: (-c.score, c.target))
                result.extend(local[strategy][:4])
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
