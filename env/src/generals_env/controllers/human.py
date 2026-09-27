"""Human input queue, independent of Tk and of the authoritative game state."""

from __future__ import annotations

from collections import deque

from ..observation import Observation
from ..state import Action, Direction, MoveMode, MoveResult, Terrain
from .base import adjacent, direction_between


class HumanController:
    max_queue = 64

    def __init__(self, player_id: int = 0, seed: int = 0):
        self.reset(player_id, seed)

    def reset(self, player_id: int, seed: int) -> None:
        self.player_id = int(player_id)
        self.selected: int | None = None
        self.anchor: int | None = None
        self.cursor: int | None = None
        self.half = False
        self.queue: deque[Action] = deque()
        self.message = "Select one of your tiles."
        self._enabled = True
        self._pending: Action | None = None
        self._observation: Observation | None = None
        self._confirmed_anchor: int | None = None

    def _observe(self, obs: Observation) -> bool:
        if obs.player_id != self.player_id:
            self.message = "Observation belongs to another player."
            return False
        self._observation = obs
        self._confirmed_anchor = None
        if not obs.alive[self.player_id]:
            self.eliminate()
        return self._enabled and not obs.terminated

    def _anchor_owned(self) -> bool:
        obs = self._observation
        return bool(self.anchor is not None and (self._confirmed_anchor == self.anchor or (obs is not None and 0 <= self.anchor < obs.size and obs.owner.ravel()[self.anchor] == self.player_id)))

    def select(self, cell: int, obs: Observation, force: bool = False) -> None:
        if not self._observe(obs) or not 0 <= cell < obs.size:
            return
        cell = int(cell)
        if self.selected is not None and not force:
            highlighted = self.cursor if self.queue else self.selected
            if cell == highlighted:
                self.toggle_half()
                return
            if self.cursor is not None:
                direction = direction_between(self.cursor, cell, obs.height, obs.width)
                if direction is not None:
                    self.enqueue(direction, obs)
                    return
        if obs.owner.ravel()[cell] != self.player_id:
            self.message = "Select a friendly tile or queue an adjacent move."
            return
        had_queue = bool(self.queue)
        self.queue.clear()
        self._pending = None
        self.selected = self.anchor = self.cursor = cell
        self.half = False
        self.message = "New source selected; previous queue cleared." if had_queue else "Source selected. Click a neighbor to move."

    def enqueue(self, direction: int, obs: Observation) -> None:
        if not self._observe(obs):
            return
        if self.selected is None or self.cursor is None:
            self.message = "Select one of your tiles first."
            return
        if len(self.queue) >= self.max_queue:
            self.message = "Queue is full (64 moves)."
            return
        target = adjacent(self.cursor, int(direction), obs.height, obs.width)
        if target is None:
            self.message = "That move leaves the map."
            return
        if obs.terrain.ravel()[target] == Terrain.MOUNTAIN:
            self.message = "Mountains cannot be entered."
            return
        self.queue.append(Action.move(self.cursor, Direction(direction), MoveMode.HALF if self.half else MoveMode.ALL_BUT_ONE))
        self.cursor = target
        self.half = False
        self.message = f"Queued {len(self.queue)} move(s)."

    def toggle_half(self) -> None:
        if self._enabled:
            self.half = not self.half
            self.message = "Next queued move: half army." if self.half else "Next queued move: all but one."

    def clear_queue(self, obs: Observation | None = None) -> None:
        # GUI feedback can occur after later players changed ownership. Use
        # its latest snapshot instead of the earlier own-turn observation.
        if obs is not None and not self._observe(obs):
            return
        self.queue.clear()
        self._pending = None
        if self._anchor_owned():
            self.cursor = self.anchor
            if self.selected is not None:
                self.selected = self.anchor
        else:
            self.selected = self.anchor = self.cursor = None
        self.message = "Queue cleared."

    def undo(self, obs: Observation | None = None) -> None:
        if obs is not None and not self._observe(obs):
            return
        if not self.queue:
            self.message = "No queued move to undo."
            return
        removed = self.queue.pop()
        self.cursor = int(removed.source)
        if not self.queue:
            if self._anchor_owned():
                self.cursor = self.anchor
            else:
                self.selected = self.anchor = self.cursor = None
        self.message = f"Last move removed; {len(self.queue)} remaining."

    def deselect(self) -> None:
        self.selected = None
        self.message = "Selection hidden; queued moves remain active."

    def eliminate(self) -> None:
        self._enabled = False
        self.queue.clear()
        self.selected = self.anchor = self.cursor = None
        self._pending = None
        self._confirmed_anchor = None
        self.half = False
        self.message = "You have been eliminated."

    def act(self, observation: Observation) -> Action:
        self._pending = None
        if not self._observe(observation):
            return Action.wait()
        obs = observation
        if not self.queue:
            if self.anchor is not None and not self._anchor_owned():
                self.selected = self.anchor = self.cursor = None
            return Action.wait()
        action = self.queue[0]
        source = int(action.source)
        if not 0 <= source < obs.size or obs.owner.ravel()[source] != self.player_id:
            self.clear_queue()
            self.selected = self.anchor = self.cursor = None
            self.message = "Source lost; queued path cleared."
            return Action.wait()
        target = adjacent(source, int(action.direction), obs.height, obs.width)
        if target is None or obs.terrain.ravel()[target] == Terrain.MOUNTAIN:
            self.clear_queue()
            self.message = "Path blocked; queued path cleared."
            return Action.wait()
        self._pending = action
        return action

    def on_result(self, result: MoveResult) -> None:
        if result.player_id != self.player_id or self._pending is None or result.action != self._pending:
            return
        self._pending = None
        if self.queue and self.queue[0] == result.action:
            self.queue.popleft()
        if result.valid and result.target_owned:
            self.anchor = int(result.target)
            self._confirmed_anchor = self.anchor
            if self.selected is not None:
                self.selected = self.anchor
            if not self.queue:
                self.cursor = self.anchor
            self.message = f"Move completed; {len(self.queue)} queued."
        else:
            self.queue.clear()
            self.anchor = int(result.source) if result.valid else None
            self._confirmed_anchor = self.anchor
            self.cursor = self.anchor
            if self.selected is not None:
                self.selected = self.anchor
            self.message = "Attack did not capture the target; path cleared." if result.valid else f"Move rejected: {result.reason}; path cleared."
