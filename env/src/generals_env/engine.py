"""Deterministic CPU rules, with no policy, clock, renderer, or random dependency."""

from collections.abc import Sequence

import numpy as np

from .state import (
    Action, ActionKind, Direction, GameState, MoveMode, MoveResult,
    Structure, Terrain, TickResult, _as_int, compute_stats,
)


_OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_MAX_ARMY = np.iinfo(np.int64).max


class CoreEngine:
    """Own a state and advance fixed-order, growth-before-movement transactions.

    begin_tick/apply_action/finish_tick allow a Runner to observe immediately
    before each player's turn. step() reuses these same methods for preselected
    actions. Invalid game moves consume a turn; malformed API calls raise before
    consuming one. A caller must finish a transaction before starting the next.
    """

    def __init__(self, state: GameState, debug: bool = False):
        if not isinstance(state, GameState):
            raise TypeError("state must be GameState")
        state.validate()
        self.state = state
        self.debug = debug
        self._active = False
        self._cursor = 0
        self._moves: list[MoveResult] = []

    @property
    def next_player(self) -> int | None:
        """Next surviving, unprocessed player; None outside/end of a tick."""
        if not self._active or self.state.terminated:
            return None
        while self._cursor < self.state.player_count and not self.state.alive[self._cursor]:
            self._cursor += 1
        return self._cursor if self._cursor < self.state.player_count else None

    def begin_tick(self) -> None:
        if self._active:
            raise RuntimeError("finish the active tick before beginning another")
        if self.state.terminated:
            raise RuntimeError("a terminated game cannot advance")
        if self.debug:
            self.state.validate()
        next_tick = self.state.tick + 1
        owned = self.state.owner >= 0
        growing = np.zeros_like(owned)
        if next_tick % 2 == 0:
            growing |= owned & (self.state.structure != Structure.NONE)
        if next_tick % 50 == 0:
            growing |= owned & (self.state.structure == Structure.NONE)
        if np.any(self.state.army[growing] == _MAX_ARMY):
            raise OverflowError("army exceeds int64 capacity")
        self.state.tick = next_tick
        self.state.army[growing] += 1
        self._active = True
        self._cursor = 0
        self._moves = []

    def apply_action(self, player_id: int, action: Action) -> MoveResult:
        if not self._active:
            raise RuntimeError("begin_tick must precede apply_action")
        player_id = _as_int(player_id, "player_id")
        if not 0 <= player_id < self.state.player_count:
            raise ValueError("player_id is outside the player slots")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        expected = self.next_player
        if expected is None:
            raise RuntimeError("no remaining player may act in this tick")
        if player_id != expected:
            raise ValueError(f"player {expected} must act next, received {player_id}")
        result = self._execute(player_id, action)
        self._moves.append(result)
        self._cursor = player_id + 1
        return result

    def _execute(self, player_id: int, action: Action) -> MoveResult:
        if action.kind == ActionKind.WAIT:
            return MoveResult(player_id, action, True, "WAIT")

        state = self.state
        source = action.source
        target = -1

        def invalid(reason: str) -> MoveResult:
            return MoveResult(player_id, action, False, reason, source=source, target=target)

        if not 0 <= source < state.layout.size:
            return invalid("OUT_OF_BOUNDS")
        y, x = divmod(source, state.layout.width)
        dy, dx = _OFFSETS[int(action.direction)]
        target_y, target_x = y + dy, x + dx
        if not (0 <= target_y < state.layout.height and 0 <= target_x < state.layout.width):
            return invalid("OUT_OF_BOUNDS")
        target = target_y * state.layout.width + target_x
        owner = state.owner.ravel()
        army = state.army.ravel()
        structure = state.structure.ravel()
        if owner[source] != player_id:
            return invalid("NOT_OWNER")
        count = int(army[source])
        if state.layout.terrain.ravel()[target] == Terrain.MOUNTAIN:
            return invalid("MOUNTAIN")
        moved = max(0, count - 1) if action.mode == MoveMode.ALL_BUT_ONE else count // 2
        defender = int(owner[target])
        defense = int(army[target])
        if defender == player_id and defense + moved > _MAX_ARMY:
            raise OverflowError("merged army exceeds int64 capacity")
        army[source] -= moved
        captured = False
        eliminated = None
        if defender == player_id:
            army[target] += moved
        elif moved > defense:
            owner[target] = player_id
            army[target] = moved - defense
            captured = True
            if structure[target] == Structure.GENERAL:
                eliminated = defender
                state.alive[defender] = False
                structure[target] = Structure.CITY
                inherited = owner == defender
                owner[inherited] = player_id
                army[inherited] = np.maximum(1, army[inherited] // 2)
                survivors = np.flatnonzero(state.alive)
                if survivors.size == 1:
                    state.terminated = True
                    state.winner_id = int(survivors[0])
        else:
            army[target] -= moved
        return MoveResult(player_id, action, True, "MOVED", source, target,
                          moved, captured, eliminated, bool(owner[target] == player_id))

    def finish_tick(self) -> TickResult:
        if not self._active:
            raise RuntimeError("there is no active tick to finish")
        if self.next_player is not None:
            raise RuntimeError("each remaining player must act before finish_tick")
        if self.debug:
            self.state.validate()
        moves = tuple(self._moves)
        result = TickResult(
            self.state.tick, moves, compute_stats(self.state),
            self.state.terminated, self.state.winner_id,
            tuple(move.eliminated for move in moves if move.eliminated is not None),
        )
        self._active = False
        self._moves = []
        return result

    def step(self, actions: Sequence[Action]) -> TickResult:
        """Execute one action per slot; already dead players require WAIT slots.

        Actions of players eliminated earlier in this tick are simply skipped.
        All input validation precedes growth, so malformed batches do not start
        a partially applied tick.
        """
        if self._active:
            raise RuntimeError("step cannot run inside an active tick")
        if self.state.terminated:
            raise RuntimeError("a terminated game cannot advance")
        if not isinstance(actions, Sequence):
            raise TypeError("actions must be a sequence indexed by player slot")
        if len(actions) != self.state.player_count:
            raise ValueError("provide exactly one action per player slot")
        for pid, action in enumerate(actions):
            if not isinstance(action, Action):
                raise TypeError("every submitted action must be an Action")
            if not self.state.alive[pid] and action.kind != ActionKind.WAIT:
                raise ValueError("players eliminated before this tick require WAIT placeholders")
        self.begin_tick()
        while (pid := self.next_player) is not None:
            self.apply_action(pid, actions[pid])
        return self.finish_tick()
