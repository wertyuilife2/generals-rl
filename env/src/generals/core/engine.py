"""Reference implementation of the standard official joint-tick rules."""
from collections.abc import Sequence
import numpy as np
from generals.core.state import (
    Action, ActionKind, GameState, MoveMode, MoveResult, Structure, Terrain,
    TickResult, NEUTRAL, compute_stats,
)
from generals.protocol.rules import MAX_GAME_TICKS, MAX_IDLE_TICKS, SURRENDER_GRACE_TICKS

_OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def mark_dead(state, pid):
    if state.alive[pid]:
        state.death_order[pid] = int(state.death_order.max()) + 1
        state.alive[pid] = False
        remaining = np.flatnonzero(state.alive)
        if remaining.size == 1:
            state.winner_id = int(remaining[0])


def update_terminal(state, reason):
    survivors = np.flatnonzero(state.alive)
    state.terminated = survivors.size <= 1
    if survivors.size:
        state.winner_id = int(survivors[0]) if survivors.size == 1 else None
    if state.terminated:
        state.end_reason = reason
        if state.winner_id is not None and state.alive[state.winner_id] and state.death_order.max() >= 0:
            last = int(state.death_order.argmax())
            if not state.capture_mask[state.winner_id] & (1 << last):
                credit_kill(state, state.winner_id, last)


def credit_kill(state, killer, victim):
    state.kills[killer] += 1
    state.capture_mask[killer] |= 1 << victim


def surrender(state, pid):
    """Depart at this boundary, leaving a capturable general for 50 ticks."""
    if not 0 <= pid < state.player_count:
        raise ValueError("invalid player id")
    if state.terminated or not state.alive[pid] or state.leave_tick[pid] >= 0:
        return
    state.leave_tick[pid] = max(1, state.tick)
    if state.tick:
        mark_dead(state, pid)
        update_terminal(state, 2)


def neutralize(state, pid):
    general = int(state.general_pos[pid])
    if general < 0:
        return
    owner = state.owner.ravel()
    if owner[general] == pid:
        land = owner == pid
        owner[land] = np.where((state.army.ravel()[land] == 0) & (state.structure.ravel()[land] == 0), -1, NEUTRAL)
        state.structure.ravel()[general] = Structure.CITY
        candidates = np.where(state.alive, state.damage[:, pid], 0)
        if candidates.max() > 0:
            credit_kill(state, int(candidates.argmax()), pid)
    state.general_pos[pid] = -1


def remember_captured_cities(state):
    # Official fog memory is keyed by original spawn, even after a trade.
    from generals.core.observation import visibility_mask
    for pid in range(state.player_count):
        visible = visibility_mask(state.owner, pid).ravel()
        for slot, spawn in enumerate(state.spawn_pos):
            if state.general_pos[slot] < 0 and spawn >= 0 and visible[spawn]:
                state.city_known[pid] |= 1 << slot


class CoreEngine:
    """All players decide on the same state; ordered moves precede growth."""

    def __init__(self, state: GameState, debug=False):
        state.validate()
        self.state, self.debug = state, debug

    def surrender(self, pid):
        surrender(self.state, pid)

    def _candidate(self, pid, action):
        state = self.state
        source, target = action.source, -1
        if action.kind == ActionKind.WAIT:
            return MoveResult(pid, action, True, "WAIT")
        reason = "OUT_OF_BOUNDS"
        if 0 <= source < state.layout.size:
            y, x = divmod(source, state.layout.width)
            dy, dx = _OFFSETS[int(action.direction)]
            if 0 <= y + dy < state.layout.height and 0 <= x + dx < state.layout.width:
                target = (y + dy) * state.layout.width + x + dx
                reason = ("NOT_OWNER" if state.owner.flat[source] != pid else
                          "MOUNTAIN" if state.layout.terrain.flat[target] == Terrain.MOUNTAIN else
                          "NO_ARMY" if state.army.flat[source] <= 1 else "MOVED")
        return MoveResult(pid, action, reason == "MOVED", reason, source, target)

    def _ordered(self, candidates):
        state = self.state
        candidates.sort(key=lambda m: (
            state.owner.flat[m.target] != m.player_id,
            state.structure.flat[m.target] == Structure.GENERAL,
            -int(state.army.flat[m.source]),
            -m.player_id if state.tick % 2 else m.player_id))
        ordered = []
        while candidates:
            index = next((i for i, move in enumerate(candidates)
                          if not any(other.target == move.source and other.source != move.target
                                     for other in candidates if other is not move)), 0)
            ordered.append(candidates.pop(index))
        return ordered

    def _transfer(self, move):
        count = int(self.state.army.flat[move.source])
        return count // 2 if move.action.mode == MoveMode.HALF else max(0, count - 1)

    def _trade_partner(self, move, pending):
        s, pid = self.state, move.player_id
        defender = int(s.owner.flat[move.target])
        if defender < 0 or defender == pid or s.general_pos[defender] != move.target:
            return None
        if not self._candidate(pid, move.action).valid:
            return None
        for other in pending:
            if (other.player_id != defender or other.target != s.general_pos[pid]
                    or not self._candidate(defender, other.action).valid):
                continue
            defense, own_defense = int(s.army.flat[move.target]), int(s.army.flat[other.target])
            if other.source == move.target:
                defense -= self._transfer(other)
            if move.source == other.target:
                own_defense -= self._transfer(move)
            if self._transfer(move) > defense and self._transfer(other) > own_defense:
                return other
        return None

    def _trade(self, move, other):
        s = self.state
        owner, army = s.owner.ravel(), s.army.ravel()
        a, b = move.player_id, other.player_id
        ga, gb = int(s.general_pos[a]), int(s.general_pos[b])
        ma, mb = self._transfer(move), self._transfer(other)
        army[move.source] -= ma
        army[other.source] -= mb
        da, db = int(army[ga]), int(army[gb])
        for cell in range(s.layout.size):
            if cell not in (ga, gb) and owner[cell] in (a, b):
                owner[cell] = b if owner[cell] == a else a
                army[cell] = army[cell] // 2 + army[cell] % 2
        owner[ga], owner[gb] = b, a
        army[ga], army[gb] = mb - da, ma - db
        s.general_pos[a], s.general_pos[b] = gb, ga
        return (MoveResult(a, move.action, True, "TRADE", move.source, gb, ma, True, None, True),
                MoveResult(b, other.action, True, "TRADE", other.source, ga, mb, True, None, True))

    def _execute(self, move):
        s, pid = self.state, move.player_id
        source, target = move.source, move.target
        owner, army, structure = s.owner.ravel(), s.army.ravel(), s.structure.ravel()
        if owner[source] != pid:
            return MoveResult(pid, move.action, False, "NOT_OWNER", source, target)
        moved = self._transfer(move)
        defender, defense = int(owner[target]), int(army[target])
        if army[source] == 0 or (defender != pid and moved == 0):
            return MoveResult(pid, move.action, False, "NO_ARMY", source, target)
        army[source] -= moved
        captured, eliminated = False, None
        if defender == pid:
            army[target] += moved
        else:
            if defender >= 0:
                s.damage[pid, defender] += min(moved, defense)
            if moved > defense:
                captured = True
                owner[target], army[target] = pid, moved - defense
                if structure[target] == Structure.GENERAL:
                    eliminated = defender if s.alive[defender] else None
                    mark_dead(s, defender)
                    credit_kill(s, pid, defender)
                    s.general_pos[defender] = -1
                    structure[target] = Structure.CITY
                    inherited = owner == defender
                    owner[inherited] = pid
                    army[inherited] = army[inherited] // 2 + army[inherited] % 2
            else:
                army[target] -= moved
        return MoveResult(pid, move.action, True, "MOVED", source, target, moved,
                          captured, eliminated, bool(owner[target] == pid))

    def step(self, actions: Sequence[Action]) -> TickResult:
        s = self.state
        if s.terminated:
            raise RuntimeError("a terminated game cannot advance")
        if not isinstance(actions, Sequence) or len(actions) != s.player_count:
            raise ValueError("provide exactly one action per player slot")
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("every submitted action must be an Action")
        if any(not s.alive[p] and a.kind != ActionKind.WAIT for p, a in enumerate(actions)):
            raise ValueError("eliminated players require WAIT placeholders")
        previous, dead_before = compute_stats(s), s.alive.copy()
        results = [self._candidate(p, a) for p, a in enumerate(actions) if s.alive[p]]
        pending = self._ordered([m for m in results if m.valid and m.reason == "MOVED"])
        results = [m for m in results if not m.valid or m.reason == "WAIT"]
        activity = False
        while pending:
            move = pending.pop(0)
            other = self._trade_partner(move, pending)
            if other is not None:
                pending.remove(other)
                results.extend(self._trade(move, other))
                activity = True
            else:
                activity |= s.owner.flat[move.source] == move.player_id
                results.append(self._execute(move))
        s.idle_ticks = 0 if activity else s.idle_ticks + 1
        reason = 1
        if s.tick > MAX_GAME_TICKS or s.idle_ticks > MAX_IDLE_TICKS:
            reason = 3 if s.tick > MAX_GAME_TICKS else 4
            ordered = sorted(np.flatnonzero(s.alive),
                             key=lambda p: (previous.army[p], previous.territory[p], p))
            for pid in ordered[:-1]:
                mark_dead(s, pid)
        owned = s.owner >= 0
        next_tick = s.tick + 1
        if next_tick > np.iinfo(np.int64).max:
            raise OverflowError('tick exceeds int64')
        for pid in range(s.player_count):
            cells = s.owner == pid
            gain = (int(np.count_nonzero(cells & (s.structure != 0))) if next_tick % 2 == 0 else 0)
            gain += int(cells.sum()) if next_tick % 50 == 0 else 0
            if sum(map(int, s.army[cells])) + gain > np.iinfo(np.int64).max:
                raise OverflowError('army growth exceeds int64')
        s.tick = next_tick
        if s.tick % 2 == 0:
            s.army[owned & (s.structure != Structure.NONE)] += 1
        if s.tick % 50 == 0:
            s.army[owned] += 1
        for pid in range(s.player_count):
            if s.leave_tick[pid] == s.tick and s.alive[pid]:
                mark_dead(s, pid)
                reason = 2
        s.warn_afk.fill(0)
        warn_turn = s.afk_turn - 22
        if s.afk_turn and warn_turn <= s.tick <= s.afk_turn:
            from generals.core.observation import visibility_mask
            for pid in range(s.player_count):
                safe = (previous.territory[pid] >= 11 or
                        np.any((s.owner == pid) & (s.structure == Structure.CITY)) or
                        np.any(visibility_mask(s.owner, pid) & (s.owner >= 0) & (s.owner != pid)))
                if s.alive[pid] and not safe:
                    if s.tick < s.afk_turn:
                        s.warn_afk[pid] = 1
                    else:
                        s.leave_tick[pid] = s.tick
                        mark_dead(s, pid)
                        reason = 5
        for pid in range(s.player_count):
            if s.leave_tick[pid] >= 0 and s.tick == s.leave_tick[pid] + SURRENDER_GRACE_TICKS:
                neutralize(s, pid)
        remember_captured_cities(s)
        update_terminal(s, reason)
        if self.debug:
            s.validate()
        eliminated = sorted(np.flatnonzero(dead_before & ~s.alive), key=lambda p: s.death_order[p])
        return TickResult(s.tick, tuple(sorted(results, key=lambda m: m.player_id)), compute_stats(s), s.terminated,
                          s.winner_id, tuple(map(int, eliminated)))
