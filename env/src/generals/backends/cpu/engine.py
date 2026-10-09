"""Single-game adapter to the shared native CPU backend."""
from collections.abc import Sequence
import numpy as np
from . import env as cpu
from generals.core.state import Action, ActionKind, GameState, MoveResult, PlayerStats, TickResult


class CPUEngine:
    """Single-lane CoreEngine.step adapter, using native state without tick copies.

    Uses CPUEnv with one lane and one OpenMP thread. The C++ kernel is the
    only rules implementation here; Python maps the public action/result types.
    State arrays are borrowed views; call invalidate_cache after direct edits.
    """
    def __init__(self, state):
        if not isinstance(state, GameState):
            raise TypeError('state must be GameState')
        self.state = state
        self.env = cpu.CPUEnv.from_states([state], threads=1)
        self.actions = np.zeros((1, state.player_count), dtype=np.int64)
        shape = state.owner.shape
        for name in ('structure', 'owner', 'army'):
            setattr(state, name, self.env.state[name][0].reshape(shape))
        for name in ('alive', 'general_pos'):
            setattr(state, name, self.env.state[name][0])

    def step(self, actions):
        if self.env._closed:
            raise RuntimeError('CPU engine is closed')
        state = self.state
        if state.terminated:
            raise RuntimeError('a terminated game cannot advance')
        if not isinstance(actions, Sequence):
            raise TypeError('actions must be a sequence indexed by player slot')
        if len(actions) != state.player_count:
            raise ValueError('provide exactly one action per player slot')
        for seat, action in enumerate(actions):
            if not isinstance(action, Action):
                raise TypeError('every submitted action must be an Action')
            if not state.alive[seat] and action.kind != ActionKind.WAIT:
                raise ValueError('players eliminated before this tick require WAIT placeholders')
            # Structured CoreEngine actions permit arbitrary integer sources;
            # keep their invalid-move semantics without overflowing int64 codes.
            self.actions[0, seat] = (0 if action.kind == ActionKind.WAIT else
                                    action.encode() if 0 <= action.source < state.layout.size else -1)
        result = self.env.step(self.actions)
        self.env.check_errors()
        state.tick = int(self.env.state['tick'][0])
        state.terminated = bool(self.env.state['terminated'][0])
        winner = int(self.env.state['winner'][0])
        state.winner_id = None if winner < 0 else winner
        stats = result.stats[0]
        stats = PlayerStats(stats[:, 0], stats[:, 1], stats[:, 2])
        reasons = ('SKIPPED', 'WAIT', 'MOVED', 'OUT_OF_BOUNDS', 'NOT_OWNER', 'MOUNTAIN')
        moves = []
        for seat, row in enumerate(result.moves[0]):
            active, valid, _, target, moved, captured, eliminated, owned, reason = map(int, row)
            if active:
                moves.append(MoveResult(seat, actions[seat], bool(valid), reasons[reason],
                    actions[seat].source, target, moved, bool(captured),
                    eliminated if eliminated >= 0 else None, bool(owned)))
        return TickResult(state.tick, tuple(moves), stats, state.terminated, state.winner_id,
                          tuple(move.eliminated for move in moves if move.eliminated is not None))

    def close(self):
        self.env.close()

    def invalidate_cache(self):
        self.env.state['tick'][0] = self.state.tick
        self.env.state['terminated'][0] = self.state.terminated
        self.env.state['winner'][0] = -1 if self.state.winner_id is None else self.state.winner_id
        self.env.invalidate_cache()
