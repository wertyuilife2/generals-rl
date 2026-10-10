"""Operational diagnostics and resource ownership outside the frozen game rules."""
import gzip
import json
import numpy as np

from generals import Action, make_engine
from generals.core.observation import snapshot
from generals.core.recording import state_digest, state_from_dict
from generals.core.state import NEUTRAL, Structure, compute_stats, GameState, MapLayout
from .league.game import RecordedGame as RulesGame, board_view


EXECUTION_POLICY = 'decision-timeout-death-v2'


def decision_timeout(details):
    phase = details.get('phase')
    # Older late-return diagnostics omitted phase, but recorded the decision
    # limit. Initialization/destruction have a distinct two-second watchdog.
    return details.get('kind') == 'timeout' and (phase == 'decision' or
        (phase is None and details.get('limit_ns') == 10_000_000))


def apply_forfeits(engine, events):
    """Arena lifecycle event; movement/combat/growth still use env's CPU kernel.

    Abandoned troops are neutral, stop growing and can be captured normally.
    This extends Arena's neutral cells beyond mapgen's starting neutral cities.
    Never reconstruct the engine from a mid-game Arena state: initialize from
    the original map and replay events, keeping env/RL's initial-state contract.
    """
    state = engine.state
    changed = False
    for event in events:
        seat = event['seat']
        if event['tick'] == state.tick and state.alive[seat]:
            state.owner[state.owner == seat] = NEUTRAL
            state.structure.ravel()[state.general_pos[seat]] = Structure.CITY
            state.alive[seat] = False
            changed = True
    if changed:
        survivors = np.flatnonzero(state.alive)
        state.terminated = len(survivors) <= 1
        state.winner_id = int(survivors[0]) if len(survivors) == 1 else None
        engine.invalidate_cache()
    return changed


def state_from_snapshot(value, forfeits):
    if not forfeits:
        return state_from_dict(value)
    # Trusted match-process snapshot, which may contain abandoned neutral land.
    return GameState(MapLayout(np.asarray(value['terrain'])),
                     *(np.asarray(value[key]) for key in
                       ('structure', 'owner', 'army', 'alive', 'general_pos')),
                     value['tick'], value['terminated'], value['winner_id'])


class _RecordedAction:
    """Restore a surviving controller's memory while replaying committed actions."""
    def __init__(self, controller, action, warm):
        self.controller, self.action, self.warm = controller, action, warm

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def act(self, observation):
        if self.warm:
            values = getattr(self.controller, 'values', None)
            saved = values.copy() if values is not None else None
            self.controller.act(observation)
            # Successful restoration is not a new tournament decision. A new
            # timeout still retains its diagnostic record and forfeits the seat.
            if saved is not None:
                values[:] = saved
        return self.action


class KillTracker:
    """Observe authoritative move results without changing the engine or rules."""

    def __init__(self, engine, players):
        self.engine = engine
        self.kills = [0] * players

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def step(self, actions):
        outcome = self.engine.step(actions)
        for move in outcome.moves:
            if move.eliminated is not None:
                self.kills[move.player_id] += 1
        return outcome


class RecordedGame(RulesGame):
    """Execution metadata is separate from the league's frozen game rules."""

    def __init__(self, plan, config, controllers, forfeits=()):
        super().__init__(plan, config, controllers)
        self.engine = KillTracker(self.engine, len(controllers))
        self.forfeits = list(forfeits)
        self._apply_forfeits()

    def _apply_forfeits(self):
        if apply_forfeits(self.engine, self.forfeits):
            self.stats = compute_stats(self.state)

    def advance(self, human_actions=None):
        actions = dict(human_actions or {})
        controllers = self.controllers
        self.controllers = list(controllers)
        for event in self.forfeits:
            if event['tick'] <= self.state.tick:
                seat = event['seat']
                self.controllers[seat] = None
                actions[seat] = Action.wait()
        try:
            super().advance(actions)
        finally:
            self.controllers = controllers
        self._apply_forfeits()

    def advance_recorded(self, actions, *, warm=False):
        controllers = self.controllers
        self.controllers = [_RecordedAction(c, actions[i], warm) if c is not None else None
                            for i, c in enumerate(controllers)]
        try:
            super().advance(dict(enumerate(actions)))
        finally:
            self.controllers = controllers
        self._apply_forfeits()

    def result(self):
        value = {**super().result(), 'kills': self.engine.kills.copy(), 'execution_policy': EXECUTION_POLICY}
        if self.forfeits:
            eliminated = self.eliminations.copy()
            for event in self.forfeits:
                seat, tick = event['seat'], event['tick']
                eliminated[seat] = tick if eliminated[seat] is None else min(eliminated[seat], tick)
            value.update(forfeits=self.forfeits, capture_ticks=self.eliminations.copy(),
                         elimination_ticks=eliminated)
            if any(e['tick'] == self.state.tick for e in self.forfeits) and self.state.terminated:
                value['reason'] = 'surrender'
        return value

    def replay_blob(self):
        if not self.forfeits:
            return super().replay_blob()
        return gzip.compress(json.dumps({'schema': 2, 'initial': self.initial,
            'actions': self.actions, 'digests': self.digests, 'forfeits': self.forfeits},
            separators=(',', ':')).encode(), mtime=0)


def mark_forfeits(board, forfeits):
    board['forfeited'] = [e['seat'] for e in forfeits if e['tick'] <= board['tick']]
    for seat in board['forfeited']:
        board['alive'][seat] = False
    return board


def verified_replay(blob, value):
    """Execute actions and lifecycle events; verify every committed state."""
    trace = json.loads(gzip.decompress(blob))
    if (trace['schema'] not in (1, 2) or len(trace['actions']) != value['ticks'] or
            len(trace['digests']) != len(trace['actions'])):
        raise ValueError('invalid replay length or schema')
    events = trace.get('forfeits', [])
    if events != value.get('forfeits', []):
        raise ValueError('replay forfeit mismatch')
    state = state_from_dict(trace['initial'])
    if state_digest(state) != value['initial_digest']:
        raise ValueError('replay initial digest mismatch')
    engine = make_engine(state)
    try:
        apply_forfeits(engine, events)
        yield state, None
        for actions, expected in zip(trace['actions'], trace['digests']):
            outcome = engine.step([Action(*a) for a in actions])
            if state_digest(state) != expected:
                raise ValueError(f'replay diverged at tick {state.tick}')
            apply_forfeits(engine, events)
            yield state, outcome
        if state_digest(state) != value['final_digest']:
            raise ValueError('replay final digest mismatch')
    finally:
        engine.close()


def replay_frames(blob, value, stride=1, compact=False):
    frames, previous = [], None
    for state, _ in verified_replay(blob, value):
        if not frames or state.tick % stride == 0 or state.tick == value['ticks']:
            current = np.stack([state.owner.ravel(), state.army.ravel(), state.structure.ravel()], axis=1)
            if compact and frames:
                changed = np.flatnonzero(np.any(current != previous, axis=1))
                stats = compute_stats(state)
                frame = {'tick':state.tick, 'cells':[[int(i), *current[i].tolist()] for i in changed],
                         'alive':state.alive.tolist(), 'stats':{key:getattr(stats,key).tolist()
                         for key in ('army','territory','cities')}}
            else:
                frame = board_view(snapshot(state))
            frames.append(mark_forfeits(frame, value.get('forfeits', [])))
            previous = current.copy()
    return frames


def timing(game):
    return [{'seat': seat, 'entrant_id': game.plan['entrant_ids'][seat], **controller.decision_metrics}
            for seat, controller in enumerate(game.controllers)
            if hasattr(controller, 'decision_metrics')]


def result(game):
    value = game.result()
    metrics = timing(game)
    return {**value, 'decision_timing': metrics} if metrics else value


def failure(game, error):
    return {**game.plan, 'ticks': game.state.tick, 'initial_digest': game.initial_digest,
            'final_digest': state_digest(game.state), 'reason': 'policy_error',
            'policy_error': str(error), 'error_detail': getattr(error, 'details', {}),
            'decision_timing': timing(game), **({'forfeits':game.forfeits} if game.forfeits else {})}


def close_game(game):
    for controller in game.controllers:
        close = getattr(controller, 'close', None)
        if close is not None:
            close()
    close = getattr(game.engine, 'close', None)
    if close is not None:
        close()
