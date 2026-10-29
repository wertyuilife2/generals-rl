"""Deterministic action traces, authoritative verification and browser frames."""
import gzip
import json

import numpy as np

from generals import Action, make_engine
from generals.core.observation import snapshot
from generals.core.recording import state_digest, state_from_dict, state_to_dict
from generals.core.state import rank_players
from generals.protocol.rules import RULES_VERSION

from .game import apply_forfeits, mark_forfeits
from .observation import board_view


def encode_replay(initial, actions, digests, forfeits=()):
    """Encode the same deterministic action trace for scalar and batch games."""
    if len(actions) != len(digests):
        raise ValueError("replay actions and digests must have the same length")
    initial = initial if isinstance(initial, dict) else state_to_dict(initial)
    value = {"schema": 3, "rules": RULES_VERSION, "initial": initial,
             "actions": actions, "digests": digests}
    if forfeits:
        value["forfeits"] = list(forfeits)
    return gzip.compress(json.dumps(value, separators=(",", ":")).encode(), mtime=0)


def verified_replay(blob, value):
    """Execute actions and lifecycle events; verify every committed state."""
    trace = json.loads(gzip.decompress(blob))
    if (trace['schema'] != 3 or trace['rules'] != RULES_VERSION or len(trace['actions']) != value['ticks'] or
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
                observation = snapshot(state)
                stats = observation.stats
                frame = {'tick':state.tick, 'cells':[[int(i), *current[i].tolist()] for i in changed],
                         'alive':state.alive.tolist(), 'ranking':rank_players(state, stats),
                         **{key:getattr(observation,key).tolist() for key in ('warn_afk','neutralize_in','has_kill')},
                         'stats':{key:getattr(stats,key).tolist()
                         for key in ('army','territory','cities')}}
            else:
                frame = board_view(state)
            frames.append(mark_forfeits(frame, value.get('forfeits', [])))
            previous = current.copy()
    return frames
