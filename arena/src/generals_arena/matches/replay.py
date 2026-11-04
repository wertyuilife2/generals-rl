"""Recorded browser frames; native action verification is an explicit audit API."""
import copy
import gzip
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
from generals import Action
from generals.core.recording import state_digest, state_from_dict, state_to_dict
from generals.core.state import PlayerStats, TickResult
from generals.protocol.rules import RULES_VERSION


def _bytes(value):
    return json.dumps(value, separators=(',', ':')).encode()


def encode_native_trace(trace, forfeits=()):
    value = dict(trace, schema=4, rules=RULES_VERSION)
    if forfeits:
        value['forfeits'] = list(forfeits)
    value['checksum'] = sha256(_bytes(value)).hexdigest()
    return gzip.compress(_bytes(value), mtime=0)


def encode_replay(initial, actions, digests, forfeits=()):
    """Legacy artifact writer. New executions record their frames natively."""
    if len(actions) != len(digests):
        raise ValueError('replay actions and digests must have the same length')
    initial = initial if isinstance(initial, dict) else state_to_dict(initial)
    value = dict(schema=3, rules=RULES_VERSION, initial=initial, actions=actions, digests=digests)
    if forfeits:
        value['forfeits'] = list(forfeits)
    return gzip.compress(_bytes(value), mtime=0)


def _trace(blob, value):
    trace = json.loads(gzip.decompress(blob))
    if (trace['schema'] not in (3, 4) or trace['rules'] != RULES_VERSION or
            len(trace['actions']) != value['ticks'] or len(trace['digests']) != len(trace['actions'])):
        raise ValueError('invalid replay length or schema')
    if trace.get('forfeits', []) != value.get('forfeits', []):
        raise ValueError('replay forfeit mismatch')
    if trace['schema'] == 4:
        checksum = trace.pop('checksum', None)
        if checksum != sha256(_bytes(trace)).hexdigest():
            raise ValueError('replay checksum mismatch')
        if len(trace['frames']) != len(trace['actions']) + 1:
            raise ValueError('invalid replay frame count')
    return trace


def _audit(trace, value, *, capture):
    from .game import RecordedGame
    state = state_from_dict(trace['initial'])
    if state_digest(state) != value['initial_digest']:
        raise ValueError('replay initial digest mismatch')
    p = state.player_count
    plan = dict(entrant_ids=[str(i) for i in range(p)], controller_seeds=[0]*p, map={}, map_seed=0)
    game = RecordedGame(plan, SimpleNamespace(max_ticks=max(1, value['ticks'])), [None]*p,
                        trace.get('forfeits', []), initial=state)
    try:
        actions = np.asarray([[Action(*a).encode() for a in row] for row in trace['actions']], np.int64).reshape(-1,p)
        game.restore(actions, capture=capture)
        actual = game.trace()
        if actual['digests'] != trace['digests']:
            raise ValueError('replay diverged: committed digest mismatch')
        if game.summary()['final_digest'] != value['final_digest']:
            raise ValueError('replay final digest mismatch')
        return actual, game.execution.query(4) if capture else None
    finally:
        game.close()


def verified_replay(blob, value):
    """Audit all actions in one native restore call, then expose immutable DTOs."""
    from .batch import _move
    trace = _trace(blob, value)
    actual, audit = _audit(trace, value, capture=True)
    frames = _expanded(actual['frames'])
    previous = None
    for index, (data, moves) in enumerate(zip(audit['states'], audit['moves'], strict=True)):
        state = state_from_dict(data)
        outcome = None
        if moves is not None:
            encoded = [Action(*a).encode() for a in trace['actions'][index-1]]
            receipts = tuple(move for seat, row in enumerate(moves)
                             if (move := _move(row, seat, encoded[seat])) is not None)
            # Verification DTOs are outside the execution path. Totals are already recorded.
            frame = frames[index]
            totals = frame['stats']
            stats = PlayerStats(*(np.asarray(totals[key], np.int64) for key in ('army','territory','cities')))
            eliminated = tuple(int(i) for i in np.flatnonzero(previous & ~state.alive))
            outcome = TickResult(state.tick, receipts, stats, state.terminated, state.winner_id, eliminated)
        previous = state.alive.copy()
        yield state, outcome


def _expanded(frames):
    output, current = [], None
    for frame in frames:
        if 'cells' not in frame:
            current = copy.deepcopy(frame)
        else:
            current = copy.deepcopy(current)
            for index, owner, army, structure in frame['cells']:
                y, x = divmod(index, current['width'])
                for key, v in (('owner',owner), ('army',army), ('structure',structure)):
                    current[key][y][x] = v
            current.update({key: copy.deepcopy(v) for key, v in frame.items() if key != 'cells'})
        output.append(current)
    return output


def replay_frames(blob, value, stride=1, compact=False):
    """Playback only reads recorded frames. Legacy traces are converted natively."""
    trace = _trace(blob, value)
    frames = trace['frames'] if trace['schema'] == 4 else _audit(trace, value, capture=False)[0]['frames']
    if compact and stride == 1:
        return frames
    full = _expanded(frames)
    selected = [f for index, f in enumerate(full) if index == 0 or f['tick'] % stride == 0 or index == len(full)-1]
    if not compact:
        return selected
    output, previous = [], None
    for frame in selected:
        if previous is None:
            output.append(frame)
        else:
            cells = []
            for y in range(frame['height']):
                for x in range(frame['width']):
                    values = [frame[k][y][x] for k in ('owner','army','structure')]
                    if values != [previous[k][y][x] for k in ('owner','army','structure')]:
                        cells.append([y*frame['width']+x, *values])
            output.append({k:v for k,v in frame.items() if k in
                ('tick','alive','ranking','warn_afk','neutralize_in','has_kill','stats','forfeited')} | {'cells':cells})
        previous = frame
    return output
