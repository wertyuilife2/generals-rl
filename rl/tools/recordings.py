"""Presentation frames from Arena's verified action replay; never policy inference."""
import gzip
from hashlib import sha256
import json
from pathlib import Path

from generals.core.recording import state_digest, state_to_dict
from generals.protocol import MOVE_REASONS
from generals_arena.matches.replay import verified_replay


def replay_arena_game(match, directory, *, metadata=None, name=None):
    """Verify every recorded state and write frames for render_recording.py."""
    source = Path(match['replay'])
    blob = source.read_bytes()
    trace = json.loads(gzip.decompress(blob))
    name = name or f"game_{str(match['game_id'])[:16]}"
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('recording name must be a simple filename')
    game = dict(match)
    game['learner_seat'] = game.get('candidate_seat', game.get('learner_seat', 0))
    details = {**(metadata or {}), 'name': name, 'game': game,
               'replay_mode': 'arena_verified_actions', 'source_replay': str(source.resolve()),
               'source_replay_sha256': sha256(blob).hexdigest(),
               'spectator_full_state_is_not_policy_input': True}
    frames = []
    for index, (state, outcome) in enumerate(verified_replay(blob, match)):
        moves = None
        if outcome is not None:
            moves = [[0]*9 for _ in state.alive]
            for move in outcome.moves:
                moves[move.player_id] = [1, int(move.valid), move.source, move.target, move.moved,
                    int(move.captured), -1 if move.eliminated is None else move.eliminated,
                    int(move.target_owned), MOVE_REASONS.index(move.reason)]
        frames.append({'state': state_to_dict(state), 'digest': state_digest(state),
                       'actions': None if index == 0 else trace['actions'][index-1], 'moves': moves,
                       'done': [bool(state.terminated), bool(index == match['ticks'] and not state.terminated)]})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/f'{name}.frames.json.gz'
    payload = {'schema_version': 1, 'metadata': details, 'frames': frames}
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    temporary.replace(destination)
    return {'game_id': match['game_id'], 'name': name, 'frames': str(destination.resolve()),
            'trace': str(source.resolve()), 'ticks': match['ticks'], 'winner_seat': match['winner_seat'],
            'native_core_replay_verified': True, 'learner_seat': game['learner_seat'],
            'opponent': match['opponent'], 'outcome': match['outcome'], 'match': match}
