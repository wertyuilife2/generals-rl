"""Tiny opaque policy package for generic Arena boundary and runner tests."""
import json
from pathlib import Path

from generals.protocol.rules import RULES_VERSION
from generals_arena.policies.code import policy_digest
from generals_arena.util import file_digest

SOURCE = '''import numpy as np
from generals import Action

def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)

def choose(owner, army, terrain, pids, width):
    count, cells = owner.shape
    can = (owner == pids[:, None]) & (army > 1)
    right = np.roll(terrain != 1, -1, axis=1)
    right[:, width-1::width] = False
    sources = np.where(can & right, np.arange(cells)[None], -1).max(axis=1)
    return np.where(sources >= 0, 1 + 8*sources + 2, 0).astype(np.int64)

class Policy:
    def __init__(self, **kwargs):
        self.reset(kwargs.get('player_id', 0), kwargs.get('seed', 0))
    def reset(self, player_id, seed):
        self.player_id = player_id
    def reset_batch(self, player_ids, seeds):
        self.player_ids = np.asarray(player_ids)
    def act_batch(self, raw):
        s = raw['state']
        actions = choose(host(s['owner']), host(s['army']), host(s['terrain']), host(raw['player_ids']), raw['width'])
        return np.where(host(raw['active']), actions, 0)
    def act(self, obs):
        value = choose(obs.owner.reshape(1,-1), obs.army.reshape(1,-1), obs.terrain.reshape(1,-1),
                       np.asarray([obs.player_id]), obs.width)[0]
        return Action.decode(int(value))
    def on_result(self, result): pass
    def on_result_batch(self, *args): pass

def create_controller(*, weights, config_path, device='cpu', player_id=0, seed=0):
    return Policy(player_id=player_id, seed=seed)
'''


def make_bundle(directory, *, name='opaque-policy', width=25, height=25, players=(2,)):
    root = Path(directory)
    root.mkdir(parents=True)
    code = root/'policy'
    code.mkdir()
    (code/'__init__.py').write_text('')
    (code/'policy.py').write_text(SOURCE)
    (root/'actor.pt').write_bytes(b'weights are opaque to Arena')
    (root/'model_config.json').write_bytes(b'also opaque, not even JSON')
    manifest = {'schema_version': 1, 'name': name, 'rules': RULES_VERSION,
                'capabilities': {'maps': {'shapes': [[width, height]]}, 'player_counts': list(players)},
                'entrypoint': 'policy:create_controller', 'artifacts': {
                    'weights': {'path': 'actor.pt', 'sha256': file_digest(root/'actor.pt')},
                    'config': {'path': 'model_config.json', 'sha256': file_digest(root/'model_config.json')},
                    'code': {'path': 'policy', 'sha256': policy_digest(code)}}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


def synthetic_placements(eliminations):
    """Explicit outcome data for isolated rating tests (no match simulation)."""
    values=[float('inf') if tick is None else tick for tick in eliminations]
    return [1+sum(other>value for other in values)+(sum(other==value for other in values)-1)/2 for value in values]
