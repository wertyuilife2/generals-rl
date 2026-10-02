"""CPU-side batch creation and state conversion, never used during a step.

Array packing owns its storage. Unpacking constructs independent GameState
copies; callers explicitly transfer GPU tensors before invoking it.
"""
import numpy as np

from generals.protocol import MapConfig
from generals.protocol.config import _integer
from .mapgen import generate_map
from .state import GameState, MapLayout


def generate_states(config, batch_size=1, *, seed=42):
    if not isinstance(config, MapConfig):
        raise TypeError("config must be MapConfig")
    batch_size = _integer("batch_size", batch_size, 1)
    seed = _integer("seed", seed, 0)
    return [generate_map(config, int(np.random.SeedSequence([seed, lane, 0]).generate_state(
        1, dtype=np.uint64)[0])) for lane in range(batch_size)]


def pack_states(states):
    states = list(states)
    if not states:
        raise ValueError("states cannot be empty")
    first = states[0]
    for state in states:
        state.validate()
        if (state.layout.terrain.shape != first.layout.terrain.shape
                or state.player_count != first.player_count):
            raise ValueError("all states must share dimensions and player count")
    arrays = {
        "terrain": np.stack([s.layout.terrain.ravel() for s in states]),
        **{name: np.stack([getattr(s, name).ravel() for s in states])
           for name in ("structure", "owner", "army", "alive", "general_pos")},
        "tick": np.asarray([s.tick for s in states], dtype=np.int64),
        "terminated": np.asarray([s.terminated for s in states], dtype=bool),
        "winner": np.asarray([-1 if s.winner_id is None else s.winner_id for s in states], dtype=np.int32),
    }
    return arrays, first.layout.width, first.layout.height


def unpack_states(arrays, *, width, height, validate=False):
    shape = (height, width)
    states = [GameState(MapLayout(arrays["terrain"][lane].reshape(shape)),
                      arrays["structure"][lane].reshape(shape), arrays["owner"][lane].reshape(shape),
                      arrays["army"][lane].reshape(shape), arrays["alive"][lane], arrays["general_pos"][lane],
                      tick=int(arrays["tick"][lane]), terminated=bool(arrays["terminated"][lane]),
                      winner_id=None if arrays["winner"][lane] < 0 else int(arrays["winner"][lane]))
              for lane in range(len(arrays["tick"]))]
    if validate:
        for state in states:
            state.validate()
    return states
