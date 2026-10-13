"""Whole-trajectory golden hashes captured from the pre-vectorization policies."""
import hashlib
import json
from pathlib import Path

import pytest

from generals import MapConfig
from generals.core.engine import CoreEngine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.recording import state_digest
from generals.core.state import Action
from generals_baselines import make_controllers


def trajectory(case):
    state=generate_map(MapConfig(**case['map']),case['seed'])
    controllers=make_controllers(case['policies'],case['seed'])
    engine=CoreEngine(state)
    digest=hashlib.sha256()
    for _ in range(case['ticks']):
        if state.terminated:break
        actions=[c.act(observe(state,seat)) if state.alive[seat] else Action.wait()
                 for seat,c in enumerate(controllers)]
        result=engine.step(actions)
        digest.update(json.dumps([a.encode() for a in actions]).encode())
        digest.update(state_digest(state).encode())
        for move in result.moves:
            controllers[move.player_id].on_result(move)
            if move.eliminated is not None:
                eliminate=getattr(controllers[move.eliminated],"eliminate",None)
                if eliminate:eliminate()
    return digest.hexdigest()


FIXTURE=Path(__file__).with_name('trajectories.json')


@pytest.mark.parametrize('case',json.loads(FIXTURE.read_text()))
def test_unchanged_actions_rng_ties_and_game_states(case):
    assert trajectory(case)==case['sha256']
