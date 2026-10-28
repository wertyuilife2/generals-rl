"""Presentation replays Arena's exact actions and rejects corrupted recordings."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from generals import Action, Direction, make_engine
from generals.core.recording import state_digest, state_to_dict
from generals.core.state import GameState, MapLayout
from generals_arena.matches.replay import encode_replay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recordings import replay_arena_game


def arena_recording(directory, adjacent):
    shape = (3,3)
    positions = (0,1) if adjacent else (0,8)
    structure = np.zeros(shape,dtype=np.uint8)
    owner = np.full(shape,-1,dtype=np.int16)
    army = np.zeros(shape,dtype=np.int64)
    for player,cell in enumerate(positions):
        structure.flat[cell],owner.flat[cell],army.flat[cell] = 2,player,1
    army.flat[0] = 20
    state = GameState(MapLayout(np.zeros(shape,dtype=np.uint8)),structure,owner,army,
                      np.ones(2,dtype=np.bool_),np.array(positions,dtype=np.int32))
    initial,initial_digest = state_to_dict(state),state_digest(state)
    actions,digests = [],[]
    engine = make_engine(state)
    try:
        for _ in range(3):
            joint = [Action.move(0,Direction.RIGHT),Action.wait()]
            engine.step(joint)
            actions.append([[int(a.kind),a.source,int(a.direction),int(a.mode)] for a in joint])
            digests.append(state_digest(state))
            if state.terminated:
                break
    finally:
        engine.close()
    path = directory/'game.json.gz'
    path.write_bytes(encode_replay(initial, actions, digests))
    return {'game_id':'testgame','opponent':'reference','candidate_seat':0,'ticks':state.tick,
            'initial_digest':initial_digest,'final_digest':state_digest(state),'winner_seat':state.winner_id,
            'replay':str(path),'outcome':'win' if state.winner_id==0 else 'draw'}


@pytest.mark.parametrize('adjacent',[True,False])
def test_authoritative_terminal_and_timeout_replay_produce_renderable_frames(tmp_path,adjacent):
    match = arena_recording(tmp_path,adjacent)
    case = replay_arena_game(match,tmp_path/'frames',metadata={'policy_id':'immutable'})
    with gzip.open(case['frames'],'rt') as stream:
        data = json.load(stream)
    assert len(data['frames']) == match['ticks']+1
    assert data['frames'][0]['actions'] is None
    assert data['frames'][-1]['digest'] == match['final_digest']
    assert data['frames'][-1]['done'] == [adjacent,not adjacent]
    assert data['metadata']['game']['learner_seat'] == 0
    assert data['metadata']['policy_id'] == 'immutable'
    assert case['match'] == match and case['native_core_replay_verified']


def test_corrupt_replay_never_produces_presentation(tmp_path):
    match = arena_recording(tmp_path,True)
    match['final_digest'] = 'bad'
    with pytest.raises(ValueError,match='final digest'):
        replay_arena_game(match,tmp_path/'not_written')
    assert not (tmp_path/'not_written').exists()
