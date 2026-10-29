"""Joint decision-boundary behavior, independent of rendering or policies."""
import numpy as np
import pytest
from generals.core.engine import CoreEngine
from generals.core.state import Action,Direction,GameState,MapLayout


def fixture():
    return GameState(MapLayout(np.zeros((1,3),np.uint8)),np.array([[2,0,2]],np.uint8),
                     np.array([[0,-1,1]],np.int16),np.array([[5,0,7]],np.int64),
                     np.ones(2,bool),np.array([0,2],np.int32))


def test_joint_contest_uses_ordered_moves():
    state=fixture();result=CoreEngine(state,debug=True).step([Action.move(0,Direction.RIGHT),Action.move(2,Direction.LEFT)])
    assert state.owner[0,1]==1 and state.army[0,1]==2
    assert result.stats.army.tolist()==[1,3]
    assert result.stats.territory.tolist()==[1,2]


def test_growth_follows_joint_combat():
    state=fixture();state.tick=1
    result=CoreEngine(state,debug=True).step([Action.move(0,Direction.RIGHT),Action.move(2,Direction.LEFT)])
    assert [m.moved for m in result.moves]==[4,6]
    assert state.army.tolist()==[[2,2,2]]


@pytest.mark.parametrize('actions',[[Action.wait()],[Action.wait(),None],{0:Action.wait(),1:Action.wait()}])
def test_invalid_joint_batch_cannot_advance(actions):
    state=fixture()
    with pytest.raises((TypeError,ValueError)):CoreEngine(state).step(actions)
    assert state.tick==0
