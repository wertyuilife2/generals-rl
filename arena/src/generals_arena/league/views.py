"""Human presentation of local fog, separate from policy/RL observations."""
from generals.core.observation import snapshot
from generals.core.state import Structure, Terrain

from .game import board_view


def human_board(state, seat):
    observation = snapshot(state, seat)
    board = board_view(observation)
    # Cities and mountains have one indistinguishable public fog marker.
    # Their actual terrain/structure/owner/army remain masked by snapshot().
    # Generals are deliberately not obstacles, so their positions stay hidden.
    obstacles = (state.layout.terrain == Terrain.MOUNTAIN) | (state.structure == Structure.CITY)
    board["fog_obstacles"] = (~observation.visible & obstacles).tolist()
    return board
