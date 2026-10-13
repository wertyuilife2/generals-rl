"""Arena-local observation construction; the environment/RL API is unchanged."""
import numpy as np

from generals.core.observation import Observation
from generals.core.state import PlayerStats, UNKNOWN, UNKNOWN_OWNER


def local_observation(state, seat, stats):
    """Match core.observe exactly, reusing this tick's public engine totals.

    Every published board array owns a fresh buffer. Public totals are immutable
    PlayerStats arrays; other players' city counts remain hidden.
    """
    owned = state.owner == seat
    horizontal = owned.copy()
    horizontal[:, 1:] |= owned[:, :-1]
    horizontal[:, :-1] |= owned[:, 1:]
    visible = horizontal.copy()
    visible[1:] |= horizontal[:-1]
    visible[:-1] |= horizontal[1:]
    cities = np.full_like(stats.cities, -1)
    cities[seat] = stats.cities[seat]
    arrays = {
        "terrain": np.where(visible, state.layout.terrain, np.uint8(UNKNOWN)),
        "structure": np.where(visible, state.structure, np.uint8(UNKNOWN)),
        "owner": np.where(visible, state.owner, np.int16(UNKNOWN_OWNER)),
        "army": np.where(visible, state.army, np.int64(-1)),
        "visible": visible,
        "alive": state.alive.copy(),
    }
    for array in arrays.values():
        array.flags.writeable = False
    return Observation(
        player_id=seat, tick=state.tick, **arrays,
        general_pos=int(state.general_pos[seat]),
        stats=PlayerStats(stats.army, stats.territory, cities),
        visibility="local", terminated=state.terminated, winner_id=state.winner_id,
    )
