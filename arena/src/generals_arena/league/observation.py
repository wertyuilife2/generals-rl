"""Arena-local observation construction; the environment/RL API is unchanged."""
from functools import lru_cache
from pathlib import Path

from generals.core.observation import Observation


@lru_cache(maxsize=1)
def _kernel():
    from generals.backends._build import build_extension
    module, _ = build_extension(Path(__file__).with_suffix('.cpp'), name='_generals_arena_observation')
    return module.observe


def local_observation(state, seat, stats):
    """Match core.observe exactly, reusing this tick's public engine totals.

    Every published board array owns a fresh buffer. Public totals are immutable
    PlayerStats arrays; other players' city counts remain hidden.
    """
    terrain, structure, owner, army, visible, alive = _kernel()(
        state.layout.terrain, state.structure, state.owner, state.army, state.alive, seat)
    return Observation(
        player_id=seat, tick=state.tick, terrain=terrain, structure=structure,
        owner=owner, army=army, visible=visible, alive=alive,
        general_pos=int(state.general_pos[seat]),
        stats=stats._for_player(seat),
        visibility="local", terminated=state.terminated, winner_id=state.winner_id,
    )
