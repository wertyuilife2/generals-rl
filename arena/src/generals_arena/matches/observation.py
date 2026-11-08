"""Compatibility snapshot API; live sessions obtain their views from the runtime."""
from types import SimpleNamespace


def board_view(state, seat=None):
    from .game import RecordedGame
    p = state.player_count
    plan = dict(entrant_ids=[str(i) for i in range(p)], controller_seeds=[0]*p, map={}, map_seed=0)
    game = RecordedGame(plan, SimpleNamespace(max_ticks=max(1,state.tick+1)), [None]*p,
                        initial=state, record=False)
    try:
        return game.view(seat)
    finally:
        game.close()
