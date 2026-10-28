"""JSON representation for browser boards and replay snapshots."""

from generals.core.observation import snapshot
from generals.core.state import rank_players


def board_view(state, seat=None):
    observation = snapshot(state, seat)
    return {"width": observation.width, "height": observation.height,
            "tick": observation.tick, "player_id": observation.player_id,
            "general_pos": observation.general_pos, "ranking": rank_players(state, observation.stats),
            **{key: getattr(observation, key).tolist() for key in
               ("terrain", "structure", "owner", "army", "visible", "alive", "fog_obstacles", "has_kill", "warn_afk", "neutralize_in")},
            "stats": {key: getattr(observation.stats, key).tolist() for key in
                      ("army", "territory", "cities")}}
