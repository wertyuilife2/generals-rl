"""JSON representation for browser boards and replay snapshots."""


def board_view(observation):
    return {"width": observation.width, "height": observation.height,
            "tick": observation.tick, "player_id": observation.player_id,
            "general_pos": observation.general_pos,
            **{key: getattr(observation, key).tolist() for key in
               ("terrain", "structure", "owner", "army", "visible", "alive")},
            "stats": {key: getattr(observation.stats, key).tolist() for key in
                      ("army", "territory", "cities")}}
