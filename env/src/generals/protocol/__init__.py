"""Small shared data protocol; importing this package loads no game backend."""
from generals.protocol.actions import (Action, ActionKind, Direction, MoveMode, Structure, Terrain,
                      EMPTY, NEUTRAL, UNKNOWN, UNKNOWN_OWNER)
from generals.protocol.config import MapConfig
from generals.protocol.rules import RULES_VERSION

# Batched storage uses [batch, height*width], with row-major cell ids. Army and
# ticks are signed int64 in every backend; encoded joint actions use [batch,P].
STATE_DTYPES = {"terrain": "uint8", "structure": "uint8", "owner": "int16",
                "army": "int64", "alive": "bool", "general_pos": "int32",
                "tick": "int64", "terminated": "bool", "winner": "int32",
                "idle_ticks": "int64", "afk_turn": "int64", "end_reason": "int32",
                "leave_tick": "int64", "death_order": "int32", "kills": "int32",
                "damage": "int64", "spawn_pos": "int32", "city_known": "uint8", "capture_mask": "uint8", "warn_afk": "uint8"}


def state_shape(name, batch, cells, players):
    if name in ("terrain", "structure", "owner", "army"):
        return (batch, cells)
    if name == "damage":
        return (batch, players, players)
    if name in ("alive", "general_pos", "leave_tick", "death_order", "kills", "spawn_pos", "city_known", "capture_mask", "warn_afk"):
        return (batch, players)
    return (batch,)
STATS_FIELDS = ("army", "territory", "cities")
COUNTER_FIELDS = ("env_ticks", "player_actions", "move_actions", "wait_actions",
                  "invalid_actions", "positive_moves", "captures",
                  "terminated_episodes", "truncated_episodes")
MOVE_FIELDS = ("active", "valid", "source", "target", "moved", "captured",
               "eliminated", "target_owned", "reason")
MOVE_REASONS = ("SKIPPED", "WAIT", "MOVED", "OUT_OF_BOUNDS", "NOT_OWNER", "MOUNTAIN", "NO_ARMY", "TRADE")
