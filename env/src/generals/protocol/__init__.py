"""Small shared data protocol; importing this package loads no game backend."""
from generals.protocol.actions import (Action, ActionKind, Direction, MoveMode, Structure, Terrain,
                      EMPTY, NEUTRAL, UNKNOWN, UNKNOWN_OWNER)
from generals.protocol.config import MapConfig
from generals.protocol.mapgen import MapGenerationConfig

# Batched storage uses [batch, height*width], with row-major cell ids. Army and
# ticks are signed int64 in every backend; encoded joint actions use [batch,P].
STATE_DTYPES = {"terrain": "uint8", "structure": "uint8", "owner": "int16",
                "army": "int64", "alive": "bool", "general_pos": "int32",
                "tick": "int64", "terminated": "bool", "winner": "int32"}
STATS_FIELDS = ("army", "territory", "cities")
COUNTER_FIELDS = ("env_ticks", "player_actions", "move_actions", "wait_actions",
                  "invalid_actions", "positive_moves", "captures",
                  "terminated_episodes", "truncated_episodes")
MOVE_FIELDS = ("active", "valid", "source", "target", "moved", "captured",
               "eliminated", "target_owned", "reason")
