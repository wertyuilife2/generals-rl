"""Optional streaming JSONL traces; replay executes actions without AI."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from generals.core.engine import CoreEngine
from generals.core.state import Action, GameState, MapLayout

from generals.protocol.rules import RULES_VERSION

_ARRAYS = ("structure", "owner", "army", "alive", "general_pos", "leave_tick",
           "death_order", "kills", "damage", "spawn_pos", "city_known", "capture_mask", "warn_afk")
_SCALARS = ("tick", "idle_ticks", "afk_turn", "end_reason")


def state_digest(state):
    digest = sha256()
    digest.update(state.layout.topology_id.encode())
    for name in _ARRAYS:
        digest.update(np.asarray(getattr(state, name), dtype="<i8").tobytes())
    digest.update(json.dumps([*[int(getattr(state, name)) for name in _SCALARS],
                              bool(state.terminated), state.winner_id]).encode())
    return digest.hexdigest()


def state_to_dict(state):
    return {"terrain": state.layout.terrain.tolist(),
            **{name: getattr(state, name).tolist() for name in
               _ARRAYS},
            **{name: int(getattr(state, name)) for name in _SCALARS},
            "terminated": bool(state.terminated), "winner_id": state.winner_id}


def state_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("state record must be an object")
    try:
        state = GameState(MapLayout(np.asarray(data["terrain"])),
                          np.asarray(data["structure"]), np.asarray(data["owner"]), np.asarray(data["army"]),
                          np.asarray(data["alive"]), np.asarray(data["general_pos"]),
                          data["tick"], data["terminated"], data["winner_id"],
                          **{name: data[name] for name in _SCALARS[1:] + _ARRAYS[5:]})
        state.validate()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid recorded state: {exc}") from exc
    return state


class TraceWriter:
    def __init__(self, path, initial_state, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self._write({"type": "header", "rules_version": RULES_VERSION,
                     "state": state_to_dict(initial_state), "metadata": metadata or {},
                     "digest": state_digest(initial_state)})

    def _write(self, row):
        self._file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write_tick(self, result, actions, state):
        # Raw structured actions also preserve rejected negative/out-of-bounds sources.
        self._write({"type": "tick", "tick": result.tick,
                     "actions": [[int(a.kind), a.source, int(a.direction), int(a.mode)] for a in actions],
                     "digest": state_digest(state)})

    def write_surrender(self, player_id, state):
        self._write({"type": "surrender", "tick": state.tick, "player_id": player_id,
                     "digest": state_digest(state)})

    def close(self):
        if not self._file.closed: self._file.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def replay(path, verify=True):
    with Path(path).open(encoding="utf-8") as stream:
        line_number = 1
        try:
            first = next(stream, None)
            if first is None:
                raise ValueError("empty trace")
            header = json.loads(first)
            if not isinstance(header, dict) or header.get("type") != "header" or header.get("rules_version") != RULES_VERSION:
                raise ValueError("unsupported trace header or rules version")
            state = state_from_dict(header["state"])
            if verify and state_digest(state) != header["digest"]:
                raise ValueError("initial state digest mismatch")
            engine = CoreEngine(state)
            for line_number, line in enumerate(stream, 2):
                row = json.loads(line)
                if isinstance(row, dict) and row.get('type') == 'surrender':
                    if row['tick'] != state.tick:
                        raise ValueError('surrender tick mismatch')
                    engine.surrender(row['player_id'])
                    if verify and state_digest(state) != row['digest']:
                        raise ValueError('surrender digest mismatch')
                    continue
                if not isinstance(row, dict) or row.get("type") != "tick":
                    raise ValueError("unknown trace record")
                if not isinstance(row.get("tick"), int) or isinstance(row["tick"], bool):
                    raise ValueError("tick must be an integer")
                if not isinstance(row.get("actions"), list):
                    raise ValueError("actions must be a list")
                result = engine.step([Action(*values) for values in row["actions"]])
                if result.tick != row["tick"]:
                    raise ValueError("trace tick order mismatch")
                if verify and state_digest(engine.state) != row["digest"]:
                    raise ValueError(f"state digest mismatch at tick {result.tick}")
            return engine.state
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(f"invalid trace at line {line_number}: {exc}") from exc
