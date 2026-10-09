"""Append-only match log. Invalid or incomplete lines fail explicitly."""
import json
import os
from pathlib import Path

from .util import canonical, integer


def read_matches(path):
    path = Path(path)
    if not path.exists():
        return []
    matches = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or not line.endswith("\n"):
                    raise ValueError("expected a complete JSON object line")
                matches.append(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid match log at line {number}: {exc}") from exc
    return matches


def append_match(path, match):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(match) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_matches(matches, schedule, rules_manifest):
    expected = {spec.game_id: spec.to_dict() for spec in schedule}
    seen = set()
    for sequence, match in enumerate(matches):
        game_id = match.get("game_id")
        if game_id not in expected or game_id in seen:
            raise ValueError("match log contains an unknown or duplicate game")
        seen.add(game_id)
        if any(match.get(key) != value for key, value in expected[game_id].items()):
            raise ValueError("recorded matchup or seeds disagree with the schedule")
        if (match.get("schema_version") != 1 or match.get("sequence") != sequence or
                match.get("rules_id") != rules_manifest["rules_id"]):
            raise ValueError("match log schema, sequence, or rules identity disagrees")
        ticks = integer("recorded ticks", match.get("ticks"), 1)
        if ticks > rules_manifest["max_ticks"]:
            raise ValueError("recorded ticks exceed the match limit")
        terminated, truncated, winner = (match.get(key) for key in ("terminated", "truncated", "winner_seat"))
        if type(terminated) is not bool or type(truncated) is not bool or terminated == truncated:
            raise ValueError("a match must either terminate or truncate")
        if terminated:
            if type(winner) is not int or winner not in (0, 1) or match.get("reason") != "capture":
                raise ValueError("a terminated match must record its winner")
        elif winner is not None or ticks != rules_manifest["max_ticks"] or match.get("reason") != "tick_limit":
            raise ValueError("a draw must finish at the tick limit without a winner")
    return seen
