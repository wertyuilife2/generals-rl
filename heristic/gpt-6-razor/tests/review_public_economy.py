"""Review public army gaps and actual orders in a recorded two-player game.

This is offline evidence, not a policy or a hidden-board reconstruction.
An unexplained public army drop followed by a recurring growth surplus can
support an investment hypothesis, but does not reveal a city or its owner.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from review_recorded_campaigns import load_frames, describe_order


def public_economy(frames):
    rows, changes = [], []
    previous = None
    for obs, actions in frames:
        if len(obs.alive) != 2 or not all(obs.alive):
            continue
        player, enemy = obs.player_id, 1 - obs.player_id
        gap = int(obs.stats.army[enemy] - obs.stats.army[player])
        visible_enemies = [c for c in range(obs.size)
                           if obs.visible.flat[c] and obs.owner.flat[c] == enemy]
        column = max(visible_enemies, key=lambda c: int(obs.army.flat[c]),
                     default=None)
        row = {
            "tick": obs.tick,
            "own_army": int(obs.stats.army[player]),
            "enemy_army": int(obs.stats.army[enemy]),
            "enemy_minus_own_army": gap,
            "own_land": int(obs.stats.territory[player]),
            "enemy_land": int(obs.stats.territory[enemy]),
            "home_army": int(obs.army.flat[obs.general_pos]),
            "own_cities": int(obs.stats.cities[player]),
            "largest_visible_enemy": None if column is None else {
                "cell": column, "army": int(obs.army.flat[column]),
                "structure": int(obs.structure.flat[column]),
            },
            "submitted_orders": [describe_order(obs, a) for a in actions],
        }
        if previous is not None:
            elapsed = obs.tick - previous["tick"]
            event = {
                "previous_tick": previous["tick"], "tick": obs.tick,
                "change": gap - previous["enemy_minus_own_army"],
                "gap_after": gap,
                "land_difference_after": row["enemy_land"] - row["own_land"],
                "phase": ("gap_in_record" if elapsed != 1 else
                          "land_growth_boundary" if obs.tick % 50 == 0 else
                          "ordinary_growth_boundary" if obs.tick % 2 == 0 else
                          "between_growth_boundaries"),
            }
            changes.append(event)
        rows.append(row)
        previous = row
    changed = [e for e in changes if e["change"]]
    histograms = {}
    for phase in ("land_growth_boundary", "ordinary_growth_boundary",
                  "between_growth_boundaries", "gap_in_record"):
        histograms[phase] = dict(sorted(Counter(
            e["change"] for e in changes if e["phase"] == phase).items()))
    return {
        "scope": "Actual legal observations and submitted orders only. Public "
                 "gap changes are measurements, not inferred city locations or "
                 "proof of opponent intent. Orders may be delayed or unexecuted. "
                 "Skipped observation ticks remain separate from one-tick changes.",
        "nonzero_gap_changes": changed,
        "gap_change_histograms": histograms,
        "timeline": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = public_economy(load_frames(args.record))
    result["record"] = str(args.record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "observations": len(result["timeline"]),
                      "histograms": result["gap_change_histograms"]}))


if __name__ == "__main__":
    main()
