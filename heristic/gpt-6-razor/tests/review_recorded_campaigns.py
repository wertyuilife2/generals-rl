"""Summarize actual online observations; never reconstruct hidden positions.

This is an offline review tool, not a policy or a counterfactual game runner.
Submitted orders and observed ownership changes are deliberately separate:
online commands can be delayed, duplicated or superseded.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from generals.backends.online.protocol import ObservationDecoder
from generals.protocol.actions import Action
from review_owned_transport import owned_transport


def load_frames(record):
    decoder = ObservationDecoder()
    frames = []
    for line in Path(record).open():
        row = json.loads(line)
        if row.get("event") == "game_start":
            decoder.start(row["data"])
        elif row.get("event") == "game_update":
            obs = decoder.update(row["data"])
            if obs is not None:
                frames.append((obs, []))
        elif row.get("kind") == "submit" and frames:
            frames[-1][1].append(Action.decode(row["action"]))
    return frames


def describe_order(obs, action):
    if not action.kind:
        return {"wait": True}
    source = action.source
    target = source + (-obs.width, 1, obs.width, -1)[int(action.direction)]
    return {
        "source": source, "target": target, "half": bool(action.mode),
        "source_army": int(obs.army.flat[source]),
        "target_army": int(obs.army.flat[target]),
        "target_owner": int(obs.owner.flat[target]),
    }


def review(record, outcome=None):
    frames = load_frames(record)
    if not frames:
        return {"record": str(record), "outcome": outcome,
                "observation_status": "no_decoded_frames"}
    first = frames[0][0]
    player = first.player_id
    ever_seen = np.zeros(first.size, dtype=bool)
    first_king = first_exposure = None
    episodes, active = [], {}
    landmarks = []
    for obs, actions in frames:
        own = obs.owner.ravel()
        structure = obs.structure.ravel()
        visible = obs.visible.ravel().astype(bool)
        ever_seen |= visible
        if first_king is None and np.any(visible & (structure == 2)
                                         & (own >= 0) & (own != player)):
            first_king = obs.tick
        gy, gx = divmod(obs.general_pos, obs.width)
        if first_exposure is None:
            neighbourhood = obs.owner[max(0, gy-1):gy+2,
                                      max(0, gx-1):gx+2]
            if np.any((neighbourhood >= 0) & (neighbourhood != player)):
                first_exposure = obs.tick
        for cell in list(active):
            episode = active[cell]
            if own[cell] != player:
                episode.update(first_observed_loss=obs.tick,
                               held_ticks=obs.tick-episode["first_owned_tick"],
                               owner_after_loss=int(own[cell]))
                episodes.append(active.pop(cell))
        for cell in np.flatnonzero((structure == 1) & (own == player)):
            cell = int(cell)
            if cell not in active:
                active[cell] = {
                    "city": cell, "first_owned_tick": obs.tick,
                    "first_owned_army": int(obs.army.flat[cell]),
                    "left_censored": obs.tick == first.tick,
                    "first_submitted_departure": None,
                }
        for action in actions:
            if action.kind and action.source in active:
                episode = active[action.source]
                if episode["first_submitted_departure"] is None:
                    episode["first_submitted_departure"] = {
                        "tick": obs.tick, **describe_order(obs, action)}
        if obs.tick % 50 == 0 or obs is frames[-1][0]:
            field = (own == player) & (structure == 0)
            landmarks.append({
                "tick": obs.tick,
                "own_army": int(obs.stats.army[player]),
                "other_armies": {str(e): int(obs.stats.army[e])
                                 for e in range(len(obs.alive)) if e != player},
                "own_land": int(obs.stats.territory[player]),
                "own_cities": int(obs.stats.cities[player]),
                "general_army": int(obs.army.flat[obs.general_pos]),
                "largest_field_army": int(max(obs.army.ravel()[field], default=0)),
                "ever_seen_cells": int(ever_seen.sum()),
                "board_cells": obs.size,
                "owned_transport": owned_transport(obs),
            })
    for episode in active.values():
        episode.update(first_observed_loss=None,
                       held_ticks=frames[-1][0].tick-episode["first_owned_tick"],
                       right_censored=True)
        episodes.append(episode)
    return {
        "record": str(record), "outcome": outcome, "player": player,
        "observation_status": "decoded",
        "first_enemy_general_visible": first_king,
        "first_home_exposed": first_exposure,
        "last_tick": frames[-1][0].tick,
        "city_episodes": sorted(episodes, key=lambda x: x["first_owned_tick"]),
        "checkpoints": landmarks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    games = []
    for game in report["games"]:
        record = Path(game["record"])
        if not record.exists():
            record = args.report.parent / record.name
        result = review(record, game.get("result", {}).get("reason"))
        result.update(attempt=game["attempt"], replay_id=game.get("replay_id"))
        games.append(result)
    losses = [g for g in games if g["outcome"] == "lost"]
    observed_losses = [g for g in losses if g["observation_status"] == "decoded"]
    output = {
        "source_report": str(args.report),
        "scope": "Actual own legal observations only. No hidden-board reconstruction, "
                 "opponent identity assertion, or counterfactual win claim. A submitted "
                 "departure is not proof of execution. City durations are observed "
                 "ownership intervals, not guarantees of future retention. Owned "
                 "transport curves use one static shortest-path tree, consume one "
                 "order per transfer and ignore growth and enemy responses; they "
                 "are not globally optimal collection bounds or survival proofs.",
        "losses": len(losses),
        "records_without_observations": [g["record"] for g in games
                                         if g["observation_status"] != "decoded"],
        "losses_without_enemy_general_seen": sum(
            g["first_enemy_general_visible"] is None for g in observed_losses),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(games),
                      "losses": len(losses),
                      "records_without_observations":
                      len(output["records_without_observations"]),
                      "losses_without_enemy_general_seen":
                      output["losses_without_enemy_general_seen"]}))


if __name__ == "__main__":
    main()
