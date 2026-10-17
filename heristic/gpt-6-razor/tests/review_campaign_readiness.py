"""Compare visible columns with the command cost of owned-territory collection.

This reads recorded legal views only. Static collection curves do not include
enemy replies or growth, and are not globally optimal collection bounds.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

from review_recorded_campaigns import load_frames, describe_order
from review_owned_transport import owned_transport


def distances(obs, owned_only):
    result = {obs.general_pos: 0}
    queue = deque(result)
    while queue:
        cell = queue.popleft()
        y, x = divmod(cell, obs.width)
        for yy, xx in ((y-1, x), (y, x+1), (y+1, x), (y, x-1)):
            if not (0 <= yy < obs.height and 0 <= xx < obs.width):
                continue
            target = yy * obs.width + xx
            if target in result or obs.terrain.flat[target] in (1, 2):
                continue
            if owned_only and obs.owner.flat[target] != obs.player_id:
                continue
            result[target] = result[cell] + 1
            queue.append(target)
    return result


def inspect(obs, actions):
    p = obs.player_id
    terrain_distance = distances(obs, False)
    own_distance = distances(obs, True)
    ours = sorted((c for c in range(obs.size) if obs.owner.flat[c] == p),
                  key=lambda c: -int(obs.army.flat[c]))[:6]
    enemies = sorted((c for c in range(obs.size) if obs.visible.flat[c]
                      and obs.owner.flat[c] >= 0 and obs.owner.flat[c] != p),
                     key=lambda c: -int(obs.army.flat[c]))[:4]
    return dict(
        tick=obs.tick, own_army=int(obs.stats.army[p]),
        other_armies={str(e): int(obs.stats.army[e])
                      for e in range(len(obs.alive)) if e != p},
        home=int(obs.army.flat[obs.general_pos]),
        own_columns=[dict(cell=c, army=int(obs.army.flat[c]),
                          structure=int(obs.structure.flat[c]),
                          owned_path_to_home=own_distance.get(c)) for c in ours],
        visible_enemy_columns=[dict(cell=c, army=int(obs.army.flat[c]),
                                    known_terrain_path_to_home=terrain_distance.get(c))
                               for c in enemies],
        transport=owned_transport(obs, (0, 4, 8, 12, 16, 24, 32)),
        submitted_orders=[describe_order(obs, a) for a in actions],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--ticks", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ticks = {int(s) for s in args.ticks.split(",")}
    rows = [inspect(o, actions) for o, actions in load_frames(args.record)
            if o.tick in ticks]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        record=str(args.record),
        record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        scope="Recorded legal observations. Terrain distances ignore combat and "
              "are not attack forecasts. Owned paths require current ownership. "
              "Static tree curves consume one command per transfer and ignore "
              "growth and enemy replies; they are not optimal collection bounds "
              "or survival certificates. No counterfactual game outcome is claimed.",
        rows=rows,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(rows=len(rows), output=str(args.output))))


if __name__ == "__main__":
    main()
