"""Bounded, adaptive scheduling with coverage and paired cyclic seat rotations."""
from dataclasses import asdict
import math
import random

from ..util import digest


def choose_opponents(target, candidates, count, ratings, coverage, anchor, rng, block):
    """60% nearby ratings, 25% underplayed opponents, 15% uniform exploration."""
    remaining = sorted(key for key in candidates if key != target)
    if len(remaining) < count - 1:
        raise ValueError("not enough distinct opponents")
    picked = []
    if block % 5 == 0 and anchor in remaining:
        picked.append(anchor)
        remaining.remove(anchor)
    while len(picked) < count - 1:
        nearby = [math.exp(-min(700, abs(ratings.get(key, 0) - ratings.get(target, 0)) / 300)) for key in remaining]
        underplayed = [1 / (1 + coverage.get(key, 0)) for key in remaining]
        weights = [.6 * n / sum(nearby) + .25 * c / sum(underplayed) + .15 / len(remaining)
                   for n, c in zip(nearby, underplayed)]
        selected = rng.choices(remaining, weights=weights, k=1)[0]
        picked.append(selected)
        remaining.remove(selected)
    return picked


def make_block(store, job, *, human=False, players=None):
    config = store.config
    entries = store.entrants()
    target = job["entrant_id"]
    candidates = [key for key, e in entries.items() if e["kind"] != "human" or key == target]
    enabled = [p for p in config.player_counts if p <= len(candidates)]
    if not enabled:
        return []
    seed = int(digest([config.seed, job["id"], job["done"]])[:16], 16)
    rng = random.Random(seed)
    count = players if players is not None else rng.choice(enabled)
    if count not in enabled:
        raise ValueError("not enough registered bots for this player count")
    coverage = store.pair_counts(target, count)
    block = sum(coverage.values()) // (count - 1) if human else job["done"]
    opponents = choose_opponents(target, candidates, count, store.ratings(count),
                                 coverage, store.anchor, rng, block)
    group = [target, *opponents]
    rng.shuffle(group)
    map_seed = rng.getrandbits(63)
    map_config = asdict(config.sample_map(count, map_seed))
    controller_seeds = {key: rng.getrandbits(63) for key in group}
    size = 1 if human else min(count, job["budget"] - job["done"])
    plans = []
    for rotation in range(size):
        seats = group[rotation:] + group[:rotation]
        game_id = digest([store.manifest["rules_id"], job["id"], job["done"], rotation])
        plans.append({"game_id": game_id, "entrant_ids": seats, "map": map_config,
                      "map_seed": map_seed, "controller_seeds": [controller_seeds[key] for key in seats],
                      "rules_id": store.manifest["rules_id"], "human": human})
    return plans
