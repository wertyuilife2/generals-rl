"""Bounded, adaptive scheduling with coverage and paired cyclic seat rotations."""
from dataclasses import asdict
import math
import random

from generals_arena.util import digest
from generals_arena.competitions.schedules import seat_rotations
from generals_arena.competitions.rating import POLICY
from generals_arena.competitions.rating import initial_rating
from generals_arena.competitions.rating import rating_mode


def choose_player_count(config, job):
    """Spend half the expected game budget on duels when both pools exist."""
    enabled = config.player_counts
    if 2 not in enabled or len(enabled) == 1:
        weights = [1 / count for count in enabled]
    else:
        weights = [(.5 if count == 2 else .5 / (len(enabled) - 1)) / count for count in enabled]
    rng = random.Random(int(digest([config.seed, job['id'], job['done'], POLICY])[:16], 16))
    return rng.choices(enabled, weights=weights)[0]


def choose_opponents(target, candidates, count, ratings, coverage, rng):
    """Duels: 80% nearby, 15% coverage, 5% uniform; FFA: 60/25/15."""
    remaining = sorted(candidates)
    if not remaining:
        raise ValueError("register at least one bot")
    picked = []
    usage = {key: int(key == target) for key in remaining}
    proximity, exploration = (.8, .05) if count == 2 else (.6, .15)
    default = initial_rating('1v1' if count == 2 else 'ffa')
    while len(picked) < count - 1:
        minimum = min(usage.values())
        available = [key for key in remaining if usage[key] == minimum]
        nearby = [math.exp(-min(700, abs(ratings.get(key, default) - ratings.get(target, default)) / 300))
                  for key in available]
        underplayed = [1 / (1 + coverage.get(key, 0)) for key in available]
        weights = [proximity * n / sum(nearby) + (1-proximity-exploration) * c / sum(underplayed)
                   + exploration / len(available) for n, c in zip(nearby, underplayed)]
        selected = rng.choices(available, weights=weights, k=1)[0]
        picked.append(selected)
        usage[selected] += 1
    return picked


def make_block(store, job, *, human=False, players=None):
    config = store.config
    entries = store.entrants()
    target = job["entrant_id"]
    candidates = [key for key, e in entries.items() if e["kind"] != "human"]
    enabled = config.player_counts
    if not candidates:
        return []
    seed = int(digest([config.seed, job["id"], job["done"]])[:16], 16)
    rng = random.Random(seed)
    # A block contains P rotations: inverse-P sampling makes the expected
    # distribution of individual games uniform over the enabled player counts.
    count = players if players is not None else (rng.choice(enabled) if human else
                choose_player_count(config, job))
    if count not in enabled:
        raise ValueError("player count is not enabled")
    mode = rating_mode({"entrant_ids": [None] * count})
    coverage = store.pair_counts(target, mode=mode)
    opponents = choose_opponents(target, candidates, count, store.ratings(mode), coverage, rng)
    group = list(enumerate([target, *opponents]))
    rng.shuffle(group)
    map_seed = rng.getrandbits(63)
    map_config = asdict(config.sample_map(count, map_seed))
    controller_seeds = {instance: rng.getrandbits(63) for instance, key in group}
    size = 1 if human else min(count, job["budget"] - job["done"])
    plans = []
    placements = seat_rotations([key for instance, key in group],
                                [controller_seeds[instance] for instance, key in group],
                                instance_ids=[instance for instance, key in group], count=size)
    for rotation, placement in enumerate(placements):
        game_id = digest([store.manifest["rules_id"], job["id"], job["done"], rotation])
        plans.append({"game_id": game_id, **placement, "map": map_config,
                      "map_seed": map_seed,
                      "rules_id": store.manifest["rules_id"], "human": human})
    return plans
