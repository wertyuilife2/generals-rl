"""Deterministic match plans shared by schedule modes and all executors."""
from dataclasses import asdict
from itertools import combinations
import random

import numpy as np

from generals_arena.util import digest
from generals_arena.util import integer
from generals_arena.competitions.rating import POLICY


def seat_rotations(entrant_ids, controller_seeds, *, count=None, instance_ids=None):
    """Rotate independent instances, carrying their RNG streams with them."""
    ids, seeds = list(entrant_ids), list(controller_seeds)
    instances = list(range(len(ids))) if instance_ids is None else list(instance_ids)
    if len(ids) != len(seeds) or len(ids) != len(instances):
        raise ValueError("one instance and controller seed are required per seat")
    for rotation in range(len(ids) if count is None else count):
        offset = rotation % len(ids)
        yield {"entrant_ids": ids[offset:] + ids[:offset],
               "instance_ids": instances[offset:] + instances[:offset],
               "controller_seeds": seeds[offset:] + seeds[:offset]}


def fixed_panel(profile, opponent, candidate):
    """Candidate versus one fixed policy, preserving the established map/seed panel."""
    players = profile["map"]["players"]
    seats = range(players) if profile["paired_seats"] else (0,)
    for index in range(profile["maps_per_opponent"]):
        map_seed = int(np.random.SeedSequence([profile["seed"], index, 0]).generate_state(1, dtype=np.uint64)[0])
        seed = int(np.random.SeedSequence([profile["seed"], index, 1]).generate_state(1, dtype=np.uint64)[0])
        for seat in seats:
            ids, instances, seeds = [], [], []
            for absolute in range(players):
                relative = (absolute - seat) % players
                ids.append(candidate["entrant_id"] if relative == 0 else opponent["entrant_id"])
                instances.append(relative)
                if relative == 0:
                    stream = seed
                elif opponent["kind"] == "baseline":
                    # Preserve this existing panel's absolute-seat baseline streams.
                    stream = int(np.random.SeedSequence(seed).spawn(players + 1)[absolute + 1].generate_state(
                        1, dtype=np.uint64)[0])
                else:
                    stream = int(np.random.SeedSequence(seed, spawn_key=(relative,)).generate_state(
                        1, dtype=np.uint64)[0])
                seeds.append(stream)
            yield {"game_id": digest([candidate["entrant_id"], opponent["entrant_id"], map_seed, seat]),
                   "entrant_ids": ids, "instance_ids": instances, "controller_seeds": seeds,
                   "map": profile["map"], "map_seed": map_seed, "map_index": index,
                   "controller_seed": seed, "candidate_seat": seat, "opponent": opponent["name"],
                   "rating_policy": profile["rating_policy"],
                   "rating_group_size": players if profile["paired_seats"] else 1}


def round_robin(entrant_ids, map_config, *, maps_per_group=1, seed=42, paired_seats=True):
    """Every distinct P-policy combination on common maps, optionally rotating seats."""
    integer("maps_per_group", maps_per_group, 1)
    integer("seed", seed, 0)
    if not isinstance(paired_seats, bool):
        raise TypeError("paired_seats must be boolean")
    ids = sorted(entrant_ids)
    if len(set(ids)) != len(ids) or len(ids) < map_config.players:
        raise ValueError("round robin needs at least P distinct entrants")
    config = asdict(map_config)
    for group in combinations(ids, map_config.players):
        for index in range(maps_per_group):
            map_seed = int(np.random.SeedSequence([seed, index, 0]).generate_state(1, dtype=np.uint64)[0])
            streams = [int(digest([seed, index, key, "controller"])[:16], 16) for key in group]
            count = map_config.players if paired_seats else 1
            for rotation, placement in enumerate(seat_rotations(group, streams, count=count)):
                yield {"game_id": digest(["round_robin", list(group), config, map_seed, rotation]),
                       **placement, "map": config, "map_seed": map_seed, "map_index": index,
                       "rating_policy": POLICY, "rating_group_size": count}


def fixed_seats(config, seats, games, seed, identity, size):
    """Every selected player participates; rotate seats on each shared map."""
    rng, plans = random.Random(seed), []
    rules_id = config.manifest()['rules_id']
    count = len(seats)
    while len(plans) < games:
        map_seed = rng.getrandbits(63)
        map_config = asdict(config.sample_map(count, map_seed))
        if size['mode'] == 'fixed':
            map_config.update(width=size['width'], height=size['height'])
        elif size['mode'] == 'range':
            map_config.update(width=rng.randint(size['min'], size['max']), height=rng.randint(size['min'], size['max']))
        seeds = [rng.getrandbits(63) for _ in seats]
        rotations = min(count, games-len(plans))
        for placement in seat_rotations(seats, seeds, count=rotations):
            plans.append({'game_id': digest([identity, len(plans)]),
                **placement, 'map_seed': map_seed,
                'map': map_config, 'rules_id': rules_id, 'human': False,
                'rating_policy': POLICY, 'rating_group_size': rotations})
    return plans


def candidate_panel(config, target, opponents, games, seed, identity):
    """Fixed independent schedule. Candidate code/name cannot change maps or RNG."""
    rng, plans, block = random.Random(seed), [], 0
    rules_id = config.manifest()['rules_id']
    counts = config.player_counts
    while len(plans) < games:
        count = rng.choices(counts, weights=[1/p for p in counts])[0]
        instances = list(enumerate([target, *[rng.choice(opponents) for _ in range(count-1)]]))
        rng.shuffle(instances)
        map_seed = rng.getrandbits(63)
        map_config = asdict(config.sample_map(count, map_seed))
        seeds = {i: rng.getrandbits(63) for i, _ in instances}
        rotations = min(count, games-len(plans))
        placements = seat_rotations([key for i, key in instances], [seeds[i] for i, key in instances],
                                    instance_ids=[i for i, key in instances], count=rotations)
        for placement in placements:
            plans.append({'game_id': digest([identity, len(plans)]),
                **placement, 'map_seed': map_seed,
                'map': map_config, 'rules_id': rules_id, 'human': False,
                'evaluation_block': block, 'rating_policy': POLICY, 'rating_group_size': rotations})
        block += 1
    return plans
