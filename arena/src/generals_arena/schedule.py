"""Reproducible group round robins with shared maps and cyclic seat rotations."""
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from .util import digest, integer


def paired_seeds(seed, map_index):
    """Also used by the training-time built-in evaluator."""
    return tuple(int(np.random.SeedSequence([seed, map_index, stream]).generate_state(1, dtype=np.uint64)[0])
                 for stream in (0, 1))


@dataclass(frozen=True)
class MatchSpec:
    game_id: str
    entrant_ids: tuple[str, ...]
    map_index: int
    map_seed: int
    controller_seeds: tuple[int, ...]

    def to_dict(self):
        result = asdict(self)
        result["entrant_ids"] = list(self.entrant_ids)
        result["controller_seeds"] = list(self.controller_seeds)
        return result


def map_count(settings):
    """Read the group budget, accepting the original pair-budget spelling."""
    if "maps_per_group" in settings and "maps_per_pair" in settings:
        raise ValueError("specify maps_per_group or maps_per_pair, not both")
    value = settings.get("maps_per_group", settings.get("maps_per_pair", 50))
    return integer("maps_per_group", value, 1)


def make_schedule(entrant_ids, *, rules_id, players=2, maps_per_group=None,
                  maps_per_pair=None, seed=1000042):
    settings = {name: value for name, value in (("maps_per_group", maps_per_group),
                                               ("maps_per_pair", maps_per_pair)) if value is not None}
    count = map_count(settings)
    integer("players", players, 2)
    if players > 8:
        raise ValueError("arena supports at most eight players per match")
    integer("seed", seed)
    ids = sorted(entrant_ids)
    if len(ids) < players or len(set(ids)) != len(ids):
        raise ValueError("a tournament needs at least players distinct entrants")
    schedule = []
    for index in range(count):
        map_seed, _ = paired_seeds(seed, index)
        for group in combinations(ids, players):
            for rotation in range(players):
                seats = group[rotation:] + group[:rotation]
                seeds = tuple(int(np.random.SeedSequence(
                    [seed, index, int(digest(entrant)[:16], 16)]).generate_state(1, dtype=np.uint64)[0])
                              for entrant in seats)
                game_id = digest({"rules_id": rules_id, "map_index": index, "entrant_ids": seats})
                schedule.append(MatchSpec(game_id, seats, index, map_seed, seeds))
    return schedule
