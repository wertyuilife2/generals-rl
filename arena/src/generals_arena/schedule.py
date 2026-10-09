"""Reproducible round robins with shared maps and both seat assignments."""
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
    entrant_ids: tuple[str, str]
    map_index: int
    map_seed: int
    controller_seeds: tuple[int, int]

    def to_dict(self):
        result = asdict(self)
        result["entrant_ids"] = list(self.entrant_ids)
        result["controller_seeds"] = list(self.controller_seeds)
        return result


def make_schedule(entrant_ids, *, rules_id, maps_per_pair=50, seed=1000042):
    integer("maps_per_pair", maps_per_pair, 1)
    integer("seed", seed)
    ids = sorted(entrant_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("a tournament needs at least two distinct entrants")
    schedule = []
    for index in range(maps_per_pair):
        map_seed, _ = paired_seeds(seed, index)
        for pair in combinations(ids, 2):
            for seats in (pair, pair[::-1]):
                seeds = tuple(int(np.random.SeedSequence(
                    [seed, index, int(digest(entrant)[:16], 16)]).generate_state(1, dtype=np.uint64)[0])
                              for entrant in seats)
                game_id = digest({"rules_id": rules_id, "map_index": index, "entrant_ids": seats})
                schedule.append(MatchSpec(game_id, seats, index, map_seed, seeds))
    return schedule
