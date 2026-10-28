"""Versioned league distribution; no changes to the environment or RL config."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from generals import MapConfig
from generals.protocol.rules import RULES_VERSION

from generals_arena.util import digest
from generals_arena.util import file_digest
from generals_arena.util import integer
from generals_arena.util import source_digest


DEFAULT_LEAGUE = Path('outputs/arena/main')


@dataclass(frozen=True)
class LeagueConfig:
    player_counts: tuple = (2, 3, 4, 5, 6, 7, 8)
    min_side: int | None = None
    max_side: int | None = None
    side_cap: int = 64
    mountain_min: float = 0.5
    mountain_max: float = 0.5
    city_min: float = 0.5
    city_max: float = 0.5
    max_ticks: int = 50002
    seed: int = 1000042
    tick_seconds: float = 0.5

    def __post_init__(self):
        counts = tuple(self.player_counts)
        if not counts or len(set(counts)) != len(counts):
            raise ValueError("player_counts must contain distinct counts from 2 to 8")
        for count in counts:
            integer("players", count, 2)
            if count > 8:
                raise ValueError("at most 8 players are supported")
        object.__setattr__(self, "player_counts", tuple(sorted(counts)))
        if (self.min_side is None) != (self.max_side is None):
            raise ValueError("specify both custom side bounds or neither")
        if self.min_side is not None:
            integer("min_side", self.min_side, 3)
            integer("max_side", self.max_side, self.min_side)
            if self.max_side > 64:
                raise ValueError("max_side must be <= 64")
        integer("side_cap", self.side_cap, self.max_side or 33)
        if self.side_cap > 128:
            raise ValueError("side_cap must be <= 128")
        integer("max_ticks", self.max_ticks, 1)
        integer("seed", self.seed)
        for low, high, maximum in ((self.mountain_min, self.mountain_max, 1.),
                                    (self.city_min, self.city_max, 1.)):
            if not (0 <= low <= high <= maximum):
                raise ValueError("invalid terrain density range")
        if not math.isfinite(self.tick_seconds) or not .05 <= self.tick_seconds <= 5:
            raise ValueError("tick_seconds must be in [0.05, 5]")

    def side_range(self, players):
        if self.min_side is None:
            base = 16 + players + max(0, players - 6) * .5
            return math.floor(base + .5), math.floor(base + 4 + players / 2 + .5)
        factor = math.sqrt(players / 2)
        return (min(self.side_cap, math.ceil(self.min_side * factor)),
                min(self.side_cap, math.ceil(self.max_side * factor)))

    def sample_map(self, players, seed):
        if players not in self.player_counts:
            raise ValueError("player count is not enabled in this league")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 714]))
        if self.min_side is None:
            width, height = None, None
        else:
            low, high = self.side_range(players)
            width, height = map(int, rng.integers(low, high + 1, size=2))
        return MapConfig(width=width, height=height, players=players,
                         mountain_density=float(rng.uniform(self.mountain_min, self.mountain_max)),
                         city_density=float(rng.uniform(self.city_min, self.city_max)))

    def manifest(self):
        package = Path(__file__).resolve().parents[1]
        files = ("competitions/config.py", "competitions/matchmaking.py", "competitions/rating.py",
                 "matches/game.py", "matches/observation.py", "web/live.py")
        value = {"schema": 7, "config": asdict(self), "rules": RULES_VERSION,
                 "engine_sha256": source_digest("generals"),
                 "implementation": {name: file_digest(package / name) for name in files},
                 "matches_sha256": source_digest("generals_arena.matches"),
                 "competitions_sha256": source_digest("generals_arena.competitions"),
                 "policies_sha256": source_digest("generals_arena.policies"),
                 "rating": "independent-relative-gor-codeforces-v3",
                 "simulation_backend": "generals-native-cpu-v1",
                 "human_observation": "local-with-indistinguishable-fog-obstacles-v1",
                 "decision_timing": "joint_decision_official_priority_then_growth"}
        return {**value, "rules_id": digest(value)}
