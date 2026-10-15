"""Versioned league distribution; no changes to the environment or RL config."""
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path

import numpy as np
from generals import MapConfig, MapGenerationConfig
from generals.core.random_map import sample_map_config

from ..util import digest, file_digest, integer, source_digest


@dataclass(frozen=True)
class LeagueConfig:
    player_counts: tuple = (2, 3, 4, 5, 6, 7, 8)
    map_generation: MapGenerationConfig | None = field(default_factory=MapGenerationConfig)
    # Historical distribution, used only when map_generation is explicitly null.
    min_side: int = 15
    max_side: int = 20
    side_cap: int = 35
    mountain_min: float = 0.15
    mountain_max: float = 0.22
    city_min: float = 0.02
    city_max: float = 0.035
    max_ticks: int = 3000
    seed: int = 1000042
    k_factor: float = 32.0
    tick_seconds: float = 0.5

    def __post_init__(self):
        if isinstance(self.map_generation, dict):
            object.__setattr__(self, 'map_generation', MapGenerationConfig(**self.map_generation))
        if self.map_generation is not None and not isinstance(self.map_generation, MapGenerationConfig):
            raise TypeError('map_generation must be MapGenerationConfig or None')
        counts = tuple(self.player_counts)
        if not counts or len(set(counts)) != len(counts):
            raise ValueError("player_counts must contain distinct counts from 2 to 8")
        for count in counts:
            integer("players", count, 2)
            if count > 8:
                raise ValueError("at most 8 players are supported")
        object.__setattr__(self, "player_counts", tuple(sorted(counts)))
        integer("min_side", self.min_side, 3)
        integer("max_side", self.max_side, self.min_side)
        if self.max_side > 64:
            raise ValueError("max_side must be <= 64 (before player-count scaling)")
        integer("side_cap", self.side_cap, self.max_side)
        if self.side_cap > 128:
            raise ValueError("side_cap must be <= 128")
        integer("max_ticks", self.max_ticks, 1)
        integer("seed", self.seed)
        for low, high, maximum in ((self.mountain_min, self.mountain_max, .4),
                                    (self.city_min, self.city_max, .2)):
            if not (0 <= low <= high <= maximum):
                raise ValueError("invalid terrain density range")
        if not math.isfinite(self.k_factor) or not 0 < self.k_factor <= 256:
            raise ValueError("k_factor must be in (0, 256]")
        if not math.isfinite(self.tick_seconds) or not .05 <= self.tick_seconds <= 5:
            raise ValueError("tick_seconds must be in [0.05, 5]")

    def side_range(self, players):
        if self.map_generation is not None:
            width = self.map_generation.side_range(players, 'width')
            height = self.map_generation.side_range(players, 'height')
            return min(width[0], height[0]), max(width[1], height[1])
        factor = math.sqrt(players / 2)
        return (min(self.side_cap, math.ceil(self.min_side * factor)),
                min(self.side_cap, math.ceil(self.max_side * factor)))

    def sample_map(self, players, seed, *, width=None, height=None):
        if players not in self.player_counts:
            raise ValueError("player count is not enabled in this league")
        if self.map_generation is not None:
            return sample_map_config(players, seed, self.map_generation, width=width, height=height)
        if (width is None) != (height is None):
            raise ValueError('provide both width and height, or neither')
        rng = np.random.default_rng(np.random.SeedSequence([seed, 714]))
        low, high = self.side_range(players)
        sampled_width, sampled_height = int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))
        return MapConfig(width=sampled_width if width is None else width,
                         height=sampled_height if height is None else height, players=players,
                         mountain_density=float(rng.uniform(self.mountain_min, self.mountain_max)),
                         city_density=float(rng.uniform(self.city_min, self.city_max)))

    def manifest(self):
        # Freeze simulation, distribution, scheduling, and scoring, not presentation code.
        files = ("config.py", "game.py", "observation.py", "rating.py", "rating_policy.py", "mode_rating.py", "scheduler.py", "views.py")
        value = {"schema": 4, "config": asdict(self),
                 "engine_sha256": source_digest("generals"),
                 "implementation": {name: file_digest(Path(__file__).with_name(name)) for name in files},
                 "rating": "independent-relative-gor-codeforces-v3",
                 "simulation_backend": "generals-native-cpu-v1",
                 "human_observation": "local-with-indistinguishable-fog-obstacles-v1",
                 "decision_timing": "joint_pre_growth_then_absolute_seat_order"}
        return {**value, "rules_id": digest(value)}
