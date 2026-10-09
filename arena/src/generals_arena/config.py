"""Fixed competition rules; a rules identity defines a separate rating pool."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

from generals import MapConfig
from .util import digest, file_digest, integer, source_digest

DECISION_TIMING = "joint_pre_growth_then_absolute_seat_order"


@dataclass(frozen=True)
class ArenaRules:
    map: MapConfig = field(default_factory=lambda: MapConfig(9, 9, 2, 0.1, 0.05))
    max_ticks: int = 2000
    visibility: str = "local"
    seed: int = 1000042

    def __post_init__(self):
        if not isinstance(self.map, MapConfig) or self.map.players != 2:
            raise ValueError("arena currently requires a two-player MapConfig")
        integer("max_ticks", self.max_ticks, 1)
        integer("seed", self.seed)
        if self.visibility not in ("local", "full"):
            raise ValueError("visibility must be local or full")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "map" in values:
            values["map"] = MapConfig(**{**asdict(cls().map), **values["map"]})
        return cls(**values)

    def manifest(self):
        values = {**asdict(self), "decision_timing": DECISION_TIMING,
                  "engine_sha256": source_digest("generals"), "schema_version": 1}
        values["execution_sha256"] = digest({name: file_digest(Path(__file__).with_name(name))
                                             for name in ("match.py", "schedule.py", "config.py")})
        return {**values, "rules_id": digest(values)}
