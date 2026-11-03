"""Map, visibility and tick limit passed to generic policy validation."""
from dataclasses import dataclass, field

from generals import MapConfig
from generals_arena.util import integer


@dataclass(frozen=True)
class MatchConfig:
    map: MapConfig = field(default_factory=lambda: MapConfig(players=2))
    max_ticks: int = 50002
    visibility: str = "local"

    def __post_init__(self):
        if not isinstance(self.map, MapConfig):
            raise ValueError("arena requires a MapConfig with 2–8 players")
        integer("max_ticks", self.max_ticks, 1)
        if self.visibility != "local":
            raise ValueError("Arena policies use local observations")
