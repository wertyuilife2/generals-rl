"""Map, visibility and tick limit passed to generic policy validation."""
from dataclasses import dataclass, field

from generals import MapConfig
from generals_arena.util import integer


@dataclass(frozen=True)
class MatchConfig:
    map: MapConfig = field(default_factory=lambda: MapConfig(9, 9, 2, 0.1, 0.05))
    max_ticks: int = 2000
    visibility: str = "local"

    def __post_init__(self):
        if not isinstance(self.map, MapConfig):
            raise ValueError("arena requires a MapConfig with 2–8 players")
        integer("max_ticks", self.max_ticks, 1)
        if self.visibility != "local":
            raise ValueError("Arena policies use local observations")
