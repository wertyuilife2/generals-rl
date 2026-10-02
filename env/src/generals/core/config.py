"""CPU runner settings and compatibility exports of shared map configuration."""
from dataclasses import dataclass
from generals.protocol.config import MapConfig, _integer


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    max_ticks: int | None = None
    visibility: str = "local"
    debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer("seed", self.seed, 0))
        if self.max_ticks is not None:
            object.__setattr__(self, "max_ticks", _integer("max_ticks", self.max_ticks, 1))
        if self.visibility not in ("local", "full"):
            raise ValueError("visibility must be 'local' or 'full'")
        if not isinstance(self.debug, bool):
            raise TypeError("debug must be a boolean")
