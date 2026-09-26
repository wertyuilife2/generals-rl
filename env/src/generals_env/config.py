"""Small, backend-independent configuration objects for the CPU prototype."""

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real


def _integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class MapConfig:
    width: int = 25
    height: int = 25
    players: int = 4
    mountain_density: float = 0.2
    city_density: float = 0.1

    def __post_init__(self) -> None:
        for name in ("width", "height", "players"):
            object.__setattr__(self, name, _integer(name, getattr(self, name), 1))
        if not 2 <= self.players <= 8:
            raise ValueError("players must be between 2 and 8")
        for name, maximum in (("mountain_density", 0.4), ("city_density", 0.2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
            object.__setattr__(self, name, float(value))
        cells = self.width * self.height
        available = cells - floor(cells * self.mountain_density) - floor(cells * self.city_density)
        if available < self.players:
            raise ValueError("not enough ordinary cells to place all generals")
        if cells > 2**31 - 1:
            raise ValueError("map is too large for int32 cell identifiers")


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
