"""Map configuration shared by independent environment implementations."""

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real
from generals.protocol.mapgen import MapGenerationConfig


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
    generation: MapGenerationConfig | None = None
    generation_attempt: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.generation, dict):
            object.__setattr__(self, 'generation', MapGenerationConfig(**self.generation))
        if self.generation is not None and not isinstance(self.generation, MapGenerationConfig):
            raise TypeError('generation must be MapGenerationConfig or None')
        object.__setattr__(self, 'generation_attempt',
                           _integer('generation_attempt', self.generation_attempt, 0))
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
        if (available if self.generation is None else cells) < self.players:
            raise ValueError("not enough ordinary cells to place all generals")
        if cells > 2**31 - 1:
            raise ValueError("map is too large for int32 cell identifiers")
