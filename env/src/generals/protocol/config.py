"""Map configuration shared by independent environment implementations."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class MapConfig:
    width: int | None = None
    height: int | None = None
    players: int = 4
    mountain_density: float = 0.5
    city_density: float = 0.5

    def __post_init__(self) -> None:
        for name in ("width", "height", "players"):
            value = getattr(self, name)
            if value is not None or name == "players":
                object.__setattr__(self, name, _integer(name, value, 1))
        if not 2 <= self.players <= 8:
            raise ValueError("players must be between 2 and 8")
        for name, maximum in (("mountain_density", 1.), ("city_density", 1.)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")
            object.__setattr__(self, name, float(value))
        cells = (self.width or 33) * (self.height or 33)
        if cells > 2**31 - 1:
            raise ValueError("map is too large for int32 cell identifiers")
