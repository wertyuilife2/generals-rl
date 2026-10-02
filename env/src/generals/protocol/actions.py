"""Backend-neutral scalar action encoding and game constants; no engine imports."""
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

EMPTY = -1
NEUTRAL = -2
UNKNOWN_OWNER = -3
UNKNOWN = 255


class Terrain(IntEnum):
    PLAIN = 0
    MOUNTAIN = 1


class Structure(IntEnum):
    NONE = 0
    CITY = 1
    GENERAL = 2


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MoveMode(IntEnum):
    ALL_BUT_ONE = 0
    HALF = 1


class ActionKind(IntEnum):
    WAIT = 0
    MOVE = 1


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    source: int = -1
    direction: Direction = Direction.UP
    mode: MoveMode = MoveMode.ALL_BUT_ONE

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ActionKind(_as_int(self.kind, "kind")))
        object.__setattr__(self, "source", _as_int(self.source, "source"))
        object.__setattr__(self, "direction", Direction(_as_int(self.direction, "direction")))
        object.__setattr__(self, "mode", MoveMode(_as_int(self.mode, "mode")))
        if self.kind == ActionKind.WAIT and (self.source != -1 or self.direction != 0 or self.mode != 0):
            raise ValueError("WAIT must use canonical source=-1, direction=0, mode=0")

    @classmethod
    def wait(cls) -> "Action":
        return cls(ActionKind.WAIT)

    @classmethod
    def move(cls, source: int, direction: Direction, mode: MoveMode = MoveMode.ALL_BUT_ONE) -> "Action":
        return cls(ActionKind.MOVE, source, direction, mode)

    def encode(self) -> int:
        """WAIT=0; MOVE=1 + source*8 + direction*2 + mode."""
        if self.kind == ActionKind.WAIT:
            return 0
        if self.source < 0:
            raise ValueError("cannot encode a negative source cell")
        return 1 + self.source * 8 + int(self.direction) * 2 + int(self.mode)

    @classmethod
    def decode(cls, encoded: int) -> "Action":
        encoded = _as_int(encoded, "encoded action")
        if encoded < 0:
            raise ValueError("encoded action cannot be negative")
        if encoded == 0:
            return cls.wait()
        source, remainder = divmod(encoded - 1, 8)
        direction, mode = divmod(remainder, 2)
        return cls.move(source, Direction(direction), MoveMode(mode))
