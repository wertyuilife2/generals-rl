"""Array contracts shared by rules, observations, controllers, and future backends.

Coordinates are always [y, x]. Actions use row-major cell_id = y * width + x.
Dynamic arrays belong to GameState; consumers receive independent observations.
"""

from dataclasses import dataclass
from enum import IntEnum
from hashlib import blake2b
from numbers import Integral

import numpy as np


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
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _integer_array(value: object, dtype: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    bounds = np.iinfo(dtype)
    if raw.size and (np.any(raw < bounds.min) or np.any(raw > bounds.max)):
        raise ValueError(f"{name} values exceed {np.dtype(dtype).name}")
    return np.array(raw, dtype=dtype, order="C", copy=True)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Use immutable backing bytes, preventing callers from re-enabling writes."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


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


@dataclass(frozen=True, eq=False)
class MapLayout:
    terrain: np.ndarray

    def __post_init__(self) -> None:
        terrain = _integer_array(self.terrain, np.uint8, "terrain")
        if terrain.ndim != 2 or min(terrain.shape) < 1:
            raise ValueError("terrain must be a nonempty [height, width] matrix")
        if np.any(terrain > Terrain.MOUNTAIN):
            raise ValueError("terrain contains an unknown terrain value")
        object.__setattr__(self, "terrain", _readonly(terrain))
        digest = blake2b(digest_size=16)
        digest.update(b"generals-four-neighbor-v1\0")
        digest.update(np.asarray(terrain.shape, dtype="<i8").tobytes())
        digest.update(terrain.tobytes())
        object.__setattr__(self, "topology_id", digest.hexdigest())

    @property
    def height(self) -> int:
        return self.terrain.shape[0]

    @property
    def width(self) -> int:
        return self.terrain.shape[1]

    @property
    def size(self) -> int:
        return self.terrain.size


@dataclass(eq=False)
class GameState:
    layout: MapLayout
    structure: np.ndarray
    owner: np.ndarray
    army: np.ndarray
    alive: np.ndarray
    general_pos: np.ndarray
    tick: int = 0
    terminated: bool = False
    winner_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, MapLayout):
            raise TypeError("layout must be MapLayout")
        self.structure = _integer_array(self.structure, np.uint8, "structure")
        self.owner = _integer_array(self.owner, np.int16, "owner")
        self.army = _integer_array(self.army, np.int64, "army")
        raw_alive = np.asarray(self.alive)
        if raw_alive.dtype != np.bool_:
            raise TypeError("alive must contain booleans")
        self.alive = np.array(raw_alive, dtype=np.bool_, order="C", copy=True)
        self.general_pos = _integer_array(self.general_pos, np.int32, "general_pos")
        self.tick = _as_int(self.tick, "tick")
        if self.winner_id is not None:
            self.winner_id = _as_int(self.winner_id, "winner_id")

    @property
    def player_count(self) -> int:
        return self.alive.size

    def copy(self) -> "GameState":
        """Deep-copy dynamic arrays while sharing the immutable topology."""
        return GameState(self.layout, self.structure, self.owner, self.army,
                         self.alive, self.general_pos, self.tick,
                         self.terminated, self.winner_id)

    def validate(self) -> None:
        shape = self.layout.terrain.shape
        expected = (("structure", np.uint8, shape), ("owner", np.int16, shape),
                    ("army", np.int64, shape), ("alive", np.bool_, (self.player_count,)),
                    ("general_pos", np.int32, (self.player_count,)))
        for name, dtype, array_shape in expected:
            value = getattr(self, name)
            if value.shape != array_shape or value.dtype != dtype or not value.flags.c_contiguous:
                raise ValueError(f"{name} has an incorrect shape, dtype, or memory layout")
        if not 2 <= self.player_count <= 8:
            raise ValueError("state must contain between 2 and 8 player slots")
        if self.tick < 0 or not isinstance(self.terminated, (bool, np.bool_)):
            raise ValueError("invalid tick or terminated flag")
        if np.any(self.structure > Structure.GENERAL) or np.any(self.army < 0):
            raise ValueError("invalid structure or negative army")
        if np.any(self.owner < NEUTRAL) or np.any(self.owner >= self.player_count):
            raise ValueError("owner is neither an empty/neutral cell nor a player")
        mountains = self.layout.terrain == Terrain.MOUNTAIN
        if np.any(self.structure[mountains] != Structure.NONE) or np.any(self.owner[mountains] != EMPTY) or np.any(self.army[mountains] != 0):
            raise ValueError("mountains cannot contain structures, ownership, or armies")
        empty = self.owner == EMPTY
        if np.any(self.structure[empty] != Structure.NONE) or np.any(self.army[empty] != 0):
            raise ValueError("empty cells cannot contain structures or armies")
        neutral = self.owner == NEUTRAL
        if np.any(self.structure[neutral] != Structure.CITY) or np.any(self.army[neutral] > 40):
            raise ValueError("neutral ownership is reserved for cities with 0–40 troops")
        if np.any(self.general_pos < 0) or np.any(self.general_pos >= self.layout.size) or np.unique(self.general_pos).size != self.player_count:
            raise ValueError("general_pos must contain unique in-bounds positions")
        live_ids = np.flatnonzero(self.alive)
        if not live_ids.size:
            raise ValueError("a game cannot have zero surviving players")
        owner = self.owner.ravel()
        structures = self.structure.ravel()
        general_cells = np.flatnonzero(structures == Structure.GENERAL)
        if set(general_cells.tolist()) != set(self.general_pos[live_ids].tolist()):
            raise ValueError("every surviving player must have exactly one general")
        for pid in range(self.player_count):
            if self.alive[pid]:
                if owner[self.general_pos[pid]] != pid:
                    raise ValueError("surviving player's general must remain their own")
            elif np.any(owner == pid):
                raise ValueError("an eliminated player cannot retain territory")
        if bool(self.terminated) != (live_ids.size == 1):
            raise ValueError("terminated must agree with the surviving player count")
        expected_winner = int(live_ids[0]) if self.terminated else None
        if self.winner_id != expected_winner:
            raise ValueError("winner_id does not agree with termination")


@dataclass(frozen=True, eq=False)
class PlayerStats:
    army: np.ndarray
    territory: np.ndarray
    cities: np.ndarray

    def __post_init__(self) -> None:
        for name in ("army", "territory", "cities"):
            values = _integer_array(getattr(self, name), np.int64, name)
            object.__setattr__(self, name, _readonly(values))


def compute_stats(state: GameState) -> PlayerStats:
    """Use integer accumulation; weighted bincount would lose large-integer precision."""
    owned = state.owner >= 0
    pids = state.owner[owned]
    counts = state.army[owned]
    army = np.zeros(state.player_count, dtype=np.int64)
    maximum = np.iinfo(np.int64).max
    # If even the global upper bound fits, every per-player sum fits too. This
    # keeps ordinary simulation on the NumPy path. Extreme loaded states use
    # Python integer sums so unsupported totals fail rather than wrap around.
    if counts.size and counts.max() > maximum // counts.size:
        for pid in range(state.player_count):
            total = sum(map(int, counts[pids == pid]))
            if total > maximum:
                raise OverflowError("player's total army exceeds int64 statistics capacity")
            army[pid] = total
    else:
        np.add.at(army, pids, counts)
    territory = np.bincount(pids, minlength=state.player_count).astype(np.int64)
    city_owners = state.owner[(state.structure == Structure.CITY) & owned]
    cities = np.bincount(city_owners, minlength=state.player_count).astype(np.int64)
    return PlayerStats(army, territory, cities)


@dataclass(frozen=True)
class MoveResult:
    player_id: int
    action: Action
    valid: bool
    reason: str
    source: int = -1
    target: int = -1
    moved: int = 0
    captured: bool = False
    eliminated: int | None = None
    target_owned: bool = False


@dataclass(frozen=True)
class TickResult:
    tick: int
    moves: tuple[MoveResult, ...]
    stats: PlayerStats
    terminated: bool
    winner_id: int | None
    eliminated: tuple[int, ...] = ()
