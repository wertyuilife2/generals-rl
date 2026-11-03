"""Array contracts shared by rules, observations, controllers, and future backends.

Coordinates are always [y, x]. Actions use row-major cell_id = y * width + x.
Dynamic arrays belong to GameState; consumers receive independent observations.
"""

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np


from generals.protocol.actions import (
    EMPTY, NEUTRAL, UNKNOWN_OWNER, UNKNOWN, Terrain, Structure, Direction,
    MoveMode, ActionKind, Action, _as_int,
)


def _integer_array(value: object, dtype: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    bounds = np.iinfo(dtype)
    if (not np.can_cast(raw.dtype, dtype, casting="safe") and raw.size
            and (np.any(raw < bounds.min) or np.any(raw > bounds.max))):
        raise ValueError(f"{name} values exceed {np.dtype(dtype).name}")
    return np.array(raw, dtype=dtype, order="C", copy=True)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Use immutable backing bytes, preventing callers from re-enabling writes."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


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
    idle_ticks: int = 0
    afk_turn: int = 0
    end_reason: int = 0
    leave_tick: np.ndarray | None = None
    death_order: np.ndarray | None = None
    kills: np.ndarray | None = None
    damage: np.ndarray | None = None
    spawn_pos: np.ndarray | None = None
    city_known: np.ndarray | None = None
    capture_mask: np.ndarray | None = None
    warn_afk: np.ndarray | None = None

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
        for name in ("idle_ticks", "afk_turn", "end_reason"):
            setattr(self, name, _as_int(getattr(self, name), name))
        p = self.player_count
        defaults = {"leave_tick": (np.int64, np.full(p, -1)),
                    "death_order": (np.int32, np.full(p, -1)),
                    "kills": (np.int32, np.zeros(p, dtype=int)),
                    "damage": (np.int64, np.zeros((p, p), dtype=int)),
                    "spawn_pos": (np.int32, self.general_pos),
                    "city_known": (np.uint8, np.zeros(p, dtype=int)),
                    "capture_mask": (np.uint8, np.zeros(p, dtype=int)),
                    "warn_afk": (np.uint8, np.zeros(p, dtype=int))}
        for name, (dtype, default) in defaults.items():
            value = getattr(self, name)
            setattr(self, name, _integer_array(default if value is None else value, dtype, name))

    @property
    def player_count(self) -> int:
        return self.alive.size

    def copy(self) -> "GameState":
        """Deep-copy dynamic arrays while sharing the immutable topology."""
        return GameState(self.layout, self.structure, self.owner, self.army,
                         self.alive, self.general_pos, self.tick,
                         self.terminated, self.winner_id, self.idle_ticks, self.afk_turn,
                         self.end_reason, self.leave_tick, self.death_order, self.kills,
                         self.damage, self.spawn_pos, self.city_known, self.capture_mask, self.warn_afk)

    def validate(self) -> None:
        shape = self.layout.terrain.shape
        expected = (("structure", np.uint8, shape), ("owner", np.int16, shape),
                    ("army", np.int64, shape), ("alive", np.bool_, (self.player_count,)),
                    ("general_pos", np.int32, (self.player_count,)),
                    ("leave_tick", np.int64, (self.player_count,)),
                    ("death_order", np.int32, (self.player_count,)),
                    ("kills", np.int32, (self.player_count,)),
                    ("damage", np.int64, (self.player_count, self.player_count)),
                    ("spawn_pos", np.int32, (self.player_count,)),
                    ("city_known", np.uint8, (self.player_count,)),
                    ("capture_mask", np.uint8, (self.player_count,)),
                    ("warn_afk", np.uint8, (self.player_count,)))
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
        if np.any((self.owner == NEUTRAL) & (self.structure == Structure.GENERAL)):
            raise ValueError("a neutral general must be converted to a city")
        positions = self.general_pos[self.general_pos >= 0]
        if (np.any(self.general_pos < -1) or np.any(positions >= self.layout.size)
                or np.unique(positions).size != positions.size):
            raise ValueError("general_pos must be -1 or a unique in-bounds position")
        if np.any(self.spawn_pos < -1) or np.any(self.spawn_pos >= self.layout.size):
            raise ValueError("spawn_pos must be -1 or an in-bounds position")
        live_ids = np.flatnonzero(self.alive)
        owner = self.owner.ravel()
        structures = self.structure.ravel()
        general_cells = np.flatnonzero(structures == Structure.GENERAL)
        if set(general_cells.tolist()) != set(positions.tolist()):
            raise ValueError("general positions must identify all current generals")
        for pid in range(self.player_count):
            if self.alive[pid]:
                if self.general_pos[pid] < 0 or owner[self.general_pos[pid]] != pid:
                    raise ValueError("surviving player's general must remain their own")
            elif np.any(owner == pid) and self.leave_tick[pid] < 0 and not self.terminated:
                raise ValueError("only departed players may retain land during the capture grace period")
        if bool(self.terminated) != (live_ids.size <= 1):
            raise ValueError("terminated must agree with the surviving player count")
        expected_winner = int(live_ids[0]) if live_ids.size == 1 else None
        if live_ids.size and self.winner_id != expected_winner:
            raise ValueError("winner_id does not agree with termination")
        if self.winner_id is not None and not 0 <= self.winner_id < self.player_count:
            raise ValueError("winner_id is outside the player range")


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


def rank_players(state: GameState, stats=None) -> list[int]:
    """Official scoreboard order; a kill means participation, not kill count."""
    stats = compute_stats(state) if stats is None else stats
    army = stats['army'] if isinstance(stats, dict) else stats.army
    land = stats['territory'] if isinstance(stats, dict) else stats.territory
    return sorted(range(state.player_count), key=lambda p: (
        -int(state.kills[p] > 0), not bool(state.alive[p]),
        -int(army[p]) if state.alive[p] else -int(state.death_order[p]),
        -int(land[p]) if state.alive[p] else 0, p))


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
