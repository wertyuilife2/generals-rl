"""Player information boundary. All published arrays are isolated copies."""
from dataclasses import dataclass
import numpy as np
from generals.core.state import GameState, PlayerStats, Terrain, Structure, UNKNOWN, UNKNOWN_OWNER, compute_stats


def readonly(value):
    result = np.array(value, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Observation:
    """Permitted player state and totals.

    Army, territory, and alive are public for every player. In LOCAL mode,
    stats.cities contains only the observer's own total; all other entries are
    -1 (unknown), including opponents whose cities are partly visible. FULL
    observations and spectator snapshots expose every true city total.
    """

    player_id: int
    tick: int
    terrain: np.ndarray
    structure: np.ndarray
    owner: np.ndarray
    army: np.ndarray
    visible: np.ndarray
    alive: np.ndarray
    general_pos: int
    stats: PlayerStats
    fog_obstacles: np.ndarray
    has_kill: np.ndarray
    warn_afk: np.ndarray
    neutralize_in: np.ndarray
    visibility: str = "local"
    terminated: bool = False
    winner_id: int | None = None

    @property
    def height(self): return self.owner.shape[0]
    @property
    def width(self): return self.owner.shape[1]
    @property
    def size(self): return self.owner.size


def visibility_mask(owner: np.ndarray, player_id: int) -> np.ndarray:
    owned = owner == player_id
    horizontal = owned.copy()
    horizontal[:, 1:] |= owned[:, :-1]
    horizontal[:, :-1] |= owned[:, 1:]
    visible = horizontal.copy()
    visible[1:] |= horizontal[:-1]
    visible[:-1] |= horizontal[1:]
    return visible


def _build(state, pid, mode, stats=None):
    visible = (np.ones(state.owner.shape, dtype=bool) if mode == "full" or state.terminated
               else visibility_mask(state.owner, pid))
    stats = compute_stats(state) if stats is None else stats
    cities = stats.cities
    if mode == "local" and not state.terminated:
        cities = np.full_like(stats.cities, -1)
        cities[pid] = stats.cities[pid]
    stats = PlayerStats(stats.army, stats.territory, cities)
    obstacles = (state.layout.terrain == Terrain.MOUNTAIN) | (state.structure == Structure.CITY)
    if mode == "local":
        for slot, spawn in enumerate(state.spawn_pos):
            if spawn >= 0 and state.general_pos[slot] < 0 and not (state.city_known[pid] & (1 << slot)):
                obstacles.flat[spawn] = False
    arrays = {
        "terrain": np.where(visible, state.layout.terrain, np.uint8(UNKNOWN)),
        "structure": np.where(visible, state.structure, np.uint8(UNKNOWN)),
        "owner": np.where(visible, state.owner, np.int16(UNKNOWN_OWNER)),
        "army": np.where(visible, state.army, np.int64(-1)),
        "visible": visible,
        "alive": state.alive.copy(),
        "fog_obstacles": obstacles & ~visible,
        "has_kill": state.kills > 0,
        "warn_afk": state.warn_afk.astype(bool),
        "neutralize_in": np.where((state.leave_tick >= 0) & (state.general_pos >= 0),
                                  np.clip((state.tick - state.leave_tick) / 50., 0, 1), -1.),
    }
    for array in arrays.values():
        array.flags.writeable = False
    return Observation(
        player_id=pid, tick=state.tick, **arrays,
        general_pos=int(state.general_pos[pid]) if pid >= 0 else -1,
        stats=stats, visibility=mode, terminated=state.terminated, winner_id=state.winner_id)


def observe(state: GameState, player_id: int, mode: str = "local", *,
            stats: PlayerStats | None = None) -> Observation:
    """Observe permitted information, optionally reusing current engine totals.

    Supplied stats must describe this state at its current tick. Published
    arrays remain isolated and read-only, including the supplied public totals.
    """
    if mode not in ("local", "full"):
        raise ValueError("visibility must be local or full")
    if not isinstance(player_id, (int, np.integer)) or isinstance(player_id, bool):
        raise TypeError("player_id must be an integer")
    if not 0 <= player_id < state.player_count:
        raise ValueError("invalid player id")
    return _build(state, int(player_id), mode, stats)


def snapshot(state: GameState, player_id: int | None = None, mode: str = "local"):
    """Explicit privileged spectator view when player_id is None."""
    return _build(state, -1, "full") if player_id is None else observe(state, player_id, mode)


def legal_action_mask(obs: Observation):
    """Return executable source moves; losing attacks remain available."""
    mask = np.zeros((obs.size, 4, 2), dtype=bool)
    pid = obs.player_id
    if pid < 0 or not obs.alive[pid] or obs.terminated:
        return mask, False
    sources = np.flatnonzero((obs.owner == pid) & (obs.army > 1))
    terrain = obs.terrain.ravel()
    h, w = obs.height, obs.width
    for source in sources:
        y, x = divmod(int(source), w)
        for direction, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and terrain[ny * w + nx] != Terrain.MOUNTAIN:
                mask[source, direction, :] = True
    return mask, True
