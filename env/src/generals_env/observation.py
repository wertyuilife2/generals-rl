"""Player information boundary. All published arrays are isolated copies."""
from dataclasses import dataclass
import numpy as np
from .state import GameState, PlayerStats, Terrain, UNKNOWN, UNKNOWN_OWNER, compute_stats


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
    h, w = owner.shape
    padded = np.pad(owned, 1)
    visible = np.zeros_like(owned)
    for dy in range(3):
        for dx in range(3):
            visible |= padded[dy:dy + h, dx:dx + w]
    return visible


def _build(state, pid, mode):
    visible = (np.ones(state.owner.shape, dtype=bool) if mode == "full"
               else visibility_mask(state.owner, pid))
    stats = compute_stats(state)
    cities = stats.cities
    if mode == "local":
        cities = np.full_like(stats.cities, -1)
        cities[pid] = stats.cities[pid]
    stats = PlayerStats(readonly(stats.army), readonly(stats.territory), readonly(cities))
    return Observation(
        player_id=pid, tick=state.tick,
        terrain=readonly(np.where(visible, state.layout.terrain, UNKNOWN).astype(np.uint8)),
        structure=readonly(np.where(visible, state.structure, UNKNOWN).astype(np.uint8)),
        owner=readonly(np.where(visible, state.owner, UNKNOWN_OWNER).astype(np.int16)),
        army=readonly(np.where(visible, state.army, -1).astype(np.int64)),
        visible=readonly(visible), alive=readonly(state.alive),
        general_pos=int(state.general_pos[pid]) if pid >= 0 else -1,
        stats=stats, visibility=mode, terminated=state.terminated, winner_id=state.winner_id)


def observe(state: GameState, player_id: int, mode: str = "local") -> Observation:
    if mode not in ("local", "full"):
        raise ValueError("visibility must be local or full")
    if not isinstance(player_id, (int, np.integer)) or isinstance(player_id, bool):
        raise TypeError("player_id must be an integer")
    if not 0 <= player_id < state.player_count:
        raise ValueError("invalid player id")
    return _build(state, int(player_id), mode)


def snapshot(state: GameState, player_id: int | None = None, mode: str = "local"):
    """Explicit privileged spectator view when player_id is None."""
    return _build(state, -1, "full") if player_id is None else observe(state, player_id, mode)


def legal_action_mask(obs: Observation):
    """Return [N, direction, mode] and WAIT; losing attacks remain legal."""
    mask = np.zeros((obs.size, 4, 2), dtype=bool)
    pid = obs.player_id
    if pid < 0 or not obs.alive[pid] or obs.terminated:
        return mask, False
    movable = np.flatnonzero((obs.owner == pid) & (obs.army >= 2))
    terrain = obs.terrain.ravel()
    h, w = obs.height, obs.width
    for source in movable:
        y, x = divmod(int(source), w)
        for direction, (dy, dx) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and terrain[ny * w + nx] != Terrain.MOUNTAIN:
                mask[source, direction, :] = True
    return mask, True
