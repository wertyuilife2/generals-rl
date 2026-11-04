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
    from generals.backends.cpu.env import _library
    board = np.ascontiguousarray(owner, dtype=np.int16)
    visible = np.empty(board.shape, dtype=bool)
    _library()[0].generals_visibility(board.ctypes.data, visible.ctypes.data,
                                     *board.shape, int(player_id))
    return visible


def _build(state, pid, mode, stats=None):
    from generals.backends.cpu.env import CPUEnv
    backend = CPUEnv.from_states([state], threads=1)
    try:
        if stats is not None:
            backend.stats[0] = np.stack((stats.army, stats.territory, stats.cities), axis=-1)
        raw = backend.observer(pid, full=mode == 'full')()
    finally:
        backend.close()
    boards = raw['state']
    shape = state.owner.shape
    arrays = {key: boards[key][0].reshape(shape) for key in
              ('terrain', 'structure', 'owner', 'army', 'fog_obstacles')}
    arrays.update({key: boards[key][0] for key in ('alive', 'has_kill', 'warn_afk', 'neutralize_in')})
    arrays['visible'] = raw['visible'][0].reshape(shape)
    for array in arrays.values():
        array.flags.writeable = False
    return Observation(player_id=pid, tick=int(boards['tick'][0]), **arrays,
        general_pos=int(raw['general_pos'][0]), stats=PlayerStats(*raw['stats'][0].T),
        visibility=mode, terminated=bool(boards['terminated'][0]),
        winner_id=None if boards['winner_id'][0] < 0 else int(boards['winner_id'][0]))


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
    from generals.backends.cpu.env import _library
    owner = np.ascontiguousarray(obs.owner, dtype=np.int16)
    army = np.ascontiguousarray(obs.army, dtype=np.int64)
    terrain = np.ascontiguousarray(obs.terrain, dtype=np.uint8)
    _library()[0].generals_legal(owner.ctypes.data, army.ctypes.data, terrain.ctypes.data,
                               mask.ctypes.data, obs.height, obs.width, pid)
    return mask, True
