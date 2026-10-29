"""Decode the official player's stream without constructing a hidden GameState."""
from dataclasses import replace
from numbers import Integral

import numpy as np

from generals.core.observation import Observation, readonly
from generals.core.state import PlayerStats
from generals.protocol.actions import (
    Action, ActionKind, Direction, MoveMode, Terrain, Structure,
    EMPTY, NEUTRAL, UNKNOWN, UNKNOWN_OWNER,
)


class OnlineProtocolError(ValueError):
    pass


class UnsupportedGameError(OnlineProtocolError):
    pass


def integer(value, name, low=0, high=(1 << 63) - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise OnlineProtocolError(f'invalid {name}')
    return int(value)


def patch(previous, diff, *, limit=12290):
    """Apply alternating [unchanged count, changed count, changed values...].

    Every element of the new array must be represented. A trailing unchanged run
    is allowed. Building a fresh array also permits city lists to shrink.
    """
    if not isinstance(diff, list):
        raise OnlineProtocolError('diff must be a list')
    output, cursor = [], 0
    while cursor < len(diff):
        unchanged = integer(diff[cursor], 'unchanged count', high=limit)
        cursor += 1
        offset = len(output)
        if offset + unchanged > len(previous):
            raise OnlineProtocolError('diff copies beyond the previous array')
        output.extend(previous[offset:offset + unchanged])
        if cursor < len(diff):
            changed = integer(diff[cursor], 'changed count', high=limit)
            cursor += 1
            if cursor + changed > len(diff) or len(output) + changed > limit:
                raise OnlineProtocolError('truncated or oversized diff')
            output.extend(integer(v, 'diff value', low=-(1 << 63))
                          for v in diff[cursor:cursor + changed])
            cursor += changed
        if len(output) > limit:
            raise OnlineProtocolError('oversized patched array')
    return output


class ObservationDecoder:
    """Standard 2–8 player games; preserve the server's raw half-turn counter."""
    def __init__(self):
        self.player_id = None
        self.players = 0
        self.replay_id = None
        self.map = []
        self.cities = []
        self.turn = -1
        self.observation = None
        self._last_update = None

    @staticmethod
    def _check_features(data):
        options = data.get('options') or {}
        if not isinstance(options, dict):
            raise OnlineProtocolError('game options must be an object')
        if data.get('modifiers') or options.get('modifiers') or data.get('chessClock'):
            raise UnsupportedGameError('game modifiers are not supported by the observation ABI')
        for key in ('swamps', 'deserts', 'lookouts', 'observatories', 'tunnels', 'strongholds'):
            diff = data.get(key + '_diff', [])
            if data.get(key) or (diff and any(value != 0 for value in diff)):
                raise UnsupportedGameError(f'{key} are not supported by the observation ABI')

    def start(self, data):
        if self.player_id is not None:
            raise OnlineProtocolError('unexpected second game_start; open a new backend per game')
        if not isinstance(data, dict):
            raise OnlineProtocolError('game_start must be an object')
        self._check_features(data)
        names = data.get('usernames')
        if not isinstance(names, list) or not 2 <= len(names) <= 8:
            raise UnsupportedGameError('online mode requires 2–8 player slots')
        player = integer(data.get('playerIndex'), 'playerIndex', high=len(names) - 1)
        teams = data.get('teams')
        if teams:
            if (not isinstance(teams, list) or len(teams) != len(names) or
                    any(not isinstance(t, (int, str)) or isinstance(t, bool) for t in teams) or
                    len(set(teams)) != len(teams)):
                raise UnsupportedGameError('team games are not supported by this observation ABI')
        replay = data.get('replay_id')
        if not isinstance(replay, str) or not replay:
            raise OnlineProtocolError('missing replay_id')
        self.player_id, self.players, self.replay_id = player, len(names), replay

    def update(self, data):
        if self.player_id is None or not isinstance(data, dict):
            raise OnlineProtocolError('game_update received without game_start')
        self._check_features(data)
        turn = integer(data.get('turn'), 'turn')
        if turn < self.turn:
            raise OnlineProtocolError('server turn moved backwards; cannot safely apply diffs')
        if data == self._last_update:
            return None
        board = patch(self.map, data['map_diff'])
        cities = patch(self.cities, data.get('cities_diff', [len(self.cities)]), limit=4096)
        if len(board) < 2:
            raise OnlineProtocolError('missing map dimensions')
        width, height = board[:2]
        if width < 1 or height < 1 or width * height > 4096:
            raise UnsupportedGameError('online board must contain 1–4096 cells')
        size = width * height
        if len(board) not in (2 + 2 * size, 2 + 3 * size):
            raise OnlineProtocolError('map array length does not match dimensions')
        # Current clients append tunnel limits; legacy maps have two planes.
        # Deliberately ignore that layer for now, but retain it in self.map so
        # later server diffs still address the complete serialized array.
        if self.map and board[:2] != self.map[:2]:
            raise OnlineProtocolError('map dimensions changed during the game')
        raw_army = np.asarray(board[2:2 + size], dtype=np.int64)
        raw_owner = np.asarray(board[2 + size:2 + 2 * size], dtype=np.int64)
        if np.any(raw_owner < -4) or np.any(raw_owner >= self.players):
            raise UnsupportedGameError('unknown official terrain code')
        visible = raw_owner >= -2
        if np.any(raw_army[visible] < 0):
            raise OnlineProtocolError('visible army is negative')
        generals = data.get('generals')
        if not isinstance(generals, list) or len(generals) != self.players:
            raise OnlineProtocolError('generals must contain one entry per player')
        generals = [integer(g, 'general position', low=-1, high=size - 1) for g in generals]
        cities = [integer(c, 'city position', high=size - 1) for c in cities]
        terrain = np.where(visible, Terrain.PLAIN, UNKNOWN).astype(np.uint8)
        terrain[raw_owner == -2] = Terrain.MOUNTAIN
        fog_obstacles = raw_owner == -4
        structure = np.where(visible, Structure.NONE, UNKNOWN).astype(np.uint8)
        owner = np.where(visible, raw_owner, UNKNOWN_OWNER).astype(np.int16)
        owner[raw_owner == -2] = EMPTY
        army = np.where(visible, raw_army, -1)
        # Neutral troops may remain after a surrender, even on ordinary land.
        owner[visible & (raw_owner == -1) & (raw_army > 0)] = NEUTRAL
        for cell in cities:
            if visible[cell]:
                if terrain[cell] == Terrain.MOUNTAIN:
                    raise OnlineProtocolError('city overlaps a mountain')
                structure[cell] = Structure.CITY
                if raw_owner[cell] == -1:
                    owner[cell] = NEUTRAL
        for pid, cell in enumerate(generals):
            if cell >= 0 and visible[cell]:
                if raw_owner[cell] != pid:
                    raise OnlineProtocolError('visible general has inconsistent ownership')
                structure[cell] = Structure.GENERAL
        scores = data.get('scores')
        if not isinstance(scores, list) or len(scores) != self.players:
            raise OnlineProtocolError('scores must contain one entry per player')
        totals, land = np.zeros(self.players, np.int64), np.zeros(self.players, np.int64)
        alive, found = np.zeros(self.players, bool), set()
        has_kill, warn_afk = np.zeros(self.players, bool), np.zeros(self.players, bool)
        neutralize_in = np.full(self.players, -1., dtype=np.float64)
        for row in scores:
            if not isinstance(row, dict):
                raise OnlineProtocolError('score must be an object')
            pid = integer(row.get('i'), 'score player', high=self.players - 1)
            if pid in found:
                raise OnlineProtocolError('duplicate score player')
            found.add(pid)
            totals[pid] = integer(row.get('total'), 'total army')
            land[pid] = integer(row.get('tiles'), 'total land', high=size)
            dead = row.get('dead', land[pid] == 0)
            if not isinstance(dead, (bool, np.bool_)):
                raise OnlineProtocolError('score dead must be boolean')
            alive[pid] = not dead
            has_kill[pid] = bool(row.get('has_kill', False))
            warn_afk[pid] = bool(row.get('warn_afk', False))
            grace = row.get('neutralize_in')
            if grace is not None:
                if isinstance(grace, bool) or not isinstance(grace, (int, float)) or not 0 <= grace <= 1:
                    raise OnlineProtocolError('neutralize_in must be a progress value in [0,1]')
                neutralize_in[pid] = grace
        own = self.player_id
        general = generals[own]
        if alive[own] and (general < 0 or not visible[general] or owner[general] != own):
            raise OnlineProtocolError('own living general is not visible')
        city_totals = np.full(self.players, -1, np.int64)
        city_totals[own] = np.count_nonzero((owner == own) & (structure == Structure.CITY))
        arrays = {name: readonly(value.reshape(height, width)) for name, value in
                  (('terrain', terrain), ('structure', structure), ('owner', owner),
                   ('army', army), ('visible', visible), ('fog_obstacles', fog_obstacles))}
        observation = Observation(player_id=own, tick=turn, **arrays,
            alive=readonly(alive), general_pos=general,
            has_kill=readonly(has_kill), warn_afk=readonly(warn_afk), neutralize_in=readonly(neutralize_in),
            stats=PlayerStats(totals, land, city_totals), visibility='local',
            terminated=not bool(alive[own]), winner_id=None)
        self.map, self.cities, self.turn = board, cities, turn
        self._last_update, self.observation = data, observation
        return observation

    def finish(self, won):
        if self.observation is not None:
            alive = self.observation.alive.copy()
            if not won:
                alive[self.player_id] = False
            self.observation = replace(self.observation, terminated=True,
                winner_id=self.player_id if won else None, alive=readonly(alive))
        return self.observation


def attack_arguments(action, observation):
    """Validate spatial/ownership constraints; the server adjudicates combat."""
    if not isinstance(action, Action):
        raise TypeError('controller must return an Action')
    if action.kind == ActionKind.WAIT:
        return None
    if observation.terminated or not observation.alive[observation.player_id]:
        raise ValueError('player cannot move after elimination')
    source = action.source
    if not 0 <= source < observation.size:
        raise ValueError('action source is outside the board')
    y, x = divmod(source, observation.width)
    dy, dx = {Direction.UP: (-1, 0), Direction.RIGHT: (0, 1),
              Direction.DOWN: (1, 0), Direction.LEFT: (0, -1)}[action.direction]
    y, x = y + dy, x + dx
    if not 0 <= y < observation.height or not 0 <= x < observation.width:
        raise ValueError('action target is outside the board')
    target = y * observation.width + x
    if observation.owner.flat[source] != observation.player_id:
        raise ValueError('action source is not owned by this player')
    if observation.army.flat[source] <= 1:
        return None  # Do not queue an order that would become executable only after growth.
    if observation.terrain.flat[target] == Terrain.MOUNTAIN:
        raise ValueError('action target is a mountain')
    return source, target, action.mode == MoveMode.HALF
