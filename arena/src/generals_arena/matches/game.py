"""Single-match state, controller lifecycle and incremental play/recovery."""
import numpy as np
from generals.core.mapgen import generate_map
from generals.core.observation import observe, snapshot
from generals.core.recording import state_digest, state_from_dict, state_to_dict
from generals.core.state import Action, NEUTRAL, Structure, compute_stats, GameState, MapLayout
from generals import MapConfig, make_engine

from generals_arena.matches.observation import board_view
from generals_arena.matches.results import EXECUTION_POLICY
from generals_arena.matches.results import build_result


def apply_forfeits(engine, events):
    """Arena lifecycle event; movement/combat/growth still use env's CPU kernel.

    Abandoned troops are neutral, stop growing and can be captured normally.
    This extends Arena's neutral cells beyond mapgen's starting neutral cities.
    Never reconstruct the engine from a mid-game Arena state: initialize from
    the original map and replay events, keeping env/RL's initial-state contract.
    """
    state = engine.state
    changed = False
    for event in events:
        seat = event['seat']
        if event['tick'] == state.tick and state.alive[seat]:
            state.owner[state.owner == seat] = NEUTRAL
            state.structure.ravel()[state.general_pos[seat]] = Structure.CITY
            state.alive[seat] = False
            changed = True
    if changed:
        survivors = np.flatnonzero(state.alive)
        state.terminated = len(survivors) <= 1
        state.winner_id = int(survivors[0]) if len(survivors) == 1 else None
        engine.invalidate_cache()
    return changed


def state_from_snapshot(value, forfeits):
    if not forfeits:
        return state_from_dict(value)
    # Trusted match-process snapshot, which may contain abandoned neutral land.
    return GameState(MapLayout(np.asarray(value['terrain'])),
                     *(np.asarray(value[key]) for key in
                       ('structure', 'owner', 'army', 'alive', 'general_pos')),
                     value['tick'], value['terminated'], value['winner_id'])


def mark_forfeits(board, forfeits):
    board['forfeited'] = [e['seat'] for e in forfeits if e['tick'] <= board['tick']]
    for seat in board['forfeited']:
        board['alive'][seat] = False
    return board


def timing(game):
    return [{'seat': seat, 'entrant_id': game.plan['entrant_ids'][seat], **controller.decision_metrics}
            for seat, controller in enumerate(game.controllers)
            if hasattr(controller, 'decision_metrics')]


def failure(game, error):
    return {**game.plan, 'ticks': game.state.tick, 'initial_digest': game.initial_digest,
            'final_digest': state_digest(game.state), 'reason': 'policy_error',
            'policy_error': str(error), 'error_detail': getattr(error, 'details', {}),
            'decision_timing': timing(game), **({'forfeits':game.forfeits} if game.forfeits else {})}


def close_game(game):
    for controller in game.controllers:
        close = getattr(controller, 'close', None)
        if close is not None:
            close()
    close = getattr(game.engine, 'close', None)
    if close is not None:
        close()


class _RecordedAction:
    """Restore a surviving controller's memory while replaying committed actions."""
    def __init__(self, controller, action, warm):
        self.controller, self.action, self.warm = controller, action, warm

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def act(self, observation):
        if self.warm:
            values = getattr(self.controller, 'values', None)
            saved = values.copy() if values is not None else None
            self.controller.act(observation)
            # Successful restoration is not a new tournament decision. A new
            # timeout still retains its diagnostic record and forfeits the seat.
            if saved is not None:
                values[:] = saved
        return self.action


class RecordedGame:
    def __init__(self, plan, config, controllers, forfeits=()):
        self.plan, self.config = plan, config
        self.controllers = controllers  # None denotes a human seat.
        count = len(plan["entrant_ids"])
        bots = [controller for controller in controllers if controller is not None]
        if (len(controllers) != count or not 2 <= count <= 8 or
                len({id(controller) for controller in bots}) != len(bots) or
                len(plan["controller_seeds"]) != count):
            raise ValueError("each seat requires an independent controller and RNG stream")
        self.map_config = MapConfig(**plan["map"])
        self.state = generate_map(self.map_config, plan["map_seed"])
        self.engine = make_engine(self.state)
        self.stats = compute_stats(self.state)
        self.initial = state_to_dict(self.state)
        self.initial_digest = state_digest(self.state)
        self.actions, self.digests = [], []
        self.eliminations = [None] * count
        self.invalid = [0] * count
        self.kills = [0] * count
        self.forfeits = list(forfeits)
        for seat, controller in enumerate(controllers):
            if controller is not None:
                controller.reset(player_id=seat, seed=plan["controller_seeds"][seat])
        self._apply_forfeits()

    @property
    def done(self):
        return self.state.terminated or self.state.tick >= self.config.max_ticks

    def _advance(self, human_actions=None):
        if self.done:
            raise ValueError("game already finished")
        human_actions = human_actions or {}
        actions = [Action.wait() if not self.state.alive[seat] else
                   human_actions.get(seat, Action.wait()) if controller is None else
                   controller.act(observe(self.state, seat, stats=self.stats))
                   for seat, controller in enumerate(self.controllers)]
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("controller must return an Action")
        result = self.engine.step(actions)
        for move in result.moves:
            if move.eliminated is not None:
                self.kills[move.player_id] += 1
        self.stats = result.stats
        self.actions.append([[int(a.kind), a.source, int(a.direction), int(a.mode)] for a in actions])
        self.digests.append(state_digest(self.state))
        for move in result.moves:
            self.invalid[move.player_id] += int(not move.valid)
            controller = self.controllers[move.player_id]
            if controller is not None:
                controller.on_result(move)
            if move.eliminated is not None:
                self.eliminations[move.eliminated] = self.state.tick
                eliminate = getattr(self.controllers[move.eliminated], "eliminate", None)
                if eliminate is not None:
                    eliminate()

    def _apply_forfeits(self):
        if apply_forfeits(self.engine, self.forfeits):
            self.stats = compute_stats(self.state)

    def advance(self, human_actions=None):
        actions = dict(human_actions or {})
        controllers = self.controllers
        self.controllers = list(controllers)
        for event in self.forfeits:
            if event['tick'] <= self.state.tick:
                seat = event['seat']
                self.controllers[seat] = None
                actions[seat] = Action.wait()
        try:
            self._advance(actions)
        finally:
            self.controllers = controllers
        self._apply_forfeits()

    def advance_recorded(self, actions, *, warm=False):
        controllers = self.controllers
        self.controllers = [_RecordedAction(c, actions[i], warm) if c is not None else None
                            for i, c in enumerate(controllers)]
        try:
            self._advance(dict(enumerate(actions)))
        finally:
            self.controllers = controllers
        self._apply_forfeits()

    def result(self):
        if not self.done:
            raise ValueError("cannot score an unfinished game")
        value = build_result(self.plan, state=self.state, stats=self.stats,
                             elimination_ticks=self.eliminations, invalid_actions=self.invalid,
                             initial_digest=self.initial_digest, kills=self.kills)
        value["execution_policy"] = EXECUTION_POLICY
        if self.forfeits:
            eliminated = self.eliminations.copy()
            for event in self.forfeits:
                seat, tick = event['seat'], event['tick']
                eliminated[seat] = tick if eliminated[seat] is None else min(eliminated[seat], tick)
            value.update(forfeits=self.forfeits, capture_ticks=self.eliminations.copy(),
                         elimination_ticks=eliminated)
            if any(e['tick'] == self.state.tick for e in self.forfeits) and self.state.terminated:
                value['reason'] = 'surrender'
        metrics = timing(self)
        if metrics:
            value['decision_timing'] = metrics
        return value

    def replay_blob(self):
        from .replay import encode_replay
        return encode_replay(self.initial, self.actions, self.digests, self.forfeits)

    def view(self, seat=None):
        return board_view(snapshot(self.state, seat))
