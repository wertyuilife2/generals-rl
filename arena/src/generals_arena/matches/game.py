"""Single-match state, controller lifecycle and incremental play/recovery."""
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.core.recording import state_digest, state_to_dict
from generals.core.state import Action, compute_stats
from generals import MapConfig, make_engine

from generals_arena.matches.observation import board_view
from generals_arena.matches.results import EXECUTION_POLICY
from generals_arena.matches.results import build_result


def apply_forfeits(engine, events):
    """Translate match departures into the environment's official surrender rule."""
    state = engine.state
    changed = False
    for event in events:
        seat = event['seat']
        if event['tick'] == state.tick and state.alive[seat]:
            engine.surrender(seat)
            changed = True
    return changed



def mark_forfeits(board, forfeits):
    board['forfeited'] = [e['seat'] for e in forfeits if e['tick'] <= board['tick']]
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
        for pid in result.eliminated:
            self.eliminations[pid] = self.state.tick
        self.stats = result.stats
        self.actions.append([[int(a.kind), a.source, int(a.direction), int(a.mode)] for a in actions])
        self.digests.append(state_digest(self.state))
        for move in result.moves:
            self.invalid[move.player_id] += int(not move.valid)
            controller = self.controllers[move.player_id]
            if controller is not None:
                controller.on_result(move)
        for pid in result.eliminated:
            eliminate = getattr(self.controllers[pid], "eliminate", None)
            if eliminate is not None:
                eliminate()

    def _apply_forfeits(self):
        before = self.state.alive.copy()
        if apply_forfeits(self.engine, self.forfeits):
            self.stats = compute_stats(self.state)
            for seat in range(self.state.player_count):
                if before[seat] and not self.state.alive[seat]:
                    self.eliminations[seat] = self.state.tick
                    eliminate = getattr(self.controllers[seat], 'eliminate', None)
                    if eliminate is not None:
                        eliminate()

    def surrender(self, seat):
        self.forfeits.append({'seat': seat, 'tick': self.state.tick, 'reason': 'surrender'})
        self._apply_forfeits()

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
                             initial_digest=self.initial_digest)
        value["execution_policy"] = EXECUTION_POLICY
        if self.forfeits:
            value['forfeits'] = self.forfeits
        metrics = timing(self)
        if metrics:
            value['decision_timing'] = metrics
        return value

    def replay_blob(self):
        from .replay import encode_replay
        return encode_replay(self.initial, self.actions, self.digests, self.forfeits)

    def view(self, seat=None):
        return board_view(self.state, seat)
