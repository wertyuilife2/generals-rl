"""Match handles: Python owns metadata; the native runtime owns execution."""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
import numpy as np

from generals import Action, MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.recording import state_from_dict
from .native import bind_loop
from .results import EXECUTION_POLICY


def mark_forfeits(board, forfeits):
    board['forfeited'] = [e['seat'] for e in forfeits if e['tick'] <= board['tick']]
    return board


def timing(game):
    return game.decision_timing()


def failure(game, error):
    result = game.summary()
    return {**result, 'reason': 'policy_error', 'policy_error': str(error),
            'error_detail': getattr(error, 'details', {})}


def close_game(game):
    game.close()


class _Input:
    """Native external-action slot shared by human input and restoration."""
    def __init__(self):
        self.actions = np.zeros(1, np.int64)

    @contextmanager
    def native_batch(self, height, width, players):
        from generals.protocol.native import ABI_VERSION, PolicyABI, PolicyBinding, address
        from generals_arena.policies.builtin import _native_library
        library = _native_library()
        yield PolicyBinding(PolicyABI(ABI_VERSION, self.actions.ctypes.data,
            address(library.arena_wait_act), None, address(library.arena_input_one)), (self, library), None)


class RecordedGame:
    def __init__(self, plan, config, controllers, forfeits=(), *, initial=None, journal=None, record=True):
        from .batch import _PolicyGroup
        self.plan, self.config, self.controllers = plan, config, controllers
        count = len(plan['entrant_ids'])
        bots = [c for c in controllers if c is not None]
        if (len(controllers) != count or not 2 <= count <= 8 or
                len({id(c) for c in bots}) != len(bots) or len(plan['controller_seeds']) != count):
            raise ValueError('each seat requires an independent controller and RNG stream')
        self.forfeits = list(forfeits)
        state = generate_map(MapConfig(**plan['map']), plan['map_seed']) if initial is None else initial
        self._stack = ExitStack()
        self._closed = False
        self._groups, self._inputs, self._views = [], {}, {}
        self.engine = self  # Compatibility handle; contains no Python engine.
        try:
            self.env = self._stack.enter_context(VecEnv.from_states([state], device='cpu', threads=1,
                                                   auto_reset=False, episode_ticks=config.max_ticks))
            for seat, controller in enumerate(controllers):
                group = _PolicyGroup.__new__(_PolicyGroup)
                group.seats, group.lanes = np.array([seat], np.int64), np.arange(1)
                group.cached = False
                if controller is None or getattr(controller, 'inactive', False):
                    controller = _Input()
                    self._inputs[seat] = controller
                    group.batch = True
                else:
                    group.batch = all(callable(getattr(controller, key, None))
                                      for key in ('reset_batch', 'act_batch'))
                    if group.batch:
                        controller.reset_batch([seat], [plan['controller_seeds'][seat]])
                    else:
                        controller.reset(player_id=seat, seed=plan['controller_seeds'][seat])
                group.controller = controller
                group.controllers = [] if group.batch else [controller]
                self._groups.append(group)
            self._active = np.ones(1, bool)
            self._joint = np.zeros((1,count), np.int64)
            self._invalid = np.zeros((1,count), np.int64)
            self._eliminations = np.full((1,count), -1, np.int64)
            self.execution = self._stack.enter_context(bind_loop(self.env, self._groups, self._joint,
                self._invalid, self._eliminations, self._active, journal,
                topologies=[state.layout.topology_id], record=record))
            self.initial_digest = self.execution.query(0)['initial_digest']
            self.execution.events(self.forfeits)
            self._timing = dict(observation_seconds=0., policy_seconds=0., step_seconds=0.)
        except BaseException:
            self._stack.close()
            for group in self._groups:
                group.close()
            raise

    @property
    def state(self):
        """Explicit compatibility snapshot; never read or mutated by the executor."""
        return state_from_dict(self._state if self._closed else self.execution.query(2))

    @property
    def tick(self):
        return self._summary['ticks'] if self._closed else int(self.env.state['tick'][0])

    @property
    def done(self):
        return not bool(self._active[0])

    @property
    def actions(self):
        return self.trace()['actions']

    @property
    def digests(self):
        return self.trace()['digests']

    def advance(self, human_actions=None):
        if self.done:
            raise ValueError('game already finished')
        for seat, controller in self._inputs.items():
            controller.actions[0] = (human_actions or {}).get(seat, Action.wait()).encode()
        self.execution.advance(1, self._timing)

    def run(self):
        if not self.done:
            self.execution.advance(self.config.max_ticks, self._timing)

    def restore(self, actions, *, warm=False, capture=False):
        self.execution.restore(actions, warm=warm, capture=capture)

    def advance_recorded(self, actions, *, warm=False):
        self.restore(np.array([[a.encode() for a in actions]], np.int64), warm=warm)

    def surrender(self, seat):
        self.execution.surrender(seat)
        self.forfeits.append(dict(seat=seat, tick=self.tick, reason='surrender'))

    def decision_timing(self):
        rows = self._metrics if self._closed else self.execution.decision_timing()
        return [{**row, 'entrant_id': self.plan['entrant_ids'][row['seat']]} for row in rows]

    def summary(self):
        value = dict(self._summary if self._closed else self.execution.query(0))
        value.update(self.plan, execution_policy=EXECUTION_POLICY)
        if self.forfeits:
            value['forfeits'] = self.forfeits
        metrics = self.decision_timing()
        if metrics:
            value['decision_timing'] = metrics
        return value

    def result(self):
        if not self.done:
            raise ValueError('cannot score an unfinished game')
        return self.summary()

    def trace(self):
        return self._trace if self._closed else self.execution.query(1)

    def replay_blob(self):
        from .replay import encode_native_trace
        return encode_native_trace(self.trace(), self.forfeits)

    def view(self, seat=None):
        key = -1 if seat is None else seat
        if not self._closed:
            self._views[key] = self.execution.query(3, seat=key)
        return self._views[key]

    def close(self):
        if self._closed:
            return
        self._summary, self._trace, self._state = (self.execution.query(i) for i in (0,1,2))
        self._metrics = self.execution.decision_timing()
        self._stack.close()
        self._closed = True
        for group in self._groups:
            group.close()
