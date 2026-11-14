"""One-time native match binding; Python handles lifecycle and result records."""
import ctypes as C
import json
from contextlib import contextmanager, ExitStack
from functools import lru_cache
from pathlib import Path

import numpy as np

from generals.protocol.native import BackendABI, PolicyABI, INCLUDE


class _Journal(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('headers', 'records', 'committed', 'actions')] + [
        ('capacity', C.c_int64)]


class _Config(C.Structure):
    _fields_ = [(name, C.c_int) for name in ('count', 'height', 'width', 'players')] + [
        ('backend', BackendABI), ('policies', C.c_void_p), ('seats', C.c_void_p),
        *[(name, C.c_void_p) for name in ('joint', 'invalid', 'eliminations', 'active', 'ending')],
        ('journal', _Journal), ('metrics', C.c_void_p), ('cursor', C.c_void_p),
        ('topologies', C.c_void_p), ('record', C.c_int), ('decision_limits', C.c_void_p)]


class _Status(C.Structure):
    _fields_ = [(name, C.c_int64) for name in ('group', 'lane', 'ticks')] + [
        (name, C.c_double) for name in ('observation', 'policy', 'step', 'bookkeeping')]


@lru_cache(maxsize=1)
def _library():
    from generals.backends._build import build_shared
    lib, _ = build_shared(Path(__file__).with_suffix('.cpp'), name='arena-loop', include_dirs=[INCLUDE, INCLUDE.parent/'backends'])
    lib.arena_loop_create.argtypes, lib.arena_loop_create.restype = [C.POINTER(_Config)], C.c_void_p
    lib.arena_loop_advance.argtypes = [C.c_void_p, C.c_int64, C.POINTER(_Status)]
    lib.arena_loop_advance.restype = C.c_int
    lib.arena_loop_destroy.argtypes, lib.arena_loop_destroy.restype = [C.c_void_p], None
    lib.arena_loop_error.argtypes, lib.arena_loop_error.restype = [], C.c_char_p
    lib.arena_loop_depart.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.arena_loop_events.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
    lib.arena_loop_restore.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int, C.c_int, C.POINTER(_Status)]
    lib.arena_loop_query.argtypes, lib.arena_loop_query.restype = [C.c_void_p, C.c_int, C.c_int, C.c_int], C.c_char_p
    return lib


def _pointer(array, dtype, shape):
    # Once at the ownership boundary. No coercion or hidden copies of outputs.
    if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or array.shape != shape
            or not array.flags.c_contiguous or not array.flags.aligned or not array.flags.writeable):
        raise ValueError(f'native buffer must be writable contiguous {dtype} {shape}')
    return array.ctypes.data


class NativeLoop:
    def __init__(self, env, bindings, backend, seats, joint, invalid, eliminations, active, journal, *, topologies=None, record=False,
                 decision_limit_ms=None, limits=None):
        from generals_arena.util import integer
        if decision_limit_ms is not None:
            integer('decision_limit_ms', decision_limit_ms, 1)
        b, p = env.num_envs, env.players
        self.env, self.bindings = env, bindings
        from generals_arena.policies.limits import execution_limits
        self.limits = limits or [execution_limits({}, decision_limit_ms) for _ in bindings]
        self.limit_values = np.asarray([[v['decision_budget_ms'] * 1_000_000,
            (v['decision_timeout_ms'] or 0) * 1_000_000] for v in self.limits], dtype=np.int64)
        self.ending = np.zeros(b, bool)
        self.status = _Status()
        policies = (PolicyABI * p)(*[binding.abi for binding in bindings])
        log = _Journal()
        self.metrics = journal.all_records if journal is not None else np.zeros((b, 8, 8), np.int64)
        self.cursor = getattr(journal, 'active_lane', np.full(1, -1, np.int64))
        if topologies is None:
            topologies = [state.layout.topology_id for state in env.to_states()]
        topology = (C.c_char_p * b)(*[key.encode() for key in topologies])
        self.seats = seats

        if journal is not None:
            capacity = journal.actions.shape[0]
            log = _Journal(_pointer(journal.headers, np.int64, (b, 4)),
                _pointer(journal.all_records, np.int64, (b, 8, 8)),
                _pointer(journal.committed, np.int64, (b, 8, 8)),
                _pointer(journal.actions, np.int64, (capacity, b, p)), capacity)
        self.config = _Config(b, env.height, env.width, p, backend.abi, C.addressof(policies),
            _pointer(seats, np.int64, (p, b)), _pointer(joint, np.int64, (b, p)),
            _pointer(invalid, np.int64, (b, p)), _pointer(eliminations, np.int64, (b, p)),
            _pointer(active, np.bool_, (b,)), _pointer(self.ending, np.bool_, (b,)), log, self.metrics.ctypes.data, self.cursor.ctypes.data,
            C.addressof(topology), record, self.limit_values.ctypes.data)
        self.owners = (backend, policies, seats, joint, invalid, eliminations, active, journal, topology)
        self.lib = _library()
        self.handle = self.lib.arena_loop_create(C.byref(self.config))
        if not self.handle:
            raise ValueError(self.lib.arena_loop_error().decode())

    def advance(self, ticks, timing):
        if not self.handle:
            raise RuntimeError('native match loop is closed')
        code = self.lib.arena_loop_advance(self.handle, ticks, C.byref(self.status))
        status = self.status
        for name in ('observation', 'policy', 'step'):
            timing[name + '_seconds'] += getattr(status, name)
        timing['native_bookkeeping_seconds'] = timing.get('native_bookkeeping_seconds', 0.) + status.bookkeeping
        self._check(code)

    def _check(self, code):
        status = self.status
        if code == 0:
            for binding in self.bindings:
                pending = getattr(binding, 'pending_error', None)
                if pending is not None:
                    pending()
        if code == 3:
            from generals_arena.policies.native import PolicyError
            seat = int(self.seats[status.group, status.lane])
            values = self.metrics[status.lane, seat]
            detail = dict(kind='timeout' if values[7] == 1 else 'policy_error', player=seat,
                lane=int(status.lane), tick=int(self.env.state['tick'][status.lane]),
                elapsed_ns=int(values[6]), limit_ns=int(self.limit_values[status.group, 0]),
                hard_timeout_ns=int(self.limit_values[status.group, 1]) or None, phase='decision')
            raise PolicyError(f'native policy decision {detail["kind"]}: {detail["elapsed_ns"]/1e6:.3f} ms', details=detail)
        if code == 1:
            binding = self.bindings[status.group]
            if binding.check_error is not None:
                binding.check_error(status.lane, int(self.env.state['tick'][status.lane]))
            raise RuntimeError(f'native policy failed in lane {status.lane}')
        if code == 2:
            raise RuntimeError(f'vector simulator reported a native error in lane {status.lane}')
        if code:
            raise RuntimeError(self.lib.arena_loop_error().decode())

    def _require_open(self):
        if not self.handle:
            raise RuntimeError('native match loop is closed')

    def query_bytes(self, kind, lane=0, seat=-1):
        self._require_open()
        data = self.lib.arena_loop_query(self.handle, kind, lane, seat)
        if data is None:
            raise RuntimeError(self.lib.arena_loop_error().decode())
        return data

    def query(self, kind, lane=0, seat=-1):
        value = json.loads(self.query_bytes(kind, lane, seat))
        if kind == 0:
            value['decision_limits'] = [self.limits[g] for g in np.argsort(self.seats[:, lane])]
        return value

    def decision_timing(self, lane=0):
        from generals_arena.policies.native import metrics
        rows = []
        for group, seats in enumerate(self.seats):
            seat = int(seats[lane])
            values = self.metrics[lane, seat]
            if values[0] or values[4] or values[5]:
                row = dict(seat=seat, **metrics(values))
                row['limit_ns'] = int(self.limit_values[group, 0])
                row['hard_timeout_ns'] = int(self.limit_values[group, 1]) or None
                if not self.bindings[group].abi.act_one:
                    row.update(measurement='batch policy wall time / active lanes; excludes observations and IPC')
                rows.append(row)
        return sorted(rows, key=lambda row: row['seat'])

    def surrender(self, seat, lane=0):
        self._require_open()
        self._check(self.lib.arena_loop_depart(self.handle, lane, seat))

    def events(self, events):
        self._require_open()
        values = np.asarray([[event['tick'], event['seat']] for event in events], np.int64)
        self._check(self.lib.arena_loop_events(self.handle, values.ctypes.data, len(events)))

    def restore(self, actions, *, warm=False, capture=False):
        self._require_open()
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        if actions.ndim != 2 or actions.shape[1] != self.env.players:
            raise ValueError('restoration requires encoded [ticks, players] actions')
        self._check(self.lib.arena_loop_restore(self.handle, actions.ctypes.data, len(actions), warm, capture, C.byref(self.status)))

    def close(self):
        if self.handle:
            self.lib.arena_loop_destroy(self.handle)
            self.handle = None
        self.owners, self.bindings = (), ()
        self.env = self.metrics = self.cursor = self.seats = None


@contextmanager
def bind_loop(env, groups, joint, invalid, eliminations, active, journal, *, native_policies=True, topologies=None, record=False,
              decision_limit_ms=None):
    """Capability-based selection; never branch on a policy's name or kind."""
    if not callable(getattr(env.unwrapped, 'native_binding', None)):
        raise ValueError('Arena requires a native backend execution binding')
    with ExitStack() as stack:
        backend = env.unwrapped.native_binding()
        bindings = [stack.enter_context(group.controller.native_batch(env.height, env.width, env.players))
                    if native_policies and group.batch and callable(getattr(group.controller, 'native_batch', None))
                    else callback_binding(group, env) for group in groups]
        from generals_arena.policies.limits import execution_limits
        limits = []
        for group, binding in zip(groups, bindings):
            supplied = getattr(group.controller, '_arena_execution', None)
            entry = getattr(group, 'entry', {})
            if supplied is not None and 'execution' not in entry:
                entry = {**entry, 'execution': supplied}
            value = execution_limits(entry, None if 'execution' in entry else decision_limit_ms)
            limits.append(value)
        loop = NativeLoop(env, bindings, backend, np.stack([group.seats for group in groups]),
                          joint, invalid, eliminations, active, journal, topologies=topologies, record=record,
                          decision_limit_ms=decision_limit_ms, limits=limits)
        stack.callback(loop.close)
        yield loop


class _Observation(C.Structure):
    _fields_ = [(key, C.c_void_p) for key in ('terrain', 'structure', 'owner', 'army', 'visible',
        'alive', 'fog_obstacles', 'has_kill', 'warn_afk', 'neutralize_in', 'stats', 'general_pos',
        'active', 'tick', 'terminated', 'winner_id')]


class _View(C.Structure):
    _fields_ = [(key, C.c_int) for key in ('count', 'height', 'width', 'players')] + [
        ('player_ids', C.c_void_p), ('observation', _Observation)]


class _Feedback(C.Structure):
    _fields_ = [('count', C.c_int)] + [(key, C.c_void_p) for key in ('moves', 'actions', 'active')]


def _array(pointer, dtype, shape):
    return np.ctypeslib.as_array(C.cast(pointer, C.POINTER(np.ctypeslib.as_ctypes_type(np.dtype(dtype)))),
                                shape=shape)


def callback_binding(group, env):
    """Budgeted Python/RL inference; native execution owns the match and observations."""
    from generals.protocol.native import PolicyBinding, ABI_VERSION, address
    from .batch import _numpy
    failure = []
    b, p, n = env.num_envs, env.players, env.height * env.width
    raw = result = None
    @C.CFUNCTYPE(C.c_int64, C.c_void_p, C.POINTER(_View), C.c_void_p)
    def act(context, view, output):
        nonlocal raw, result
        try:
            if failure:
                raise failure[0]
            if raw is None:
                # Storage and output addresses are stable for this native lease.
                o = view.contents.observation
                boards = {key: _array(getattr(o, key), dtype, shape) for key, dtype, shape in (
                    ('terrain', 'u1', (b,n)), ('structure', 'u1', (b,n)), ('owner', 'i2', (b,n)),
                    ('army', 'i8', (b,n)), ('alive', '?', (b,p)), ('fog_obstacles', '?', (b,n)),
                    ('has_kill', '?', (b,p)), ('warn_afk', '?', (b,p)), ('neutralize_in', 'f8', (b,p)),
                    ('tick', 'i8', (b,)), ('terminated', '?', (b,)), ('winner_id', 'i4', (b,)))}
                raw = dict(state=boards, visible=_array(o.visible, '?', (b,n)),
                    stats=_array(o.stats, 'i8', (b,p,3)), general_pos=_array(o.general_pos, 'i4', (b,)),
                    active=_array(o.active, '?', (b,)), player_ids=group.seats,
                    width=env.width, height=env.height, visibility='local')
                result = _array(output, 'i8', (b,))
            # CPU tensors share storage; device tensors explicitly synchronize
            # their small action vector back to the host backend.
            result[:] = _numpy(group.act(raw))
            return -1
        except BaseException as exc:
            failure[:] = [exc]
            return 0
    moves, joint = np.zeros((b,p,9), np.int64), np.zeros((b,p), np.int64)
    lanes = np.arange(b)
    @C.CFUNCTYPE(None, C.c_void_p, C.POINTER(_Feedback))
    def feedback(context, data):
        try:
            data = data.contents
            moves[lanes, group.seats] = _array(data.moves, 'i8', (b,9))
            joint[lanes, group.seats] = _array(data.actions, 'i8', (b,))
            group.feedback({'_moves': moves}, joint, _array(data.active, '?', (b,)))
        except BaseException as exc:
            failure[:] = [exc]
    def check_error(lane, tick):
        if failure:
            raise failure[0]
    needs_feedback = getattr(group.controller, 'needs_feedback', True)
    # The match loop accounts and guards the entire batch callback.
    binding = PolicyBinding(PolicyABI(ABI_VERSION, None, address(act), address(feedback) if needs_feedback else None,
                                     None, flags=0),
                            (group, act, feedback, failure), check_error)
    binding.pending_error = lambda: check_error(0, 0)
    return binding
