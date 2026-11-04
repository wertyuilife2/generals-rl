"""Resource reuse must preserve games, mutable memory and artifact boundaries."""
from generals import MapConfig
from generals_arena.matches.cache import initial_map
from generals_arena.matches.runner import MatchRunner
from generals_arena.policies.entries import prepare_entrant
from execution_test_utils import without_timing
from test_match_runner import plans_for


def test_worker_reuses_package_controllers_across_requests(tmp_path, monkeypatch):
    import policy_test_utils
    source = policy_test_utils.SOURCE.replace(
        "    return Policy(player_id=player_id, seed=seed)",
        "    with open(str(weights)+'.loads', 'a') as stream: stream.write('load\\n')\n"
        "    return Policy(player_id=player_id, seed=seed)")
    monkeypatch.setattr(policy_test_utils, 'SOURCE', source)
    bundle = policy_test_utils.make_bundle(tmp_path/'bundle')
    entry = prepare_entrant(dict(name='model', kind='package', bundle=str(bundle)))
    plans = plans_for(entry, width=25)
    for plan in plans:
        plan['map']['height'] = 25
    entries = {entry['entrant_id']: entry}
    with MatchRunner(entries, max_ticks=8, batch_size=1, record_replays=True) as runner:
        first, _ = runner.run(plans)
        again, _ = runner.run(plans)
    assert without_timing(first) == without_timing(again)
    assert (bundle/'actor.pt.loads').read_text().splitlines() == ['load', 'load']


def test_map_cache_is_immutable_and_shared_with_batch_execution(monkeypatch):
    from generals_arena.matches import cache
    from generals_arena.matches.batch import execute_batch
    from test_native_loop import schedule
    initial_map.cache_clear()
    calls = []
    original = cache.generate_map
    def generate(config, seed):
        calls.append((config, seed))
        return original(config, seed)
    monkeypatch.setattr(cache, 'generate_map', generate)
    entries, plans = schedule(['WAIT','random'], count=2)
    maps = {(MapConfig(**p['map']), p['map_seed']): initial_map(MapConfig(**p['map']), p['map_seed']) for p in plans}
    before = [state.owner.copy() for state in maps.values()]
    first, _ = execute_batch(entries, plans, max_ticks=10, initial_states=maps)
    second, _ = execute_batch(entries, plans, max_ticks=10, initial_states=maps)
    assert len(calls) == 2 and without_timing(first) == without_timing(second)
    for state, owner in zip(maps.values(), before):
        assert not state.owner.flags.writeable and (state.owner == owner).all()
    initial_map.cache_clear()


def test_batch_keys_are_computed_once_per_plan_not_per_scan(monkeypatch, in_process_batch):
    from generals_arena.matches import runner as module
    entry = prepare_entrant(dict(name='random', kind='baseline', policy='random'))
    plans = plans_for(entry, count=12)
    for index, plan in enumerate(plans):
        plan['map']['width'] = 9+index%3
    calls = []
    original = module._batch_key
    def key(plan):
        calls.append(plan['game_id'])
        return original(plan)
    monkeypatch.setattr(module, '_batch_key', key)
    with MatchRunner({entry['entrant_id']:entry}, max_ticks=2, batch_size=2) as runner:
        output, _ = runner.run(plans)
    assert [row['game_id'] for row in output] == [p['game_id'] for p in plans]
    assert len(calls) <= len(plans)*2


def test_device_actions_cross_explicit_host_boundary(monkeypatch):
    import numpy as np
    from generals_arena.matches.batch import _PolicyGroup, execute_batch
    from test_native_loop import schedule
    transfers = []
    original = _PolicyGroup.act
    class DeviceActions:
        def __init__(self, values):
            self.values, self.on_host = values, False
        def __array__(self, *args, **kwargs):
            raise AssertionError('implicit device-to-NumPy conversion')
        def detach(self):
            return self
        def cpu(self):
            transfers.append(True)
            self.on_host = True
            return self
        def numpy(self):
            assert self.on_host
            return np.asarray(self.values)
    monkeypatch.setattr(_PolicyGroup, 'act', lambda self, raw: DeviceActions(original(self, raw)))
    entries, plans = schedule(['WAIT','random'], count=2)
    actual, _ = execute_batch(entries, plans, max_ticks=8, native=False, record_replays=True)
    expected, _ = execute_batch(entries, plans, max_ticks=8, record_replays=True)
    assert transfers and without_timing(actual) == without_timing(expected)


def test_native_backend_projection_and_completion_barrier(monkeypatch):
    import ctypes as C
    from generals.backends.cpu.env import CPUEnv
    from generals.protocol.native import HOST_BUFFERS, address
    from generals_arena.matches.batch import execute_batch
    from test_native_loop import schedule
    entries, plans = schedule(['WAIT','random'], count=2)
    expected, _ = execute_batch(entries, plans, max_ticks=8, record_replays=True)
    original = CPUEnv.native_binding
    barriers, operations = [], []
    def deferred(self):
        binding = original(self)
        abi = binding.abi
        assert abi.capabilities & HOST_BUFFERS and abi.snapshot and not hasattr(abi, 'state')
        observe_type = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p)
        step_type = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)
        sync_type = C.CFUNCTYPE(None, C.c_void_p)
        observe, step = observe_type(abi.observe), step_type(abi.step)
        pending = []
        @observe_type
        def queued_observe(context, seats, selected, output):
            pending.append(lambda: observe(context,seats,selected,output))
        @step_type
        def queued_step(context, actions):
            operations.append('step')
            pending.append(lambda: step(context,actions))
        @sync_type
        def synchronize(context):
            barriers.append(True)
            while pending:
                pending.pop(0)()
        abi.observe, abi.step, abi.synchronize = map(address, (queued_observe,queued_step,synchronize))
        abi.device = 1  # A queued device implementation with explicit host staging.
        binding.owner = (binding.owner,queued_observe,queued_step,synchronize)
        return binding
    monkeypatch.setattr(CPUEnv,'native_binding',deferred)
    actual, _ = execute_batch(entries,plans,max_ticks=8,record_replays=True)
    assert operations == ['step']*8 and barriers
    assert without_timing(actual) == without_timing(expected)
