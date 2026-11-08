"""The native loop removes Python dispatch while retaining every game receipt."""
from execution_test_utils import without_timing
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np
import pytest

from generals import MapConfig
from generals_arena.matches.batch import execute_batch
from generals_arena.matches.batch_worker import BatchJournal
from generals_arena.policies.builtin import wait_spec
from generals_arena.policies.entries import prepare_entrant


def schedule(lineup, count=4):
    entries, ids = {}, []
    for name in lineup:
        entry = prepare_entrant(wait_spec() if name == 'WAIT' else
                                dict(name=name, kind='baseline', policy=name))
        entries[entry['entrant_id']] = entry
        ids.append(entry['entrant_id'])
    p = len(ids)
    plans = []
    for lane in range(count):
        order = [(i + lane) % p for i in range(p)]
        plans.append(dict(game_id=str(lane), map=asdict(MapConfig(25, 25, p)), map_seed=900+lane,
            entrant_ids=[ids[i] for i in order], instance_ids=order,
            controller_seeds=[2**48 + lane*p + i for i in order]))
    return entries, plans


@pytest.mark.parametrize('lineup', [
    ['random', 'expansion', 'defensive', 'aggressive'],
    ['WAIT', 'random', 'WAIT'], ['WAIT', 'WAIT'],
])
def test_native_and_python_loops_preserve_full_replays_and_nonreplay_results(lineup):
    entries, plans = schedule(lineup)
    native, timing = execute_batch(entries, plans, max_ticks=280, record_replays=True)
    reference, _ = execute_batch(entries, plans, max_ticks=280, record_replays=True, native=False)
    unrecorded, _ = execute_batch(entries, plans, max_ticks=280)
    assert 'native_bookkeeping_seconds' in timing
    assert without_timing(native) == without_timing(reference)
    assert without_timing([r for r, _ in unrecorded]) == without_timing([r for r, _ in reference])
    assert all(blob is None for _, blob in unrecorded)


def test_native_loop_never_calls_python_backend_observer_step_policy_or_feedback(monkeypatch):
    from generals.backends.cpu.env import CPUEnv
    from generals_baselines.base import BaseAI
    from generals_baselines.batch import PolicyBatch
    from generals_arena.policies.builtin import WaitController
    entries, plans = schedule(['WAIT', 'random', 'expansion', 'defensive'])
    def forbidden(*args, **kwargs):
        pytest.fail('per-tick Python dispatch reached from a native match')
    for cls, names in [(CPUEnv, ('step', 'observer')),
                       (BaseAI, ('act_batch', 'on_result_batch')),
                       (PolicyBatch, ('act', 'on_result')),
                       (WaitController, ('act_batch',))]:
        for name in names:
            monkeypatch.setattr(cls, name, forbidden)
    records, timing = execute_batch(entries, plans, max_ticks=40)
    assert len(records) == len(plans)
    assert 'native_bookkeeping_seconds' in timing


def test_bound_planners_keep_rng_memory_and_receipts_identical():
    entries, plans = schedule(['random', 'expansion', 'defensive', 'aggressive'])
    caches = [{}, {}]
    try:
        for native, cache in zip((False, True), caches):
            execute_batch(entries, plans, max_ticks=240, native=native, runtime_cache=cache)
        assert caches[0].keys() == caches[1].keys()
        for key in caches[0]:
            left, right = [cache[key]._batch.controllers for cache in caches]
            for a, b in zip(left, right):
                assert a.rng.bit_generator.state == b.rng.bit_generator.state
                assert a.path == b.path and a._pending == b._pending
                for name in ('terrain', 'structure', 'owner', 'army', 'last_seen'):
                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        # Rebinding cached controllers creates a new seed/seat lease, including
        # different batch size and changing seat assignments.
        actual, _ = execute_batch(entries, plans[-1:], max_ticks=30, runtime_cache=caches[1])
        expected, _ = execute_batch(entries, plans[-1:], max_ticks=30, native=False)
        assert without_timing(actual) == without_timing(expected)
    finally:
        for cache in caches:
            for controller in cache.values():
                controller.close()


def test_native_journal_matches_reference_actions_and_publication(tmp_path):
    entries, plans = schedule(['WAIT', 'random', 'expansion'])
    journals = [BatchJournal(tmp_path/f'{i}.bin', len(plans), 280, 3, create=True) for i in range(2)]
    try:
        outputs = [execute_batch(entries, plans, max_ticks=280, journal=journal, native=native)[0]
                   for native, journal in zip((False, True), journals)]
        assert without_timing(outputs[0]) == without_timing(outputs[1])
        for name in ('actions',):
            np.testing.assert_array_equal(getattr(journals[0], name), getattr(journals[1], name))
        assert journals[1].headers[:, 0].tolist() == [r['ticks'] for r, _ in outputs[1]]
    finally:
        for journal in journals:
            journal.close()


def test_native_sdk_calls_same_boundary_without_python_act(monkeypatch, tmp_path):
    from generals_arena.policies import native as sdk
    from test_native_batch import policy, plans as native_plans
    entry = policy(tmp_path, '''
    if(o->previous && o->previous->player!=o->player) return -1;
    return arena_move(o->general_pos, (s.seed+o->tick)%4, false);
''')
    plans = native_plans(entry, 3, [[10, 20, 30], [40, 50, 60]])
    entries = {entry['entrant_id']: entry}
    with ExitStack() as stack:
        journals = [BatchJournal(tmp_path/f'sdk-{i}.bin', 2, 40, 3, create=True) for i in range(2)]
        # DIAGNOSTICS must release mmap-backed ndarray views before closing.
        for journal in journals:
            stack.callback(journal.close)
        stack.callback(setattr, sdk, 'DIAGNOSTICS', None)
        monkeypatch.setattr(sdk, 'DIAGNOSTICS', journals[0])
        expected, _ = execute_batch(entries, plans, max_ticks=40, record_replays=True,
                                    journal=journals[0], native=False)
        def forbidden(*args, **kwargs):
            pytest.fail('native SDK used per-tick Python dispatch')
        monkeypatch.setattr(sdk.NativeController, 'act_batch', forbidden)
        monkeypatch.setattr(sdk.NativeController, 'on_result_batch', forbidden)
        sdk.DIAGNOSTICS = journals[1]
        actual, timing = execute_batch(entries, plans, max_ticks=40, record_replays=True, journal=journals[1])
        assert 'native_bookkeeping_seconds' in timing
        for rows in (actual, expected):
            for record, _ in rows:
                metrics = record.pop('decision_timing')
                assert all(m['calls'] == 40 and not m['errors'] and not m['timeouts'] for m in metrics)
        assert without_timing(actual) == without_timing(expected)
        np.testing.assert_array_equal(journals[0].actions, journals[1].actions)
        np.testing.assert_array_equal(journals[0].headers[:,0], journals[1].headers[:,0])


def test_native_bind_rejects_replaced_backend_storage():
    from generals import VecEnv
    with VecEnv.from_config(MapConfig(25, 25, 2), episode_ticks=10) as env:
        env.state['army'] = env.state['army'].copy()
        with pytest.raises(ValueError, match='replacing their storage'):
            env.unwrapped.native_binding()


def test_native_bind_rejects_bad_abi_and_noncontiguous_outputs():
    from generals import VecEnv
    from generals_arena.matches.native import NativeLoop
    from generals_arena.policies.builtin import WaitController
    with VecEnv.from_config(MapConfig(25, 25, 2), batch_size=2, episode_ticks=10) as env:
        backend = env.unwrapped.native_binding()
        with WaitController().native_batch(25, 25, 2) as policy:
            outputs = dict(env=env, bindings=[policy, policy], backend=backend,
                seats=np.array([[0, 1], [1, 0]], np.int64), joint=np.zeros((2, 2), np.int64),
                invalid=np.zeros((2, 2), np.int64), eliminations=np.full((2, 2), -1, np.int64),
                active=np.ones(2, bool), journal=None)
            with pytest.raises(ValueError, match='contiguous'):
                NativeLoop(**(outputs | {'joint': outputs['joint'].T}))
            policy.abi.abi = 999
            with pytest.raises(ValueError, match='unsupported native policy'):
                NativeLoop(**outputs)
