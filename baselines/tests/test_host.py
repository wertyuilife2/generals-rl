"""Differential checks against the original ctypes ABI, including RNG/memory."""
import ctypes
from dataclasses import replace

import numpy as np
import pytest

from generals import Action, MapConfig, make_engine
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals_baselines import create_controller
from generals_baselines.base import _Observation, _Options


def ctypes_act(policy, obs):
    arrays = [np.ascontiguousarray(getattr(obs, name)) for name in
              ('terrain', 'structure', 'owner', 'army', 'visible', 'alive')]
    native = _Observation(*(a.ctypes.data for a in arrays), obs.height, obs.width,
                          len(obs.alive), obs.player_id, obs.tick, bool(obs.terminated))
    options = _Options(policy.max_sources, policy.max_path_attempts, policy.path_lifetime,
                       policy.memory_ttl, policy.temperature, (ctypes.c_double*3)(*policy.weights.values()))
    with policy.rng.bit_generator.lock:
        result = policy._lib.baseline_act(policy._handle, ctypes.byref(native),
                                         ctypes.byref(options), ctypes.byref(policy._random))
    assert result >= 0
    return Action.decode(result)


@pytest.mark.parametrize('kind', ['random', 'aggressive', 'defensive', 'expansion'])
def test_native_binding_preserves_live_options_memory_paths_and_random_stream(kind):
    state = generate_map(MapConfig(15, 17, 2), 15)
    left, right = create_controller(kind, seed=73), create_controller(kind, seed=73)
    opponent = create_controller('random', player_id=1, seed=74)
    engine = make_engine(state)
    try:
        for tick in range(180):
            if state.terminated:
                break
            for policy in (left, right):
                policy.max_sources = 1 + tick % 6
                policy.max_path_attempts = tick % 13
                policy.path_lifetime = 1 + tick % 20
                policy.memory_ttl = tick % 51
                policy.temperature = .25 + (tick % 3) * .25
            obs = observe(state, 0)
            if tick % 2:
                obs = replace(obs, **{k: np.asfortranarray(getattr(obs, k)) for k in
                                     ('terrain', 'structure', 'owner', 'army', 'visible')})
            actual = left.act(obs)
            assert actual == ctypes_act(right, obs)
            assert left.rng.bit_generator.state == right.rng.bit_generator.state
            assert left.path == right.path and left.last_search_attempts == right.last_search_attempts
            for name in ('terrain', 'structure', 'owner', 'army', 'last_seen'):
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
            result = engine.step([actual, opponent.act(observe(state, 1))])
            for move in result.moves:
                if move.player_id == 0:
                    left.on_result(move); right.on_result(move)
                else:
                    opponent.on_result(move)
    finally:
        engine.close()
        for policy in (left, right, opponent):
            policy.close()


def test_bad_observations_and_options_do_not_consume_rng():
    policy = create_controller('random', seed=19)
    obs = observe(generate_map(MapConfig(15, 15, 2), 4), 0)
    initial = policy.rng.bit_generator.state
    for changes in ({'army': obs.army.astype(np.float64)}, {'visible': obs.visible.astype(np.uint8)},
                    {'owner': obs.owner[:, :-1]}, {'alive': np.ones(9, dtype=bool)}):
        with pytest.raises(ValueError):
            policy.act(replace(obs, **changes))
        assert policy.rng.bit_generator.state == initial
    policy.max_path_attempts = -1
    with pytest.raises(ValueError, match='negative'):
        policy.act(obs)
    assert policy.rng.bit_generator.state == initial
    policy.close()


def test_options_keep_original_ctypes_integer_conversion():
    obs = observe(generate_map(MapConfig(15, 15, 2), 3), 0)
    left, right = create_controller('aggressive', seed=9), create_controller('aggressive', seed=9)
    try:
        for policy in (left, right):
            policy.max_sources = 2**100 + 6
            policy.max_path_attempts = 2**100 + 12
            policy.path_lifetime = 2**100 + 20
            policy.memory_ttl = 2**100 + 50
        assert left.act(obs) == ctypes_act(right, obs)
        assert left.rng.bit_generator.state == right.rng.bit_generator.state
    finally:
        left.close(); right.close()
