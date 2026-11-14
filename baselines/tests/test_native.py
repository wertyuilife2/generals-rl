"""Native policy ownership, information boundaries and build identity."""
import subprocess
import sys

import numpy as np
import pytest

from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals_baselines import create_controller


def test_native_policy_does_not_load_training_or_use_other_seats_memory():
    state = generate_map(MapConfig(15, 17, 4), 94)
    left = create_controller('random', 0, 7)
    right = create_controller('random', 1, 7)
    left.act(observe(state, 0)); right.act(observe(state, 1))
    assert left._handle != right._handle
    assert left.native_metadata['source_sha256']
    saved = left.terrain
    saved[:] = 12
    assert not np.any(left.terrain == 12)
    old = right.rng.bit_generator.state
    left.act(observe(state, 0))
    assert right.rng.bit_generator.state == old
    left.close()
    assert left.path == [] and left.terrain is None
    with pytest.raises(RuntimeError, match='closed'):
        left.path = [0, 1]
    with pytest.raises(RuntimeError, match='closed'):
        left.act(observe(state, 0))
    left.reset(0, 7)
    fresh = create_controller('random', 0, 7)
    assert left.act(observe(state, 0)) == fresh.act(observe(state, 0))
    code = "from generals_baselines import create_controller; create_controller('aggressive'); import sys; assert 'torch' not in sys.modules and 'generals_rl' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


def test_noncontiguous_observations_are_copied_without_modification():
    from dataclasses import replace
    state = generate_map(MapConfig(15, 17, 2), 5)
    obs = observe(state, 0)
    changes = {key: np.asfortranarray(getattr(obs, key)) for key in
               ('terrain', 'structure', 'owner', 'army', 'visible')}
    alternate = replace(obs, **changes)
    for policy in ('aggressive', 'expansion', 'defensive', 'random'):
        a, b = create_controller(policy, seed=55), create_controller(policy, seed=55)
        assert a.act(obs) == b.act(alternate)
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
    for key in changes:
        np.testing.assert_array_equal(changes[key], getattr(obs, key))


@pytest.mark.parametrize('kind', ['aggressive', 'expansion', 'defensive', 'random'])
@pytest.mark.parametrize('source_limit', [0, 1, 12])
def test_planner_scratch_storage_survives_map_size_changes(kind, source_limit):
    reused = create_controller(kind, seed=31)
    reused.max_sources = source_limit
    try:
        for width, height in [(15, 17), (33, 35), (15, 63), (63, 15), (17, 15), (15, 17)]:
            state = generate_map(MapConfig(width, height, 2), 45)
            general = int(state.general_pos[0])
            y, x = divmod(general, width)
            yy, xx = np.indices(state.owner.shape)
            domain = ((abs(yy-y) + abs(xx-x) <= 4)
                      & (state.layout.terrain == 0)
                      & (state.structure != 2))
            state.owner[domain] = 0
            state.army[domain] = 20
            state.army.ravel()[general] = 30
            obs = observe(state, 0)
            fresh = create_controller(kind, seed=0)
            try:
                fresh.max_sources = source_limit
                fresh.rng.bit_generator.state = reused.rng.bit_generator.state
                assert reused.act(obs) == fresh.act(obs)
                assert reused.path == fresh.path
                assert reused.rng.bit_generator.state == fresh.rng.bit_generator.state
            finally:
                fresh.close()
    finally:
        reused.close()
