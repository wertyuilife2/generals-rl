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
