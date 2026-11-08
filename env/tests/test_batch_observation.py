"""Scalar/batch CPU/CUDA/MPS public observations carry identical information."""
import numpy as np
import pytest

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.observation import observe


@pytest.mark.parametrize('players', (2, 4, 8))
@pytest.mark.parametrize('device', ('cpu', 'cuda:0', 'mps'))
def test_raw_observation_matches_scalar_and_does_not_alias(players, device):
    if device != 'cpu':
        torch = pytest.importorskip('torch')
        if not (torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader') if device == 'mps' else torch.cuda.is_available()):
            pytest.skip('GPU unavailable')
    states = [generate_map(MapConfig(25, 25, players, .1, .1), seed) for seed in range(3)]
    ids = np.arange(3) % players
    with VecEnv.from_states(states, device=device, auto_reset=False) as env:
        bound = env.observer(ids)
        for _ in range(3):
            raw = bound()
            def host(value):
                return value if isinstance(value, np.ndarray) else value.cpu().numpy()
            for lane, state in enumerate(env.to_states()):
                expected = observe(state, int(ids[lane]))
                for key in ('terrain', 'structure', 'owner', 'army', 'fog_obstacles'):
                    np.testing.assert_array_equal(host(raw['state'][key])[lane], getattr(expected, key).ravel())
                for key in ('has_kill', 'warn_afk', 'neutralize_in'):
                    np.testing.assert_array_equal(host(raw['state'][key])[lane], getattr(expected, key))
                np.testing.assert_array_equal(host(raw['visible'])[lane], expected.visible.ravel())
                np.testing.assert_array_equal(host(raw['state']['alive'])[lane], expected.alive)
                for index, name in enumerate(('army', 'territory', 'cities')):
                    np.testing.assert_array_equal(host(raw['stats'])[lane, :, index], getattr(expected.stats, name))
                assert host(raw['general_pos'])[lane] == expected.general_pos
            raw['state']['army'][...] = 999
            assert not np.any(host(env.state['army']) == 999)
            raw['player_ids'][...] = (raw['player_ids'] + 1) % players
            np.testing.assert_array_equal(host(bound()['player_ids']), ids)
            actions = np.zeros((3, players), np.int64)
            env.step(actions if device == 'cpu' else torch.as_tensor(actions, device=device))


def test_invalid_seats_fail_at_binding():
    with VecEnv.from_config(MapConfig(18, 18, 2), batch_size=2) as env:
        for ids in ((-1, 0), (0, 2), (0,), (True, False)):
            with pytest.raises(ValueError):
                env.observer(ids)


def test_native_cpu_observer_matches_numpy_through_departure_reset_and_city_memory():
    from generals.core.batch_observation import batch_observer
    with VecEnv.from_config(MapConfig(25, 27, 4), batch_size=3, threads=1) as env:
        seats = np.array([0, 2, 3])
        native, reference = env.observer(seats), batch_observer(env, seats)

        def compare():
            left, right = native(), reference()
            for key in right:
                if key == 'state':
                    for field in right[key]:
                        np.testing.assert_array_equal(left[key][field], right[key][field], err_msg=field)
                elif isinstance(right[key], np.ndarray):
                    np.testing.assert_array_equal(left[key], right[key], err_msg=key)
                else:
                    assert left[key] == right[key]
            return left

        saved = compare()
        env.surrender(1)
        for _ in range(55):
            env.step(np.zeros((3, 4), np.int64))
            compare()
        env.state['city_known'][:] = 0
        compare()
        env.state['city_known'][:] = 15
        compare()
        env.state['terminated'][:] = True
        compare()
        env.reset(np.array([True, False, True]))
        compare()
        np.testing.assert_array_equal(saved['state']['tick'], [0, 0, 0])
