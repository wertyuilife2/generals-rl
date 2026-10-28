"""Scalar/batch CPU/CUDA public observations carry identical information."""
import numpy as np
import pytest

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals.core.observation import observe


@pytest.mark.parametrize('players', (2, 4, 8))
@pytest.mark.parametrize('device', ('cpu', 'cuda:0'))
def test_raw_observation_matches_scalar_and_does_not_alias(players, device):
    if device != 'cpu':
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
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
                for key in ('terrain', 'structure', 'owner', 'army'):
                    np.testing.assert_array_equal(host(raw['state'][key])[lane], getattr(expected, key).ravel())
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
