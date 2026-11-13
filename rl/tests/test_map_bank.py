"""Cache integrity, parallel equivalence and deterministic bounded map renewal."""
from contextlib import ExitStack
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest
import torch

from generals import MapConfig, VecEnv
from generals.core.mapgen import generate_map
from generals_rl.map_bank import build_bank, cache_identity, load_bank
from generals_rl.map_curriculum import MapCurriculumConfig
from generals_rl.map_sampling import MapSampler

CONFIG = MapConfig(9, 9, 2, .5, .5)
CURRICULUM = MapCurriculumConfig(100, 200, levels=3)


def same_bank(a, b):
    for name in a[0]:
        np.testing.assert_array_equal(a[0][name], b[0][name])
    for x, y in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(x, y)


def test_parallel_generation_matches_serial_for_each_curriculum_level():
    options = dict(width=9, height=9, curriculum=CURRICULUM)
    same_bank(build_bank(CONFIG, 260, 47, workers=1, **options),
              build_bank(CONFIG, 260, 47, workers=2, **options))


def cache_job(directory):
    return load_bank(CONFIG, 64, 47, width=9, height=9, curriculum=CURRICULUM, cache_dir=directory)[1]


def test_cache_concurrent_build_corruption_and_copy_on_write(tmp_path):
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(cache_job, [str(tmp_path)]*2))
    assert sorted(r['cache_hit'] for r in results) == [False, True]
    assert results[0]['cache_key'] == results[1]['cache_key']
    args = dict(width=9, height=9, curriculum=CURRICULUM, cache_dir=tmp_path)
    bank, meta = load_bank(CONFIG, 64, 47, **args)
    expected = bank[0]['army'].copy()
    bank[0]['army'][:] = 12345
    again, _ = load_bank(CONFIG, 64, 47, **args)
    np.testing.assert_array_equal(again[0]['army'], expected)
    (tmp_path/meta['cache_key']/'army.npy').write_bytes(b'interrupted/corrupt cache')
    repaired, meta = load_bank(CONFIG, 64, 47, **args)
    assert not meta['cache_hit']
    np.testing.assert_array_equal(repaired[0]['army'], expected)


def test_cache_key_tracks_generation_not_schedule(monkeypatch):
    def key(config=CONFIG, size=16, seed=47, curriculum=CURRICULUM):
        return cache_identity(config, size, seed, 9, 9, curriculum)
    assert key() == key(curriculum=replace(CURRICULUM, warmup_transitions=1000, end_transitions=2000))
    assert all(key() != other for other in (key(size=17), key(seed=48),
        key(config=replace(CONFIG, mountain_density=.4)), key(curriculum=replace(CURRICULUM, levels=4))))
    original = key()
    monkeypatch.setattr('generals_rl.map_bank.source_digest', lambda: 'different-generator')
    assert original != key()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_stream_is_deterministic_independent_of_prefetch_and_preserves_live_games(device):
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    with ExitStack() as stack:
        samplers = []
        for workers in (1, 2):
            env = stack.enter_context(VecEnv.from_states([generate_map(CONFIG, 91)]*4,
                auto_reset=False, device=device, backend='cpu' if device == 'cpu' else 'gpu'))
            sampler = MapSampler(env, CONFIG, 6, seed=47, curriculum=CURRICULUM,
                generation_workers=workers, refresh_reuse=1, refresh_chunk_size=4,
                prefetch_chunks=2*workers+1)
            stack.callback(sampler.close)
            samplers.append(sampler)
        mask = np.array([True, False, True, False]) if device == 'cpu' else torch.tensor(
            [True, False, True, False], device=device)
        copy = lambda x: x.copy() if isinstance(x, np.ndarray) else x.clone()
        a, b = samplers
        initial = {k: copy(v) for k, v in a.backend.state.items()}
        first_bank = copy(a.arrays['army'])
        for step in range(8):
            # Different readiness, same logical replacement points and samples.
            b.stream.pending[0][2].result()
            for sampler in samplers:
                sampler.set_transitions(100+step*20)
                sampler.reset(mask)
                if step == 0:
                    assert sampler.stream.introduced == 0  # Initial fill is never discarded unused.
            for k, value in a.backend.state.items():
                if device == 'cpu':
                    np.testing.assert_array_equal(value, b.backend.state[k])
                    np.testing.assert_array_equal(value[~mask], initial[k][~mask])
                else:
                    torch.testing.assert_close(value, b.backend.state[k], rtol=0, atol=0)
                    torch.testing.assert_close(value[~mask], initial[k][~mask], rtol=0, atol=0)
        assert a.stream.introduced == b.stream.introduced == 12
        assert (len(a.stream.pending), len(b.stream.pending)) == (3, 5)
        assert not (np.array_equal(first_bank, a.arrays['army']) if device == 'cpu'
                    else torch.equal(first_bank, a.arrays['army']))
        assert a.arrays['army'].shape == (6*CURRICULUM.levels, 81)
