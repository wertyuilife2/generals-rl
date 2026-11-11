"""D4 commutes with native rules, legal actions and selective map-bank reset."""
from contextlib import ExitStack

import numpy as np
import pytest
import torch

from generals import MapConfig, VecEnv
from generals.core.batch import generate_states, pack_states, unpack_states
from generals_rl.config import task_config
from generals_rl.map_symmetry import MapSymmetry
from generals_rl.task import GeneralsTask
from test_paired_ablation import configuration


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
def test_all_eight_transforms_commute_with_native_steps_masks_and_rewards(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg = configuration(device=device)
    cfg['task'].update(num_envs=8, map_pool_size=0, max_episode_steps=20)
    states = generate_states(MapConfig(9, 9, 2, .3, .3), 8, seed=121)
    arrays, _, _ = pack_states(states)
    transformed, _ = MapSymmetry(9).apply(arrays, np.ones((8, 81), dtype=bool), np.arange(8))
    others = unpack_states(transformed, width=9, height=9, validate=True)
    symmetry = MapSymmetry(9, device)
    transforms = torch.arange(8, device=device)
    with ExitStack() as stack:
        tasks = []
        for initial in (states, others):
            backend = VecEnv.from_states(initial, device=device, backend=cfg['task']['backend'],
                                         auto_reset=False, episode_ticks=20)
            tasks.append(stack.enter_context(GeneralsTask(task_config(cfg['task']), backend=backend)))
        first, second = tasks
        generator = torch.Generator(device=device).manual_seed(71)
        for _ in range(45):
            mask = first.get_observations()['action_mask']
            actions = torch.multinomial(mask.float(), 1, generator=generator).flatten()
            mapped = symmetry.actions(actions.reshape(8, 2), transforms).flatten()
            assert second.get_observations()['action_mask'].gather(1, mapped[:, None]).all()
            a, b = first.step(actions), second.step(mapped)
            for field in ('reward', 'terminated', 'truncated'):
                torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
            original = {k: torch.as_tensor(v, device=device) for k, v in first.backend.state.items()}
            expected, _ = symmetry.apply(original, torch.ones((8, 81), dtype=torch.bool, device=device), transforms)
            for key in expected:
                torch.testing.assert_close(expected[key], torch.as_tensor(second.backend.state[key], device=device), rtol=0, atol=0, msg=key)


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
def test_d4_reset_preserves_bank_sampling_rng_and_live_rows(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg = configuration(device=device)
    cfg['task']['num_envs'] = 4
    with ExitStack() as stack:
        plain = stack.enter_context(GeneralsTask(task_config(cfg['task'])))
        cfg['task']['d4_augmentation'] = True
        augmented = stack.enter_context(GeneralsTask(task_config(cfg['task'])))
        for _ in range(3):
            before = {k: v.clone() for k, v in augmented.get_observations().items()}
            mask = torch.tensor([True, True, False, False, True, True, False, False], device=device)
            plain.reset(mask)
            augmented.reset(mask)
            for key, value in before.items():
                torch.testing.assert_close(augmented.get_observations()[key][~mask], value[~mask], rtol=0, atol=0)
            a, b = plain.map_sampler, augmented.map_sampler
            if device == 'cuda':
                assert torch.equal(a.generator.get_state(), b.generator.get_state())
                assert torch.equal(a.stats, b.stats)
            else:
                assert a.generator.bit_generator.state == b.generator.bit_generator.state
                np.testing.assert_array_equal(a.stats, b.stats)
