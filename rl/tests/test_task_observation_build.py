"""Selective resets and timeout snapshots preserve the full observation path."""
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from generals import VecEnv
from generals.core.state import Action, Direction
from generals_rl.mdp.rewards import RewardConfig
from generals_rl.task import GeneralsTask, TaskConfig
from test_mdp import capture_state


def make_task(stack, *, device='cpu', dual=True, feature_set='temporal_paths', privileged=True):
    config = TaskConfig(width=3, height=3, num_envs=3, backend='cpu' if device == 'cpu' else 'gpu',
        device=device, random_maps=False, map_pool_size=0, max_episode_steps=3,
        history_length=8 if feature_set == 'temporal_paths' else 4, feature_set=feature_set,
        opponent='current_self_play' if dual else 'wait', train_both_sides=dual,
        persistent_knowledge=True, combat_features=True, king_threat_distance=True,
        critic_privileged=privileged, rotate_seats=True,
        reward=RewardConfig(territory=0, cities=0, army=0, tick=0, potential_gamma=1))
    backend = VecEnv.from_states([capture_state() for _ in range(3)], device=device,
        backend=config.backend, episode_ticks=3, auto_reset=False)
    return stack.enter_context(GeneralsTask(config, backend=backend))


def same_fields(first, second):
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0, msg=key)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('dual,feature_set,privileged', [
    (True, 'temporal_paths', True), (True, 'stacked', False), (False, 'stacked', True)])
def test_training_path_matches_full_snapshots_through_partial_termination_and_timeout(device, dual, feature_set, privileged):
    if device != 'cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.set_num_threads(1)
    with ExitStack() as stack:
        reference, optimized = [make_task(stack, device=device, dual=dual, feature_set=feature_set,
                                          privileged=privileged) for _ in range(2)]
        retained = []
        saw_termination = saw_timeout = saw_partial = False
        for tick in range(10):
            actions = torch.zeros(reference.num_envs, dtype=torch.long, device=device)
            if tick == 0:
                actions[0] = Action.move(3, Direction.RIGHT).encode()
            previous = optimized.get_observations()
            retained.append((previous, {key: value.clone() for key, value in previous.items()}))
            first = reference.step(actions)
            second = optimized.step(actions, capture_all_terminals=False)
            same_fields(first.observations, second.observations)
            same_fields(first.metrics, second.metrics)
            for name in ('reward', 'terminated', 'truncated'):
                torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)
            if second.terminal_observations is None:
                assert not second.truncated.any()
            else:
                for key, value in second.terminal_observations.items():
                    torch.testing.assert_close(value[second.truncated], first.terminal_observations[key][first.truncated],
                                               rtol=0, atol=0, msg=key)
                retained.append((second.terminal_observations,
                                 {key: value.clone() for key, value in second.terminal_observations.items()}))
            saw_termination |= bool(second.terminated.any())
            saw_timeout |= bool(second.truncated.any())
            count = int((second.terminated | second.truncated).sum())
            saw_partial |= 0 < count < reference.num_envs
        assert saw_termination and saw_timeout and saw_partial
        for actual, saved in retained:
            same_fields(actual, saved)
        for name in ('completed_episodes', 'truncated_episodes', 'terminated_episodes'):
            assert getattr(reference, name) == getattr(optimized, name)
        same_fields(reference.batch().state, optimized.batch().state)


def test_natural_terminal_needs_one_full_encode_and_only_resets_ended_memory_rows():
    with ExitStack() as stack:
        task = make_task(stack)
        actions = torch.tensor([Action.move(3, Direction.RIGHT).encode(), 0, 0, 0, 0, 0])
        with patch.object(task.observation_encoder, 'encode', wraps=task.observation_encoder.encode) as encode, \
                patch.object(task.knowledge, 'update', wraps=task.knowledge.update) as memory:
            result = task.step(actions, capture_all_terminals=False)
        assert result.terminal_observations is None
        assert result.terminated.tolist() == [True, True, False, False, False, False]
        assert [(call.args[0].num_envs, call.kwargs.get('privileged', False))
                for call in encode.call_args_list] == [(6, False), (6, True)]
        assert [call.args[0].num_envs for call in memory.call_args_list] == [6, 2]
