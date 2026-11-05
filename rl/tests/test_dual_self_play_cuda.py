"""CPU/CUDA agreement for paired trajectories on padded official maps."""
from contextlib import ExitStack

import numpy as np
import pytest
import torch

from generals import MapConfig
from generals.core.mapgen import generate_map
from generals.vector import VecEnv
from generals_rl.cts import GeneralsCTSVecEnv
from generals_rl.map_sampling import pad_state
from generals_rl.mdp.rewards import RewardConfig
from generals_rl.task import GeneralsTask, TaskConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_paired_gpu_steps_match_cpu_with_history_memory_and_timeouts():
    torch.set_num_threads(1)
    states, valid = zip(*(pad_state(generate_map(MapConfig(w, h, 2), 98 + i), 23, 23)
                           for i, (w, h) in enumerate(((18, 20), (23, 18), (20, 23)))))
    with ExitStack() as stack:
        tasks = []
        for backend, device in (('cpu', 'cpu'), ('gpu', 'cuda:0')):
            config = TaskConfig(num_envs=3, backend=backend, device=device,
                max_episode_steps=3, opponent='current_self_play', train_both_sides=True,
                random_maps=False, map_pool_size=0, combat_features=True,
                king_threat_distance=True, persistent_knowledge=True,
                reward=RewardConfig(potential_gamma=.99))
            env = VecEnv.from_states(states, backend=backend, device=device,
                                      auto_reset=False, episode_ticks=3)
            task = stack.enter_context(GeneralsTask(config, backend=env))
            task._valid = torch.as_tensor(np.stack(valid), device=device)
            task.reset()
            tasks.append(task)
        envs = [GeneralsCTSVecEnv(task) for task in tasks]
        observations = [env.get_observations() for env in envs]
        for _ in range(9):
            for key in observations[0].keys():
                torch.testing.assert_close(observations[0][key], observations[1][key].cpu(),
                                           rtol=2e-6, atol=2e-7, msg=key)
            mask = observations[0]['action_mask']
            actions = (mask.long() * torch.arange(mask.shape[1])).argmax(1)
            old = observations[1].clone()
            reference = envs[0].step(actions)
            actual = envs[1].step(actions.cuda())
            # Pre-step immutable views still describe the sampled actions.
            for key in old.keys():
                torch.testing.assert_close(old[key], observations[1][key], rtol=0, atol=0)
            for index in (1, 2):
                torch.testing.assert_close(reference[index], actual[index].cpu(), rtol=1e-6, atol=1e-7)
            for key in reference[3]['terminal_observations'].keys():
                torch.testing.assert_close(reference[3]['terminal_observations'][key],
                    actual[3]['terminal_observations'][key].cpu(), rtol=2e-6, atol=2e-7, msg=key)
            torch.testing.assert_close(reference[3]['time_outs'], actual[3]['time_outs'].cpu())
            observations = [reference[0], actual[0]]
        assert tasks[0].completed_episodes == tasks[1].completed_episodes
        assert tasks[0].truncated_episodes == tasks[1].truncated_episodes
