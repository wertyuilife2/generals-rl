"""Finite-horizon draws and bootstrap preserve distinct episode end causes."""

import pytest
import torch

from generals_rl.config import configure_observations, task_config
from generals_rl.cts import GeneralsCTSVecEnv
from generals_rl.task import GeneralsTask
from generals_rl.student_ppo import StudentPPO
from rsl_rl.storage import RolloutStorageCTS
from test_shared_critic import model_for
from test_value_distribution import recipe


def configuration(mode='bootstrap', device='cpu'):
    cfg = recipe('hl_gauss', device)
    cfg['task'].update(feature_set='temporal_paths', time_limit_mode=mode)
    return configure_observations(cfg)


@pytest.mark.parametrize('mode', ('bootstrap', 'terminal_draw'))
def test_timeout_gae_keeps_end_causes_but_changes_only_bootstrap(mode):
    torch.set_num_threads(1)
    cfg = configuration(mode)
    cfg['task']['max_episode_steps'] = 1
    with GeneralsTask(task_config(cfg['task'])) as task:
        env = GeneralsCTSVecEnv(task)
        model, obs = model_for(task, cfg)
        storage = RolloutStorageCTS('rl', task.num_envs, 0, 1, obs, [1], action_dtype=torch.long,
                                    store_action_mean=False, store_action_std=False)
        alg = StudentPPO(model, storage, task.num_envs, num_learning_epochs=1, num_mini_batches=1,
            gamma=1., lam=.95, physical_minibatch_size=4,
            use_clipped_value_loss=False, ppo_monitor=False)
        with torch.no_grad():
            alg.act(obs)
            next_obs, reward, done, extras = env.step(torch.zeros(task.num_envs, dtype=torch.long))
            assert done.all() and task.truncated_episodes == task.num_games
            assert (extras['log']['Episode/truncated'] == 1).all()
            assert not reward.any()
            expected = model.evaluate(extras['terminal_observations']) if mode == 'bootstrap' else torch.zeros_like(storage.values[0])
            assert bool(extras['time_outs'].all()) == (mode == 'bootstrap')
            alg.process_env_step(next_obs, reward, done, extras)
            alg.compute_returns(next_obs)
        torch.testing.assert_close(storage.returns[0], expected, rtol=0, atol=1e-7)
        losses = alg.update()
        assert all(torch.isfinite(torch.tensor(x)) for x in losses.values())


def test_capture_on_time_limit_keeps_terminal_win_and_loss():
    from generals import VecEnv
    from generals.core.state import Action, Direction
    from test_mdp import capture_state
    cfg = configuration(mode='terminal_draw')
    cfg['task'].update(width=3, height=3, num_envs=1, map_pool_size=0, max_episode_steps=1)
    backend = VecEnv.from_states([capture_state()], auto_reset=False, episode_ticks=1)
    with GeneralsTask(task_config(cfg['task']), backend=backend) as task:
        result = task.step(torch.tensor([Action.move(3, Direction.RIGHT).encode(), 0]), capture_all_terminals=False)
        assert result.terminated.all() and not result.truncated.any()
        assert result.reward.tolist() == [1., -1.]
