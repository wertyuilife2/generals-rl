"""Paired PPO, finite time limits and a single zero-sum game critic."""
from unittest.mock import patch

import pytest
import torch

from generals_rl.config import configure_observations, task_config
from generals_rl.cts import GeneralsCTSVecEnv
from generals_rl.task import GeneralsTask
from generals_rl.student_ppo import StudentPPO
from rsl_rl.storage import RolloutStorageCTS
from test_shared_critic import model_for
from test_value_distribution import recipe, assert_train_resume_export
from test_student_ppo import algorithm, observation


def configuration(joint=False, mode='bootstrap', device='cpu'):
    cfg = recipe('hl_gauss', device)
    cfg['task'].update(feature_set='temporal_paths', time_limit_mode=mode)
    cfg['runner']['algorithm']['pair_minibatches'] = True
    cfg['runner']['policy']['joint_critic'] = joint
    return configure_observations(cfg)


def test_paired_minibatches_visit_every_sample_once_without_breaking_games():
    alg = algorithm(observation(8), pair_minibatches=True, physical_minibatch_size=4)
    steps = alg.storage.num_transitions_per_env
    (_, indices), _ = alg._advantage_sample_indices()
    assert sorted(indices.tolist()) == list(range(8 * steps))
    for batch in indices.chunk(alg.num_mini_batches):
        envs, ticks = batch // steps, batch % steps
        assert torch.equal(envs[::2] + 1, envs[1::2])
        assert (envs[::2] % 2 == 0).all()
        assert torch.equal(ticks[::2], ticks[1::2])


@pytest.mark.parametrize('joint', (False, True))
@pytest.mark.parametrize('mode', ('bootstrap', 'terminal_draw'))
def test_timeout_gae_keeps_end_causes_but_changes_only_bootstrap(joint, mode):
    torch.set_num_threads(1)
    cfg = configuration(joint, mode)
    cfg['task']['max_episode_steps'] = 1
    with GeneralsTask(task_config(cfg['task'])) as task:
        env = GeneralsCTSVecEnv(task)
        model, obs = model_for(task, cfg)
        storage = RolloutStorageCTS('rl', task.num_envs, 0, 1, obs, [1], action_dtype=torch.long,
                                    store_action_mean=False, store_action_std=False)
        alg = StudentPPO(model, storage, task.num_envs, num_learning_epochs=1, num_mini_batches=1,
            gamma=1., lam=.95, physical_minibatch_size=4, pair_minibatches=True,
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


@pytest.mark.parametrize('device', ('cpu', 'cuda'))
def test_joint_critic_reads_both_histories_without_actor_leak_and_uses_one_game_pass(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.set_num_threads(1)
    cfg = configuration(device=device)
    with GeneralsTask(task_config(cfg['task'])) as task, torch.no_grad():
        torch.manual_seed(17)
        baseline, obs = model_for(task, cfg)
        rng = torch.get_rng_state()
        cfg['runner']['policy']['joint_critic'] = True
        torch.manual_seed(17)
        model, _ = model_for(task, cfg)
        assert torch.equal(torch.get_rng_state(), rng)
        for name, weight in baseline.actor.state_dict().items():
            torch.testing.assert_close(model.actor.state_dict()[name], weight, rtol=0, atol=0)
        counts = []
        from generals_rl.policy.tokens import token_blocks
        def counted(blocks, inputs, *args, **kwargs):
            counts.append(inputs.numel() // (81 * 128))
            return token_blocks(blocks, inputs, *args, **kwargs)
        with patch('generals_rl.policy.tokens.token_blocks', counted), \
                patch('generals_rl.scaled_ppo.token_blocks', counted):
            probabilities, values = model.action_probabilities_and_value(obs)
        assert counts == [task.num_envs, task.num_games]
        torch.testing.assert_close(values[::2], -values[1::2], rtol=0, atol=0)
        raw = model.evaluate(obs, raw_value=True)
        torch.testing.assert_close(raw[::2], raw[1::2].flip(-1), rtol=0, atol=0)
        other = obs.clone()
        other['knowledge'][1::2] += .5
        changed, new_values = model.action_probabilities_and_value(other)
        torch.testing.assert_close(probabilities[::2], changed[::2], rtol=0, atol=0)
        assert not torch.equal(values[::2], new_values[::2])


@pytest.mark.parametrize('mode', ('bootstrap', 'terminal_draw'))
def test_joint_training_resume_and_actor_export(tmp_path, mode):
    assert_train_resume_export(tmp_path, configuration(True, mode))


def test_pairing_configuration_rejects_incomplete_or_filtered_pairs():
    with pytest.raises(ValueError, match='paired'):
        algorithm(observation(7), pair_minibatches=True)
    with pytest.raises(ValueError, match='paired'):
        algorithm(observation(8), pair_minibatches=True, physical_minibatch_size=3)
    cfg = configuration(True)
    cfg['runner']['algorithm']['pair_minibatches'] = False
    with pytest.raises(ValueError, match='joint critic'):
        configure_observations(cfg)


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
