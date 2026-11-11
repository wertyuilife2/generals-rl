"""Bounded sparse-reward critics preserve scalar GAE and deployment contracts."""
from contextlib import redirect_stdout
import io
import json

import pytest
import torch

from generals_rl.config import configure_observations, task_config
from generals_rl.export import export_policy
from generals_rl.policy.actor import build_actor
from generals_rl.task import GeneralsTask
from generals_rl.train import train
from generals_rl.value_distribution import ValueDistribution
from test_shared_critic import configuration, model_for



def recipe(value='scalar', device='cpu'):
    cfg = configuration(True, True, device)
    cfg['task'].update(width=9, height=9, random_maps=False, train_both_sides=True,
                       opponent='current_self_play', max_episode_steps=3)
    cfg['task']['reward'].update(territory=0., cities=0., army=0., tick=0., potential_gamma=1.)
    cfg['runner']['policy'].update(map_width=9, map_height=9, critic_fusion='early_shared',
                                    patch_size=1, value_distribution=value)
    cfg['runner']['algorithm'].update(gamma=1., use_clipped_value_loss=value=='scalar', ppo_monitor=True,
        ppo_probe_samples=8, ppo_probe_batch_size=4)
    return configure_observations(cfg)



@pytest.mark.parametrize('kind', ('hl_gauss', 'binary_ce'))
def test_bounded_scalar_decode_soft_targets_and_gradient(kind):
    torch.set_num_threads(1)
    distribution = ValueDistribution(kind)
    returns = torch.tensor([[-1.], [-.73], [0.], [.15], [1.]], requires_grad=True)
    target = distribution.targets(returns)
    assert not target.requires_grad
    output = torch.zeros(5, distribution.outputs, requires_grad=True)
    prediction = distribution.decode(output)
    assert prediction.shape == (5, 1)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=1e-7, rtol=0)
    if kind == 'binary_ce':
        torch.testing.assert_close(target * 2 - 1, returns)
        optimum = torch.logit(target.clamp(1e-6, 1-1e-6))
        torch.testing.assert_close(distribution.decode(optimum), returns, atol=3e-6, rtol=0)
    else:
        torch.testing.assert_close(target.sum(-1), torch.ones(5))
        expected = (target * distribution.centers).sum(-1)
        assert -.98 < expected[0] < -.96  # Explicit truncated-Gaussian edge bias.
        assert .96 < expected[-1] < .98
        torch.testing.assert_close(expected[1:-1], returns.detach()[1:-1, 0], atol=1e-5, rtol=0)
        torch.testing.assert_close(target[0], target[-1].flip(0))
    distribution.loss(output, returns).backward()
    assert returns.grad is None and torch.isfinite(output.grad).all() and output.grad.norm() > 0



def assert_train_resume_export(tmp_path, cfg):
    torch.set_num_threads(1)
    value = cfg['runner']['policy']['value_distribution']
    with redirect_stdout(io.StringIO()):
        train(cfg, tmp_path/'run', iterations=2)
    checkpoint = tmp_path/'run/checkpoint.pt'
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with redirect_stdout(io.StringIO()):
        train(cfg, tmp_path/'resume', iterations=1, resume=checkpoint)
    bundle = export_policy(checkpoint, tmp_path/'bundle')
    spec = json.loads((bundle/'model_config.json').read_text())
    actor = build_actor(spec['network'], spec['features'])
    actor.load_state_dict(torch.load(bundle/'actor.pt', weights_only=True)['actor_state_dict'])
    with GeneralsTask(task_config(cfg['task'])) as task, torch.no_grad():
        model, obs = model_for(task, cfg)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        values = model.evaluate(obs)
        assert values.shape == (task.num_envs, 1) and torch.isfinite(values).all()
        if value != 'scalar':
            assert values.abs().max() <= 1
        torch.testing.assert_close(actor(obs['policy'], obs['knowledge'], obs['valid_cells']),
                                   model._forward(obs), atol=0, rtol=0)



@pytest.mark.parametrize('value', ('scalar', 'hl_gauss', 'binary_ce'))
def test_real_train_resume_export_and_scalar_rollout(tmp_path, value):
    assert_train_resume_export(tmp_path, recipe(value))


@pytest.mark.parametrize('kind', ('hl_gauss', 'binary_ce'))
def test_classification_gae_and_timeout_use_scalar_units(kind):
    from generals_rl.student_ppo import StudentPPO
    from rsl_rl.storage import RolloutStorageCTS
    cfg=recipe(value=kind)
    with GeneralsTask(task_config(cfg['task'])) as task:
        model,obs=model_for(task,cfg)
        storage=RolloutStorageCTS('rl',task.num_envs,0,2,obs,[1],action_dtype=torch.long,
            store_action_std=False,store_action_mean=False)
        algorithm=StudentPPO(model,storage,task.num_envs,num_learning_epochs=1,num_mini_batches=1,
            gamma=1.,lam=.95,physical_minibatch_size=4,use_clipped_value_loss=False,ppo_monitor=False)
        assert storage.values.shape == (2,task.num_envs,1)
        with torch.no_grad():
            expected=model.evaluate(obs).clone()
            algorithm.act(obs)
            done=torch.ones(task.num_envs,dtype=torch.bool)
            timeout=torch.tensor([True,False,True,False])
            rewards=torch.tensor([0.,1.,0.,-1.])
            algorithm.process_env_step(obs,rewards,done,{'time_outs':timeout,'terminal_observations':obs})
            torch.testing.assert_close(storage.rewards[0,:,0], rewards+expected[:,0]*timeout)
            algorithm.act(obs)
            algorithm.process_env_step(obs,torch.zeros_like(rewards),~done,{'time_outs':~done,'terminal_observations':obs})
            algorithm.compute_returns(obs)
        torch.testing.assert_close(storage.returns[0,:,0],rewards+expected[:,0]*timeout)
        assert storage.returns.abs().max()<=1.00001
        loss=algorithm.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
