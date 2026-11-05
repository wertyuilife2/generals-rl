"""Shared computation, private information routing and dual-trajectory PPO."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from generals_rl.config import configure_observations, default_training_config, task_config
from generals_rl.export import export_policy
from generals_rl.policy.actor import build_actor
from generals_rl.policy.runtime import create_controller
from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy
from generals_rl.task import GeneralsTask
from generals_rl.train import train


def configuration(shared, privileged, device='cpu'):
    config = default_training_config()
    config['task'].update(num_envs=2, map_pool_size=4, max_episode_steps=2,
                          backend='gpu' if device == 'cuda' else 'cpu', device=device)
    config['runner'].update(num_steps_per_env=2, save_interval=1)
    config['runner']['policy'].update(share_backbone=shared, critic_privileged=privileged,
                                      patch_size=3, amp_bfloat16=device == 'cuda',
                                      compile_forward=False, inference_chunk_size=16)
    config['runner']['algorithm'].update(num_learning_epochs=1, num_mini_batches=1,
        physical_minibatch_size=4, amp_bfloat16=device == 'cuda', ppo_monitor=False)
    config['evaluation'] = {'enabled': False}
    config['curriculum'] = {}
    return configure_observations(config)


def model_for(task, config):
    options = deepcopy(config['runner']['policy'])
    options.pop('class_name')
    obs = TensorDict(task.get_observations(), [task.num_envs], device=task.device)
    return ScaledInformationSpatialPPOPolicy(obs, config['runner']['obs_groups'],
        task.action_space.size, **options).to(task.device), obs


@pytest.mark.parametrize('patch_size', [1, 3])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('privileged', [False, True])
def test_1m_total_budget_is_preserved_with_a_wider_shared_backbone(patch_size, shared, privileged):
    torch.set_num_threads(1)
    config = configuration(shared, privileged)
    config['runner']['policy']['patch_size'] = patch_size
    with GeneralsTask(task_config(config['task'])) as task:
        model, _ = model_for(task, config)
        backbone = model.actor.encoder.backbone
        assert (backbone.out_channels, backbone.depth, backbone.num_heads) == (
            (128, 4, 4) if shared else (96, 4, 3))
        total = sum(p.numel() for p in model.parameters())
        assert model.parameter_counts()['total'] == total
        assert 900_000 <= total <= 1_300_000


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('privileged', [False, True])
def test_joint_reuses_local_encoder_and_private_information_cannot_change_actor(shared, privileged):
    torch.set_num_threads(1)
    config = configuration(shared, privileged)
    with GeneralsTask(task_config(config['task'])) as task:
        model, obs = model_for(task, config)
        assert ('critic' in obs) == privileged
        assert (model.critic.privileged_encoder is not None) == privileged
        assert (model.critic.encoder is None) == shared
        if not shared:
            assert type(model.critic.encoder) is type(model.actor.encoder)
            for a, b in zip(model.actor.encoder.parameters(), model.critic.encoder.parameters(), strict=True):
                assert a.data_ptr() != b.data_ptr()
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        parameters = list(model.named_parameters(remove_duplicate=False))
        assert len(parameters) == len({id(p) for _, p in parameters})
        calls = []
        hook = model.actor.encoder.register_forward_hook(lambda *_: calls.append('local'))
        probabilities, values = model.action_probabilities_and_value(obs)
        hook.remove()
        assert calls == ['local']
        torch.testing.assert_close(probabilities, model.action_probabilities(obs), rtol=0, atol=0)
        torch.testing.assert_close(values, model.evaluate(obs), rtol=0, atol=0)
        altered = obs.clone()
        if privileged:
            assert obs['critic'].shape == (task.num_envs, 11 * 529 + 13)
            altered['critic'] = torch.randn_like(obs['critic'])
            assert not torch.equal(model.evaluate(altered), values)
        else:
            altered['critic'] = torch.randn(task.num_envs, 11 * 529 + 13)
            torch.testing.assert_close(model.evaluate(altered), values, rtol=0, atol=0)
        changed_probabilities, _ = model.action_probabilities_and_value(altered)
        torch.testing.assert_close(changed_probabilities, probabilities, rtol=0, atol=0)

        model.zero_grad(set_to_none=True)
        model.evaluate(obs).square().mean().backward()
        assert all(p.grad is None for p in model.actor.cell_head.parameters())
        actor_grad = sum(float(p.grad.abs().sum()) for p in model.actor.encoder.parameters() if p.grad is not None)
        assert (actor_grad > 0) == shared
        if privileged:
            assert sum(float(p.grad.abs().sum()) for p in model.critic.privileged_encoder.parameters()
                       if p.grad is not None) > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_disabled_privilege_skips_encoding_and_default_flag_wires_task_and_storage():
    config = configuration(True, False)
    assert 'critic' not in config['runner']['storage_obs_keys']
    assert config['runner']['obs_groups']['critic'] == ['policy', 'knowledge', 'valid_cells']
    with patch('generals_rl.task.encode_privileged', side_effect=AssertionError('private encode called')):
        with GeneralsTask(task_config(config['task'])) as task:
            for _ in range(3):
                task.step(torch.zeros(task.num_envs, dtype=torch.long))
                assert 'critic' not in task.get_observations()


def test_private_maps_are_exactly_the_eleven_informative_current_channels():
    config = configuration(True, True)
    with GeneralsTask(task_config(config['task'])) as task:
        full = task.observation_encoder.encode(task.batch(), task.player_ids, privileged=True)
        compact = task.get_observations()['critic']
        torch.testing.assert_close(compact[:, :11 * 529], full[:, 529:12 * 529], rtol=0, atol=0)
        torch.testing.assert_close(compact[:, -13:], full[:, -13:], rtol=0, atol=0)
        assert not full[:, 12 * 529:13 * 529].any()
        torch.testing.assert_close(full[:, :529].bool(), task.get_observations()['valid_cells'])


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('privileged', [False, True])
def test_dual_ppo_checkpoint_resume_and_export_for_each_configuration(tmp_path, shared, privileged):
    torch.set_num_threads(1)
    config = configuration(shared, privileged)
    with redirect_stdout(io.StringIO()):
        summary = train(config, tmp_path/'run', iterations=1)
        checkpoint = tmp_path/'run/checkpoint.pt'
        train(config, tmp_path/'resumed', iterations=1, resume=checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert summary['iterations'] == 1
    assert any(key.startswith('actor.encoder.') for key in saved['model_state_dict'])
    assert any(key.startswith('critic.privileged_encoder.') for key in saved['model_state_dict']) == privileged
    assert any(key.startswith('critic.encoder.') for key in saved['model_state_dict']) == (not shared)
    bundle = export_policy(checkpoint, tmp_path/'bundle')
    controller = create_controller(weights=bundle/'actor.pt', config_path=bundle/'model_config.json')
    spec = json.loads((bundle/'model_config.json').read_text())
    assert 'critic_privileged' not in spec['network']
    assert 'share_backbone' not in spec['network']
    assert spec['network']['vit_config'] == ([128, 4, 4] if shared else [96, 4, 3])
    with GeneralsTask(task_config(config['task'])) as task, torch.no_grad():
        model, obs = model_for(task, config)
        model.load_state_dict(saved['model_state_dict'])
        expected = model.actor(obs['policy'], obs['knowledge'], obs['valid_cells'])
        actor = build_actor(spec['network'], spec['features'])
        actor.load_state_dict(controller.actor.state_dict())
        torch.testing.assert_close(actor(obs['policy'], obs['knowledge'], obs['valid_cells']), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('privileged', [False, True])
def test_cuda_compiled_joint_forward_backward_matches_eager(shared, privileged):
    torch.set_num_threads(1)
    config = configuration(shared, privileged, 'cuda')
    with GeneralsTask(task_config(config['task'])) as task:
        model, obs = model_for(task, config)
        reference = tuple(x.detach() for x in model._forward(obs, joint=True))
        model.compile_forward = True
        actual = model._forward(obs, joint=True)
        for a, b in zip(actual, reference, strict=True):
            torch.testing.assert_close(a, b, rtol=.03, atol=.003)
        loss = actual[0][obs['action_mask']].square().mean() + actual[1].square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
