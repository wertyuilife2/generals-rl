"""Early shared-weight and post-backbone privileged attention contracts."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from unittest.mock import patch

import pytest
import torch

from generals_rl.export import export_policy
from generals_rl.policy.actor import build_actor
from generals_rl.task import GeneralsTask
from generals_rl.config import task_config
from generals_rl.train import train
from test_shared_critic import configuration, model_for
from test_vit_padding import observations, policy


MODES = ('early_shared', 'post_attention')


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('patch_size', (1, 3))
@pytest.mark.parametrize('privileged', (False, True))
def test_actor_capacity_private_routing_and_shared_gradient(mode, patch_size, privileged):
    torch.set_num_threads(1)
    obs = observations()
    torch.manual_seed(91)
    original = policy(obs, patch_size=patch_size)
    torch.manual_seed(91)
    if not privileged:
        obs = obs.exclude('critic')
    options = configuration(True, privileged)['runner']['obs_groups']
    # Build with the matching observation contract when privilege is disabled.
    from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy
    model_options = deepcopy(configuration(True, privileged)['runner']['policy'])
    model_options.pop('class_name')
    model_options.update(critic_fusion=mode, patch_size=patch_size)
    model = ScaledInformationSpatialPPOPolicy(obs, options, 4233, **model_options)
    for key, value in original.actor.state_dict().items():
        torch.testing.assert_close(model.actor.state_dict()[key], value, rtol=0, atol=0)
    parameters = list(model.named_parameters(remove_duplicate=False))
    assert len(parameters) == len({id(p) for _, p in parameters})
    assert model.parameter_counts()['total'] == sum(p.numel() for p in model.parameters())
    assert model.actor.encoder.backbone.depth == 4

    calls = []
    hook = model.actor.encoder.backbone.transformer.blocks[0].register_forward_hook(
        lambda *_: calls.append('shared'))
    probabilities, values = model.action_probabilities_and_value(obs)
    hook.remove()
    assert len(calls) == (2 if mode == 'early_shared' else 1)
    torch.testing.assert_close(probabilities, original.action_probabilities(obs), rtol=0, atol=0)
    torch.testing.assert_close(probabilities, model.action_probabilities(obs), rtol=0, atol=0)
    torch.testing.assert_close(values, model.evaluate(obs), rtol=0, atol=0)

    changed = obs.clone()
    changed['critic'] = torch.randn(len(obs), 11 * 529 + 13)
    changed_probabilities, changed_values = model.action_probabilities_and_value(changed)
    torch.testing.assert_close(changed_probabilities, probabilities, rtol=0, atol=0)
    assert torch.equal(changed_values, values) != privileged
    torch.testing.assert_close(model.action_probabilities(obs.exclude('critic', 'single_obs')),
                               probabilities, rtol=0, atol=0)

    model.zero_grad(set_to_none=True)
    model.action_probabilities(obs)[:, 0].log().mean().backward()
    assert all(p.grad is None for p in model.critic.parameters())
    model.zero_grad(set_to_none=True)
    model.evaluate(obs).square().mean().backward()
    assert all(p.grad is None for p in model.actor.cell_head.parameters())
    assert all(p.grad is None for p in model.actor.encoder.projection.parameters())
    assert model.actor.encoder.backbone.transformer.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    if privileged:
        assert model.critic.privileged_embedding.weight.grad.abs().sum() > 0
    if mode == 'post_attention':
        assert len(model.critic.blocks) == 2
        assert model.critic.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('patch_size', (1, 3))
def test_fused_critic_excludes_padding_and_matches_unmasked_canvas(mode, patch_size):
    torch.set_num_threads(1)
    obs = observations()
    model = policy(obs, critic_fusion=mode, patch_size=patch_size)
    altered = obs.clone()
    valid = obs['valid_cells']
    altered['policy'].reshape(5, 4, 13 * 529 + 13)[..., :13 * 529].reshape(
        5, 4, 13, 529).masked_fill_(~valid[:, None, None], 10000.)
    altered['knowledge'].reshape(5, 11, 529).masked_fill_(~valid[:, None], -10000.)
    altered['critic'][:, :11 * 529].reshape(5, 11, 529).masked_fill_(~valid[:, None], 10000.)
    with torch.no_grad():
        torch.testing.assert_close(model.evaluate(altered), model.evaluate(obs), rtol=0, atol=0)
        expected = model.evaluate(obs)
        model.inference_chunk_size = 2
        torch.testing.assert_close(model.evaluate(obs), expected, rtol=2e-5, atol=2e-6)
    # With every cell valid, masked and ordinary dense execution agree.
    obs['valid_cells'].fill_(True)
    unmasked = policy(obs, critic_fusion=mode, patch_size=patch_size, mask_padding_tokens=False)
    unmasked.load_state_dict(model.state_dict())
    torch.testing.assert_close(unmasked.evaluate(obs), model.evaluate(obs), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('historical', (False, True))
def test_cpu_ppo_checkpoint_resume_and_actor_export(tmp_path, mode, historical):
    torch.set_num_threads(1)
    config = configuration(True, True)
    config['runner']['policy']['critic_fusion'] = mode
    if historical:
        config['task'].update(opponent='self_play', train_both_sides=False)
        config['self_play'].update(pool_size=2, snapshot_interval=1, retain_seed=False)
    with redirect_stdout(io.StringIO()):
        summary = train(config, tmp_path/'run', iterations=1)
        checkpoint = tmp_path/'run/checkpoint.pt'
        train(config, tmp_path/'resumed', iterations=1, resume=checkpoint)
    assert summary['iterations'] == 1
    saved = torch.load(checkpoint, weights_only=True)
    states = saved['optimizer_state_dict']['state'].values()
    assert {int(state['step']) for state in states} == {1}
    assert all(torch.isfinite(value).all() for value in saved['model_state_dict'].values())
    bundle = export_policy(checkpoint, tmp_path/'bundle')
    spec = json.loads((bundle/'model_config.json').read_text())
    assert 'critic_fusion' not in spec['network']
    assert spec['network']['vit_config'] == [128, 4, 4]
    actor = build_actor(spec['network'], spec['features'])
    actor.load_state_dict(torch.load(bundle/'actor.pt', weights_only=True)['actor_state_dict'])
    with GeneralsTask(task_config(config['task'])) as task, torch.no_grad():
        model, obs = model_for(task, config)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        torch.testing.assert_close(actor(obs['policy'], obs['knowledge'], obs['valid_cells']),
                                   model._forward(obs), rtol=0, atol=0)


@pytest.mark.parametrize('options', (
    {'critic_fusion': 'unknown'},
    {'critic_fusion': 'early_shared', 'share_backbone': False},
    {'critic_fusion': 'post_attention', 'spatial_mode': 'pyramid'},
    {'critic_fusion': 'post_attention', 'backbone': 'resnet'},
    {'critic_fusion': 'post_attention', 'critic_attention_layers': 0},
    {'critic_fusion': 'post_attention', 'critic_attention_layers': True},
    {'critic_fusion': 'post_attention', 'critic_attention_layers': 5},
))
def test_invalid_fusion_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        policy(observations(), **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('patch_size', (1, 3))
def test_cuda_packed_fusion_matches_dense_outputs_and_gradients(mode, patch_size):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    obs = observations('cuda')
    model = policy(obs, critic_fusion=mode, patch_size=patch_size, amp_bfloat16=True)
    outputs, gradients = [], []
    for reference in (True, False):
        model.zero_grad(set_to_none=True)
        with ExitStack() as stack:
            if reference:
                stack.enter_context(patch('generals_rl.policy.tokens.pack_tokens', lambda x, _: x))
                stack.enter_context(patch('generals_rl.scaled_ppo.pack_tokens', lambda x, _: x))
            logits, value = model._forward(obs, joint=True)
            logits = logits[obs['action_mask']]
            (value.square().mean() + logits.square().mean()).backward()
        outputs.append((logits.detach(), value.detach()))
        gradients.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=.03, atol=.003)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[1][name], gradients[0][name], rtol=.06, atol=2e-4, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('patch_size', (1, 3))
def test_cuda_compiled_fusion_forward_backward(mode, patch_size):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    obs = observations('cuda')
    model = policy(obs, critic_fusion=mode, patch_size=patch_size, amp_bfloat16=True)
    expected = tuple(x.detach() for x in model._forward(obs, joint=True))
    model.compile_forward = True
    actual = model._forward(obs, joint=True)
    # BF16 fused matmuls can differ by a few 1e-3 near zero logits.
    # Also bound the actual categorical update, which PPO depends on.
    torch.testing.assert_close(actual[0], expected[0], rtol=.03, atol=.005)
    torch.testing.assert_close(actual[1], expected[1], rtol=.03, atol=.003)
    with torch.no_grad():
        p = model._probabilities_from_logits(expected[0], obs['action_mask'])
        q = model._probabilities_from_logits(actual[0], obs['action_mask'])
        kl = torch.where(p > 0, p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log()), 0).sum(-1)
        assert kl.abs().max() < 1e-4
    (actual[0][obs['action_mask']].square().mean() + actual[1].square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
