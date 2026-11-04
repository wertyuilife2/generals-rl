"""Padding semantics, packed attention gradients and exported actor parity."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from generals_rl.config import default_training_config
from generals_rl.export import make_policy_spec
from generals_rl.policy.actor import build_actor
from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy


def observations(device='cpu'):
    generator = torch.Generator().manual_seed(715)
    shapes = ((18, 19), (23, 23), (20, 22), (19, 21), (22, 18))
    frames = torch.rand(5, 4, 13 * 529 + 13, generator=generator)
    maps = frames[..., :13 * 529].reshape(5, 4, 13, 23, 23)
    maps[:, :, 4:6] = 0
    maps[:, :, 4:6, 3, 4] = 1
    valid = torch.zeros(5, 23, 23, dtype=torch.bool)
    for row, (height, width) in enumerate(shapes):
        valid[row, :height, :width] = True
    mask = torch.cat((torch.ones(5, 1, dtype=torch.bool), valid.flatten(1).repeat_interleave(8, 1)), 1)
    return TensorDict(dict(policy=frames.flatten(1), single_obs=frames[:, -1].clone(),
                           critic=frames[:, -1].clone(), knowledge=torch.rand(5, 11 * 529, generator=generator),
                           action_mask=mask, valid_cells=valid.flatten(1)), [5]).to(device)


def policy(obs, **overrides):
    options = deepcopy(default_training_config()['runner']['policy'])
    options.pop('class_name')
    options.update(amp_bfloat16=False, compile_forward=False, mask_padding_tokens=True)
    options.update(overrides)
    return ScaledInformationSpatialPPOPolicy(obs,
        default_training_config()['runner']['obs_groups'], 4233, **options).to(obs.device)


def dense_reference(transformer, inputs, valid, tokens):
    count, _, height, width = inputs.shape
    x = transformer.norm_pre(transformer.patch_drop(transformer._pos_embed(transformer.patch_embed(inputs))))
    for block in transformer.blocks:
        x = block(x, attn_mask=valid[:, None, None, :])
    return transformer.norm(x).transpose(1, 2).reshape(count, -1, height, width)


def test_padding_cannot_change_actor_valid_actions_and_export_matches():
    torch.set_num_threads(1)
    obs = observations()
    model = policy(obs)
    config = default_training_config()
    config['runner']['policy']['mask_padding_tokens'] = True
    spec = make_policy_spec(config)
    actor = build_actor(spec['network'], spec['features'])
    actor.load_state_dict(model.actor.state_dict())
    valid = obs['valid_cells']
    altered = obs.clone()
    altered['knowledge'].reshape(5, 11, 529).masked_fill_(~valid[:, None], 12345.)
    altered['policy'].reshape(5, 4, 13 * 529 + 13)[..., :13 * 529].reshape(5, 4, 13, 529).masked_fill_(~valid[:, None, None], -12345.)
    altered['critic'][:, :13 * 529].reshape(5, 13, 529).masked_fill_(~valid[:, None], 12345.)
    with torch.no_grad():
        expected = model.actor(obs['policy'], obs['knowledge'], valid)
        actual = model.actor(altered['policy'], altered['knowledge'], valid)
        torch.testing.assert_close(actual[obs['action_mask']], expected[obs['action_mask']], rtol=0, atol=0)
        torch.testing.assert_close(actor(obs['policy'], obs['knowledge'], valid), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.evaluate(altered), model.evaluate(obs), rtol=0, atol=0)
    assert spec['network']['mask_padding_tokens'] is True
    with pytest.raises(ValueError, match='dense ViT'):
        policy(obs, backbone='resnet')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_packed_actor_and_critic_match_masked_dense_outputs_and_gradients():
    torch.set_num_threads(1)
    obs = observations('cuda')
    model = policy(obs, amp_bfloat16=True)
    for critic, module_name in ((False, 'generals_rl.policy.scaled'), (True, 'generals_rl.scaled_ppo')):
        network = model.critic if critic else model.actor
        outputs, gradients = [], []
        for reference in (True, False):
            network.zero_grad(set_to_none=True)
            with patch(module_name+'.valid_token_features', dense_reference) if reference else torch.enable_grad():
                output = model._forward(obs, critic=critic)
                selected = output if critic else output[obs['action_mask']]
                selected.square().mean().backward()
            outputs.append(selected.detach())
            gradients.append({name: p.grad.detach().clone() for name, p in network.named_parameters() if p.grad is not None})
        torch.testing.assert_close(outputs[1], outputs[0], rtol=.02, atol=2e-3)
        assert gradients[0].keys() == gradients[1].keys()
        for name in gradients[0]:
            assert torch.isfinite(gradients[1][name]).all(), name
            torch.testing.assert_close(gradients[1][name], gradients[0][name], rtol=.05, atol=1e-4, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_compiled_packed_policy_training_and_frozen_batch_padding():
    torch.set_num_threads(1)
    obs = observations('cuda')
    model = policy(obs, amp_bfloat16=True)
    for critic in (False, True):
        model.compile_forward = False
        reference = model._forward(obs, critic=critic).detach()
        model.compile_forward = True
        actual = model._forward(obs, critic=critic)
        torch.testing.assert_close(actual, reference, rtol=.03, atol=3e-3)
        actual.square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        model.compile_forward = False
        reference = model.action_probabilities(obs)
        model.compile_forward = True
        actual = model.action_probabilities(obs)
        torch.testing.assert_close(actual, reference, rtol=.04, atol=1e-4)
        # Vary valid-token counts inside the same frozen batch bucket. Dummy
        # rows introduced by the bucket have zero-length Flash sequences.
        for height, width in ((18, 18), (18, 19), (19, 18), (19, 19), (20, 18), (18, 20),
                              (20, 20), (21, 18), (22, 18), (23, 18), (23, 19), (23, 20)):
            changed = obs.clone()
            valid = changed['valid_cells'].reshape(5, 23, 23)
            valid[0].zero_()
            valid[0, :height, :width] = True
            model.compile_forward = False
            reference = model.action_probabilities(changed)
            model.compile_forward = True
            actual = model.action_probabilities(changed)
            torch.testing.assert_close(actual, reference, rtol=.04, atol=1e-4)
    assert torch.isfinite(actual).all()
