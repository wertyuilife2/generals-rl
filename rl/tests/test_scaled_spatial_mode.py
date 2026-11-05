"""Spatial pyramid retains complete capacity and independent information branches."""
from copy import deepcopy

import pytest
import torch

from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy
from test_scaled_ppo import GROUPS, observations


CASES = [(family, size) for family in ('resnet', 'vit') for size in ('1m', '10m', '100m')]


@pytest.mark.parametrize('family,size', CASES)
def test_pyramid_preserves_every_dense_parameter_and_full_local_resolution(family, size):
    with torch.device('meta'):
        obs = observations(1).to('meta')
        pyramid = ScaledInformationSpatialPPOPolicy(obs, GROUPS, 649, backbone=family, model_size=size,
                                                    share_backbone=False, mask_padding_tokens=False, spatial_mode='pyramid', map_width=9, map_height=9)
        dense = ScaledInformationSpatialPPOPolicy(obs, GROUPS, 649, backbone=family, model_size=size,
                                                  share_backbone=False, mask_padding_tokens=False, map_width=9, map_height=9)
        assert pyramid.parameter_counts()['total'] > dense.parameter_counts()['total']
        dense_params, pyramid_params = dict(dense.named_parameters()), dict(pyramid.named_parameters())
        assert all(pyramid_params[key].shape == value.shape for key, value in dense_params.items())
        assert set(pyramid_params) - set(dense_params) == {
            f'{branch}.encoder.local_skip.{suffix}' for branch in ('actor', 'critic') for suffix in ('weight', 'bias')}
        for branch in (pyramid.actor.encoder, pyramid.critic.encoder):
            local, deep = branch.backbone.forward_pyramid(torch.zeros(1, branch.backbone.in_channels, 9, 9))
            assert local.shape == (1, branch.backbone.local_channels, 9, 9)
            assert deep.shape[-2:] == ((2, 2) if family == 'resnet' else (3, 3))
        assert pyramid.actor(obs['policy'], obs['knowledge']).shape == (1, 649)
        assert pyramid.evaluate(obs).shape == (1, 1)
        assert pyramid.semantics.endswith('_spatial_pyramid_v1')
        assert dense.spatial_mode == 'dense' and 'pyramid' not in dense.semantics


@pytest.mark.parametrize('family', ('resnet', 'vit'))
def test_pyramid_decoder_and_critic_keep_local_information_and_fp32_mask_boundary(family):
    torch.set_num_threads(1)
    obs = observations()
    policy = ScaledInformationSpatialPPOPolicy(obs, GROUPS, 649, backbone=family,
                    share_backbone=False, mask_padding_tokens=False, spatial_mode='pyramid', inference_chunk_size=1, map_width=9, map_height=9)
    coarse, local = [], []
    hooks = [policy.actor.encoder.projection.register_forward_hook(lambda module, args, output: coarse.append(tuple(output.shape))),
             policy.actor.encoder.local_skip.register_forward_hook(lambda module, args, output: local.append(tuple(output.shape)))]
    with torch.no_grad():
        expected = policy.action_probabilities(obs)
        changed = obs.clone()
        changed['critic'].fill_(99)
        changed['single_obs'].fill_(float('nan'))
        torch.testing.assert_close(policy.action_probabilities(changed), expected, rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    assert all(shape[1] == 128 and shape[-1] == (2 if family == 'resnet' else 3) for shape in coarse)
    assert all(shape[1:] == (128, 9, 9) for shape in local)
    assert not ({id(p) for p in policy.actor.parameters()} & {id(p) for p in policy.critic.parameters()})
    for key in ('policy', 'knowledge', 'critic'):
        obs[key].requires_grad_()
    obs['action_mask'][0].zero_(); obs['action_mask'][0, 0] = True
    with torch.autocast('cpu', dtype=torch.bfloat16):
        probabilities, values = policy.action_probabilities(obs), policy.evaluate(obs)
        loss = values.square().mean() - probabilities[:, 0].log().mean()
    assert probabilities.dtype == values.dtype == torch.float32
    assert probabilities[0, 0] == 1 and probabilities[~obs['action_mask']].count_nonzero() == 0
    loss.backward()
    assert (obs['policy'].grad.reshape(2, 4, 1066).abs().sum((0, 2)) > 0).all()
    assert (obs['knowledge'].grad.reshape(2, 11, 81).abs().sum((0, 2)) > 0).all()
    assert obs['critic'].grad.abs().sum() > 0
    assert all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32 and
                                           torch.isfinite(p.grad).all()) for p in policy.parameters())
    restored = deepcopy(policy)
    assert restored.spatial_mode == 'pyramid' and restored.distribution is None


@pytest.mark.parametrize('family', ('resnet', 'vit'))
@pytest.mark.parametrize('mode', (None, False, 'unknown'))
def test_invalid_spatial_mode_is_rejected(family, mode):
    with pytest.raises(ValueError, match='spatial_mode'):
        ScaledInformationSpatialPPOPolicy(observations(), GROUPS, 649, backbone=family, spatial_mode=mode, map_width=9, map_height=9)
