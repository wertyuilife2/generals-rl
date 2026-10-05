"""Fold redundant P2 seat input into P0+bias; initialize distance weight zero."""
from collections import OrderedDict
from copy import deepcopy

import torch

from .combat_migration import FRAME_DIM, _dense_input_specs


def fold_king_threat_input_weights(state_dict, policy_config, obs_groups, *,
                                  history_length=4, players=2, width=9, height=9):
    specs, latent_dim = _dense_input_specs(
        state_dict, policy_config, obs_groups, history_length=history_length,
        players=players, width=width, height=height)
    result = OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())
    if hasattr(state_dict, "_metadata"):
        result._metadata = deepcopy(state_dict._metadata)
    changed = []
    for key, (frames, prefix) in specs.items():
        weight = result[key]
        bias_key = key.removesuffix("weight") + "bias"
        bias = result.get(bias_key)
        if (bias is None or bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype
                or not bias.is_floating_point() or bias.layout != torch.strided):
            raise ValueError(f"unsupported bias layout for {key}")
        for frame in range(frames):
            seat0 = prefix + frame * FRAME_DIM + FRAME_DIM - 2
            seat1 = seat0 + 1
            old_seat1 = weight[:, seat1].clone()
            weight[:, seat0] -= old_seat1
            bias += old_seat1
            weight[:, seat1] = 0
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError(f"seat folding overflowed {key}")
        changed.append({"weight": key, "bias": bias_key, "frames": frames, "latent_prefix": prefix})
    return result, {"schema_version": 1, "frame_dim": FRAME_DIM,
        "history_length": history_length, "latent_dim": latent_dim,
        "observation_semantics": "king_threat_distance_v1",
        "initialization": "warm_start_weights", "resume_supported": False,
        "optimizer": "fresh", "historical_pool": "fresh_reseed", "changed_layers": changed}
