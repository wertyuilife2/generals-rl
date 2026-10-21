"""Controlled information-input ablations on the unchanged v17 light CNN.

These adapters share the categorical PPO lifecycle, not actor/critic weights.
New input weights start at zero: the randomly initialized baseline function
and its RNG stream are retained, and the new features can learn immediately.
No old checkpoint or optimizer is loaded.
"""
from copy import deepcopy

import torch
from torch import nn

from .spatial_ppo import (FRAME_DIM, HISTORY, MAP_PLANES, MAP_SIZE, PUBLIC_DIM,
                          LightSpatialPPOPolicy, OwnKingRelativeGeometry,
                          _validate_history_length)


def _extend_conv(old, extra_channels, trailing_channels=0):
    """Preserve old inputs and append zero-initialized information before XY."""
    devices = [old.weight.device.index] if old.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        new = nn.Conv2d(old.in_channels + extra_channels, old.out_channels,
                        old.kernel_size, stride=old.stride, padding=old.padding,
                        dilation=old.dilation, groups=old.groups,
                        bias=old.bias is not None, padding_mode=old.padding_mode,
                        device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        prefix = old.in_channels - trailing_channels
        new.weight[:, :prefix].copy_(old.weight[:, :prefix])
        if trailing_channels:
            new.weight[:, -trailing_channels:].copy_(old.weight[:, -trailing_channels:])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class HistoryStateCritic(nn.Module):
    """Fuse privileged maps, local history and optional player memory per cell."""
    def __init__(self, baseline, knowledge_channels, *, history_length=HISTORY):
        super().__init__()
        _validate_history_length(history_length)
        self.history_length = history_length
        self.knowledge_channels = knowledge_channels
        self.trunk = baseline.trunk
        self.trunk[0] = _extend_conv(self.trunk[0], history_length * MAP_PLANES + knowledge_channels)
        old = baseline.context[0]
        devices = [old.weight.device.index] if old.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            new = nn.Linear(old.in_features + PUBLIC_DIM, old.out_features,
                            bias=old.bias is not None, device=old.weight.device,
                            dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.context = nn.Sequential(new, baseline.context[1])
        self.value_head = baseline.value_head

    def forward(self, privileged, history, knowledge=None):
        if history.ndim != 2 or history.shape[-1] != self.history_length * FRAME_DIM:
            raise ValueError("critic history shape does not match history_length")
        if privileged.shape != (history.shape[0], FRAME_DIM):
            raise ValueError("critic privileged observation shape does not match history batch")
        frames = history.reshape(-1, self.history_length, FRAME_DIM)
        maps = privileged[:, :-PUBLIC_DIM].reshape(-1, MAP_PLANES, MAP_SIZE, MAP_SIZE)
        local = frames[..., :-PUBLIC_DIM].reshape(-1, self.history_length * MAP_PLANES, MAP_SIZE, MAP_SIZE)
        inputs = [maps, local]
        if self.knowledge_channels:
            if knowledge is None or knowledge.shape != (history.shape[0], self.knowledge_channels * MAP_SIZE * MAP_SIZE):
                raise ValueError("critic requires the actor's configured knowledge planes")
            inputs.append(knowledge.reshape(-1, self.knowledge_channels, MAP_SIZE, MAP_SIZE))
        pooled = self.trunk(torch.cat(inputs, 1)).mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, privileged[:, -PUBLIC_DIM:],
                                          frames[:, -1, -PUBLIC_DIM:]), -1))
        return self.value_head(torch.cat((pooled, context), -1))


class InformationSpatialPPOPolicy(LightSpatialPPOPolicy):
    """The memory × history-state critic 2×2, without changing the PPO core."""
    def __init__(self, obs, obs_groups, num_actions, *, critic_actor_information=False,
                 knowledge_channels=0, **kwargs):
        if not isinstance(critic_actor_information, bool):
            raise TypeError("critic_actor_information must be boolean")
        if isinstance(knowledge_channels, bool) or not isinstance(knowledge_channels, int) or knowledge_channels < 0:
            raise ValueError("knowledge_channels must be a nonnegative integer")
        actor_keys = ["policy"] + (["knowledge"] if knowledge_channels else [])
        critic_keys = ["critic"] + (actor_keys if critic_actor_information else [])
        expected = {"policy": actor_keys, "critic": critic_keys}
        if obs_groups != expected:
            raise ValueError(f"information policy requires obs_groups={expected}")
        if knowledge_channels and ("knowledge" not in obs or obs["knowledge"].shape != (
                obs.batch_size[0], knowledge_channels * MAP_SIZE * MAP_SIZE)):
            raise ValueError("knowledge observation shape does not match knowledge_channels")
        # Complete the v17 construction first, including own-king geometry.
        # Critic-only changes cannot perturb the actor's initial weights/RNG.
        super().__init__(obs, {"policy": ["policy"], "critic": ["critic"]}, num_actions, **kwargs)
        self.obs_groups = deepcopy(obs_groups)
        self.critic_actor_information = critic_actor_information
        self.actor.knowledge_channels = knowledge_channels
        if knowledge_channels:
            geometry = isinstance(self.actor.trunk[0], OwnKingRelativeGeometry)
            index = int(geometry)
            self.actor.trunk[index] = _extend_conv(self.actor.trunk[index], knowledge_channels,
                                                  trailing_channels=2 if geometry else 0)
            if geometry:
                self.actor.trunk[0].knowledge_channels = knowledge_channels
            self.num_actor_obs += knowledge_channels * MAP_SIZE * MAP_SIZE
        if critic_actor_information:
            self.critic = HistoryStateCritic(self.critic, knowledge_channels, history_length=self.history_length)
            self.num_critic_obs += self.history_length * FRAME_DIM + knowledge_channels * MAP_SIZE * MAP_SIZE
        self.semantics += f"_knowledge{knowledge_channels}_critic_actor_info{int(critic_actor_information)}_v1"

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if not self.critic_actor_information or obs.batch_size[0] == 0:
            return super().evaluate(obs, is_teacher=is_teacher, **kwargs)
        if is_teacher:
            raise ValueError("information student policy has no nonempty teacher value branch")
        return self.critic(obs["critic"], obs["policy"], obs.get("knowledge"))
