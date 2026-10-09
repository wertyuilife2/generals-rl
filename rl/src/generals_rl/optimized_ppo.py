"""Packed/aligned v32 execution and an optional independent critic projection.

With ``critic_projection_channels=0``, parameter names and shapes are identical
to InformationSpatialPPOPolicy. Alignment adds constant zero channels only at
execution time. The projected critic is an architectural ablation, while the
actor and the categorical PPO lifecycle retain their original capacity.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .information_ppo import InformationSpatialPPOPolicy
from .spatial_ppo import FRAME_DIM, MAP_PLANES, MAP_SIZE, NUM_ACTIONS, PUBLIC_DIM, OwnKingRelativeGeometry


class AlignedInputConv2d(nn.Conv2d):
    """Keep logical/checkpoint weights; convolve prepacked, zero-aligned inputs."""

    def __init__(self, *args, input_channel_alignment=4, **kwargs):
        super().__init__(*args, **kwargs)
        if self.groups != 1:
            raise ValueError("input alignment currently requires groups=1")
        self.input_channel_alignment = input_channel_alignment
        self.input_padding = (-self.in_channels) % input_channel_alignment if input_channel_alignment else 0
        self.compute_in_channels = self.in_channels + self.input_padding

    @classmethod
    def from_conv(cls, old, alignment):
        devices = [old.weight.device.index] if old.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            new = cls(old.in_channels, old.out_channels, old.kernel_size, stride=old.stride,
                      padding=old.padding, dilation=old.dilation, groups=old.groups,
                      bias=old.bias is not None, padding_mode=old.padding_mode,
                      device=old.weight.device, dtype=old.weight.dtype,
                      input_channel_alignment=alignment)
        new.weight, new.bias = old.weight, old.bias
        new.train(old.training)
        return new

    def forward(self, inputs):
        weight = F.pad(self.weight, (0, 0, 0, 0, 0, self.input_padding)) if self.input_padding else self.weight
        return self._conv_forward(inputs, weight, self.bias)


def _unpack_history(history, history_length, consumer):
    if history.ndim != 2 or history.shape[-1] != history_length * FRAME_DIM:
        raise ValueError(f"{consumer} history shape does not match history_length")
    count = history.shape[0]
    frames = history.reshape(count, history_length, FRAME_DIM)
    maps = frames[..., :-PUBLIC_DIM].reshape(count, history_length * MAP_PLANES, MAP_SIZE, MAP_SIZE)
    return frames, maps


def _knowledge_maps(knowledge, channels, count, consumer):
    if knowledge is None or knowledge.shape != (count, channels * MAP_SIZE * MAP_SIZE):
        raise ValueError(f"{consumer} requires the configured non-stacked knowledge planes")
    return knowledge.reshape(count, channels, MAP_SIZE, MAP_SIZE)


def _padding_part(module, buffer_name, count):
    padding = getattr(module, buffer_name)
    return [padding.expand(count, -1, -1, -1)] if padding.shape[1] else []


class _PackedSpatialActor(nn.Module):
    def __init__(self, baseline, alignment):
        super().__init__()
        self.history_length = baseline.history_length
        self.knowledge_channels = baseline.knowledge_channels
        self.own_king_context = baseline.own_king_context
        self.trunk, self.context = baseline.trunk, baseline.context
        self.cell_head, self.wait_head = baseline.cell_head, baseline.wait_head
        self._geometry_index = int(isinstance(self.trunk[0], OwnKingRelativeGeometry))
        first = AlignedInputConv2d.from_conv(self.trunk[self._geometry_index], alignment)
        self.trunk[self._geometry_index] = first
        self.register_buffer("_input_zero_padding", first.weight.new_zeros(1, first.input_padding, MAP_SIZE, MAP_SIZE),
                             persistent=False)

    def forward(self, history, knowledge=None, *, unpacked=None):
        frames, maps = unpacked if unpacked is not None else _unpack_history(history, self.history_length, "actor")
        count = history.shape[0]
        parts = [maps]
        if self.knowledge_channels:
            parts.append(_knowledge_maps(knowledge, self.knowledge_channels, count, "actor"))
        current = maps[:, (self.history_length - 1) * MAP_PLANES:self.history_length * MAP_PLANES]
        marker = None
        if self._geometry_index or self.own_king_context:
            marker = (current[:, 4].bool() & current[:, 5].bool()).flatten(1)
        if self._geometry_index:
            if not maps.is_floating_point():
                raise TypeError("own-king geometry expects floating-point map features")
            geometry = self.trunk[0]
            king = marker.long().argmax(1)
            dx = (geometry.column.to(dtype=maps.dtype) - (king % MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
            dy = (geometry.row.to(dtype=maps.dtype) - (king // MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
            parts.append(torch.where(marker.sum(1).eq(1)[:, None, None, None], torch.stack((dx, dy), 1), 0))
        parts.extend(_padding_part(self, "_input_zero_padding", count))
        # All extra actor planes enter one materialization, in the legacy order.
        spatial = torch.cat(parts, 1)
        for index in range(self._geometry_index, len(self.trunk)):
            spatial = self.trunk[index](spatial)
        pooled = spatial.mean(dim=(-2, -1))
        context_features = [pooled]
        if self.own_king_context:
            position = marker.long().argmax(1)
            king = spatial.flatten(2).gather(2, position[:, None, None].expand(-1, spatial.shape[1], 1)).squeeze(2)
            context_features.append(torch.where(marker.sum(1).eq(1)[:, None], king, 0))
        context = self.context(torch.cat((*context_features, frames[:, -1, -PUBLIC_DIM:]), -1))
        fused = F.elu(spatial + context[..., None, None])
        cell_logits = self.cell_head(fused)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        moves = cell_logits.permute(0, 2, 3, 1).reshape(count, NUM_ACTIONS - 1)
        return torch.cat((wait, moves), -1)


class _PackedHistoryCritic(nn.Module):
    def __init__(self, baseline, alignment, projection_channels):
        super().__init__()
        self.history_length = baseline.history_length
        self.knowledge_channels = baseline.knowledge_channels
        self.projection_channels = projection_channels
        self.trunk, self.context, self.value_head = baseline.trunk, baseline.context, baseline.value_head
        old = self.trunk[0]
        if projection_channels:
            # Preserve the initial v32 value function: its history/knowledge
            # extension weights are zero. The random projector becomes trainable
            # as soon as the initially zero downstream block receives an update.
            devices = [old.weight.device.index] if old.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.history_projection = AlignedInputConv2d(
                    self.history_length * MAP_PLANES + self.knowledge_channels, projection_channels, 1,
                    device=old.weight.device, dtype=old.weight.dtype, input_channel_alignment=alignment)
                first = AlignedInputConv2d(MAP_PLANES + projection_channels, old.out_channels, old.kernel_size,
                    stride=old.stride, padding=old.padding, dilation=old.dilation,
                    bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype,
                    input_channel_alignment=alignment)
            with torch.no_grad():
                first.weight.zero_()
                first.weight[:, :MAP_PLANES].copy_(old.weight[:, :MAP_PLANES])
                if old.bias is not None:
                    first.bias.copy_(old.bias)
            self.register_buffer("_history_zero_padding", old.weight.new_zeros(
                1, self.history_projection.input_padding, MAP_SIZE, MAP_SIZE), persistent=False)
        else:
            first = AlignedInputConv2d.from_conv(old, alignment)
        self.trunk[0] = first
        self.register_buffer("_input_zero_padding", old.weight.new_zeros(1, first.input_padding, MAP_SIZE, MAP_SIZE),
                             persistent=False)

    def forward(self, privileged, history, knowledge=None, *, unpacked=None):
        frames, local = unpacked if unpacked is not None else _unpack_history(history, self.history_length, "critic")
        count = history.shape[0]
        if privileged.shape != (count, FRAME_DIM):
            raise ValueError("critic privileged observation shape does not match history batch")
        memory = [_knowledge_maps(knowledge, self.knowledge_channels, count, "critic")] if self.knowledge_channels else []
        maps = privileged[:, :-PUBLIC_DIM].reshape(count, MAP_PLANES, MAP_SIZE, MAP_SIZE)
        if self.projection_channels:
            history_inputs = torch.cat((local, *memory, *_padding_part(self, "_history_zero_padding", count)), 1)
            information = [self.history_projection(history_inputs)]
        else:
            information = [local, *memory]
        inputs = torch.cat((maps, *information, *_padding_part(self, "_input_zero_padding", count)), 1)
        pooled = self.trunk(inputs).mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, privileged[:, -PUBLIC_DIM:], frames[:, -1, -PUBLIC_DIM:]), -1))
        return self.value_head(torch.cat((pooled, context), -1))


@dataclass
class _PendingHistory:
    history: weakref.ReferenceType
    version: int
    grad_enabled: bool
    inference_mode: bool
    maps: torch.Tensor


class OptimizedInformationSpatialPPOPolicy(InformationSpatialPPOPolicy):
    """v32 actor capacity with packed/aligned execution and optional critic HK projection.

    ``critic_projection_channels=0`` is checkpoint-compatible equivalent mode.
    A positive width (at least 16; 32 recommended) independently projects history
    and knowledge before privileged concatenation and both spatial convolutions.

    History sharing is a one-use actor→critic handoff. It never caches a graph
    for gradient-bearing inputs or tensors without a mutation counter (inference
    tensors); both safely fall back to unpacking. Actor-only observations, greedy
    inference, failed forwards, device changes and serialization clear the cache.
    """

    def __init__(self, obs, obs_groups, num_actions, *, input_channel_alignment=4,
                 critic_projection_channels=0, share_history_unpack=True, **kwargs):
        if (isinstance(input_channel_alignment, bool) or not isinstance(input_channel_alignment, int)
                or input_channel_alignment < 0):
            raise ValueError("input_channel_alignment must be a nonnegative integer (0 disables alignment)")
        if (isinstance(critic_projection_channels, bool) or not isinstance(critic_projection_channels, int)
                or critic_projection_channels != 0 and critic_projection_channels < 16):
            raise ValueError("critic_projection_channels must be 0 or an integer of at least 16")
        if not isinstance(share_history_unpack, bool):
            raise TypeError("share_history_unpack must be boolean")
        if critic_projection_channels and not kwargs.get("critic_actor_information", False):
            raise ValueError("critic projection requires critic_actor_information=True")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.input_channel_alignment = input_channel_alignment
        self.critic_projection_channels = critic_projection_channels
        self.share_history_unpack = share_history_unpack
        self._pending_history = None
        self.actor = _PackedSpatialActor(self.actor, input_channel_alignment)
        if self.critic_actor_information:
            self.critic = _PackedHistoryCritic(self.critic, input_channel_alignment, critic_projection_channels)
        self.semantics += f"_packed_aligned{input_channel_alignment}_critic_projection{critic_projection_channels}_v1"

    def clear_input_cache(self):
        self._pending_history = None

    @staticmethod
    def _history_version(history):
        try:
            return history._version
        except RuntimeError:  # Tensors created in inference_mode have no counter.
            return None

    def _remember_history(self, obs, unpacked):
        history = obs["policy"]
        version = self._history_version(history)
        if (self.share_history_unpack and self.critic_actor_information and "critic" in obs
                and self.history_length > 1 and not history.requires_grad and version is not None):
            # Weak history ownership lets transient opponent/observation batches
            # disappear without retaining their frames or the packed map copy.
            owner = weakref.ref(self)
            def expired(_reference):
                policy = owner()
                current = None if policy is None else policy._pending_history
                if current is not None and current.history is _reference:
                    policy.clear_input_cache()
            pending = _PendingHistory(weakref.ref(history, expired), version, torch.is_grad_enabled(),
                                      torch.is_inference_mode_enabled(), unpacked[1])
            self._pending_history = pending

    def _take_history(self, history):
        pending = self._pending_history
        self.clear_input_cache()
        if (pending is not None and pending.history() is history and not history.requires_grad
                and pending.version == self._history_version(history)
                and pending.grad_enabled == torch.is_grad_enabled()
                and pending.inference_mode == torch.is_inference_mode_enabled()):
            frames = history.reshape(history.shape[0], self.history_length, FRAME_DIM)
            return frames, pending.maps
        return None

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        self.clear_input_cache()
        if is_teacher:
            raise ValueError("spatial student policy has no teacher inference branch")
        unpacked = _unpack_history(obs["policy"], self.history_length, "actor")
        try:
            logits = self.actor(obs["policy"], obs.get("knowledge"), unpacked=unpacked)
            probabilities = self._probabilities_from_logits(logits, obs.get("action_mask"), validate_mask=validate_mask)
            self._remember_history(obs, unpacked)
            return probabilities
        except Exception:
            self.clear_input_cache()
            raise

    def act_inference(self, obs):
        try:
            return super().act_inference(obs)
        finally:
            self.clear_input_cache()

    def evaluate(self, obs, is_teacher=False, **kwargs):
        unpacked = self._take_history(obs["policy"]) if "policy" in obs else None
        if not self.critic_actor_information or obs.batch_size[0] == 0:
            return super().evaluate(obs, is_teacher=is_teacher, **kwargs)
        if is_teacher:
            raise ValueError("information student policy has no nonempty teacher value branch")
        value = self.critic(obs["critic"], obs["policy"], obs.get("knowledge"), unpacked=unpacked)
        if self.value_potential_offset:
            potential = obs.get("critic_potential")
            if potential is None or potential.shape != (obs.batch_size[0], 1):
                raise ValueError("value_potential_offset requires obs['critic_potential'] with shape [N, 1]")
            value = value - potential.detach()
        return value

    def _apply(self, fn, recurse=True):
        self.clear_input_cache()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["_pending_history"] = None
        # The latest distribution is also transient and may hold a non-leaf
        # autograd graph, which cannot be deep-copied for a self-play snapshot.
        state["distribution"] = None
        return state
