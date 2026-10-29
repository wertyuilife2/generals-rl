"""Packed/aligned v32 execution with checkpoint-compatible actor and critic.

Parameter names and shapes match InformationSpatialPPOPolicy. Alignment adds
constant zero channels only at execution time; history unpacking is shared.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import torch
from torch import nn

from .information_ppo import InformationSpatialPPOPolicy
from .policy.cnn import (AlignedInputConv2d, _unpack_history, _knowledge_maps,
                         _padding_part, _PackedSpatialActor)


class _PackedHistoryCritic(nn.Module):
    def __init__(self, baseline, alignment):
        super().__init__()
        self.layout = baseline.layout
        self.history_length = baseline.history_length
        self.knowledge_channels = baseline.knowledge_channels
        self.trunk, self.context, self.value_head = baseline.trunk, baseline.context, baseline.value_head
        old = self.trunk[0]
        first = AlignedInputConv2d.from_conv(old, alignment)
        self.trunk[0] = first
        self.register_buffer("_input_zero_padding", old.weight.new_zeros(1, first.input_padding, self.layout.height, self.layout.width),
                             persistent=False)

    def forward(self, privileged, history, knowledge=None, *, unpacked=None):
        frames, local = unpacked if unpacked is not None else _unpack_history(history, self.layout)
        count = history.shape[0]
        if privileged.shape != (count, self.layout.frame_dim):
            raise ValueError("critic privileged observation shape does not match history batch")
        memory = [_knowledge_maps(knowledge, self.knowledge_channels, count, "critic", self.layout)] if self.knowledge_channels else []
        maps = privileged[:, :-self.layout.public_dim].reshape(count, self.layout.map_planes, self.layout.height, self.layout.width)
        inputs = torch.cat((maps, local, *memory, *_padding_part(self, "_input_zero_padding", count)), 1)
        pooled = self.trunk(inputs).mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, privileged[:, -self.layout.public_dim:], frames[:, -1, -self.layout.public_dim:]), -1))
        return self.value_head(torch.cat((pooled, context), -1))


@dataclass
class _PendingHistory:
    history: weakref.ReferenceType
    version: int
    grad_enabled: bool
    inference_mode: bool
    maps: torch.Tensor


class OptimizedInformationSpatialPPOPolicy(InformationSpatialPPOPolicy):
    """v32 actor and critic capacity with packed/aligned execution.

    ``critic_projection_channels=0`` is retained in checkpoint metadata; the
    projected LightCNN experiment is retired. Scaled policies have their own
    independent projection implementation.

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
                or critic_projection_channels != 0):
            raise ValueError("LightCNN critic_projection_channels experiments are retired; expected 0")
        if not isinstance(share_history_unpack, bool):
            raise TypeError("share_history_unpack must be boolean")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.input_channel_alignment = input_channel_alignment
        self.critic_projection_channels = critic_projection_channels
        self.share_history_unpack = share_history_unpack
        self._pending_history = None
        self.actor = _PackedSpatialActor(self.actor, input_channel_alignment)
        if self.critic_actor_information:
            self.critic = _PackedHistoryCritic(self.critic, input_channel_alignment)
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
            frames = history.reshape(history.shape[0], self.history_length, self.layout.frame_dim)
            return frames, pending.maps
        return None

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        self.clear_input_cache()
        if is_teacher:
            raise ValueError("spatial student policy has no teacher inference branch")
        unpacked = _unpack_history(obs["policy"], self.layout)
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
        return self.critic(obs["critic"], obs["policy"], obs.get("knowledge"), unpacked=unpacked)

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
