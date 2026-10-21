"""Shared variable-map lifecycle and chunked inference for scaled policies."""
from contextlib import nullcontext

import torch

from .spatial_ppo import SpatialPPOPolicy


class MapPPOPolicy(SpatialPPOPolicy):
    """StudentPPO lifecycle with dynamic geometry and chunked inference.

Chunks only split deterministic forward computation. Categorical construction
and action sampling still occur once per original batch, preserving its RNG
protocol. Training activation memory is controlled separately by StudentPPO's
physical_minibatch_size, not by concatenating graphs of inference chunks.
"""
    semantics = "map_p2_cell_heads_resnetv2_deit3_v1"

    def __init__(self, *args, **kwargs):
        raise ValueError("standalone MapPPOPolicy is retired; use ScaledInformationSpatialPPOPolicy")

    def _update_dimensions(self):
        self.num_actions = self.layout.num_actions
        self.num_actor_obs = self.layout.history_dim + self.layout.knowledge_dim
        self.num_single_obs = self.layout.frame_dim
        self.num_critic_obs = self.layout.frame_dim + (self.num_actor_obs if self.critic_actor_information else 0)

    def _validate_observations(self, obs, num_actions):
        if num_actions != self.layout.num_actions:
            raise ValueError("action vocabulary does not match map geometry")
        self.layout.frames(obs["policy"])
        self.layout.current_maps(obs["single_obs"])
        self.layout.current_maps(obs["critic"])
        if self.layout.knowledge_channels:
            self.layout.memory_maps(obs.get("knowledge"), obs.batch_size[0])

    def set_geometry(self, width, height):
        """Select a homogeneous map bucket without replacing any parameters."""
        self.layout = self.layout.with_geometry(width, height)
        self.actor.layout = self.critic.layout = self.layout
        self._update_dimensions()
        self.distribution = None

    def _forward(self, obs, *, critic=False):
        reference = obs["critic"] if critic else obs["policy"]
        if not len(reference):
            return reference.new_empty((0, 1 if critic else self.num_actions))
        # Microbatch training must backward/release each graph before another
        # chunk. Do not retain all those graphs under a misleading memory flag.
        chunk = len(reference) if torch.is_grad_enabled() else self.inference_chunk_size
        values = []
        for start in range(0, len(reference), chunk):
            part = obs[start:start + chunk]
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if self.amp_bfloat16 and reference.is_cuda else nullcontext())
            with autocast:
                if critic:
                    value = self.critic(part["critic"], part.get("policy"), part.get("knowledge"))
                else:
                    value = self.actor(part["policy"], part.get("knowledge"))
            values.append(value.float())
        return torch.cat(values, 0)

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        if is_teacher:
            raise ValueError("map student policy has no teacher branch")
        # The same masking/factorization implementation as the v17 actor.
        logits = self._forward(obs)
        return self._probabilities_from_logits(logits, obs.get("action_mask"), validate_mask=validate_mask)

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if obs.batch_size[0] == 0:
            return obs["critic"].new_empty((0, 1))
        if is_teacher:
            raise ValueError("map student policy has no nonempty teacher value branch")
        return self._forward(obs, critic=True)
