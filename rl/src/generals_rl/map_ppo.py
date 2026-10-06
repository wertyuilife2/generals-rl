"""Variable-map categorical PPO heads over mature ResNetV2 and ViT backbones.

Map geometry belongs to ObservationLayout, not trainable layers. Homogeneous
map-size buckets can reuse weights after set_geometry(); mixed padded batches
are deliberately not implemented. H4 remains the experimental default.
"""
from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

from .backbones import build_backbone
from .mdp.layout import ObservationLayout, own_king_coordinates
from .spatial_ppo import SpatialPPOPolicy


class GridActor(nn.Module):
    def __init__(self, layout, kind, size, head_channels=128):
        super().__init__()
        self.layout = layout
        self.knowledge_channels = layout.knowledge_channels
        inputs = layout.history_length * layout.map_planes + self.knowledge_channels + 2
        self.backbone = build_backbone(kind, size, inputs)
        self.projection = nn.Conv2d(self.backbone.out_channels, head_channels, 1)
        self.context = nn.Sequential(nn.Linear(head_channels + layout.public_dim, head_channels), nn.SiLU())
        self.cell_head = nn.Conv2d(head_channels, 8, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * head_channels, head_channels), nn.SiLU(), nn.Linear(head_channels, 1))

    def forward(self, history, knowledge=None):
        layout = self.layout
        frames = layout.frames(history)
        maps = layout.history_maps(history)
        current = maps[:, -layout.map_planes:]
        inputs = [maps]
        if self.knowledge_channels:
            inputs.append(layout.memory_maps(knowledge, len(history)))
        inputs.append(own_king_coordinates(current))
        spatial = torch.nn.functional.silu(self.projection(self.backbone(torch.cat(inputs, 1))))
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -layout.public_dim:]), -1))
        fused = torch.nn.functional.silu(spatial + context[..., None, None])
        moves = self.cell_head(fused).permute(0, 2, 3, 1).reshape(len(history), 8 * layout.cells)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        return torch.cat((wait, moves), -1)


class GridCritic(nn.Module):
    """One independent, fixed 1M ResNet critic for all actor architecture pairs."""
    def __init__(self, layout, actor_information, head_channels=128):
        super().__init__()
        self.layout = layout
        self.actor_information = actor_information
        inputs = layout.map_planes + 2
        public = layout.public_dim
        if actor_information:
            inputs += layout.history_length * layout.map_planes + layout.knowledge_channels
            public *= 2
        self.backbone = build_backbone("resnet", "1m", inputs)
        self.value_head = nn.Sequential(nn.Linear(self.backbone.out_channels + public, head_channels),
                                        nn.SiLU(), nn.Linear(head_channels, 1))

    def forward(self, privileged, history=None, knowledge=None):
        layout = self.layout
        current = layout.current_maps(privileged)
        inputs = [current]
        public = [privileged[:, -layout.public_dim:]]
        if self.actor_information:
            inputs.append(layout.history_maps(history))
            public.append(layout.frames(history)[:, -1, -layout.public_dim:])
            if layout.knowledge_channels:
                inputs.append(layout.memory_maps(knowledge, len(privileged)))
        inputs.append(own_king_coordinates(current))
        pooled = self.backbone(torch.cat(inputs, 1)).mean(dim=(-2, -1))
        return self.value_head(torch.cat((pooled, *public), -1))


class MapPPOPolicy(SpatialPPOPolicy):
    """StudentPPO lifecycle with dynamic geometry and chunked inference.

Chunks only split deterministic forward computation. Categorical construction
and action sampling still occur once per original batch, preserving its RNG
protocol. Training activation memory is controlled separately by StudentPPO's
physical_minibatch_size, not by concatenating graphs of inference chunks.
"""
    semantics = "map_p2_cell_heads_resnetv2_deit3_v1"

    def __init__(self, obs, obs_groups, num_actions, *, backbone="resnet", model_size="1m",
                 map_width=9, map_height=9, history_length=4, players=2,
                 map_planes=12, public_dim=13, knowledge_channels=0,
                 critic_actor_information=True, action_distribution="categorical",
                 actor_obs_normalization=False, critic_obs_normalization=False,
                 wait_move_factorization=True, value_potential_offset=False,
                 inference_chunk_size=512, amp_bfloat16=False):
        nn.Module.__init__(self)
        if (players, map_planes, public_dim) != (2, 12, 13):
            raise ValueError("map policy requires the declared P2 twelve-plane/public13 schema")
        if action_distribution != "categorical" or actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("map policy uses categorical actions and unnormalized explicit features")
        for name, value in (("critic_actor_information", critic_actor_information),
                            ("wait_move_factorization", wait_move_factorization),
                            ("value_potential_offset", value_potential_offset),
                            ("amp_bfloat16", amp_bfloat16)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if isinstance(inference_chunk_size, bool) or not isinstance(inference_chunk_size, int) or inference_chunk_size < 1:
            raise ValueError("inference_chunk_size must be a positive integer")
        self.layout = ObservationLayout(map_width, map_height, history_length, map_planes, public_dim, knowledge_channels)
        actor_keys = ["policy"] + (["knowledge"] if knowledge_channels else [])
        expected = {"policy": actor_keys, "critic": ["critic"] + (actor_keys if critic_actor_information else [])}
        if obs_groups != expected:
            raise ValueError(f"map policy requires obs_groups={expected}")
        self._validate_observations(obs, num_actions)
        self.obs_groups = deepcopy(obs_groups)
        self.action_distribution = "categorical"
        self.state_dependent_std = False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.single_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.teacher_encoder = nn.Identity()
        self.critic_actor_information = critic_actor_information
        self.wait_move_factorization = wait_move_factorization
        self.value_potential_offset = value_potential_offset
        self.inference_chunk_size = inference_chunk_size
        self.amp_bfloat16 = amp_bfloat16
        self.actor = GridActor(self.layout, backbone, model_size)
        self.critic = GridCritic(self.layout, critic_actor_information)
        self.distribution = None
        self._update_dimensions()

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

    def action_probabilities(self, obs, is_teacher=False):
        if is_teacher:
            raise ValueError("map student policy has no teacher branch")
        # The same masking/factorization implementation as the v17 actor.
        logits = self._forward(obs)
        return self._probabilities_from_logits(logits, obs.get("action_mask"))

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if obs.batch_size[0] == 0:
            return obs["critic"].new_empty((0, 1))
        if is_teacher:
            raise ValueError("map student policy has no nonempty teacher value branch")
        value = self._forward(obs, critic=True)
        if self.value_potential_offset:
            potential = obs.get("critic_potential")
            if potential is None or potential.shape != value.shape:
                raise ValueError("value_potential_offset requires matching critic_potential")
            value = value - potential.detach()
        return value
