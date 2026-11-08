"""Local-information actor/value encoders with an optional privileged branch."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache
from types import FunctionType

import torch
from torch import nn
from torch.nn import functional as F

from .spatial_ppo import SpatialPPOPolicy
from .policy.layout import ObservationLayout
from .policy.scaled import _ScaledInformationActor, _ViTBackbone, RESNET_TOTAL_PRESETS, vit_configuration
from .policy.tokens import (pack_tokens, paired_token_blocks, patch_mask, prepare_tokens, spatial_features,
                            token_blocks, valid_mean, valid_token_features)


class _ScaledInformationCritic(nn.Module):
    """Fuse local features with a small, value-only full-state encoder."""
    def __init__(self, layout, encoder, head_channels, patch_size, mask_padding_tokens, privileged):
        super().__init__()
        self.layout, self.encoder = layout, encoder
        self.mask_padding_tokens = mask_padding_tokens
        self.privileged_encoder = (_ViTBackbone(layout.privileged_planes, (64, 1, 4),
            map_shape=(layout.height, layout.width), patch_size=patch_size) if privileged else None)
        self.fusion = (nn.Conv2d(head_channels + 64, head_channels, 1) if privileged else None)
        public_inputs = layout.public_dim * (2 if privileged else 1)
        self.context = nn.Sequential(nn.Linear(head_channels + public_inputs, head_channels), nn.SiLU())
        self.value_head = nn.Sequential(nn.Linear(2 * head_channels, head_channels), nn.SiLU(), nn.Linear(head_channels, 1))

    def forward(self, encoded, public, privileged, valid_cells, tokens):
        spatial, pooled = encoded
        statistics = [public]
        if self.privileged_encoder is not None:
            maps = self.layout.privileged_maps(privileged)
            if self.mask_padding_tokens:
                maps = maps.masked_fill(~valid_cells.reshape(-1, 1, self.layout.height, self.layout.width), 0)
                extra = valid_token_features(self.privileged_encoder.transformer, maps, tokens[0], tokens)
            else:
                extra = self.privileged_encoder(maps)
            spatial = F.silu(self.fusion(torch.cat((spatial, extra), 1)))
            pooled = (valid_mean(spatial, tokens[0]) if self.mask_padding_tokens
                      else spatial.mean((-2, -1)))
            statistics.append(privileged[:, -self.layout.public_dim:])
        context = self.context(torch.cat((pooled, *statistics), -1))
        return self.value_head(torch.cat((pooled, context), -1))


class _AttentionInformationCritic(nn.Module):
    """Fuse state before shared attention, or before private post-attention."""
    def __init__(self, layout, local_encoder, mode, privileged, layers):
        super().__init__()
        self.layout = layout
        self.patch_size = local_encoder.patch_size
        transformer = local_encoder.backbone.transformer
        width = transformer.embed_dim
        head_channels = local_encoder.projection.out_channels
        self.grid_shape = transformer.patch_embed.dynamic_feat_size((layout.height, layout.width))
        self.projection = deepcopy(local_encoder.projection)
        # Common heads/embedding have identical initialization across both
        # fusion variants. Shared modules are registered only under actor.
        public_inputs = layout.public_dim * (2 if privileged else 1)
        self.context = nn.Sequential(nn.Linear(head_channels + public_inputs, head_channels), nn.SiLU())
        self.value_head = nn.Sequential(nn.Linear(2 * head_channels, head_channels), nn.SiLU(), nn.Linear(head_channels, 1))
        self.privileged_embedding = (nn.Conv2d(layout.privileged_planes, width, self.patch_size,
                                              stride=self.patch_size) if privileged else None)
        self.blocks = deepcopy(transformer.blocks[-layers:]) if mode == "post_attention" else None
        self.norm = deepcopy(transformer.norm) if self.blocks is not None else None
        self.fusion_norm = nn.LayerNorm(2 * width, eps=1e-6) if privileged and self.blocks is not None else None
        self.fusion = nn.Linear(2 * width, width) if self.fusion_norm is not None else None

    def embed(self, local, privileged, valid_cells, tokens):
        if self.privileged_embedding is not None:
            maps = self.layout.privileged_maps(privileged)
            if tokens is not None:
                maps = maps.masked_fill(~valid_cells.reshape(-1, 1, self.layout.height, self.layout.width), 0)
            maps = F.pad(maps, (0, -self.layout.width % self.patch_size,
                                0, -self.layout.height % self.patch_size))
            extra = pack_tokens(self.privileged_embedding(maps).flatten(2).transpose(1, 2), tokens)
            local = (local + extra if self.fusion is None else
                     local + self.fusion(self.fusion_norm(torch.cat((local, extra), -1))))
        return local

    def decode(self, x, public, privileged, tokens, transformer):
        norm = transformer.norm if self.norm is None else self.norm
        spatial = F.silu(self.projection(spatial_features(norm(x), self.grid_shape, tokens)))
        pooled = valid_mean(spatial, tokens[0]) if tokens is not None else spatial.mean((-2, -1))
        statistics = [public]
        if self.privileged_embedding is not None:
            statistics.append(privileged[:, -self.layout.public_dim:])
        context = self.context(torch.cat((pooled, *statistics), -1))
        return self.value_head(torch.cat((pooled, context), -1))

    def forward(self, local, public, privileged, valid_cells, tokens, transformer):
        local = self.embed(local, privileged, valid_cells, tokens)
        blocks = transformer.blocks if self.blocks is None else self.blocks
        x = token_blocks(blocks, local, tokens, self.grid_shape[0] * self.grid_shape[1])
        return self.decode(x, public, privileged, tokens, transformer)


@lru_cache(maxsize=32)
def _compiled_numeric_forward(mode, shared, privileged, fusion, max_autotune=False, batch_shared=False):
    # Separate frame caches keep the three forward modes and architecture
    # choices from consuming one Dynamo frame's shape-recompilation budget.
    original = ScaledInformationSpatialPPOPolicy._numeric_forward
    name = (f"scaled_{mode}_shared{int(shared)}_priv{int(privileged)}_{fusion}"
            f"_maxauto{int(max_autotune)}_batched{int(batch_shared)}")
    forward = FunctionType(original.__code__.replace(co_name=name), original.__globals__, name,
                           original.__defaults__, original.__closure__)
    options = {"triton.cudagraphs": False, "freezing": False}
    if max_autotune:
        options.update(emulate_precision_casts=False, max_autotune=True)
    return torch.compile(forward, dynamic=True, fullgraph=True, options=options)


@lru_cache(maxsize=32)
def _compiled_frozen_actor_forward(batch_size, max_autotune=False):
    """Static batch buckets cache unbound functions, never snapshot modules."""
    original = _ScaledInformationActor.forward
    name = f"frozen_actor_batch_{batch_size}_maxauto{int(max_autotune)}"
    forward = FunctionType(original.__code__.replace(co_name=name), original.__globals__, name,
                           original.__defaults__, original.__closure__)
    options = {"triton.cudagraphs": False, "freezing": False}
    if max_autotune:
        options.update(emulate_precision_casts=False, max_autotune=True)
    return torch.compile(forward, dynamic=False, fullgraph=True, options=options)


class ScaledInformationSpatialPPOPolicy(SpatialPPOPolicy):
    """Copy the local encoder for independent weights, or reuse its features.

    ViT1M uses a wider local encoder when shared to retain a comparable total
    parameter budget. parameter_counts() reports actual capacity. Only the
    value branch can read privileged maps and statistics.
    """
    semantics = "official_p2_hist4_local13_priv11_encoder_v1"

    def __init__(self, obs, obs_groups, num_actions, *, backbone="vit", model_size="1m",
                 map_width=23, map_height=23, history_length=4, players=2,
                 map_planes=13, public_dim=13, knowledge_channels=11,
                 share_backbone=True, critic_privileged=True, own_king_geometry=True,
                 input_channel_alignment=4, head_channels=128, action_distribution="categorical",
                 actor_obs_normalization=False, critic_obs_normalization=False,
                 wait_move_factorization=True, inference_chunk_size=128, amp_bfloat16=False,
                 compile_forward=False, spatial_mode="dense", mask_padding_tokens=None, patch_size=1,
                 critic_fusion="late", critic_attention_layers=2, batch_shared_forward=False):
        nn.Module.__init__(self)
        if backbone not in ("resnet", "vit"):
            raise ValueError("backbone must be resnet or vit")
        if spatial_mode not in ("dense", "pyramid"):
            raise ValueError("spatial_mode must be dense or pyramid")
        if critic_fusion not in ("late", "early_shared", "post_attention"):
            raise ValueError("unsupported critic_fusion")
        if critic_fusion in ("early_shared", "post_attention") and (
                backbone != "vit" or spatial_mode != "dense" or share_backbone is not True):
            raise ValueError("attention critic fusion requires a shared dense ViT")
        if mask_padding_tokens is None:
            mask_padding_tokens = backbone == "vit" and spatial_mode == "dense"
        if type(mask_padding_tokens) is not bool or mask_padding_tokens and (backbone != 'vit' or spatial_mode != 'dense'):
            raise ValueError('mask_padding_tokens requires a dense ViT')
        if not isinstance(model_size, str) or model_size.lower() not in RESNET_TOTAL_PRESETS:
            raise ValueError("model_size must be 1m, 10m, or 100m")
        self.backbone_kind, self.model_size = backbone, model_size.lower()
        schema = (players, map_width, map_height, history_length, map_planes, public_dim, knowledge_channels)
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in schema)
                or (players, history_length, map_planes, public_dim, knowledge_channels) != (2, 4, 13, 13, 11)
                or min(map_width, map_height) < 1):
            raise ValueError("scaled policy requires P2 H4/K11 thirteen-plane/public13 features")
        if action_distribution != "categorical" or actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("scaled policy uses categorical actions and unnormalized explicit features")
        for name, value in (("share_backbone", share_backbone), ("critic_privileged", critic_privileged),
                            ("own_king_geometry", own_king_geometry), ("wait_move_factorization", wait_move_factorization),
                            ("amp_bfloat16", amp_bfloat16), ("compile_forward", compile_forward),
                            ("batch_shared_forward", batch_shared_forward)):
            if type(value) is not bool:
                raise TypeError(f"{name} must be boolean")
        if not own_king_geometry or not wait_move_factorization:
            raise ValueError("scaled policy requires own-king geometry and WAIT/MOVE factorization")
        if type(input_channel_alignment) is not int or input_channel_alignment < 0:
            raise ValueError("input_channel_alignment must be a nonnegative integer")
        for name, value in (("head_channels", head_channels), ("inference_chunk_size", inference_chunk_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.layout = ObservationLayout(map_width, map_height, history_length, map_planes, public_dim, knowledge_channels)
        self.share_backbone, self.critic_privileged = share_backbone, critic_privileged
        local_keys = ["policy", "knowledge", "valid_cells"]
        expected = {"policy": local_keys, "critic": (["critic"] if critic_privileged else []) + local_keys}
        if obs_groups != expected:
            raise ValueError(f"scaled policy requires obs_groups={expected}")
        self._validate_observations(obs, num_actions)
        self.obs_groups = deepcopy(obs_groups)
        self.action_distribution, self.state_dependent_std = "categorical", False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.own_king_geometry, self.wait_move_factorization = own_king_geometry, wait_move_factorization
        self.inference_chunk_size, self.amp_bfloat16 = inference_chunk_size, amp_bfloat16
        self.compile_forward, self.spatial_mode = compile_forward, spatial_mode
        self.mask_padding_tokens, self.patch_size = mask_padding_tokens, patch_size
        self.input_channel_alignment = input_channel_alignment
        self.history_length, self.head_channels = history_length, head_channels
        self.critic_fusion = critic_fusion
        self.batch_shared_forward = batch_shared_forward
        vit_config = vit_configuration(self.model_size, share_backbone) if backbone == "vit" else None
        self.actor = _ScaledInformationActor(self.layout, backbone, self.model_size, input_channel_alignment,
                                             head_channels, spatial_mode, mask_padding_tokens, patch_size, vit_config)
        if critic_fusion == "late":
            value_encoder = None if share_backbone else deepcopy(self.actor.encoder)
            self.critic = _ScaledInformationCritic(self.layout, value_encoder, head_channels, patch_size,
                                                   mask_padding_tokens, critic_privileged)
        else:
            if critic_fusion == "post_attention" and (type(critic_attention_layers) is not int
                    or not 1 <= critic_attention_layers <= self.actor.encoder.backbone.depth):
                raise ValueError("critic_attention_layers must be between 1 and ViT depth")
            self.critic = _AttentionInformationCritic(self.layout, self.actor.encoder, critic_fusion,
                                                      critic_privileged, critic_attention_layers)
        self.distribution = None
        self.num_actions = self.layout.num_actions
        self.num_actor_obs = self.layout.history_dim + self.layout.knowledge_dim
        self.num_single_obs = self.layout.frame_dim
        self.num_critic_obs = self.num_actor_obs + (self.layout.privileged_dim if critic_privileged else 0)
        self.semantics += f"_{backbone}_{self.model_size}_head{head_channels}_shared{int(share_backbone)}_priv{int(critic_privileged)}"
        if critic_fusion != "late":
            self.semantics += f"_fusion_{critic_fusion}_v1"
            if critic_fusion == "post_attention":
                self.semantics += f"_layers{critic_attention_layers}"
        if spatial_mode == "pyramid":
            self.semantics += "_spatial_pyramid_v1"
        if mask_padding_tokens:
            self.semantics += "_valid_tokens_v1"
        if patch_size != 1:
            self.semantics += f"_patch{patch_size}"

    def _validate_observations(self, obs, num_actions):
        if num_actions != self.layout.num_actions:
            raise ValueError("action vocabulary does not match map geometry")
        self.layout.frames(obs["policy"])
        if self.critic_privileged:
            self.layout.privileged_maps(obs["critic"])
        if obs['valid_cells'].dtype != torch.bool or obs['valid_cells'].shape != (obs.batch_size[0], self.layout.cells):
            raise ValueError('valid_cells must be bool [batch, cells]')
        self.layout.memory_maps(obs.get("knowledge"), obs.batch_size[0])

    def parameter_counts(self):
        actor = sum(parameter.numel() for parameter in self.actor.parameters())
        critic = sum(parameter.numel() for parameter in self.critic.parameters())
        return {"actor": actor, "critic": critic, "total": actor + critic}

    def _numeric_forward(self, history, knowledge, valid_cells, privileged, tokens, mode):
        if mode == "actor":
            return self.actor(history, knowledge, valid_cells, tokens)
        if self.critic_fusion != "late":
            encoder = self.actor.encoder
            transformer = encoder.backbone.transformer
            local = encoder.embed(history, knowledge, valid_cells, tokens)
            public = self.layout.frames(history)[:, -1, -self.layout.public_dim:]
            if (mode == "joint" and self.critic_fusion == "early_shared"
                    and (self.batch_shared_forward or not self.critic_privileged)):
                maximum = self.critic.grid_shape[0] * self.critic.grid_shape[1]
                if self.critic_privileged:
                    value_input = self.critic.embed(local, privileged, valid_cells, tokens)
                    shared, value_features = paired_token_blocks(
                        transformer.blocks, local, value_input, tokens, maximum)
                else:
                    shared = value_features = token_blocks(transformer.blocks, local, tokens, maximum)
                logits = self.actor.decode(encoder.project_tokens(shared, tokens), public)
                values = self.critic.decode(value_features, public, privileged, tokens, transformer)
                return logits, values
            if mode == "joint" or self.critic_fusion == "post_attention":
                shape = self.critic.grid_shape
                shared = token_blocks(transformer.blocks, local, tokens, shape[0] * shape[1])
            if mode == "joint":
                logits = self.actor.decode(encoder.project_tokens(shared, tokens), public)
            value_input = local if self.critic_fusion == "early_shared" else transformer.norm(shared)
            values = self.critic(value_input, public, privileged, valid_cells, tokens, transformer)
            return (logits, values) if mode == "joint" else values
        encoder = self.actor.encoder if self.share_backbone or mode == "joint" else self.critic.encoder
        encoded = encoder(history, knowledge, valid_cells, tokens)
        public = self.layout.frames(history)[:, -1, -self.layout.public_dim:]
        if mode == "joint":
            logits = self.actor.decode(encoded, public)
            if not self.share_backbone:
                encoded = self.critic.encoder(history, knowledge, valid_cells, tokens)
        values = self.critic(encoded, public, privileged, valid_cells, tokens)
        return (logits, values) if mode == "joint" else values

    def _numeric_network_forward(self, part, mode):
        history, knowledge, valid = part["policy"], part.get("knowledge"), part["valid_cells"]
        compile_network = self.compile_forward and (self.backbone_kind == "vit" or not torch.is_grad_enabled())
        max_autotune = (self.backbone_kind == "vit" and self.model_size == "100m"
                        and self.spatial_mode == "pyramid" and history.is_cuda)
        if (compile_network and mode == "actor" and not torch.is_grad_enabled() and not self.training
                and all(not p.requires_grad for p in self.actor.parameters())):
            count = history.shape[0]
            bucket = 1 << (count - 1).bit_length()
            if bucket != count:
                history = F.pad(history, (0, 0, 0, bucket - count))
                knowledge = F.pad(knowledge, (0, 0, 0, bucket - count))
                valid = F.pad(valid, (0, 0, 0, bucket - count))
            tokens = (prepare_tokens(patch_mask(valid, self.layout.height, self.layout.width, self.patch_size))
                      if self.mask_padding_tokens else None)
            return _compiled_frozen_actor_forward(bucket, max_autotune)(self.actor, history, knowledge, valid, tokens)[:count]
        tokens = (prepare_tokens(patch_mask(valid, self.layout.height, self.layout.width, self.patch_size))
                  if self.mask_padding_tokens else None)
        privileged = part["critic"] if mode != "actor" and self.critic_privileged else None
        args = (history, knowledge, valid, privileged, tokens, mode)
        if compile_network:
            return _compiled_numeric_forward(mode, self.share_backbone, self.critic_privileged,
                                              self.critic_fusion, max_autotune, self.batch_shared_forward)(self, *args)
        return self._numeric_forward(*args)

    def _network_forward(self, part, mode):
        reference = part['policy']
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.amp_bfloat16 and reference.is_cuda else nullcontext()
        with autocast:
            result = self._numeric_network_forward(part, mode)
        return tuple(value.float() for value in result) if mode == "joint" else result.float()

    def _forward(self, obs, *, critic=False, joint=False):
        reference = obs["policy"]
        count, mode = len(reference), "joint" if joint else "critic" if critic else "actor"
        if not count:
            logits, values = reference.new_empty((0, self.num_actions)), reference.new_empty((0, 1))
            return (logits, values) if joint else values if critic else logits
        if torch.is_grad_enabled():
            return self._network_forward(obs, mode)
        widths = (self.num_actions, 1) if joint else (1 if critic else self.num_actions,)
        output = tuple(reference.new_empty((count, width), dtype=torch.float32) for width in widths)
        for start in range(0, count, self.inference_chunk_size):
            end = min(count, start + self.inference_chunk_size)
            result = self._network_forward(obs[start:end], mode)
            values = result if joint else (result,)
            for target, value in zip(output, values):
                target[start:end].copy_(value)
        return output if joint else output[0]

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        if is_teacher:
            raise ValueError("scaled student policy has no teacher branch")
        logits = self._forward(obs)
        with torch.autocast(logits.device.type, enabled=False):
            return self._probabilities_from_logits(logits, obs.get("action_mask"), validate_mask=validate_mask)

    def action_probabilities_and_value(self, obs, *, validate_mask=True):
        logits, values = self._forward(obs, joint=True)
        with torch.autocast(logits.device.type, enabled=False):
            probabilities = self._probabilities_from_logits(logits, obs.get("action_mask"), validate_mask=validate_mask)
        return probabilities, values

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if is_teacher and obs.batch_size[0]:
            raise ValueError("scaled student policy has no teacher value branch")
        return self._forward(obs, critic=True)

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["distribution"] = None
        return state
