"""Independent spatial actor and critic with nominal 1M/10M/100M budgets.

Actual parameter counts include the configured board geometry and feature schema;
``parameter_counts()`` reports the instantiated model rather than historical totals.
"""
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
from .policy.cnn import AlignedInputConv2d
from .policy.scaled import (_build_total_backbone, _ScaledInformationActor, RESNET_TOTAL_PRESETS)


class _ScaledInformationCritic(nn.Module):
    """Independent full backbone after privileged + all history/knowledge input."""
    def __init__(self, layout, kind, size, alignment, projection_channels, head_channels,
                 spatial_mode="dense"):
        super().__init__()
        self.layout, self.projection_channels = layout, projection_channels
        self.spatial_mode = spatial_mode
        self.knowledge_channels = layout.knowledge_channels
        history_inputs = layout.history_length * layout.map_planes + layout.knowledge_channels
        spatial_inputs = layout.map_planes + (projection_channels or history_inputs)
        if projection_channels:
            self.history_projection = AlignedInputConv2d(history_inputs, projection_channels, 1,
                                                         input_channel_alignment=alignment)
            self.register_buffer("_history_zero_padding", self.history_projection.weight.new_zeros(
                1, self.history_projection.input_padding, layout.height, layout.width), persistent=False)
        first = AlignedInputConv2d(spatial_inputs, 32, 3, padding=1, input_channel_alignment=alignment)
        # Both spatial layers survive the projection ablation for either family;
        # the entire ResNet/ViT then adds capacity to the independent critic.
        self.spatial_stem = nn.Sequential(first, nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.backbone = _build_total_backbone(kind, size, 32, spatial_mode, (layout.height, layout.width))
        self.projection = nn.Conv2d(self.backbone.out_channels, head_channels, 1)
        self.context = nn.Sequential(nn.Linear(head_channels + 2 * layout.public_dim, head_channels), nn.SiLU())
        self.value_head = nn.Sequential(nn.Linear(2 * head_channels, head_channels), nn.SiLU(), nn.Linear(head_channels, 1))
        self.register_buffer("_input_zero_padding", first.weight.new_zeros(
            1, first.input_padding, layout.height, layout.width), persistent=False)
        if spatial_mode == "pyramid":
            self.local_skip = nn.Conv2d(self.backbone.local_channels, head_channels, 1)

    def forward(self, privileged, history=None, knowledge=None):
        layout, count = self.layout, privileged.shape[0]
        frames = layout.frames(history)
        local, memory = layout.history_maps(history), layout.memory_maps(knowledge, count)
        if self.projection_channels:
            parts = [local, memory]
            if self._history_zero_padding.shape[1]:
                parts.append(self._history_zero_padding.expand(count, -1, -1, -1))
            information = [self.history_projection(torch.cat(parts, 1))]
        else:
            information = [local, memory]
        parts = [layout.current_maps(privileged), *information]
        if self._input_zero_padding.shape[1]:
            parts.append(self._input_zero_padding.expand(count, -1, -1, -1))
        spatial = self.spatial_stem(torch.cat(parts, 1))
        if self.spatial_mode == "pyramid":
            local, deep = self.backbone.forward_pyramid(spatial)
            pooled = (F.silu(self.projection(deep)).mean(dim=(-2, -1))
                      + F.silu(self.local_skip(local)).mean(dim=(-2, -1)))
        else:
            spatial = F.silu(self.projection(self.backbone(spatial)))
            pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, privileged[:, -layout.public_dim:],
                                          frames[:, -1, -layout.public_dim:]), -1))
        return self.value_head(torch.cat((pooled, context), -1))


@lru_cache(maxsize=4)
def _compiled_numeric_forward(critic, max_autotune=False):
    """Cache unbound functions, never a learner or a snapshot module instance."""
    forward = _ScaledInformationCritic.forward if critic else _ScaledInformationActor.forward
    options = {"triton.cudagraphs": False, "freezing": False}
    if max_autotune:
        options.update(emulate_precision_casts=False, max_autotune=True)
    return torch.compile(forward, dynamic=True, fullgraph=True, options=options)


@lru_cache(maxsize=32)
def _compiled_frozen_actor_forward(batch_size, max_autotune=False):
    """Static batch buckets isolate frozen inference from learner graph guards.

    Dynamo caches per code object. A distinct code name for each bucket keeps
    shape variants from exhausting one frame's recompile limit. Only unbound
    functions are cached; snapshot modules and weights remain ordinary inputs.
    """
    original = _ScaledInformationActor.forward
    code = original.__code__.replace(co_name=f"frozen_actor_batch_{batch_size}_maxauto{int(max_autotune)}")
    forward = FunctionType(code, original.__globals__, code.co_name,
                           original.__defaults__, original.__closure__)
    options = {"triton.cudagraphs": False, "freezing": False}
    if max_autotune:
        options.update(emulate_precision_casts=False, max_autotune=True)
    return torch.compile(forward, dynamic=False, fullgraph=True, options=options)


class ScaledInformationSpatialPPOPolicy(SpatialPPOPolicy):
    """Mature dense backbones, balanced TOTAL capacity and v32 information/action semantics.

    No BatchNorm, nonzero Dropout, or stochastic depth is present. GroupNorm and
    LayerNorm preserve independent samples across opponent groups and PPO chunks.
    Master parameters remain float32; CUDA bf16 autocasts only network forwards,
    then logits/probabilities/log-probabilities and values return to float32.
    """
    semantics = "official_p2_hist4_knowledge11_independent_critic_v2"

    def __init__(self, obs, obs_groups, num_actions, *, backbone="resnet", model_size="1m",
                 map_width=23, map_height=23, history_length=4, players=2,
                 map_planes=14, public_dim=13, knowledge_channels=11,
                 critic_actor_information=True, own_king_geometry=True,
                 critic_projection_channels=32, input_channel_alignment=4,
                 head_channels=128, action_distribution="categorical",
                 actor_obs_normalization=False, critic_obs_normalization=False,
                 wait_move_factorization=True,
                 inference_chunk_size=128, amp_bfloat16=False, compile_forward=False,
                 spatial_mode="dense"):
        nn.Module.__init__(self)
        if backbone not in ("resnet", "vit"):
            raise ValueError("backbone must be resnet or vit")
        if spatial_mode not in ("dense", "pyramid"):
            raise ValueError("spatial_mode must be dense or pyramid")
        if not isinstance(model_size, str) or model_size.lower() not in RESNET_TOTAL_PRESETS:
            raise ValueError("model_size must be 1m, 10m, or 100m")
        self.backbone_kind, self.model_size = backbone, model_size.lower()
        schema = (players, map_width, map_height, history_length, map_planes, public_dim, knowledge_channels)
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in schema)
                or (players, history_length, map_planes, public_dim, knowledge_channels) != (2, 4, 14, 13, 11)
                or min(map_width, map_height) < 1):
            raise ValueError("scaled policy requires P2 H4/K11 fourteen-plane/public13 features")
        if action_distribution != "categorical" or actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("scaled policy uses categorical actions and unnormalized explicit features")
        for name, value in (("critic_actor_information", critic_actor_information), ("own_king_geometry", own_king_geometry),
                            ("wait_move_factorization", wait_move_factorization),
                            ("amp_bfloat16", amp_bfloat16), ("compile_forward", compile_forward)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if not critic_actor_information or not own_king_geometry or not wait_move_factorization:
            raise ValueError("scaled policy requires actor-information critic, own-king geometry and WAIT/MOVE factorization")
        if (isinstance(critic_projection_channels, bool) or not isinstance(critic_projection_channels, int)
                or critic_projection_channels != 0 and critic_projection_channels < 16):
            raise ValueError("critic_projection_channels must be 0 or an integer of at least 16")
        if isinstance(input_channel_alignment, bool) or not isinstance(input_channel_alignment, int) or input_channel_alignment < 0:
            raise ValueError("input_channel_alignment must be a nonnegative integer")
        for name, value in (("head_channels", head_channels), ("inference_chunk_size", inference_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.layout = ObservationLayout(map_width, map_height, history_length, map_planes, public_dim, knowledge_channels)
        expected = {"policy": ["policy", "knowledge"], "critic": ["critic", "policy", "knowledge"]}
        if obs_groups != expected:
            raise ValueError(f"scaled policy requires obs_groups={expected}")
        self._validate_observations(obs, num_actions)
        self.obs_groups = deepcopy(obs_groups)
        self.action_distribution, self.state_dependent_std = "categorical", False
        self.actor_obs_normalization = self.critic_obs_normalization = False
        self.critic_actor_information, self.own_king_geometry = critic_actor_information, own_king_geometry
        self.wait_move_factorization = wait_move_factorization
        self.inference_chunk_size, self.amp_bfloat16 = inference_chunk_size, amp_bfloat16
        self.compile_forward = compile_forward
        self.spatial_mode = spatial_mode
        self.critic_projection_channels, self.input_channel_alignment = critic_projection_channels, input_channel_alignment
        self.history_length, self.head_channels = history_length, head_channels
        self.actor = _ScaledInformationActor(self.layout, backbone, self.model_size, input_channel_alignment,
                                             head_channels, spatial_mode)
        self.critic = _ScaledInformationCritic(self.layout, backbone, self.model_size, input_channel_alignment,
                                               critic_projection_channels, head_channels, spatial_mode)
        self.distribution = None
        self._update_dimensions()
        self.semantics += f"_{backbone}_{self.model_size}_projection{critic_projection_channels}_head{head_channels}"
        if spatial_mode == "pyramid":
            self.semantics += "_spatial_pyramid_v1"

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

    def evaluate(self, obs, is_teacher=False, **kwargs):
        if obs.batch_size[0] == 0:
            return obs["critic"].new_empty((0, 1))
        if is_teacher:
            raise ValueError("map student policy has no nonempty teacher value branch")
        return self._forward(obs, critic=True)

    def parameter_counts(self):
        actor = sum(parameter.numel() for parameter in self.actor.parameters())
        critic = sum(parameter.numel() for parameter in self.critic.parameters())
        return {"actor": actor, "critic": critic, "total": actor + critic}

    def _numeric_network_forward(self, part, critic):
        # ResNet compiled backward does not meet the validated CUDA numeric
        # thresholds. Keep its training path eager; no_grad inference is eligible.
        if self.compile_forward and (self.backbone_kind == "vit" or not torch.is_grad_enabled()):
            reference = part["critic"] if critic else part["policy"]
            if (not critic and not torch.is_grad_enabled() and not self.training
                    and not self.actor.projection.weight.requires_grad
                    and all(not p.requires_grad for p in self.actor.parameters())):
                count = reference.shape[0]
                bucket = 1 << (count - 1).bit_length()
                history, knowledge = part["policy"], part.get("knowledge")
                if bucket != count:
                    history = F.pad(history, (0, 0, 0, bucket - count))
                    if knowledge is not None:
                        knowledge = F.pad(knowledge, (0, 0, 0, bucket - count))
                max_autotune = (self.backbone_kind == "vit" and self.model_size == "100m"
                                and self.spatial_mode == "pyramid" and reference.is_cuda)
                return _compiled_frozen_actor_forward(bucket, max_autotune)(
                    self.actor, history, knowledge)[:count]
            if (self.backbone_kind == "vit" and self.model_size == "100m"
                    and self.spatial_mode == "pyramid" and reference.is_cuda):
                forward = _compiled_numeric_forward(critic, max_autotune=True)
            else:
                forward = _compiled_numeric_forward(critic)
            return (forward(self.critic, part["critic"], part["policy"], part.get("knowledge"))
                    if critic else forward(self.actor, part["policy"], part.get("knowledge")))
        if critic:
            return self.critic(part["critic"], part["policy"], part.get("knowledge"))
        return self.actor(part["policy"], part.get("knowledge"))

    def _network_forward(self, part, critic, reference):
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.amp_bfloat16 and reference.is_cuda else nullcontext()
        with autocast:
            result = self._numeric_network_forward(part, critic)
        return result.float()

    def _forward(self, obs, *, critic=False):
        reference = obs["critic"] if critic else obs["policy"]
        count, width = len(reference), 1 if critic else self.num_actions
        if not count:
            return reference.new_empty((0, width), dtype=torch.float32)
        if torch.is_grad_enabled():
            # Training memory is controlled by StudentPPO, which immediately
            # backwards each physical minibatch rather than retaining all graphs.
            return self._network_forward(obs, critic, reference)
        output = reference.new_empty((count, width), dtype=torch.float32)
        for start in range(0, count, self.inference_chunk_size):
            end = min(count, start + self.inference_chunk_size)
            output[start:end].copy_(self._network_forward(obs[start:end], critic, reference))
        return output

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        if is_teacher:
            raise ValueError("scaled student policy has no teacher branch")
        logits = self._forward(obs)
        # The caller may have an outer AMP context (StudentPPO/evaluation).
        # Explicitly keep masking/gating and Categorical normalization in FP32.
        with torch.autocast(logits.device.type, enabled=False):
            return self._probabilities_from_logits(logits, obs.get("action_mask"), validate_mask=validate_mask)

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["distribution"] = None
        return state
