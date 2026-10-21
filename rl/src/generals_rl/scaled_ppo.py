"""Balanced total actor+critic budgets for the H4/K11 v32 information task.

The six targets reuse pinned timm ResNetV2/DeiT-III blocks and established
cell-preserving adapters. Budgets count BOTH independently trainable branches,
including observation projections and heads. Default projection32 counts:

             actor       critic       total
ResNet 1m      514,537      536,737    1,051,274
ResNet 10m   5,079,049    5,098,945   10,177,994
ResNet 100m 49,046,457   49,058,289   98,104,746
ViT 1m         526,985      548,417    1,075,402
ViT 10m      4,957,961    4,975,297    9,933,258
ViT 100m    49,477,001   49,481,025   98,958,026

Projection0 retains all raw history/knowledge and adds 5,856 critic parameters.
H8 uses the same layout with eight chronological frames and wider input layers;
the tabulated budgets and the checkpoint-compatible default remain H4.
Rollout chunks bound hidden activation memory; PPO training uses StudentPPO's
physical_minibatch_size to backward/release each graph before the next chunk.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache, partial
from types import FunctionType

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import GroupNormAct
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer

from .backbones import RESNET_DEPTHS, ResNetBackbone, ViTBackbone
from .map_ppo import MapPPOPolicy
from .mdp.layout import ObservationLayout, own_king_coordinates
from .optimized_ppo import AlignedInputConv2d


# Each preset applies equally to the actor and the independent critic. ResNet
# uses the unchanged 16-bottleneck [3,4,6,3] stage pattern at every capacity.
RESNET_TOTAL_PRESETS = {
    "1m": ((32, 64, 128, 272), 16),
    "10m": ((112, 224, 464, 928), 32),
    "100m": ((368, 736, 1472, 2944), 88),
}
VIT_TOTAL_PRESETS = {"1m": (96, 4, 3), "10m": (224, 8, 7), "100m": (640, 10, 10)}
DEFAULT_PARAMETER_COUNTS = {
    ("resnet", "1m"): (514_537, 536_737),
    ("resnet", "10m"): (5_079_049, 5_098_945),
    ("resnet", "100m"): (49_046_457, 49_058_289),
    ("vit", "1m"): (526_985, 548_417),
    ("vit", "10m"): (4_957_961, 4_975_297),
    ("vit", "100m"): (49_477_001, 49_481_025),
}


class _TotalResNetBackbone(ResNetBackbone):
    """Balanced-budget ResNetV2 with optional coarse stages and a dense skip."""
    def __init__(self, in_channels, size, spatial_mode="dense"):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.spatial_mode = spatial_mode
        channels, stem_channels = RESNET_TOTAL_PRESETS[size]
        self.local_channels = channels[0]
        self.out_channels, self.depth = channels[-1], sum(RESNET_DEPTHS)
        self.encoder = ResNetV2(
            layers=list(RESNET_DEPTHS), channels=channels, in_chans=in_channels,
            num_classes=0, global_pool="", stem_type="deep", stem_chs=stem_channels,
            avg_down=False, norm_layer=partial(GroupNormAct, num_groups=4),
            output_stride=32, drop_rate=0.0, drop_path_rate=0.0, zero_init_last=False)
        self.encoder.stem.pool = nn.Identity()
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)
        for feature in self.encoder.feature_info:
            feature["reduction"] = 1
        if spatial_mode == "pyramid":
            for stage in list(self.encoder.stages)[1:]:
                stage.blocks[0].conv2.stride = (2, 2)
                stage.blocks[0].downsample.conv.stride = (2, 2)
            for feature in self.encoder.feature_info:
                if feature["module"].startswith("stages."):
                    feature["reduction"] = 2 ** int(feature["module"].rsplit(".", 1)[1])

    def forward_pyramid(self, x):
        """Return full-resolution stage0 and normalized coarse deep features."""
        x = self.encoder.stem(x)
        local = self.encoder.stages[0](x)
        x = local
        for stage in list(self.encoder.stages)[1:]:
            x = stage(x)
        return local, self.encoder.norm(x)


class _TotalViTBackbone(ViTBackbone):
    """Existing one-token-per-cell timm adapter at balanced total budgets."""
    def __init__(self, in_channels, size, spatial_mode="dense"):
        nn.Module.__init__(self)
        if spatial_mode not in ("dense", "pyramid"):
            raise ValueError("spatial_mode must be dense or pyramid")
        self.spatial_mode = spatial_mode
        self.in_channels = in_channels
        self.out_channels, self.depth, self.num_heads = VIT_TOTAL_PRESETS[size]
        self.local_channels = self.out_channels
        self.transformer = VisionTransformer(
            img_size=(9, 9), patch_size=1, in_chans=in_channels,
            num_classes=0, global_pool="", class_token=False,
            embed_dim=self.out_channels, depth=self.depth, num_heads=self.num_heads,
            mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=1e-4, dynamic_img_size=True,
            drop_rate=0.0, pos_drop_rate=0.0, patch_drop_rate=0.0,
            proj_drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0)


    def forward_pyramid(self, x):
        """All blocks survive; block0 keeps81 cells, then deep tokens pool25→9.

        This changes spatial computation. It does not establish equal quality
        or measured efficiency. Dense forward remains the inherited ViT path.
        """
        if self.spatial_mode != "pyramid":
            raise ValueError("forward_pyramid requires spatial_mode=pyramid")
        if x.ndim != 4 or x.shape[1] != self.in_channels or x.shape[-2:] != (9, 9):
            raise ValueError("ViT pyramid requires packed NCHW9x9 inputs")
        transformer = self.transformer
        count = x.shape[0]
        tokens = transformer.patch_embed(x)
        # Add the original complete81-entry position table before either pool.
        tokens = transformer.norm_pre(transformer.patch_drop(transformer._pos_embed(tokens)))
        tokens = transformer.blocks[0](tokens)
        local = tokens.transpose(1, 2).reshape(count, self.out_channels, 9, 9)
        coarse = F.avg_pool2d(local, 3, stride=2, padding=1, count_include_pad=False)
        tokens = coarse.flatten(2).transpose(1, 2)
        middle_blocks = 1 if self.depth == 4 else 2
        for index in range(1, self.depth):
            if index == middle_blocks + 1:
                coarse = tokens.transpose(1, 2).reshape(count, self.out_channels, 5, 5)
                coarse = F.avg_pool2d(coarse, 3, stride=2, padding=1, count_include_pad=False)
                tokens = coarse.flatten(2).transpose(1, 2)
            tokens = transformer.blocks[index](tokens)
        deep = transformer.norm(tokens).transpose(1, 2).reshape(count, self.out_channels, 3, 3)
        return local, deep


def _build_total_backbone(kind, size, inputs, spatial_mode="dense"):
    return {"resnet": _TotalResNetBackbone, "vit": _TotalViTBackbone}[kind](inputs, size, spatial_mode)


class _ScaledInformationActor(nn.Module):
    def __init__(self, layout, kind, size, alignment, head_channels, spatial_mode="dense"):
        super().__init__()
        self.layout = layout
        self.spatial_mode = spatial_mode
        self.knowledge_channels = layout.knowledge_channels
        self.logical_inputs = layout.history_length * layout.map_planes + layout.knowledge_channels + 2
        padding = (-self.logical_inputs) % alignment if alignment else 0
        self.backbone = _build_total_backbone(kind, size, self.logical_inputs + padding, spatial_mode)
        self.projection = nn.Conv2d(self.backbone.out_channels, head_channels, 1)
        self.context = nn.Sequential(nn.Linear(head_channels + layout.public_dim, head_channels), nn.SiLU())
        self.cell_head = nn.Conv2d(head_channels, 8, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * head_channels, head_channels), nn.SiLU(), nn.Linear(head_channels, 1))
        self.register_buffer("_input_zero_padding", self.projection.weight.new_zeros(
            1, padding, layout.height, layout.width), persistent=False)
        if spatial_mode == "pyramid":
            self.local_skip = nn.Conv2d(self.backbone.local_channels, head_channels, 1)

    def forward(self, history, knowledge=None):
        layout, count = self.layout, history.shape[0]
        frames = layout.frames(history)
        maps = layout.history_maps(history)
        parts = [maps, layout.memory_maps(knowledge, count), own_king_coordinates(maps[:, -layout.map_planes:])]
        if self._input_zero_padding.shape[1]:
            parts.append(self._input_zero_padding.expand(count, -1, -1, -1))
        if self.spatial_mode == "pyramid":
            local, deep = self.backbone.forward_pyramid(torch.cat(parts, 1))
            coarse = self.projection(deep)
            if (coarse.is_cuda and torch.is_grad_enabled()
                    and isinstance(self.backbone, _TotalResNetBackbone)):
                coarse = coarse.contiguous(memory_format=torch.channels_last)
            spatial = F.silu(F.interpolate(coarse, size=(layout.height, layout.width),
                                          mode="bilinear", align_corners=False) + self.local_skip(local))
        else:
            spatial = F.silu(self.projection(self.backbone(torch.cat(parts, 1))))
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -layout.public_dim:]), -1))
        fused = F.silu(spatial + context[..., None, None])
        moves = self.cell_head(fused).permute(0, 2, 3, 1).reshape(count, 8 * layout.cells)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        return torch.cat((wait, moves), -1)


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
        self.backbone = _build_total_backbone(kind, size, 32, spatial_mode)
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


class ScaledInformationSpatialPPOPolicy(MapPPOPolicy):
    """Mature dense backbones, balanced TOTAL capacity and v32 information/action semantics.

    No BatchNorm, nonzero Dropout, or stochastic depth is present. GroupNorm and
    LayerNorm preserve independent samples across opponent groups and PPO chunks.
    Master parameters remain float32; CUDA bf16 autocasts only network forwards,
    then logits/probabilities/log-probabilities and values return to float32.
    """
    semantics = "scaled_v32_p2_9x9_hist4_knowledge11_independent_critic_v1"

    def __init__(self, obs, obs_groups, num_actions, *, backbone="resnet", model_size="1m",
                 map_width=9, map_height=9, history_length=4, players=2,
                 map_planes=12, public_dim=13, knowledge_channels=11,
                 critic_actor_information=True, own_king_geometry=True,
                 critic_projection_channels=32, input_channel_alignment=4,
                 head_channels=128, action_distribution="categorical",
                 actor_obs_normalization=False, critic_obs_normalization=False,
                 wait_move_factorization=True, value_potential_offset=False,
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
        schema = (players, map_width, map_height, map_planes, public_dim, knowledge_channels)
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in schema)
                or schema != (2, 9, 9, 12, 13, 11)
                or isinstance(history_length, bool) or not isinstance(history_length, int)
                or history_length not in (4, 8)):
            raise ValueError("scaled policy requires the v32 P2 9x9 H4 or H8/K11 twelve-plane/public13 schema")
        if action_distribution != "categorical" or actor_obs_normalization is not False or critic_obs_normalization is not False:
            raise ValueError("scaled policy uses categorical actions and unnormalized explicit features")
        for name, value in (("critic_actor_information", critic_actor_information), ("own_king_geometry", own_king_geometry),
                            ("wait_move_factorization", wait_move_factorization), ("value_potential_offset", value_potential_offset),
                            ("amp_bfloat16", amp_bfloat16), ("compile_forward", compile_forward)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if value_potential_offset:
            raise ValueError("analytic value_potential_offset is retired; expected False")
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
        self.actor_obs_normalizer = self.single_obs_normalizer = self.critic_obs_normalizer = nn.Identity()
        self.teacher_encoder = nn.Identity()
        self.critic_actor_information, self.own_king_geometry = critic_actor_information, own_king_geometry
        self.wait_move_factorization, self.value_potential_offset = wait_move_factorization, value_potential_offset
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
        self.semantics = self.semantics.replace("hist4", f"hist{history_length}")
        self.semantics += f"_{backbone}_{self.model_size}_projection{critic_projection_channels}_head{head_channels}"
        if spatial_mode == "pyramid":
            self.semantics += "_spatial_pyramid_v1"

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

    def set_geometry(self, width, height):
        if (width, height) != (9, 9):
            raise ValueError("scaled v32 policy requires 9x9 maps")

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["distribution"] = None
        return state
