"""Shared trainable scaled actor/backbones; no PPO, optimizer or critic."""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import GroupNormAct
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer
from .backbones import RESNET_DEPTHS, ResNetBackbone, ViTBackbone
from .layout import own_king_coordinates


RESNET_TOTAL_PRESETS = {
    "1m": ((32, 64, 128, 272), 16),
    "10m": ((112, 224, 464, 928), 32),
    "100m": ((368, 736, 1472, 2944), 88),
}


VIT_TOTAL_PRESETS = {"1m": (96, 4, 3), "10m": (224, 8, 7), "100m": (640, 10, 10)}


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
        if not count:
            layout.memory_maps(knowledge, 0)
            return history.new_empty((0, layout.num_actions))
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
