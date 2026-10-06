"""Cell-preserving adapters over pinned, mature timm implementations.

Upstream: timm 1.0.30, commit 0df212b369a5385b16dfe513d5143a7311ea1ddc.
The residual/attention blocks are imported unchanged. Only classification
stems/pooling are adapted to dense game grids; policy/value heads live outside
this module. See doc/rl_architecture_design.md for design and provenance.
"""
from __future__ import annotations

from functools import partial

from torch import Tensor, nn
from timm.layers import GroupNormAct
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer


# Full ResNet-50V2 stage pattern, with globally scaled/rounded stage widths.
# Each ResNet preset is (four stage output widths, deep-stem width); ViT presets
# are (embedding width, transformer block count, attention head count).
# Scales describe actor budgets with a small common head, not exact equality
# for every possible observation channel count or critic configuration.
RESNET_DEPTHS = (3, 4, 6, 3)
RESNET_PRESETS = {
    "1m": ((48, 96, 208, 400), 16),
    "10m": ((160, 320, 656, 1312), 40),
    "100m": ((512, 1040, 2064, 4144), 128),
}
VIT_PRESETS = {"1m": (128, 5, 4), "10m": (320, 8, 5), "100m": (896, 10, 14)}


def _check_input(x: Tensor, in_channels: int) -> None:
    if x.ndim != 4 or x.shape[1] != in_channels:
        raise ValueError(f"backbone expects [N,{in_channels},H,W]")
    if x.shape[2] < 1 or x.shape[3] < 1:
        raise ValueError("backbone map dimensions must be positive")


class ResNetBackbone(nn.Module):
    """Full-resolution BiT/ResNetV2 with GroupNorm and standardized weights.

    The complete upstream ResNetV2 constructs its canonical four stages,
    projections, deep stem and initialization. Dense-grid adaptation only
    removes spatial strides/pooling. There are no batch running statistics,
    stochastic depth, or train/eval-dependent layers.
    """

    def __init__(self, in_channels: int, size: str):
        super().__init__()
        self.in_channels = in_channels
        channels, stem_channels = RESNET_PRESETS[size]
        self.out_channels = channels[-1]
        self.depth = sum(RESNET_DEPTHS)
        # Four groups divide all narrow scaled widths, including the deep
        # stem's internal channels; use the same rule across all three sizes.
        self.encoder = ResNetV2(
            layers=list(RESNET_DEPTHS), channels=channels,
            in_chans=in_channels, num_classes=0, global_pool="",
            stem_type="deep", stem_chs=stem_channels, avg_down=False,
            norm_layer=partial(GroupNormAct, num_groups=4),
            output_stride=32, drop_rate=0.0, drop_path_rate=0.0,
            zero_init_last=False,
        )
        # Construct at the canonical output_stride so upstream does not replace
        # strides with dilation. Then preserve every game cell by removing only
        # spatial downsampling, including each stage's official shortcut conv.
        self.encoder.stem.pool = nn.Identity()
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)
        for feature in self.encoder.feature_info:
            feature["reduction"] = 1

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.in_channels)
        return self.encoder.forward_features(x)


class ViTBackbone(nn.Module):
    """DeiT-III-style Pre-LN/LayerScale ViT, one token per game cell.

    timm's dynamic image-size path interpolates the learned 2D positional table
    while leaving game observations untouched. Accepting new sizes does not
    establish generalization to maps absent from training. Homogeneous-size
    batches need no padding/attention masks.
    """

    def __init__(self, in_channels: int, size: str):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels, self.depth, self.num_heads = VIT_PRESETS[size]
        self.transformer = VisionTransformer(
            img_size=(9, 9), patch_size=1, in_chans=in_channels,
            num_classes=0, global_pool="", class_token=False,
            embed_dim=self.out_channels, depth=self.depth,
            num_heads=self.num_heads, mlp_ratio=4.0, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=1e-4, dynamic_img_size=True,
            drop_rate=0.0, pos_drop_rate=0.0, patch_drop_rate=0.0,
            proj_drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.in_channels)
        n, _, h, w = x.shape
        tokens = self.transformer.forward_features(x)
        return tokens.transpose(1, 2).reshape(n, self.out_channels, h, w)


def build_backbone(kind: str, size: str, in_channels: int) -> nn.Module:
    """Construct ``resnet``/``vit`` at ``1m``/``10m``/``100m`` capacity.

    Inputs and outputs preserve H/W at runtime, including rectangular maps.
    Call under ``torch.device('meta')`` for allocation-free parameter auditing.
    """
    if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels < 1:
        raise ValueError("in_channels must be a positive integer")
    if not isinstance(size, str) or size.lower() not in RESNET_PRESETS:
        raise ValueError("backbone size must be 1m, 10m, or 100m")
    constructors = {"resnet": ResNetBackbone, "vit": ViTBackbone}
    if kind not in constructors:
        raise ValueError("backbone kind must be resnet or vit")
    return constructors[kind](in_channels, size.lower())
