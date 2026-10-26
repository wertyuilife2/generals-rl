"""Dense-grid forward adapters shared by balanced-budget timm backbones.

Scaled policy constructors own the supported total actor+critic budgets. The
underlying ResNetV2 and DeiT-III blocks remain pinned to timm 1.0.30.
"""
from __future__ import annotations

from torch import Tensor, nn


RESNET_DEPTHS = (3, 4, 6, 3)


def _check_input(x: Tensor, in_channels: int) -> None:
    if x.ndim != 4 or x.shape[1] != in_channels:
        raise ValueError(f"backbone expects [N,{in_channels},H,W]")
    if x.shape[2] < 1 or x.shape[3] < 1:
        raise ValueError("backbone map dimensions must be positive")


class ResNetBackbone(nn.Module):
    """Cell-preserving ResNetV2 forward; construction lives in scaled_ppo."""

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.in_channels)
        return self.encoder.forward_features(x)


class ViTBackbone(nn.Module):
    """One-token-per-cell ViT forward; construction lives in scaled_ppo."""

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.in_channels)
        n, _, h, w = x.shape
        tokens = self.transformer.forward_features(x)
        return tokens.transpose(1, 2).reshape(n, self.out_channels, h, w)
