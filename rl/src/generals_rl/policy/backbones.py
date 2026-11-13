"""Dense-grid forward adapter for the timm ViT backbone."""
from __future__ import annotations

from torch import Tensor, nn


def _check_input(x: Tensor, in_channels: int) -> None:
    if x.ndim != 4 or x.shape[1] != in_channels:
        raise ValueError(f"backbone expects [N,{in_channels},H,W]")
    if x.shape[2] < 1 or x.shape[3] < 1:
        raise ValueError("backbone map dimensions must be positive")




class ViTBackbone(nn.Module):
    """Non-overlapping patch-grid ViT forward; construction lives in policy.scaled."""

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, self.in_channels)
        n, _, h, w = x.shape
        h, w = self.transformer.patch_embed.dynamic_feat_size((h, w))
        tokens = self.transformer.forward_features(x)
        return tokens.transpose(1, 2).reshape(n, self.out_channels, h, w)
