"""Bounded value heads for terminal +/-1 tasks; GAE always receives a scalar.

HL-Gauss follows AverageJoe's linear centers, half-bin CDF integration and
renormalization (sigma in reward units). Truncation makes endpoint targets
slightly inward-biased; this is measured, not silently corrected.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ValueDistribution(nn.Module):
    def __init__(self, kind="scalar", bins=128, sigma=.04):
        super().__init__()
        if kind not in ("scalar", "hl_gauss", "binary_ce"):
            raise ValueError("value_distribution must be scalar, hl_gauss or binary_ce")
        if type(bins) is not int or bins < 2 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("value bins/sigma must be >=2 and finite positive")
        self.kind, self.bins, self.sigma = kind, bins, sigma
        self.outputs = bins if kind == "hl_gauss" else 1
        self.register_buffer("centers", torch.linspace(-1., 1., bins), persistent=False)

    def decode(self, output):
        output = output.float()
        if self.kind == "scalar":
            return output
        if self.kind == "binary_ce":
            return 2 * output.sigmoid() - 1
        return (output.softmax(-1) * self.centers).sum(-1, keepdim=True)

    def targets(self, returns):
        target = returns.detach().float().clamp(-1, 1)
        if self.kind == "binary_ce":
            return (target + 1) / 2
        half = 1. / (self.bins - 1)
        delta = self.centers - target
        upper = torch.erf((delta + half) / (self.sigma * math.sqrt(2)))
        lower = torch.erf((delta - half) / (self.sigma * math.sqrt(2)))
        mass = (upper - lower).clamp_min(0)
        return mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)

    def loss(self, output, returns):
        """FP32 soft CE replaces value MSE/clipping, never the actor objective."""
        output, target = output.float(), self.targets(returns)
        if self.kind == "binary_ce":
            return F.binary_cross_entropy_with_logits(output, target)
        if self.kind == "hl_gauss":
            return -(target * output.log_softmax(-1)).sum(-1).mean()
        raise ValueError("scalar values use the existing PPO regression loss")

    @torch.no_grad()
    def diagnostics(self, output, returns):
        values, target = self.decode(output), returns.float()
        metrics = {"value_mse": (values - target).square().mean(),
                   "value_target_out_of_support": (target.abs() > 1 + 1e-5).float().mean(),
                   "value_abs_prediction": values.abs().mean()}
        if self.kind == "hl_gauss":
            mass = self.targets(target)
            encoded = (mass * self.centers).sum(-1, keepdim=True)
            metrics["value_encoding_abs_bias"] = (encoded - target.clamp(-1, 1)).abs().mean()
            probabilities = output.float().softmax(-1)
            metrics["value_endpoint_mass"] = probabilities[:, [0, -1]].sum(-1).mean()
        return metrics
