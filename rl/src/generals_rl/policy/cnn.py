"""Shared trainable CNN actor primitives, without PPO or critic modules."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from .layout import own_king_coordinates


def _validate_history_length(history_length):
    if not isinstance(history_length, int) or isinstance(history_length, bool) or history_length < 1:
        raise ValueError("history_length must be a positive integer")


class SpatialTrunk(nn.Module):
    def __init__(self, inputs, channels, blocks):
        super().__init__()
        self.projection = nn.Conv2d(inputs, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                          nn.ELU(), nn.Conv2d(channels, channels, 1))
            for _ in range(blocks))

    def forward(self, maps):
        features = torch.nn.functional.elu(self.projection(maps))
        for block in self.blocks:
            features = torch.nn.functional.elu(features + block(features))
        return features


class SpatialActor(nn.Module):
    def __init__(self, channels, blocks, trunk_factory=SpatialTrunk, *, layout):
        super().__init__()
        self.layout = layout
        self.history_length = layout.history_length
        self.trunk = trunk_factory(layout.history_length * layout.map_planes, channels, blocks)
        self.knowledge_channels = 0
        self.context = nn.Sequential(nn.Linear(channels + layout.public_dim, channels), nn.ELU())
        self.cell_head = nn.Conv2d(channels, 8, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ELU(), nn.Linear(channels, 1))

    def forward(self, history, knowledge=None):
        layout = self.layout
        frames, maps = _unpack_history(history, layout)
        if self.knowledge_channels:
            maps = torch.cat((maps, _knowledge_maps(knowledge, self.knowledge_channels,
                                                   history.shape[0], "actor", layout)), 1)
        spatial = self.trunk(maps)
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -layout.public_dim:]), -1))
        fused = F.elu(spatial + context[..., None, None])
        moves = self.cell_head(fused).permute(0, 2, 3, 1).reshape(history.shape[0], layout.num_actions - 1)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        return torch.cat((wait, moves), -1)


class OwnKingRelativeGeometry(nn.Module):
    """Coordinates derived only from the latest locally observed own general."""
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.knowledge_channels = 0

    def forward(self, maps):
        layout = self.layout
        start = (layout.history_length - 1) * layout.map_planes
        return torch.cat((maps, own_king_coordinates(maps[:, start:start + layout.map_planes])), 1)


def _extend_conv(old, extra_channels, trailing_channels=0):
    """Preserve old inputs and append zero-initialized information before XY."""
    devices = [old.weight.device.index] if old.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        new = nn.Conv2d(old.in_channels + extra_channels, old.out_channels,
                        old.kernel_size, stride=old.stride, padding=old.padding,
                        dilation=old.dilation, groups=old.groups,
                        bias=old.bias is not None, padding_mode=old.padding_mode,
                        device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        prefix = old.in_channels - trailing_channels
        new.weight[:, :prefix].copy_(old.weight[:, :prefix])
        if trailing_channels:
            new.weight[:, -trailing_channels:].copy_(old.weight[:, -trailing_channels:])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


class AlignedInputConv2d(nn.Conv2d):
    """Keep logical/checkpoint weights; convolve prepacked, zero-aligned inputs."""

    def __init__(self, *args, input_channel_alignment=4, **kwargs):
        super().__init__(*args, **kwargs)
        if self.groups != 1:
            raise ValueError("input alignment currently requires groups=1")
        self.input_channel_alignment = input_channel_alignment
        self.input_padding = (-self.in_channels) % input_channel_alignment if input_channel_alignment else 0
        self.compute_in_channels = self.in_channels + self.input_padding

    @classmethod
    def from_conv(cls, old, alignment):
        devices = [old.weight.device.index] if old.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            new = cls(old.in_channels, old.out_channels, old.kernel_size, stride=old.stride,
                      padding=old.padding, dilation=old.dilation, groups=old.groups,
                      bias=old.bias is not None, padding_mode=old.padding_mode,
                      device=old.weight.device, dtype=old.weight.dtype,
                      input_channel_alignment=alignment)
        new.weight, new.bias = old.weight, old.bias
        new.train(old.training)
        return new

    def forward(self, inputs):
        weight = F.pad(self.weight, (0, 0, 0, 0, 0, self.input_padding)) if self.input_padding else self.weight
        return self._conv_forward(inputs, weight, self.bias)


def _unpack_history(history, layout):
    frames = layout.frames(history)
    maps = frames[..., :layout.map_planes * layout.cells].reshape(
        history.shape[0], layout.history_length * layout.map_planes, layout.height, layout.width)
    return frames, maps


def _knowledge_maps(knowledge, channels, count, consumer, layout):
    if knowledge is None or knowledge.shape != (count, channels * layout.cells):
        raise ValueError(f"{consumer} requires the configured non-stacked knowledge planes")
    return knowledge.reshape(count, channels, layout.height, layout.width)


def _padding_part(module, buffer_name, count):
    padding = getattr(module, buffer_name)
    return [padding.expand(count, -1, -1, -1)] if padding.shape[1] else []


class _PackedSpatialActor(nn.Module):
    def __init__(self, baseline, alignment):
        super().__init__()
        self.layout = baseline.layout
        self.history_length = baseline.history_length
        self.knowledge_channels = baseline.knowledge_channels
        self.trunk, self.context = baseline.trunk, baseline.context
        self.cell_head, self.wait_head = baseline.cell_head, baseline.wait_head
        self._geometry_index = int(isinstance(self.trunk[0], OwnKingRelativeGeometry))
        first = AlignedInputConv2d.from_conv(self.trunk[self._geometry_index], alignment)
        self.trunk[self._geometry_index] = first
        self.register_buffer("_input_zero_padding", first.weight.new_zeros(
            1, first.input_padding, self.layout.height, self.layout.width), persistent=False)

    def forward(self, history, knowledge=None, *, unpacked=None):
        layout = self.layout
        frames, maps = unpacked if unpacked is not None else _unpack_history(history, layout)
        count = history.shape[0]
        parts = [maps]
        if self.knowledge_channels:
            parts.append(_knowledge_maps(knowledge, self.knowledge_channels, count, "actor", layout))
        if self._geometry_index:
            start = (layout.history_length - 1) * layout.map_planes
            parts.append(own_king_coordinates(maps[:, start:start + layout.map_planes]))
        parts.extend(_padding_part(self, "_input_zero_padding", count))
        spatial = torch.cat(parts, 1)
        for index in range(self._geometry_index, len(self.trunk)):
            spatial = self.trunk[index](spatial)
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -layout.public_dim:]), -1))
        fused = F.elu(spatial + context[..., None, None])
        moves = self.cell_head(fused).permute(0, 2, 3, 1).reshape(count, layout.num_actions - 1)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        return torch.cat((wait, moves), -1)


def light_trunk(inputs, channels, _residual_blocks):
    return nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.ELU(),
                         nn.Conv2d(channels, channels, 3, padding=1), nn.ELU())


def add_own_king_geometry(actor):
    first, *layers = actor.trunk.children()
    replacement = nn.Conv2d(first.in_channels + 2, first.out_channels, first.kernel_size,
                            stride=first.stride, padding=first.padding,
                            device=first.weight.device, dtype=first.weight.dtype)
    actor.trunk = nn.Sequential(OwnKingRelativeGeometry(actor.layout), replacement, *layers)


def add_actor_knowledge(actor, knowledge_channels):
    actor.knowledge_channels = knowledge_channels
    if knowledge_channels:
        geometry = isinstance(actor.trunk[0], OwnKingRelativeGeometry)
        index = int(geometry)
        actor.trunk[index] = _extend_conv(actor.trunk[index], knowledge_channels,
                                          trailing_channels=2 if geometry else 0)
        if geometry:
            actor.trunk[0].knowledge_channels = knowledge_channels
