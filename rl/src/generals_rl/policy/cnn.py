"""Shared trainable CNN actor primitives, without PPO or critic modules."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

MAP_PLANES, MAP_SIZE, PUBLIC_DIM, HISTORY = 12, 9, 13, 4
FRAME_DIM = MAP_PLANES * MAP_SIZE * MAP_SIZE + PUBLIC_DIM
NUM_ACTIONS = 1 + MAP_SIZE * MAP_SIZE * 8


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
    def __init__(self, channels, blocks, trunk_factory=SpatialTrunk, *, history_length=HISTORY):
        super().__init__()
        _validate_history_length(history_length)
        self.history_length = history_length
        self.trunk = trunk_factory(history_length * MAP_PLANES, channels, blocks)
        self.knowledge_channels = 0
        self.context = nn.Sequential(nn.Linear(channels + PUBLIC_DIM, channels), nn.ELU())
        self.cell_head = nn.Conv2d(channels, 8, 1)
        self.wait_head = nn.Sequential(nn.Linear(2 * channels, channels), nn.ELU(), nn.Linear(channels, 1))

    def forward(self, history, knowledge=None):
        if history.ndim != 2 or history.shape[-1] != self.history_length * FRAME_DIM:
            raise ValueError("actor history shape does not match history_length")
        frames = history.reshape(history.shape[0], self.history_length, FRAME_DIM)
        # History is oldest -> newest. Only the newest 13 public features enter
        # context; old public scalars and the redundant single_obs are ignored.
        maps = frames[..., :FRAME_DIM - PUBLIC_DIM].reshape(-1, self.history_length * MAP_PLANES, MAP_SIZE, MAP_SIZE)
        if self.knowledge_channels:
            if knowledge is None or knowledge.shape != (history.shape[0], self.knowledge_channels * MAP_SIZE * MAP_SIZE):
                raise ValueError("actor requires the configured non-stacked knowledge planes")
            maps = torch.cat((maps, knowledge.reshape(-1, self.knowledge_channels, MAP_SIZE, MAP_SIZE)), 1)
        spatial = self.trunk(maps)
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -PUBLIC_DIM:]), dim=-1))
        fused = torch.nn.functional.elu(spatial + context[..., None, None])
        cell_logits = self.cell_head(fused)
        wait = self.wait_head(torch.cat((pooled, context), dim=-1))
        # WAIT=0; move=1+source*8+direction*2+mode, source in row-major order.
        # A channel-major flatten here would silently scramble the action IDs.
        moves = cell_logits.permute(0, 2, 3, 1).reshape(history.shape[0], NUM_ACTIONS - 1)
        return torch.cat((wait, moves), dim=-1)


class OwnKingRelativeGeometry(nn.Module):
    """Append relative coordinates of the latest locally observed own general."""

    def __init__(self, history_length=HISTORY):
        super().__init__()
        _validate_history_length(history_length)
        self.history_length = history_length
        self.knowledge_channels = 0
        self.register_buffer("row", torch.arange(MAP_SIZE, dtype=torch.float32)[:, None]
                             .expand(MAP_SIZE, MAP_SIZE), persistent=False)
        self.register_buffer("column", torch.arange(MAP_SIZE, dtype=torch.float32)[None, :]
                             .expand(MAP_SIZE, MAP_SIZE), persistent=False)

    def forward(self, maps):
        if maps.ndim != 4 or maps.shape[1:] != (self.history_length * MAP_PLANES + self.knowledge_channels, MAP_SIZE, MAP_SIZE):
            raise ValueError("own-king geometry expects history maps plus configured knowledge planes")
        if not maps.is_floating_point():
            raise TypeError("own-king geometry expects floating-point map features")
        current = maps[:, (self.history_length - 1) * MAP_PLANES:self.history_length * MAP_PLANES]
        own_king = (current[:, 4].bool() & current[:, 5].bool()).flatten(1)
        valid = own_king.sum(1).eq(1)
        king = own_king.long().argmax(1)
        dx = (self.column.to(dtype=maps.dtype) - (king % MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
        dy = (self.row.to(dtype=maps.dtype) - (king // MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
        geometry = torch.where(valid[:, None, None, None], torch.stack((dx, dy), 1), 0)
        return torch.cat((maps, geometry), 1)


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


def _unpack_history(history, history_length, consumer):
    if history.ndim != 2 or history.shape[-1] != history_length * FRAME_DIM:
        raise ValueError(f"{consumer} history shape does not match history_length")
    count = history.shape[0]
    frames = history.reshape(count, history_length, FRAME_DIM)
    maps = frames[..., :-PUBLIC_DIM].reshape(count, history_length * MAP_PLANES, MAP_SIZE, MAP_SIZE)
    return frames, maps


def _knowledge_maps(knowledge, channels, count, consumer):
    if knowledge is None or knowledge.shape != (count, channels * MAP_SIZE * MAP_SIZE):
        raise ValueError(f"{consumer} requires the configured non-stacked knowledge planes")
    return knowledge.reshape(count, channels, MAP_SIZE, MAP_SIZE)


def _padding_part(module, buffer_name, count):
    padding = getattr(module, buffer_name)
    return [padding.expand(count, -1, -1, -1)] if padding.shape[1] else []


class _PackedSpatialActor(nn.Module):
    def __init__(self, baseline, alignment):
        super().__init__()
        self.history_length = baseline.history_length
        self.knowledge_channels = baseline.knowledge_channels
        self.trunk, self.context = baseline.trunk, baseline.context
        self.cell_head, self.wait_head = baseline.cell_head, baseline.wait_head
        self._geometry_index = int(isinstance(self.trunk[0], OwnKingRelativeGeometry))
        first = AlignedInputConv2d.from_conv(self.trunk[self._geometry_index], alignment)
        self.trunk[self._geometry_index] = first
        self.register_buffer("_input_zero_padding", first.weight.new_zeros(1, first.input_padding, MAP_SIZE, MAP_SIZE),
                             persistent=False)

    def forward(self, history, knowledge=None, *, unpacked=None):
        frames, maps = unpacked if unpacked is not None else _unpack_history(history, self.history_length, "actor")
        count = history.shape[0]
        parts = [maps]
        if self.knowledge_channels:
            parts.append(_knowledge_maps(knowledge, self.knowledge_channels, count, "actor"))
        if self._geometry_index:
            current = maps[:, (self.history_length - 1) * MAP_PLANES:self.history_length * MAP_PLANES]
            marker = (current[:, 4].bool() & current[:, 5].bool()).flatten(1)
            if not maps.is_floating_point():
                raise TypeError("own-king geometry expects floating-point map features")
            geometry = self.trunk[0]
            king = marker.long().argmax(1)
            dx = (geometry.column.to(dtype=maps.dtype) - (king % MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
            dy = (geometry.row.to(dtype=maps.dtype) - (king // MAP_SIZE)[:, None, None]) / (MAP_SIZE - 1)
            parts.append(torch.where(marker.sum(1).eq(1)[:, None, None, None], torch.stack((dx, dy), 1), 0))
        parts.extend(_padding_part(self, "_input_zero_padding", count))
        # All extra actor planes enter one materialization, in the legacy order.
        spatial = torch.cat(parts, 1)
        for index in range(self._geometry_index, len(self.trunk)):
            spatial = self.trunk[index](spatial)
        pooled = spatial.mean(dim=(-2, -1))
        context = self.context(torch.cat((pooled, frames[:, -1, -PUBLIC_DIM:]), -1))
        fused = F.elu(spatial + context[..., None, None])
        cell_logits = self.cell_head(fused)
        wait = self.wait_head(torch.cat((pooled, context), -1))
        moves = cell_logits.permute(0, 2, 3, 1).reshape(count, NUM_ACTIONS - 1)
        return torch.cat((wait, moves), -1)


def light_trunk(inputs, channels, _residual_blocks):
    return nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.ELU(),
                         nn.Conv2d(channels, channels, 3, padding=1), nn.ELU())


def add_own_king_geometry(actor):
    first, *layers = actor.trunk.children()
    replacement = nn.Conv2d(first.in_channels + 2, first.out_channels, first.kernel_size,
                            stride=first.stride, padding=first.padding,
                            device=first.weight.device, dtype=first.weight.dtype)
    actor.trunk = nn.Sequential(OwnKingRelativeGeometry(actor.history_length), replacement, *layers)


def add_actor_knowledge(actor, knowledge_channels):
    actor.knowledge_channels = knowledge_channels
    if knowledge_channels:
        geometry = isinstance(actor.trunk[0], OwnKingRelativeGeometry)
        index = int(geometry)
        actor.trunk[index] = _extend_conv(actor.trunk[index], knowledge_channels,
                                          trailing_channels=2 if geometry else 0)
        if geometry:
            actor.trunk[0].knowledge_channels = knowledge_channels
