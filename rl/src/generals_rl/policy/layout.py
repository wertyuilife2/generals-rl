"""Observation and action geometry, independent of a network architecture."""
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class ObservationLayout:
    width: int
    height: int
    history_length: int = 4
    map_planes: int = 12
    public_dim: int = 13
    knowledge_channels: int = 0

    def __post_init__(self):
        for name in ("width", "height", "history_length", "map_planes", "public_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.knowledge_channels, bool) or not isinstance(self.knowledge_channels, int) or self.knowledge_channels < 0:
            raise ValueError("knowledge_channels must be a nonnegative integer")

    @property
    def cells(self):
        return self.width * self.height

    @property
    def frame_dim(self):
        return self.map_planes * self.cells + self.public_dim

    @property
    def history_dim(self):
        return self.history_length * self.frame_dim

    @property
    def knowledge_dim(self):
        return self.knowledge_channels * self.cells

    @property
    def num_actions(self):
        return 1 + 8 * self.cells

    def frames(self, history):
        if history.ndim != 2 or history.shape[-1] != self.history_dim:
            raise ValueError("policy history shape does not match observation layout")
        return history.reshape(history.shape[0], self.history_length, self.frame_dim)

    def history_maps(self, history):
        frames = self.frames(history)
        return frames[..., :self.map_planes * self.cells].reshape(
            history.shape[0], self.history_length * self.map_planes, self.height, self.width)

    def current_maps(self, frame):
        if frame.ndim != 2 or frame.shape[-1] != self.frame_dim:
            raise ValueError("single observation shape does not match observation layout")
        return frame[:, :self.map_planes * self.cells].reshape(
            frame.shape[0], self.map_planes, self.height, self.width)

    def memory_maps(self, knowledge, batch_size):
        if knowledge is None or knowledge.shape != (batch_size, self.knowledge_dim):
            raise ValueError("knowledge shape does not match observation layout")
        return knowledge.reshape(batch_size, self.knowledge_channels, self.height, self.width)

    def with_geometry(self, width, height):
        return replace(self, width=width, height=height)


def own_king_coordinates(current_maps):
    """Relative XY derived exclusively from the supplied current observation."""
    n, _, height, width = current_maps.shape
    marker = (current_maps[:, 4].bool() & current_maps[:, 5].bool()).flatten(1)
    valid = marker.sum(1).eq(1)
    position = marker.long().argmax(1)
    rows = torch.arange(height, device=current_maps.device, dtype=current_maps.dtype)
    cols = torch.arange(width, device=current_maps.device, dtype=current_maps.dtype)
    dx = (cols[None, None, :] - (position % width)[:, None, None]) / max(1, width - 1)
    dy = (rows[None, :, None] - (position // width)[:, None, None]) / max(1, height - 1)
    geometry = torch.stack((dx.expand(n, height, width), dy.expand(n, height, width)), 1)
    return torch.where(valid[:, None, None, None], geometry, 0)
