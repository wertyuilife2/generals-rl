"""Lossless storage of P2 fourteen-plane observations; no model/input changes."""
import torch
import torch.nn.functional as F
from tensordict import TensorDict


def _pack_maps(value, cells):
    frames = value.unflatten(-1, (-1, 14 * cells + 13))
    maps = frames[..., :14 * cells].unflatten(-1, (14, cells))
    flags = torch.zeros_like(maps[..., 0, :], dtype=torch.int16)
    for bit, channel in enumerate((*range(9), 12, 13)):
        flags = flags | (maps[..., channel, :].to(torch.int16) << bit)
    return flags, maps[..., 9:12, :].contiguous(), frames[..., 14 * cells:].contiguous()


def _gather_maps(flags, continuous, public, times, envs):
    flags, continuous, public = flags[times, envs], continuous[times, envs], public[times, envs]
    binary = [((flags >> bit) & 1).float() for bit in range(11)]
    maps = torch.stack((*binary[:9], *continuous.unbind(-2), *binary[9:]), -2).flatten(-2)
    return torch.cat((maps, public), -1).flatten(-2)


def _pack_knowledge(value, cells):
    planes = value.unflatten(-1, (11, cells))
    flags = torch.zeros_like(planes[..., 0, :], dtype=torch.int16)
    for bit in range(10):
        flags = flags | (planes[..., bit, :].to(torch.int16) << bit)
    return flags, planes[..., 10, :]


def _gather_knowledge(flags, age, times, envs):
    flags, age = flags[times, envs], age[times, envs]
    return torch.stack((*[((flags >> bit) & 1).float() for bit in range(10)], age), -2).flatten(-2)


def _pack_mask(value):
    padded = F.pad(value, (0, (-value.shape[-1]) % 8)).unflatten(-1, (-1, 8))
    shifts = torch.arange(8, device=value.device, dtype=torch.int32)
    return (padded.to(torch.int32) << shifts).sum(-1, dtype=torch.int32).to(torch.uint8)


def _gather_mask(value, times, envs, actions):
    packed = value[times, envs]
    shifts = torch.arange(8, device=value.device, dtype=torch.int32)
    return (((packed[..., None].to(torch.int32) >> shifts) & 1).flatten(-2)[..., :actions]).bool()


class GeneralsObservationCodec:
    """Pack binary planes/masks; retain every continuous/public value in FP32.

    CUDA compiles each pack and fused gather/decode operation. CPU uses the same
    tensor expressions. The codec applies only to rollout storage, not inference
    packages, live task observations or the environment protocol.
    """

    def __init__(self, observations):
        self.actions = observations['action_mask'].shape[-1]
        self.cells = (self.actions - 1) // 8
        if self.cells < 1 or self.actions != 1 + 8 * self.cells:
            raise ValueError('codec requires encoded Generals actions')
        self.frame_dim = 14 * self.cells + 13
        self.map_keys = {'policy', 'critic', 'single_obs'} & set(observations.keys())
        for key in self.map_keys:
            value = observations[key]
            if value.dtype != torch.float32 or value.shape[-1] % self.frame_dim:
                raise ValueError('codec requires fourteen-plane/public13 float32 observations')
        if 'knowledge' in observations and (observations['knowledge'].dtype != torch.float32
                or observations['knowledge'].shape[-1] != 11 * self.cells):
            raise ValueError('codec requires K11 float32 knowledge')
        functions = (_pack_maps, _gather_maps, _pack_knowledge, _gather_knowledge, _pack_mask, _gather_mask)
        if observations['action_mask'].device.type == 'cuda':
            functions = tuple(torch.compile(fn, fullgraph=True, dynamic=True) for fn in functions)
        (self._pack_maps, self._gather_maps, self._pack_knowledge,
         self._gather_knowledge, self._pack_mask, self._gather_mask) = functions

    def encode(self, key, value):
        if key in self.map_keys:
            flags, continuous, public = self._pack_maps(value, self.cells)
            return TensorDict(dict(flags=flags, continuous=continuous, public=public),
                              batch_size=value.shape[:-1], device=value.device)
        if key == 'knowledge':
            flags, age = self._pack_knowledge(value, self.cells)
            return TensorDict(dict(flags=flags, age=age), batch_size=value.shape[:-1], device=value.device)
        if key == 'action_mask':
            return self._pack_mask(value)
        return value

    def gather(self, key, value, times, envs):
        if key in self.map_keys:
            return self._gather_maps(value['flags'], value['continuous'], value['public'], times, envs)
        if key == 'knowledge':
            return self._gather_knowledge(value['flags'], value['age'], times, envs)
        if key == 'action_mask':
            return self._gather_mask(value, times, envs, self.actions)
        return value[times, envs]
