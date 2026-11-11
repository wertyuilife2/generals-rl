"""Lossless storage of local13 and privileged11 observations."""
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .policy.features import feature_recipe


def _pack_maps(value, cells, planes):
    frames = value.unflatten(-1, (-1, planes * cells + 13))
    maps = frames[..., :planes * cells].unflatten(-1, (planes, cells))
    private = planes == 11
    dtype = torch.uint8 if private else torch.int16
    channels = range(8) if private else (*range(9), 12)
    continuous = 8 if private else 9
    flags = torch.zeros_like(maps[..., 0, :], dtype=dtype)
    for bit, channel in enumerate(channels):
        flags = flags | (maps[..., channel, :].to(dtype) << bit)
    return flags, maps[..., continuous:continuous + 3, :].contiguous(), frames[..., planes * cells:].contiguous()


def _gather_maps(flags, continuous, public, times, envs, planes):
    flags, continuous, public = flags[times, envs], continuous[times, envs], public[times, envs]
    offset, bits = (8, 8) if planes == 11 else (9, 10)
    binary = [((flags >> bit) & 1).float() for bit in range(bits)]
    maps = torch.stack((*binary[:offset], *continuous.unbind(-2), *binary[offset:]), -2).flatten(-2)
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


def _gather_mask(value, times, envs, size):
    packed = value[times, envs]
    shifts = torch.arange(8, device=value.device, dtype=torch.int32)
    return (((packed[..., None].to(torch.int32) >> shifts) & 1).flatten(-2)[..., :size]).bool()


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
        self.mask_sizes = {key: observations[key].shape[-1] for key in ("action_mask", "valid_cells") if key in observations}
        self.map_planes = {key: 11 if key == 'critic' else 13
                           for key in {'policy', 'critic', 'single_obs'} & set(observations.keys())}
        for key, planes in self.map_planes.items():
            value = observations[key]
            if value.dtype != torch.float32 or value.shape[-1] % (planes * self.cells + 13):
                raise ValueError('codec requires local13/privileged11/public13 float32 observations')
        if 'knowledge' in observations and (observations['knowledge'].dtype != torch.float32
                or observations['knowledge'].shape[-1] != 11 * self.cells):
            raise ValueError('codec requires K11 float32 knowledge')
        functions = (_pack_maps, _gather_maps, _pack_knowledge, _gather_knowledge, _pack_mask, _gather_mask)
        if observations['action_mask'].device.type == 'cuda':
            functions = tuple(torch.compile(fn, fullgraph=True, dynamic=True) for fn in functions)
        (self._pack_maps, self._gather_maps, self._pack_knowledge,
         self._gather_knowledge, self._pack_mask, self._gather_mask) = functions

    def encode(self, key, value):
        if key in self.map_planes:
            flags, continuous, public = self._pack_maps(value, self.cells, self.map_planes[key])
            return TensorDict(dict(flags=flags, continuous=continuous, public=public),
                              batch_size=value.shape[:-1], device=value.device)
        if key == 'knowledge':
            flags, age = self._pack_knowledge(value, self.cells)
            return TensorDict(dict(flags=flags, age=age), batch_size=value.shape[:-1], device=value.device)
        if key in self.mask_sizes:
            return self._pack_mask(value)
        return value

    def gather(self, key, value, times, envs):
        if key in self.map_planes:
            return self._gather_maps(value['flags'], value['continuous'], value['public'], times, envs, self.map_planes[key])
        if key == 'knowledge':
            return self._gather_knowledge(value['flags'], value['age'], times, envs)
        if key in self.mask_sizes:
            return self._gather_mask(value, times, envs, self.mask_sizes[key])
        return value[times, envs]


def _pack_half_maps(value, cells, planes):
    flags, continuous, public = _pack_maps(value, cells, planes)
    return flags, continuous.half(), public.half()


def _pack_half_knowledge(value, cells):
    flags, age = _pack_knowledge(value, cells)
    return flags, age.half()


class CompactObservationCodec(GeneralsObservationCodec):
    """Bit-packed flags plus FP16 normalized features, quantized before acting.

    Combat risk/margin are continuous: never reinterpret them as identity bits.
    Decoding returns the exact float32 values seen by the rollout policy.
    """
    dynamic_memory = False
    feature_set = 'compact'

    def __init__(self, observations):
        # Extended memory is encoded separately; validate its standard prefix.
        initial = observations.clone(recurse=False)
        if self.dynamic_memory:
            cells = (observations['action_mask'].shape[-1] - 1) // 8
            recipe = feature_recipe(self.feature_set)
            if observations['knowledge'].shape[-1] != recipe.knowledge_channels * cells + recipe.public_extra_dim:
                raise ValueError('dynamic memory does not match the feature recipe')
            initial['knowledge'] = observations['knowledge'][..., :11*cells]
        super().__init__(initial)
        for key in self.map_planes.keys() | {'knowledge'}:
            value = observations[key]
            exact = (torch.isfinite(value) & (value == value.half().float())).all()
            if value.is_cuda:
                torch._assert_async(exact, 'compact observations must be quantized before acting')
            elif not bool(exact):
                raise ValueError('compact observations must be quantized before acting')
        functions = (_pack_half_maps, _pack_half_knowledge)
        if observations['action_mask'].is_cuda:
            functions = tuple(torch.compile(fn, fullgraph=True, dynamic=True) for fn in functions)
        self._pack_maps, self._pack_knowledge = functions

    def encode(self, key, value):
        base = value[..., :11*self.cells] if key == 'knowledge' and self.dynamic_memory else value
        result = super().encode(key, base)
        if key == 'knowledge' and self.dynamic_memory:
            result['dynamics'] = value[..., 11*self.cells:].half()
        return result

    def gather(self, key, value, times, envs):
        result = super().gather(key, value, times, envs)
        if key == 'knowledge' and self.dynamic_memory:
            result = torch.cat((result, value['dynamics'][times, envs].float()), -1)
        return result


class DynamicObservationCodec(CompactObservationCodec):
    dynamic_memory = True
    feature_set = 'temporal'


class PathObservationCodec(DynamicObservationCodec):
    """Store the two distance planes from each decision, alongside dynamic memory."""
    feature_set = 'temporal_paths'
