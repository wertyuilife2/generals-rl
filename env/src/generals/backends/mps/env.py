"""Device-resident Metal backend with the CPU/CUDA batch protocol.

Public tensors retain [batch, ...] shapes. Each board uses contiguous 16-byte
cell records; player fields and outputs use field-major views. No CPU rule
execution or state readback occurs in step. Integer semantics match CUDA: armies,
totals and ticks must fit signed int64; arithmetic overflow is not checked.
"""
import torch

from generals.backends.gpu.env import GPUEnv


def field_major(value):
    if value.ndim < 2:
        return value
    return value.flatten(1).t().contiguous().t().reshape(value.shape)


def pack_tensors(tensors, *, cells=False):
    """Pack public tensor views in one aligned Metal buffer.

    Views share a version counter, so two storage versions cover all state and
    initial-state edits. Separate output storage keeps stats edits independent.
    Cell records match the shader's Cell layout; other fields retain strides.
    """
    cell_fields = {"army": 0, "owner": 8, "structure": 10, "terrain": 11} if cells else {}
    offsets, size = {}, tensors["army"].numel() * 16 if cells else 0
    for name, value in tensors.items():
        if name in cell_fields:
            offsets[name] = cell_fields[name]
            continue
        size = (size + 15) // 16 * 16
        offsets[name] = size
        size += value.numel() * value.element_size()
    storage = torch.empty(size, dtype=torch.uint8, device=next(iter(tensors.values())).device)
    views = {}
    for name, value in tensors.items():
        begin = offsets[name]
        if name in cell_fields:
            end = value.numel() * 16
            stride = 16 // value.element_size()
            view = storage[begin:end].view(value.dtype).as_strided(value.shape, (value.shape[1] * stride, stride))
        else:
            end = begin + value.numel() * value.element_size()
            view = storage[begin:end].view(value.dtype).as_strided(value.shape, value.stride())
        view.copy_(value)
        views[name] = view
    return storage, views, tuple(offsets.items())


class MPSEnv(GPUEnv):
    kernel_kind = "metal"

    def _resolve_device(self, device):
        requested = torch.device(device)
        if requested.type != "mps" or requested.index not in (None, 0):
            raise ValueError("MPSEnv requires device=mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS simulation requires an available Apple GPU and MPS-enabled PyTorch")
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("MPS simulation requires a PyTorch build with torch.mps.compile_shader")
        return torch.device("mps:0")

    def __init__(self, initial, *, width, height, device="mps", **kwargs):
        super().__init__(initial, width=width, height=height, device=device, **kwargs)
        for mapping in (self.state, self.initial):
            for key, value in mapping.items():
                mapping[key] = value if key in ("terrain", "structure", "owner", "army") else field_major(value)
        for name in ("stats", "_initial_stats", "counters", "last_moves", "done", "_wait_actions"):
            setattr(self, name, field_major(getattr(self, name)))
        self._state_storage, self.state, self._state_offsets = pack_tensors(self.state, cells=True)
        self._initial_storage, self.initial, self._initial_offsets = pack_tensors(self.initial, cells=True)
        self._output_storage, outputs, self._output_offsets = pack_tensors({
            name: getattr(self, name) for name in ("stats", "counters", "last_moves", "done", "errors")})
        for name, value in outputs.items():
            setattr(self, name, value)
        from generals.protocol.vector import BatchStep
        self._step_result = BatchStep(self.last_moves, self.stats, self.done, self.errors)

    @classmethod
    def from_states(cls, states, *, device="mps", **kwargs):
        return super().from_states(states, device=device, **kwargs)

    @classmethod
    def from_config(cls, config, batch_size=1, *, device="mps", **kwargs):
        return super().from_config(config, batch_size=batch_size, device=device, **kwargs)

    def configure_step(self, *, threads_per_group=128):
        self._require_open()
        from .kernel import StepKernel
        if self._step_kernel is None:
            self._step_kernel = StepKernel(self, threads_per_group=threads_per_group)
        else:
            self._step_kernel.configure_group(threads_per_group)

    def step(self, actions):
        self._require_open()
        if not isinstance(actions, torch.Tensor) or actions.dtype != torch.int64:
            raise TypeError("actions must be an int64 MPS tensor")
        if actions.shape != (self.batch_size, self.players) or actions.device != self.device:
            raise ValueError("actions must have shape [batch, players] on the environment device")
        if self._step_kernel is None:
            self.configure_step()
        # The external action ABI is ordinary contiguous [B,P]; internal cell
        # records do not require callers to repack their actions.
        self._step_kernel.launch(actions.contiguous())
        return self._step_result

    def step_metadata(self):
        result = {"step_implementation": "metal", "state_layout": "cell-record-16B/output-field-major"}
        if self._step_kernel is not None:
            result["kernel"] = self._step_kernel.metadata()
        return result

    def synchronize(self):
        self._require_open()
        torch.mps.synchronize()

    def close(self):
        super().close()
        self._state_storage = self._initial_storage = self._output_storage = None
