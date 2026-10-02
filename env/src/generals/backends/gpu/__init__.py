"""CUDA execution backend, selected lazily by generals.VecEnv."""
from generals.backends.gpu.env import GPUEnv, BatchStep

__all__ = ["GPUEnv", "BatchStep"]
