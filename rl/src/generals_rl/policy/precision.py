"""Scoped PyTorch precision flags shared by training and deployed inference."""
from contextlib import contextmanager

import torch


def validate_precision(precision=None, cudnn_allow_tf32=None, cudnn_benchmark=None):
    if precision is not None and precision not in ("highest", "high", "medium"):
        raise ValueError("matmul_precision must be highest, high, medium or null")
    for name, value in (("cudnn_allow_tf32", cudnn_allow_tf32), ("cudnn_benchmark", cudnn_benchmark)):
        if value is not None and type(value) is not bool:
            raise TypeError(f"{name} must be boolean or null")


@contextmanager
def matmul_precision(precision=None, cudnn_allow_tf32=None, cudnn_benchmark=None):
    """Apply explicit flags for one forward and restore even after an error."""
    validate_precision(precision, cudnn_allow_tf32, cudnn_benchmark)
    if precision is None and cudnn_allow_tf32 is None and cudnn_benchmark is None:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        if precision is not None:
            torch.set_float32_matmul_precision(precision)
        if cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = cudnn_benchmark
        yield
    finally:
        if cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = previous_cudnn
        if precision is not None:
            torch.set_float32_matmul_precision(previous)
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = previous_benchmark
