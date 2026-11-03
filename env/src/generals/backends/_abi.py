"""Pointer-only native state ABI, shared by NumPy and Torch launchers."""
import ctypes
from generals.protocol import STATE_DTYPES


class Batch(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in STATE_DTYPES]
