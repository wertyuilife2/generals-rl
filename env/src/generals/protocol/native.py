"""Internal native execution leases. Public raw-array APIs remain checked.

Bindings are trusted, process-local capabilities, not serializable pointers.
The executor owns the controllers and backend for the entire lease; no reset,
close, storage replacement, or concurrent calls are permitted inside it.
"""
import ctypes as C
from dataclasses import dataclass
from pathlib import Path

INCLUDE = Path(__file__).parent
ABI_VERSION = 3
HOST_BUFFERS = 1


class PolicyABI(C.Structure):
    _fields_ = [('abi', C.c_uint64)] + [(name, C.c_void_p) for name in
                                      ('context', 'act', 'feedback', 'act_one')] + [('flags', C.c_uint64)]

    def __init__(self, abi, context, act, feedback, act_one, flags=1):
        super().__init__(abi, context, act, feedback, act_one, flags)


class BackendABI(C.Structure):
    _fields_ = [('abi', C.c_uint64)] + [(name, C.c_void_p) for name in
        ('context', 'observe', 'step', 'alive', 'done', 'tick', 'moves', 'errors', 'depart', 'snapshot', 'stats', 'legal')] + [
        ('capabilities', C.c_uint64), ('device', C.c_int32), ('synchronize', C.c_void_p)]


def address(function):
    return C.cast(function, C.c_void_p).value


@dataclass
class PolicyBinding:
    abi: PolicyABI
    owner: object
    check_error: object


@dataclass
class BackendBinding:
    abi: BackendABI
    owner: object
