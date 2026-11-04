"""Internal native execution leases. Public raw-array APIs remain checked.

Bindings are trusted, process-local capabilities, not serializable pointers.
The executor owns the controllers and backend for the entire lease; no reset,
close, storage replacement, or concurrent calls are permitted inside it.
"""
import ctypes as C
from dataclasses import dataclass
from pathlib import Path

INCLUDE = Path(__file__).parent
ABI_VERSION = 2


class PolicyABI(C.Structure):
    _fields_ = [('abi', C.c_uint64)] + [(name, C.c_void_p) for name in
                                      ('context', 'act', 'feedback', 'act_one')]


class BackendABI(C.Structure):
    _fields_ = [('abi', C.c_uint64)] + [(name, C.c_void_p) for name in
        ('context', 'observe', 'step', 'alive', 'done', 'tick', 'moves', 'errors', 'depart', 'state', 'stats', 'legal')]


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
