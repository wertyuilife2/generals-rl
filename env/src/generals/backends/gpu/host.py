"""CPython launch bridge, without a PyTorch C++ ABI dependency."""
import ctypes
from functools import lru_cache
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile

import torch


@lru_cache(maxsize=1)
def _module():
    source = Path(__file__).with_suffix('.cpp')
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if compiler is None:
        raise RuntimeError('native GPU launch needs a C++ compiler and Python development headers')
    version = subprocess.run([compiler, '--version'], capture_output=True, text=True, check=True, timeout=15).stdout
    flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '-I' + sysconfig.get_path('include')]
    digest = sha256(source.read_bytes() + repr((compiler, version, flags, sys.version)).encode()).hexdigest()
    root = Path(tempfile.gettempdir()) / f'generals-cuda-host-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('native launch cache must be a private owned directory')
    target = root / (digest + sysconfig.get_config_var('EXT_SUFFIX'))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            output = Path(scratch) / target.name
            result = subprocess.run([compiler, *flags, str(source), '-o', str(output)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError('native launch compilation failed:\n' + result.stderr)
            os.replace(output, target)
    spec = importlib.util.spec_from_file_location('_generals_cuda_host', target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {'source_sha256': sha256(source.read_bytes()).hexdigest(),
                    'build_sha256': digest, 'compiler_flags': flags}


def _stream_address(index):
    return torch.cuda.current_stream(index).cuda_stream


def bind(kernel):
    module, metadata = _module()
    kernel._tensor_type = torch.Tensor
    # Feature-detected fast stream accessor. Fall back to the public API on
    # PyTorch builds which do not provide it; no torch headers are compiled.
    kernel._stream_address = getattr(torch._C, '_cuda_getCurrentRawStream', _stream_address)
    address = lambda function: ctypes.cast(function, ctypes.c_void_p).value
    fields = type(kernel.args)
    kernel._host_context = module.bind(kernel, ctypes.addressof(kernel.args),
        address(kernel.lib.generals_launch), address(kernel.lib.cudaGetDevice),
        address(kernel.lib.cudaSetDevice), address(kernel.lib.generals_error),
        (fields.actions.offset, fields.auto_reset.offset, fields.horizon.offset))
    return module, {**metadata, 'raw_stream_accessor': kernel._stream_address is not _stream_address}
