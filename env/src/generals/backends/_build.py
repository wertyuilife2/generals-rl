"""Small shared-library builds, cached by source, compiler and exact flags."""
import ctypes
from hashlib import sha256
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

_openmp_runtime = None


def openmp_flags():
    global _openmp_runtime
    if sys.platform == 'linux':
        return ['-march=native', '-fopenmp', '-fPIC', '-shared']
    if sys.platform != 'darwin':
        raise RuntimeError('native CPU backend supports Linux and macOS')
    prefixes = [Path(os.environ['GENERALS_LIBOMP_PREFIX'])] if 'GENERALS_LIBOMP_PREFIX' in os.environ else [
        Path('/opt/homebrew/opt/libomp'), Path('/usr/local/opt/libomp')]
    prefix = next((p for p in prefixes if (p/'include/omp.h').is_file()
                   and (p/'lib/libomp.dylib').is_file()), None)
    if prefix is None:
        raise RuntimeError('macOS CPU backend requires libomp; set GENERALS_LIBOMP_PREFIX to its installation prefix')
    library = prefix/'lib'
    # Use one OpenMP runtime even if torch is imported later. Discover its
    # bundled library without importing this optional package or initializing it.
    spec = find_spec('torch')
    if spec is not None and spec.origin:
        bundled = Path(spec.origin).parent/'lib'
        if (bundled/'libomp.dylib').is_file():
            library = bundled
    # Some wheels retain their build machine's absolute install name. Loading
    # the actual file first resolves that name and keeps one shared runtime.
    _openmp_runtime = ctypes.CDLL(str(library/'libomp.dylib'), mode=ctypes.RTLD_GLOBAL)
    return ['-Xpreprocessor', '-fopenmp', f'-I{prefix}/include', f'-L{library}',
            f'-Wl,-rpath,{library}', '-lomp', '-dynamiclib']


def build_shared(source, *, name, flags=None):
    source = Path(source)
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if compiler is None:
        raise RuntimeError('native execution requires a C++17 compiler (set CXX)')
    version = subprocess.check_output([compiler, '--version'], text=True, timeout=15).splitlines()[0]
    if flags is None:
        if sys.platform not in ('linux', 'darwin'):
            raise RuntimeError('native execution supports Linux and macOS')
        flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-fPIC', '-shared']
    flags = ['-std=c++17', '-O3', '-ffp-contract=off', *flags]
    key = sha256(source.read_bytes()+repr((compiler, version, flags)).encode()).hexdigest()
    root = Path(tempfile.gettempdir())/f'generals-{name}-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('native compilation cache must be a private owned directory')
    target = root/(key+('.dylib' if sys.platform == 'darwin' else '.so'))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            output = Path(scratch)/target.name
            result = subprocess.run([compiler, *flags, str(source), '-o', str(output)],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'native compilation failed:\n{result.stderr}')
            os.replace(output, target)
    return ctypes.CDLL(str(target)), {'compiler': version, 'compiler_flags': flags,
                                     'source_sha256': sha256(source.read_bytes()).hexdigest(),
                                     'build_sha256': key}
