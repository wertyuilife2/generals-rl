"""Arena-owned C++ policy build, immutable artifacts and direct timed C ABI.

This is a policy host, not a game backend. Native code is trusted local code;
the match process contains crashes/hangs but is not an OS security sandbox.
"""
from functools import lru_cache
from hashlib import sha256
import io
import ctypes as C
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import numpy as np
from generals import Action
from .util import canonical, digest, file_digest

SDK = Path(__file__).with_name('sdk')
DECISION_NS = 10_000_000
PROTOCOL = 'arena-cpp-library-v2'


class PolicyError(RuntimeError):
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


def _private_cache():
    root = Path(tempfile.gettempdir()) / f'generals-arena-policies-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('policy cache must be a private owned directory')
    return root


def source_identity(directory, parameters=None):
    supplied = Path(directory)
    if supplied.is_symlink():
        raise ValueError('policy sources cannot contain symlinks')
    directory = supplied.resolve()
    if not directory.exists():
        raise ValueError(f'policy source does not exist: {supplied}')
    files = {}
    if directory.is_file():
        if directory.suffix != '.cpp':
            raise ValueError('policy source must be a .cpp file or a directory')
        root = directory.parent
        pending = [directory]
    else:
        root = directory
        pending = sorted(directory.rglob('*'))
    seen = set()
    while pending:
        path = pending.pop(0)
        if path.is_symlink():
            raise ValueError('policy sources cannot contain symlinks')
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and path.suffix in ('.cpp', '.h', '.hpp'):
            name = path.relative_to(root).as_posix()
            if path.name == 'arena_policy.h':
                if path.read_bytes() != (SDK/'arena_policy.h').read_bytes():
                    raise ValueError('arena_policy.h must match the Arena SDK')
                continue
            files[name] = path.read_bytes()
            if directory.is_file():
                for include in re.findall(rb'^\s*#\s*include\s*"([^"\r\n]+)"', files[name], re.M):
                    include = include.decode()
                    local = path.parent/include
                    if not local.exists():
                        local = root/include
                    if include == 'arena_policy.h' and not local.exists():
                        continue  # Supplied by Arena, even without a workspace header.
                    if local.is_symlink() or not local.resolve().is_relative_to(root):
                        raise ValueError('local policy includes must stay inside the source directory')
                    if not local.is_file() or local.suffix not in ('.h', '.hpp'):
                        raise ValueError(f'local include must be a .h/.hpp file: {include}')
                    pending.append(local.resolve())
    if not any(name.endswith('.cpp') for name in files):
        raise ValueError('source directory must contain a .cpp file')
    if sum(map(len, files.values())) > 20*1024*1024:
        raise ValueError('policy sources exceed 20 MiB')
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise RuntimeError('a C++17 compiler is required (CXX)')
    version = subprocess.check_output([compiler, '--version'], text=True, timeout=15).splitlines()[0]
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict):
        raise ValueError('policy parameters must be a JSON object')
    flags = ['-std=c++17', '-O3', '-ffp-contract=off', '-fPIC', '-fvisibility=hidden',
             '-dynamiclib' if platform.system() == 'Darwin' else '-shared']
    identity = {'protocol': PROTOCOL, 'platform': [platform.system(), platform.machine()],
                'compiler': version, 'compiler_path': compiler, 'flags': flags,
                'sources': {name: sha256(data).hexdigest() for name, data in files.items()},
                'sdk': {name: file_digest(SDK/name) for name in ('arena_policy.h', 'adapter.cpp')},
                'parameters': parameters, 'decision_limit_ns': DECISION_NS}
    # Validate parameters before creating any registered artifact.
    return digest(identity), identity, files


def build_bundle(directory, destination, parameters=None, *, snapshot=None):
    """Snapshot first, compile only the snapshot, then atomically publish a zip."""
    key, metadata, files = source_identity(directory, parameters) if snapshot is None else snapshot
    with tempfile.TemporaryDirectory(dir=_private_cache()) as scratch:
        root = Path(scratch)
        source = root/'source'
        source.mkdir()
        for name, contents in files.items():
            target = source/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        for name in ('arena_policy.h', 'adapter.cpp'):
            (root/name).write_bytes((SDK/name).read_bytes())
        result = subprocess.run([metadata['compiler_path'], *metadata['flags'], '-I', str(root),
            '-I', str(source), str(root/'adapter.cpp'),
            *(str(source/name) for name in files if name.endswith('.cpp')),
            '-o', str(root/'policy.so')], capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError(f'C++ policy compilation failed:\n{result.stderr}')
        metadata = {**metadata, 'version': key, 'library_sha256': file_digest(root/'policy.so')}
        data = {'manifest.json': canonical(metadata).encode(),
                'params.json': canonical(metadata['parameters']).encode(),
                'policy.so': (root/'policy.so').read_bytes(),
                'sdk/arena_policy.h': (root/'arena_policy.h').read_bytes(),
                'sdk/adapter.cpp': (root/'adapter.cpp').read_bytes(),
                **{'source/'+name: content for name, content in files.items()}}
        output = io.BytesIO()
        with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
            for name, content in sorted(data.items()):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                archive.writestr(info, content)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(output.getvalue())
    return metadata


def bundle_info(bundle):
    with ZipFile(bundle) as archive:
        metadata = json.loads(archive.read('manifest.json'))
    if metadata.get('protocol') != PROTOCOL or metadata.get('decision_limit_ns') != DECISION_NS:
        raise ValueError('unsupported C++ policy protocol or decision limit')
    if metadata['platform'] != [platform.system(), platform.machine()]:
        raise ValueError('policy binary is for another platform; register its source here')
    return metadata


def _unpack(bundle, expected):
    # The OS may purge its temporary directory while a long-running Arena
    # process is still alive.  Re-check the extracted artifact on every new
    # controller instead of caching a Path that may no longer exist.
    if file_digest(bundle) != expected:
        raise ValueError('policy bundle changed')
    metadata = bundle_info(bundle)
    target = _private_cache()/expected
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            root = Path(temporary)/'artifact'
            root.mkdir(mode=0o700)
            with ZipFile(bundle) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 100*1024*1024:
                    raise ValueError('invalid or oversized policy bundle')
                for info in archive.infolist():
                    path = PurePosixPath(info.filename)
                    if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError('unsafe policy artifact path')
                    item = root/path
                    item.parent.mkdir(parents=True, exist_ok=True)
                    item.write_bytes(archive.read(info))
            if file_digest(root/'policy.so') != metadata['library_sha256']:
                raise ValueError('policy executable checksum mismatch')
            if (root/'params.json').read_text() != canonical(metadata['parameters']):
                raise ValueError('policy parameter checksum mismatch')
            (root/'policy.so').chmod(0o500)
            try:
                root.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
    if file_digest(target/'policy.so') != metadata['library_sha256']:
        raise ValueError('cached policy executable changed')
    return target


# Set only inside Arena's single match process. No policy runs in the web server.
DIAGNOSTICS = None
METRIC_FIELDS = ('calls', 'total_ns', 'max_ns', 'max_tick', 'timeouts', 'errors', 'last_ns', 'status')


class Move(C.Structure):
    _fields_ = [(key, C.c_int64) for key in
                ('player', 'action', 'valid', 'source', 'target', 'moved', 'captured', 'eliminated', 'target_owned')]


class Observation(C.Structure):
    _fields_ = [(key, C.c_int) for key in ('width', 'height', 'players', 'player')] + [
        ('tick', C.c_int64), ('general_pos', C.c_int)] + [
        (key, C.POINTER(kind)) for key, kind in (
            ('terrain', C.c_uint8), ('structure', C.c_uint8), ('owner', C.c_int16),
            ('army', C.c_int64), ('visible', C.c_uint8), ('alive', C.c_uint8),
            ('total_army', C.c_int64), ('total_land', C.c_int64), ('total_cities', C.c_int64))] + [
        ('previous', C.POINTER(Move))]


@lru_cache(maxsize=128)
def _library(path):
    lib = C.CDLL(str(path))
    metric = C.POINTER(C.c_int64)
    lib.arena_host_create.argtypes = [C.c_int, C.c_uint64, C.c_char_p, metric]
    lib.arena_host_create.restype = C.c_void_p
    lib.arena_host_act.argtypes = [C.c_void_p, C.POINTER(Observation), metric]
    lib.arena_host_act.restype = C.c_int64
    lib.arena_host_destroy.argtypes = [C.c_void_p, metric]
    lib.arena_host_destroy.restype = None
    return lib


def metrics(values):
    data = {name: int(values[i]) for i, name in enumerate(METRIC_FIELDS)}
    return {key: data[key] for key in METRIC_FIELDS[:6]} | {
        'limit_ns': DECISION_NS, 'native_max_ns': data['max_ns'],
        'measurement': 'single-thread C++ arena_act wall time; excludes Arena and IPC'}


class NativeController:
    """Direct C ABI call. All policies and the env CPU engine share one match process."""
    def __init__(self, *, bundle, player_id=0, seed=0):
        self.bundle = str(Path(bundle).resolve())
        self.bundle_sha256 = file_digest(self.bundle)
        self.root = _unpack(self.bundle, self.bundle_sha256)
        self.lib, self.instance = None, None
        self.previous, self.player_id = None, player_id
        self.values = np.zeros(8, dtype=np.int64)

    def _activate(self, stage, tick=0):
        if DIAGNOSTICS is not None:
            DIAGNOSTICS.header[1:4] = [self.player_id, tick, stage]

    def _idle(self):
        if DIAGNOSTICS is not None:
            DIAGNOSTICS.header[1] = -1

    def _check(self, tick, phase):
        status = int(self.values[7])
        if status:
            details = {'kind': 'timeout' if status == 1 else 'policy_error', 'player': self.player_id,
                       'tick': int(tick), 'elapsed_ns': int(self.values[6]),
                       'limit_ns': DECISION_NS if phase == 'decision' else 2_000_000_000,
                       'bundle_sha256': self.bundle_sha256, 'phase': phase}
            raise PolicyError(f'C++ policy {phase} {details["kind"]}: '
                              f'{details["elapsed_ns"]/1e6:.3f} ms', details=details)

    def reset(self, player_id, seed):
        self.close()
        self.player_id, self.previous = int(player_id), None
        self.values = (DIAGNOSTICS.records[self.player_id] if DIAGNOSTICS is not None
                       else np.zeros(8, dtype=np.int64))
        self.values[:] = 0
        self._activate(1)
        self.lib = _library(self.root/'policy.so')
        self.instance = self.lib.arena_host_create(self.player_id, int(seed),
            (self.root/'params.json').read_bytes(), self.values.ctypes.data_as(C.POINTER(C.c_int64)))
        self._idle()
        self._check(0, 'initialization')

    @property
    def decision_metrics(self):
        return metrics(self.values)

    def act(self, obs):
        if self.instance is None:
            raise RuntimeError('policy must be reset before acting')
        fields = [(obs.terrain, np.uint8, C.c_uint8), (obs.structure, np.uint8, C.c_uint8),
                  (obs.owner, np.int16, C.c_int16), (obs.army, np.int64, C.c_int64),
                  (obs.visible, np.uint8, C.c_uint8), (obs.alive, np.uint8, C.c_uint8),
                  (obs.stats.army, np.int64, C.c_int64), (obs.stats.territory, np.int64, C.c_int64),
                  (obs.stats.cities, np.int64, C.c_int64)]
        arrays = [np.ascontiguousarray(value, dtype=dtype) for value, dtype, _ in fields]
        previous = Move(*self.previous) if self.previous is not None else None
        native_obs = Observation(obs.width, obs.height, len(obs.alive), obs.player_id,
            obs.tick, obs.general_pos,
            *(array.ctypes.data_as(C.POINTER(kind)) for array, (_, _, kind) in zip(arrays, fields)),
            C.pointer(previous) if previous is not None else None)
        self._activate(2, obs.tick)
        encoded = self.lib.arena_host_act(self.instance, C.byref(native_obs),
                                         self.values.ctypes.data_as(C.POINTER(C.c_int64)))
        self._idle()
        self._check(obs.tick, 'decision')
        if not 0 <= encoded <= obs.size*8:
            self.values[5] += 1
            raise PolicyError('C++ policy returned an invalid action encoding', details={
                'kind': 'policy_error', 'player': self.player_id, 'tick': obs.tick,
                'elapsed_ns': int(self.values[6]), 'limit_ns': DECISION_NS})
        return Action.decode(encoded)

    def on_result(self, move):
        self.previous = [move.player_id, move.action.encode(), int(move.valid), move.source,
            move.target, move.moved, int(move.captured), -1 if move.eliminated is None else move.eliminated,
            int(move.target_owned)]

    def close(self):
        if self.instance is not None:
            instance, self.instance = self.instance, None
            tick = int(DIAGNOSTICS.header[0]) if DIAGNOSTICS is not None else 0
            self._activate(3, tick)
            self.lib.arena_host_destroy(instance, self.values.ctypes.data_as(C.POINTER(C.c_int64)))
            self._idle()
            self._check(tick, 'destruction')


def create_controller(*, bundle, player_id, seed):
    return NativeController(bundle=bundle, player_id=player_id, seed=seed)
