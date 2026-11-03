"""Content-addressed local policy packages; no RL or tensor dependency."""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import re
import shutil
import sys
import threading

from generals_arena.util import digest
from generals_arena.util import file_digest


_imports = threading.RLock()
_entrypoint = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # The identity covers source bytes, never a potentially stale .pyc.
        return compile(Path(self.path).read_bytes(), self.path, 'exec', dont_inherit=True)


class _SourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('_generals_arena_policy_'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.origin:
            if not spec.origin.endswith('.py'):
                raise ImportError('frozen policy modules must have verified Python source')
            spec.loader = _SourceLoader(fullname, spec.origin)
        return spec


_finder = _SourceFinder()


def _files(directory):
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("policy must be a local source package directory")
    paths = []
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.relative_to(root).parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_symlink():
            raise ValueError("policy source cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix not in ('.py', '.json', '.txt', '.md'):
            raise ValueError(f"unsupported policy source file: {path.name}")
        paths.append(path)
    if root/'__init__.py' not in paths:
        raise ValueError("policy source needs __init__.py")
    return root, paths


def policy_digest(directory):
    root, paths = _files(directory)
    return digest({path.relative_to(root).as_posix(): file_digest(path) for path in paths})


def snapshot_policy_code(source, target):
    """Copy source files and verify the complete package after copying."""
    source, paths = _files(source)
    expected = policy_digest(source)
    target = Path(target)
    target.mkdir()
    try:
        for path in paths:
            destination = target/path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        if policy_digest(target) != expected:
            raise ValueError("policy source changed while snapshotting")
    except BaseException:
        shutil.rmtree(target)
        raise


def load_policy_factory(directory, entrypoint, expected_sha):
    """Load relative imports under a content/location-specific namespace.

    Names never overlap between two source versions. The location suffix keeps
    lazy imports valid when an identical original directory is later removed.
    As with other Arena Python policies, this runs trusted local source code.
    """
    root = Path(directory).resolve()
    if not isinstance(entrypoint, str) or not _entrypoint.fullmatch(entrypoint):
        raise ValueError("policy entrypoint must be module:callable")
    if policy_digest(root) != expected_sha:
        raise ValueError("policy source SHA256 mismatch")
    namespace = f"_generals_arena_policy_{expected_sha}_{digest(str(root))[:12]}"
    module_name, factory_name = entrypoint.split(':')
    with _imports:
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
        created = namespace not in sys.modules
        try:
            if created:
                spec = importlib.util.spec_from_file_location(namespace, root/'__init__.py',
                                                              loader=_SourceLoader(namespace, str(root/'__init__.py')),
                                                              submodule_search_locations=[str(root)])
                module = importlib.util.module_from_spec(spec)
                sys.modules[namespace] = module
                spec.loader.exec_module(module)
            module = importlib.import_module(f'{namespace}.{module_name}')
            factory = getattr(module, factory_name, None)
            if not callable(factory):
                raise ValueError("policy entrypoint is not callable")
            if policy_digest(root) != expected_sha:
                raise ValueError("policy source changed while importing")
            return factory
        except BaseException:
            if created:
                for name in list(sys.modules):
                    if name == namespace or name.startswith(namespace+'.'):
                        sys.modules.pop(name, None)
            raise
