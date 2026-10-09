"""Stable identities and atomic JSON reports."""
from hashlib import sha256
from importlib.util import find_spec
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    value = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(chunk)
    return value.hexdigest()


def source_digest(package):
    """Hash installed Python sources, including local uncommitted changes."""
    spec = find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"install the {package} package before registering this entrant")
    root = Path(next(iter(spec.submodule_search_locations)))
    files = {str(path.relative_to(root)): file_digest(path) for path in sorted(root.rglob("*.py"))}
    if not files:
        raise ValueError(f"no source files found for {package}")
    return digest(files)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value
