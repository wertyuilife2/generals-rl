"""Immutable policy bundles; config and weights are opaque to Arena."""
import json
from pathlib import Path
import re
import shutil
import tempfile
from zipfile import ZipFile

from generals_arena.util import canonical, digest, file_digest, process_lock, write_json
from .code import load_policy_factory, policy_digest, snapshot_policy_code


SCHEMA_VERSION = 1
from generals.protocol.rules import RULES_VERSION as RULES_PROTOCOL


def _object(value, name, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError(f"invalid {name} fields")
    return value


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _label(value, name, maximum):
    if (not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f"invalid {name}")
    return value.strip()


def _read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def _json_object(value):
    value = _read_json(value) if isinstance(value, (str, Path)) else value
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    # Independent trees, deterministic serialization and no nonfinite values.
    return json.loads(canonical(value))


def _sha256(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ValueError(f"invalid {name} SHA256")
    return value.lower()


def load_manifest(manifest):
    """Validate a manifest without reading config or importing policy code."""
    value = _json_object(manifest)
    _object(value, "policy manifest", ("schema_version", "name", "rules", "artifacts", "capabilities", "entrypoint"))
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported policy manifest schema_version")
    if value["rules"] != RULES_PROTOCOL:
        raise ValueError("unsupported rules protocol")
    if not isinstance(value["entrypoint"], str) or not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value["entrypoint"]):
        raise ValueError("policy entrypoint must be module:callable")
    value["name"] = _label(value["name"], "name", 128)
    artifacts = _object(value["artifacts"], "artifacts", ("weights", "config", "code"))
    paths = []
    for role, artifact in artifacts.items():
        _object(artifact, f"{role} artifact", ("path", "sha256"))
        path = artifact["path"]
        if (not isinstance(path, str) or not path or Path(path).is_absolute()
                or any(part in (".", "..") for part in path.split("/")) or "\\" in path):
            raise ValueError("artifact path must be a normalized relative path")
        paths.append(path)
        artifact["sha256"] = _sha256(artifact["sha256"], role)
    if len(set(paths)) != 3 or "manifest.json" in paths:
        raise ValueError("artifact paths must be distinct and exclude manifest.json")
    if any(Path(left) in Path(right).parents for left in paths for right in paths if left != right):
        raise ValueError("artifact paths must not overlap")
    capabilities = _object(value["capabilities"], "capabilities", ("maps", "player_counts"))
    counts = capabilities["player_counts"]
    if not isinstance(counts, list) or not counts:
        raise ValueError("player_counts must be a nonempty list")
    for count in counts:
        _integer(count, "player_counts entry", 2, 8)
    if len(set(counts)) != len(counts):
        raise ValueError("duplicate player_counts")
    capabilities["player_counts"] = sorted(counts)
    maps = capabilities["maps"]
    if isinstance(maps, dict) and set(maps) == {"shapes"}:
        shapes = maps["shapes"]
        if not isinstance(shapes, list) or not shapes:
            raise ValueError("maps.shapes must be a nonempty list of [width, height]")
        for shape in shapes:
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError("map shape must be [width, height]")
            for side in shape:
                _integer(side, "map side")
        if len({tuple(shape) for shape in shapes}) != len(shapes):
            raise ValueError("duplicate map shapes")
        maps["shapes"] = sorted(shapes)
    else:
        _object(maps, "map range", ("width", "height"))
        for axis in ("width", "height"):
            bounds = maps[axis]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"maps.{axis} must be [minimum, maximum]")
            _integer(bounds[0], f"maps.{axis} minimum")
            _integer(bounds[1], f"maps.{axis} maximum", bounds[0])
    return value


def capabilities_supports(capabilities, width, height, players):
    """Whether one map/player combination is in validated v1 capabilities.

    All declared shapes/ranges apply to every declared player count. A future
    implementation must validate that whole product before accepting a policy.
    """
    if any(type(value) is not int or value < 1 for value in (width, height, players)):
        return False
    if players not in capabilities["player_counts"]:
        return False
    maps = capabilities["maps"]
    return ([width, height] in maps["shapes"] if "shapes" in maps else
            maps["width"][0] <= width <= maps["width"][1]
            and maps["height"][0] <= height <= maps["height"][1])


def validate_bundle(bundle):
    """Validate a complete package without importing its code or parsing config."""
    root = Path(bundle)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("bundle must be a local directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("bundle artifacts cannot contain symlinks")
    manifest = load_manifest(manifest_path)
    paths = {}
    for role, artifact in manifest["artifacts"].items():
        path = root / artifact["path"]
        if any(candidate.is_symlink() for candidate in (path, *path.parents) if candidate != root.parent):
            raise ValueError("bundle artifacts cannot contain symlinks")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("artifact escapes bundle directory")
        actual = policy_digest(path) if role == "code" else file_digest(path)
        if actual != artifact["sha256"]:
            raise ValueError(f"{role} SHA256 mismatch")
        paths[role] = str(path.resolve())
    return {"policy_id": digest({"kind": "package", "manifest": manifest}),
            "name": manifest["name"], "manifest": manifest, "paths": paths}


def _spec(root, info, device):
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
        raise ValueError("policy device must be cpu or cuda[:index]")
    return {"name": info["name"], "kind": "package", "version": info["policy_id"],
            "bundle": str(root / "policies" / info["policy_id"]), "device": device}


def resolve_bundle(directory, policy_id, *, device="cpu"):
    if not isinstance(policy_id, str) or not re.fullmatch(r"[a-f0-9]{64}", policy_id):
        raise ValueError("policy_id must be a full lowercase SHA256")
    root = Path(directory).resolve()
    info = validate_bundle(root / "policies" / policy_id)
    if info["policy_id"] != policy_id:
        raise ValueError("policy snapshot identity changed")
    return _spec(root, info, device)


def snapshot_bundle(directory, bundle, *, device="cpu"):
    """Atomically import complete immutable bytes, preserving alias ownership."""
    info = validate_bundle(bundle)
    root = Path(directory).resolve()
    spec = _spec(root, info, device)
    root.mkdir(parents=True, exist_ok=True)
    with process_lock(root, ".registry.lock", blocking=True):
        for path in root.glob("policies/*/manifest.json"):
            previous = load_manifest(path)
            if previous["name"] == info["name"] and path.parent.name != info["policy_id"]:
                raise ValueError("policy alias already identifies a different bundle; choose a new name")
        parent = root / "policies"
        parent.mkdir(exist_ok=True)
        target = parent / info["policy_id"]
        if target.exists() or target.is_symlink():
            return resolve_bundle(root, info["policy_id"], device=device)
        with tempfile.TemporaryDirectory(prefix=".package-", dir=parent) as temporary:
            staged = Path(temporary) / "bundle"
            staged.mkdir()
            for role, artifact in info["manifest"]["artifacts"].items():
                target_path = staged / artifact["path"]
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if role == "code":
                    snapshot_policy_code(info["paths"][role], target_path)
                else:
                    shutil.copyfile(info["paths"][role], target_path)
            (staged / "manifest.json").write_text(canonical(info["manifest"]) + "\n", encoding="utf-8")
            if validate_bundle(staged)["policy_id"] != info["policy_id"]:
                raise ValueError("bundle changed while snapshotting")
            staged.rename(target)
    return spec


def create_controller(*, bundle, device='cpu', player_id=0, seed=0):
    """Construct an opaque policy package through its verified entrypoint."""
    info = validate_bundle(bundle)
    manifest = info['manifest']
    factory = load_policy_factory(info['paths']['code'], manifest['entrypoint'],
                                  manifest['artifacts']['code']['sha256'])
    return factory(weights=info['paths']['weights'], config_path=info['paths']['config'],
                   device=device, player_id=player_id, seed=seed)


def _resolve_policy(directory, version, *, device=None, entries=None):
    if not re.fullmatch('[a-f0-9]{64}', version):
        raise ValueError('policy/ requires a full version hash from a run or commit report')
    path = Path(directory).resolve() / 'policies' / version
    if (path / 'manifest.json').is_file():
        entry = next((entry for entry in (entries or {}).values()
                      if entry['kind'] == 'package' and entry['version'] == version), None)
        placement = device or (entry['factory_kwargs']['device'] if entry else 'cpu')
        return {**resolve_bundle(directory, version, device=placement), 'name': f'package-{version[:16]}'}
    metadata = json.loads((path / 'build.json').read_text())
    bundle = path / 'policy.zip'
    if metadata['version'] != version or file_digest(bundle) != metadata['bundle_sha256']:
        raise ValueError('archived policy artifact changed')
    return {'name': f'cpp-{version[:16]}', 'kind': 'cpp', 'version': version, 'bundle': str(bundle)}


def snapshot_policy(directory, source, parameters=None):
    """Archive a native policy using a path, independent of competition storage."""
    from .native import build_bundle, source_identity
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot = source_identity(source, parameters)
    version = snapshot[0]
    parent = root / 'policies'
    with process_lock(root, '.registry.lock', blocking=True):
        parent.mkdir(exist_ok=True)
        target = parent / version
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix='.native-', dir=parent) as temporary:
                staged = Path(temporary) / 'snapshot'
                staged.mkdir()
                metadata = build_bundle(source, staged / 'policy.zip', parameters, snapshot=snapshot)
                write_json(staged / 'build.json', {**metadata, 'bundle_sha256': file_digest(staged / 'policy.zip')})
                with ZipFile(staged / 'policy.zip') as archive:
                    for name in archive.namelist():
                        if name.startswith(('source/', 'sdk/')) or name == 'params.json':
                            path = staged / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(archive.read(name))
                staged.rename(target)
    return _resolve_policy(root, version)


def select_policy(directory, value, parameters=None, device=None, entries=None):
    """Resolve one public selector without accessing a database or competition."""
    from .builtin import wait_spec
    from .entries import verify_artifacts
    value, entries = str(value), entries or {}
    if value.startswith('baseline/'):
        kind = value[len('baseline/'):]
        if kind == 'wait':
            return wait_spec()
        from generals_baselines import AI_KINDS
        if kind not in AI_KINDS:
            raise ValueError(f'unknown baseline; choose from {", ".join((*AI_KINDS, "wait"))}')
        return {'name': 'random' if kind == 'random' else f'baseline/{kind}',
                'kind': 'baseline', 'policy': kind}
    if value.startswith('policy/'):
        return _resolve_policy(directory, value[len('policy/'):], device=device, entries=entries)
    if value.startswith('@'):
        entry = next((entry for entry in entries.values() if entry['name'] == value[1:]), None)
        if entry is None:
            raise ValueError('@name must identify a registered native or packaged policy')
        if entry['name'] == 'WAIT' and entry['kind'] == 'custom' and entry['factory'] == wait_spec()['factory']:
            verify_artifacts(entry)
            return wait_spec()
        if entry['kind'] not in ('cpp', 'package'):
            raise ValueError('@name must identify a registered native or packaged policy')
        verify_artifacts(entry)
        if entry['kind'] == 'package':
            return {'name': f'package-{entry["version"][:16]}', 'kind': 'package',
                    'version': entry['version'], 'bundle': entry['factory_kwargs']['bundle'],
                    'device': device or entry['factory_kwargs']['device']}
        return {'name': f'cpp-{entry["version"][:16]}', 'kind': 'cpp',
                'version': entry['version'], 'bundle': entry['artifact_paths']['bundle']}
    if (Path(value) / 'manifest.json').is_file():
        if parameters is not None:
            raise ValueError('--params applies to native policies')
        spec = snapshot_bundle(directory, value, device=device or 'cpu')
        return {**spec, 'name': f'package-{spec["version"][:16]}'}
    return snapshot_policy(directory, value, parameters)
