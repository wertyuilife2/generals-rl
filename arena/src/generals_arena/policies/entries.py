"""Resolve named, immutable player versions without importing their algorithms."""
from contextlib import suppress
from importlib import import_module
from pathlib import Path

from generals.protocol.controller import Controller

from generals_arena.util import canonical, digest, file_digest, source_digest


def _policy_implementation(policy_sha256):
    return {"custom_version": policy_sha256}


def prepare_entrant(spec, *, base_dir=Path(".")):
    """Separate portable version identity from local artifact locations."""
    spec = dict(spec)
    name, kind = spec.get("name"), spec.get("kind")
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise ValueError("each entrant needs a nonempty name of at most 128 characters")
    common = {"name", "kind", "version", "execution"}
    allowed = {"baseline": {"policy"},
               "custom": {"factory", "kwargs", "artifacts"}, "cpp": {"bundle"},
               "package": {"bundle", "device"}}
    if kind not in allowed or set(spec) - common - allowed[kind]:
        raise ValueError(f"invalid entrant kind or unknown fields: {spec}")
    version = spec.get("version", "")
    if not isinstance(version, str):
        raise ValueError("entrant version must be a string")
    from .limits import execution_limits
    execution = execution_limits(spec)
    hashes, artifacts = {}, {}

    def artifact(key, value):
        path = Path(value)
        path = (Path(base_dir) / path).resolve() if not path.is_absolute() else path.resolve()
        hashes[key], artifacts[key] = file_digest(path), str(path)
        return str(path)

    if kind == "baseline":
        factory = "generals_baselines.registry:create_controller"
        from generals_baselines import AI_KINDS
        if spec.get("policy") not in AI_KINDS:
            raise ValueError(f"baseline policy must be one of {AI_KINDS}")
        kwargs = {"kind": spec["policy"]}
        implementation = {"generals_baselines": source_digest("generals_baselines")}
    elif kind == "cpp":
        from .native import bundle_info
        bundle = artifact("bundle", spec["bundle"])
        metadata = bundle_info(bundle)
        if version and version != metadata['version']:
            raise ValueError('C++ version must match the immutable bundle')
        version = metadata['version']
        factory = 'generals_arena.policies.native:create_controller'
        kwargs = {'bundle': bundle}
        implementation = {'custom_version': version}
    elif kind == "package":
        from generals_arena.policies.artifacts import validate_bundle
        root = Path(spec["bundle"])
        root = (Path(base_dir) / root).resolve() if not root.is_absolute() else root.resolve()
        info = validate_bundle(root)
        if version and version != info["policy_id"]:
            raise ValueError("package version must match the immutable bundle")
        version = info["policy_id"]
        factory = "generals_arena.policies.artifacts:create_controller"
        kwargs = {"bundle": str(root), "device": spec.get("device", "cpu")}
        artifacts["bundle"] = str(root)
        hashes["bundle"] = version
        implementation = _policy_implementation(info["manifest"]["artifacts"]["code"]["sha256"])
    else:
        factory = spec.get("factory")
        if not isinstance(factory, str) or factory.count(":") != 1 or not all(factory.split(":")):
            raise ValueError("custom factory must be module:callable")
        if not version.strip():
            raise ValueError("custom entrants require a code version (for example a Git commit)")
        kwargs = dict(spec.get("kwargs", {}))
        for key, value in spec.get("artifacts", {}).items():
            if key in kwargs:
                raise ValueError(f"artifact {key} duplicates a factory kwarg")
            kwargs[key] = artifact(key, value)
        implementation = {"custom_version": version}
    if {"player_id", "seed"} & kwargs.keys():
        raise ValueError("player_id and seed are supplied by the arena")
    identity_kwargs = {key: ({"sha256": hashes[key]} if key in hashes else value)
                       for key, value in kwargs.items()}
    identity = {"name": name, "kind": kind, "version": version, "factory": factory,
                "kwargs": identity_kwargs, "implementation": implementation}
    # Validate JSON values before opening any output files.
    canonical(identity)
    return {"entrant_id": digest(identity), **identity, "factory_kwargs": kwargs,
            "artifact_paths": artifacts, "artifact_sha256": hashes, "execution": execution}


def verify_artifacts(entrant):
    """Verify registered bytes and return package metadata when applicable."""
    info = None
    for key, path in entrant["artifact_paths"].items():
        if key == "bundle" and entrant["kind"] == "package":
            from generals_arena.policies.artifacts import validate_bundle
            info = validate_bundle(path)
            actual = info["policy_id"]
        else:
            actual = file_digest(path)
        if actual != entrant["artifact_sha256"][key]:
            raise ValueError(f"artifact changed after registration: {entrant['name']}/{key}")
    return info


def load_controller(entrant, rules=None, *, resources=None):
    """Load a controller; defer map validation when its rules are not known yet."""
    kwargs = dict(entrant["factory_kwargs"])
    if entrant["kind"] == "cpp":
        # The native loader verifies the exact bytes it subsequently extracts.
        kwargs["expected_sha256"] = entrant["artifact_sha256"]["bundle"]
        if resources is not None:
            from .native import load_resources
            key = kwargs['expected_sha256']
            if key not in resources:
                resources[key] = load_resources(kwargs['bundle'], key)
            if hasattr(resources, 'move_to_end'):
                resources.move_to_end(key)
            kwargs['_resources'] = resources[key]
            while len(resources) > 32:
                del resources[next(iter(resources))]

    else:
        verify_artifacts(entrant)
    module, name = entrant["factory"].split(":")
    factory = getattr(import_module(module), name, None)
    if not callable(factory):
        raise ValueError(f"controller factory is not callable: {entrant['factory']}")
    controller = factory(player_id=0, seed=0, **kwargs)
    from .limits import execution_limits
    controller._arena_execution = execution_limits(entrant)
    try:
        if not isinstance(controller, Controller) or not all(
                callable(getattr(controller, method, None)) for method in ("reset", "act", "on_result")):
            raise TypeError(f"{entrant['name']} must implement reset, act, and on_result")
        validate = getattr(controller, "validate_rules", None)
        if validate is not None and rules is not None:
            validate(rules.map, rules.visibility)
        return controller
    except BaseException:
        close = getattr(controller, 'close', None)
        if callable(close):
            with suppress(Exception):
                close()
        raise
