"""Resolve named, immutable player versions without importing their algorithms."""
from contextlib import suppress
from importlib import import_module
from pathlib import Path

from generals.protocol.controller import Controller

from generals_arena.util import canonical, digest, file_digest, source_digest


def _policy_implementation(policy_sha256):
    return {"custom_version": digest({
        "policy_sha256": policy_sha256,
        "loader_sha256": digest({name: file_digest(Path(__file__).with_name(name))
                                  for name in ("artifacts.py", "code.py")}),
    })}


def prepare_entrant(spec, *, base_dir=Path(".")):
    """Separate portable version identity from local artifact locations."""
    spec = dict(spec)
    name, kind = spec.get("name"), spec.get("kind")
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise ValueError("each entrant needs a nonempty name of at most 128 characters")
    common = {"name", "kind", "version"}
    allowed = {"baseline": {"policy"},
               "custom": {"factory", "kwargs", "artifacts"}, "cpp": {"bundle"},
               "package": {"bundle", "device"}}
    if kind not in allowed or set(spec) - common - allowed[kind]:
        raise ValueError(f"invalid entrant kind or unknown fields: {spec}")
    version = spec.get("version", "")
    if not isinstance(version, str):
        raise ValueError("entrant version must be a string")
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
        from .native import bundle_info, execution_fingerprint
        bundle = artifact("bundle", spec["bundle"])
        metadata = bundle_info(bundle)
        if version and version != metadata['version']:
            raise ValueError('C++ version must match the immutable bundle')
        version = metadata['version']
        factory = 'generals_arena.policies.native:create_controller'
        kwargs = {'bundle': bundle}
        implementation = {'custom_version': digest({'bundle_version': version,
                                                     'host_sha256': execution_fingerprint()})}
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
            "artifact_paths": artifacts, "artifact_sha256": hashes}


def verify_artifacts(entrant):
    """Verify registered bytes and return package metadata when applicable."""
    info = None
    if entrant["kind"] == "cpp":
        from .native import execution_fingerprint
        implementation = {'custom_version': digest({'bundle_version': entrant['version'],
                                                    'host_sha256': execution_fingerprint()})}
        if entrant['implementation'] != implementation:
            raise ValueError("native host changed after registration; use its recorded execution implementation")
    for key, path in entrant["artifact_paths"].items():
        if key == "bundle" and entrant["kind"] == "package":
            from generals_arena.policies.artifacts import validate_bundle
            info = validate_bundle(path)
            actual = info["policy_id"]
            if entrant["implementation"] != _policy_implementation(info["manifest"]["artifacts"]["code"]["sha256"]):
                raise ValueError("policy loader changed after registration; use the original loader or a new deployment")
        else:
            actual = file_digest(path)
        if actual != entrant["artifact_sha256"][key]:
            raise ValueError(f"artifact changed after registration: {entrant['name']}/{key}")
    return info


def load_controller(entrant, rules):
    """Load one factory and check the shared controller and rules protocol."""
    verify_artifacts(entrant)
    module, name = entrant["factory"].split(":")
    factory = getattr(import_module(module), name, None)
    if not callable(factory):
        raise ValueError(f"controller factory is not callable: {entrant['factory']}")
    controller = factory(player_id=0, seed=0, **entrant["factory_kwargs"])
    try:
        if not isinstance(controller, Controller) or not all(
                callable(getattr(controller, method, None)) for method in ("reset", "act", "on_result")):
            raise TypeError(f"{entrant['name']} must implement reset, act, and on_result")
        validate = getattr(controller, "validate_rules", None)
        if validate is not None:
            validate(rules.map, rules.visibility)
        verify_artifacts(entrant)
        return controller
    except BaseException:
        close = getattr(controller, 'close', None)
        if callable(close):
            with suppress(Exception):
                close()
        raise
