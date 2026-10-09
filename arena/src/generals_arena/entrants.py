"""Resolve named, immutable player versions without importing their algorithms."""
from pathlib import Path

from .util import canonical, digest, file_digest, source_digest


def prepare_entrant(spec, *, base_dir=Path(".")):
    """Separate portable version identity from local artifact locations."""
    spec = dict(spec)
    name, kind = spec.get("name"), spec.get("kind")
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise ValueError("each entrant needs a nonempty name of at most 128 characters")
    common = {"name", "kind", "version"}
    allowed = {"baseline": {"policy"}, "rl": {"checkpoint", "config", "device"},
               "custom": {"factory", "kwargs", "artifacts"}}
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
    elif kind == "rl":
        factory = "generals_rl.arena_adapter:CTSController"
        checkpoint = artifact("checkpoint", spec["checkpoint"])
        config = (artifact("config_path", spec["config"]) if "config" in spec else
                  artifact("config_path", Path(checkpoint).with_name("run_config.json")))
        kwargs = {"checkpoint": checkpoint, "config_path": config, "device": spec.get("device", "cpu")}
        implementation = {package: source_digest(package) for package in ("generals_rl", "rsl_rl")}
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
    for key, path in entrant["artifact_paths"].items():
        if file_digest(path) != entrant["artifact_sha256"][key]:
            raise ValueError(f"artifact changed after registration: {entrant['name']}/{key}")
