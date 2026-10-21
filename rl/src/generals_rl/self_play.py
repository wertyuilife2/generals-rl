"""Versioned, frozen policy snapshots for episode-stable self-play opponents.

The pool has no dependency on RSL or a particular policy class. A factory builds
fresh modules and snapshots contain only a detached ``state_dict``. In
particular, we never deepcopy a live policy whose cached distribution may hold
non-leaf autograd tensors. Callers retain ``PolicySnapshot`` handles throughout
an episode: pruning a candidate from the pool does not change an active game.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch
from torch import nn


SCHEMA_VERSION = 1


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class PolicySnapshot:
    snapshot_id: int
    model: nn.Module
    metadata: Mapping[str, Any]


class HistoricalPolicyPool:
    """Uniform candidate pool with bounded disk storage and frozen opponents.

    ``policy_spec`` describes the architecture/MDP dimensions and must match on
    reload. ``resume_from`` copies validated snapshots into a new destination;
    ``resume_iteration`` excludes models newer than the resumed checkpoint.
    Active handles may temporarily keep pruned models in memory until their
    episodes finish. They are never sampled again after pruning.
    ``retain_seed=False`` (the default) keeps the most recent ``max_size``
    publications once the pool fills, allowing the initial policy to expire.
    Setting it to true preserves the original seed plus the newest policies.
    """

    def __init__(self, directory, *, policy_factory: Callable[[], nn.Module],
                 policy_spec: Mapping[str, Any], device="cpu", max_size=8,
                 resume_from=None, resume_iteration=None, read_only=False,
                 retain_seed=False):
        if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size < 2:
            raise ValueError("max_size must be at least 2")
        if resume_iteration is not None and resume_from is None:
            raise ValueError("resume_iteration requires resume_from")
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be boolean")
        if not isinstance(retain_seed, bool):
            raise TypeError("retain_seed must be boolean")
        if read_only and resume_from is not None:
            raise ValueError("read-only pools open the writer's destination; only the writer resumes")
        self.directory = Path(directory).resolve()
        self.policy_factory = policy_factory
        self.policy_spec = _json_copy(dict(policy_spec))
        self.policy_spec_sha256 = _json_hash(self.policy_spec)
        self.device = torch.device(device)
        self.max_size = max_size
        self.read_only = read_only
        self.retain_seed = retain_seed
        self._snapshots: dict[int, PolicySnapshot] = {}
        self._next_id = 0
        self._resume_source = None

        if read_only:
            self.refresh()
            return

        manifest_path = self.directory / "manifest.json"
        if resume_from is not None:
            source = Path(resume_from).resolve()
            if source.is_file():
                if source.name != "manifest.json":
                    raise ValueError("resume_from must be a pool directory or manifest.json")
                source = source.parent
            if source == self.directory:
                raise ValueError("resuming a pool must use a new destination directory")
            if self.directory.exists() and any(self.directory.iterdir()):
                raise FileExistsError(f"pool destination is not empty: {self.directory}")
            manifest, snapshots = self._read_pool(source)
            if resume_iteration is not None:
                snapshots = [s for s in snapshots if s.metadata["iteration"] <= resume_iteration]
                if self.retain_seed and snapshots and not snapshots[0].metadata["seed"]:
                    raise ValueError("resumed pool must retain its original seed snapshot")
                if manifest["snapshots"] and not snapshots:
                    raise ValueError("resume_iteration predates the available pool snapshots")
            snapshots = self._bounded(snapshots)
            self.directory.mkdir(parents=True, exist_ok=True)
            for snapshot in snapshots:
                # Copy exact bytes so hashes remain comparable with the source.
                filename = snapshot.metadata["filename"]
                (self.directory / filename).write_bytes((source / filename).read_bytes())
                self._snapshots[snapshot.snapshot_id] = snapshot
            self._next_id = manifest["next_id"]
            self._resume_source = {"directory": str(source),
                                   "manifest_sha256": _file_hash(source / "manifest.json"),
                                   "iteration_cutoff": resume_iteration}
            self._write_manifest()
        elif manifest_path.exists():
            manifest, snapshots = self._read_pool(self.directory)
            kept = self._bounded(snapshots)
            self._snapshots = {s.snapshot_id: s for s in kept}
            self._next_id = manifest["next_id"]
            self._resume_source = manifest.get("resume_source")
            self._write_manifest()
            for snapshot in snapshots:
                if snapshot.snapshot_id not in self._snapshots:
                    (self.directory / snapshot.metadata["filename"]).unlink()
        else:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise FileExistsError(f"pool directory has no manifest and is not empty: {self.directory}")
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    @property
    def ids(self):
        return tuple(self._snapshots)

    def __len__(self):
        return len(self._snapshots)

    def get(self, snapshot_id):
        return self._snapshots[int(snapshot_id)]

    def refresh(self):
        """Read a completed publication, retaining unchanged frozen handles.

        Distributed callers synchronize with the writer before refreshing, so
        checkpoint pruning cannot race these reads. Active episodes keep their
        own references to handles no longer present in the candidate manifest.
        """
        if not self.read_only:
            raise RuntimeError("refresh requires a read-only pool")
        manifest, snapshots = self._read_pool(self.directory, reuse=self._snapshots)
        capacity = manifest.get("max_size")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 2:
            raise ValueError("invalid shared historical pool capacity")
        self._snapshots = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
        self.max_size = capacity
        self._next_id = manifest["next_id"]
        self._resume_source = manifest.get("resume_source")

    def sample(self, count, *, generator: torch.Generator):
        """Return shared immutable handles; sampling only happens on reset."""
        if count < 0:
            raise ValueError("sample count cannot be negative")
        if count == 0:
            return []
        candidates = list(self._snapshots.values())
        if not candidates:
            raise RuntimeError("publish an initial snapshot before sampling opponents")
        positions = torch.randint(len(candidates), (count,), generator=generator,
                                  device=generator.device).cpu().tolist()
        return [candidates[index] for index in positions]

    def publish(self, source_policy: nn.Module, *, iteration: int, source=None,
                seed=False):
        """Freeze and persist weights, then make the new snapshot selectable.

        The first snapshot is marked as the bootstrap seed. ``source`` is arbitrary
        JSON provenance, for example the checkpoint path and SHA256.
        """
        if self.read_only:
            raise PermissionError("cannot publish to a read-only historical pool")
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < -1:
            raise ValueError("iteration must be an integer >= -1")
        if seed and self._snapshots:
            raise ValueError("only the first snapshot may be the seed")
        seed = not self._snapshots
        source = _json_copy(dict(source or {}))
        snapshot_id = self._next_id
        state = {key: value.detach().cpu().clone()
                 for key, value in source_policy.state_dict().items()}
        if any(not bool(torch.isfinite(value).all()) for value in state.values()
               if value.is_floating_point() or value.is_complex()):
            raise FloatingPointError("cannot publish non-finite policy weights")
        model = self._make_model(state)
        filename = f"snapshot_{snapshot_id:08d}.pt"
        target = self.directory / filename
        temporary = self.directory / (filename + ".tmp")
        if target.exists():
            raise FileExistsError(f"snapshot file already exists: {target}")
        payload = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
                   "iteration": iteration, "seed": seed, "source": source,
                   "policy_spec_sha256": self.policy_spec_sha256,
                   "model_state_dict": state}
        try:
            torch.save(payload, temporary)
            temporary.replace(target)
            metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
            metadata.update(filename=filename, sha256=_file_hash(target),
                            created_utc=datetime.now(timezone.utc).isoformat(),
                            tensor_count=len(state))
            snapshot = PolicySnapshot(snapshot_id, model, MappingProxyType(metadata))
            candidates = list(self._snapshots.values()) + [snapshot]
            kept = self._bounded(candidates)
            previous = self._snapshots
            self._snapshots = {s.snapshot_id: s for s in kept}
            self._next_id += 1
            try:
                self._write_manifest()
            except BaseException:
                self._snapshots = previous
                self._next_id -= 1
                target.unlink(missing_ok=True)
                raise
            for obsolete in candidates:
                if obsolete.snapshot_id not in self._snapshots:
                    (self.directory / obsolete.metadata["filename"]).unlink()
            return snapshot
        finally:
            temporary.unlink(missing_ok=True)

    def describe(self):
        """JSON-safe metadata suitable for a training report/checkpoint info."""
        return {"directory": str(self.directory), **self._manifest()}

    def _bounded(self, snapshots):
        snapshots = list(snapshots)
        if not self.retain_seed:
            return snapshots[-self.max_size:]
        while len(snapshots) > self.max_size:
            index = next(i for i, value in enumerate(snapshots) if not value.metadata["seed"])
            del snapshots[index]
        return snapshots

    def _make_model(self, state):
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        # Factory initialization must not perturb the learner's sampling RNG.
        with torch.random.fork_rng(devices=devices):
            model = self.policy_factory()
            if not isinstance(model, nn.Module):
                raise TypeError("policy_factory must return a torch.nn.Module")
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise ValueError("snapshot weights do not match policy_factory") from exc
            model = model.to(self.device)
        model.requires_grad_(False)
        model.eval()
        return model

    def _manifest(self):
        return {"schema_version": SCHEMA_VERSION, "format": "torch_state_dict",
                "policy_spec": self.policy_spec, "policy_spec_sha256": self.policy_spec_sha256,
                "max_size": self.max_size, "retain_seed": self.retain_seed, "next_id": self._next_id,
                "resume_source": self._resume_source,
                "snapshots": [dict(value.metadata) for value in self._snapshots.values()]}

    def _write_manifest(self):
        temporary = self.directory / "manifest.json.tmp"
        try:
            temporary.write_text(json.dumps(self._manifest(), indent=2, allow_nan=False) + "\n")
            temporary.replace(self.directory / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _read_pool(self, directory, *, reuse=None):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("format") != "torch_state_dict":
            raise ValueError("unsupported historical pool version or format")
        # Old manifests predate this option and always retained their seed.
        retain_seed = manifest.get("retain_seed", True)
        if not isinstance(retain_seed, bool) or retain_seed != self.retain_seed:
            raise ValueError("historical pool retain_seed differs from current retention mode")
        if (manifest.get("policy_spec_sha256") != self.policy_spec_sha256
                or _json_hash(manifest.get("policy_spec")) != self.policy_spec_sha256):
            raise ValueError("historical pool policy_spec differs from current architecture/MDP")
        entries = manifest.get("snapshots")
        next_id = manifest.get("next_id")
        if not isinstance(entries, list) or not isinstance(next_id, int) or next_id < 0:
            raise ValueError("invalid pool snapshot list/next_id")
        snapshots = []
        seen = set()
        for index, metadata in enumerate(entries):
            snapshot_id = metadata.get("snapshot_id")
            filename = metadata.get("filename")
            if (not isinstance(snapshot_id, int) or snapshot_id < 0 or snapshot_id in seen
                    or snapshot_id >= next_id or filename != f"snapshot_{snapshot_id:08d}.pt"
                    or not isinstance(metadata.get("seed"), bool)
                    or metadata["seed"] != (snapshot_id == 0)
                    or self.retain_seed and index == 0 and not metadata["seed"]):
                raise ValueError("invalid historical snapshot identity/seed")
            if seen and snapshot_id <= max(seen):
                raise ValueError("snapshot IDs must be strictly increasing")
            seen.add(snapshot_id)
            cached = None if reuse is None else reuse.get(snapshot_id)
            if cached is not None:
                if dict(cached.metadata) != metadata:
                    raise ValueError("published historical snapshot metadata changed")
                snapshots.append(cached)
                continue
            path = directory / filename
            if _file_hash(path) != metadata.get("sha256"):
                raise ValueError(f"historical snapshot SHA256 mismatch: {filename}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("schema_version", "snapshot_id", "iteration", "seed", "source", "policy_spec_sha256"):
                if payload.get(key) != metadata.get(key):
                    raise ValueError(f"historical snapshot metadata mismatch: {filename}: {key}")
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported snapshot schema")
            state = payload.get("model_state_dict")
            if not isinstance(state, dict) or len(state) != metadata.get("tensor_count"):
                raise ValueError("invalid snapshot state dictionary")
            if any(not isinstance(value, torch.Tensor) for value in state.values()):
                raise ValueError("snapshot state must contain only tensors")
            if any(not bool(torch.isfinite(value).all()) for value in state.values()
                   if value.is_floating_point() or value.is_complex()):
                raise ValueError("snapshot contains non-finite weights")
            model = self._make_model(state)
            snapshots.append(PolicySnapshot(snapshot_id, model, MappingProxyType(_json_copy(metadata))))
        return manifest, snapshots
