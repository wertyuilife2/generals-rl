"""Frozen opponents, episode-held handles, and historical pool provenance."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import Categorical

from generals_rl.self_play import HistoricalPolicyPool


class TinyPolicy(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.actor = nn.Linear(width, 3)
        self.register_buffer("clock", torch.tensor(0.0))
        self.distribution = None

    def act_inference(self, obs):
        logits = self.actor(obs["policy"])
        logits = logits.masked_fill(~obs["action_mask"], -1e9)
        self.distribution = Categorical(logits=logits)
        return self.distribution.probs.argmax(-1, keepdim=True)


SPEC = {"policy_class": "TinyPolicy", "features": 3, "actions": 3, "observation_version": 1}


def pool(path, **kwargs):
    return HistoricalPolicyPool(path, policy_factory=TinyPolicy,
                                policy_spec=SPEC, **kwargs)


def file_hashes(path):
    return {value.name: sha256(value.read_bytes()).hexdigest()
            for value in Path(path).iterdir() if value.is_file()}


class HistoricalPoolTests(unittest.TestCase):
    def test_default_pool_evicts_bootstrap_and_samples_only_newest_eight(self):
        with tempfile.TemporaryDirectory() as directory:
            history = pool(directory, max_size=8)
            source = TinyPolicy()
            bootstrap = history.publish(source, iteration=-1)
            original = bootstrap.model.actor.weight.clone()
            for iteration in range(12):
                with torch.no_grad():
                    source.actor.weight.add_(0.1)
                history.publish(source, iteration=iteration)
            self.assertEqual(history.ids, tuple(range(5, 13)))
            self.assertFalse((Path(directory) / bootstrap.metadata["filename"]).exists())
            torch.testing.assert_close(bootstrap.model.actor.weight, original, rtol=0, atol=0)
            candidates = history.sample(1000, generator=torch.Generator().manual_seed(8))
            self.assertEqual({snapshot.snapshot_id for snapshot in candidates}, set(range(5, 13)))
            self.assertFalse(any(snapshot.metadata["seed"] for snapshot in candidates))
            self.assertEqual(len(list(Path(directory).glob("snapshot_*.pt"))), 8)
            self.assertFalse(history.describe()["retain_seed"])

    def test_recent_pool_resume_and_reader_preserve_retention_after_seed_expiry(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            history = pool(source, max_size=3, retain_seed=False)
            policy = TinyPolicy()
            for iteration in range(6):
                history.publish(policy, iteration=iteration)
            reader = pool(source, read_only=True, retain_seed=False)
            self.assertEqual(reader.ids, (3, 4, 5))
            active = reader.get(3)
            history.publish(policy, iteration=6)
            reader.refresh()
            self.assertEqual(reader.ids, (4, 5, 6))
            self.assertFalse(active.metadata["seed"])
            before = file_hashes(source)
            resumed = pool(Path(directory) / "resumed", max_size=2, retain_seed=False,
                           resume_from=source, resume_iteration=5)
            self.assertEqual(resumed.ids, (4, 5))
            self.assertEqual(resumed.publish(policy, iteration=6).snapshot_id, 7)
            self.assertEqual(resumed.ids, (5, 7))
            reloaded = pool(resumed.directory, max_size=2, retain_seed=False)
            self.assertEqual(reloaded.describe(), resumed.describe())
            self.assertEqual(file_hashes(source), before)
            with self.assertRaisesRegex(ValueError, "predates"):
                pool(Path(directory) / "too_old", retain_seed=False, resume_from=source,
                     resume_iteration=3)
            for kwargs in ({}, {"read_only": True}, {"resume_from": source}):
                destination = source if "resume_from" not in kwargs else Path(directory) / "wrong_mode"
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "retain_seed"):
                    pool(destination, retain_seed=True, **kwargs)
            self.assertEqual(file_hashes(source), before)

    def test_legacy_manifest_without_retention_field_uses_original_pinned_seed_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            history = pool(directory, max_size=2, retain_seed=True)
            for iteration in range(4):
                history.publish(TinyPolicy(), iteration=iteration)
            path = Path(directory) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.pop("retain_seed")
            path.write_text(json.dumps(manifest))
            before = file_hashes(directory)
            reader = pool(directory, read_only=True, retain_seed=True)
            self.assertEqual(reader.ids, (0, 3))
            self.assertTrue(reader.retain_seed)
            self.assertEqual(file_hashes(directory), before)
            reloaded = pool(directory, retain_seed=True)
            self.assertEqual(reloaded.ids, (0, 3))
            with self.assertRaisesRegex(ValueError, "retain_seed"):
                pool(directory)
            with self.assertRaisesRegex(TypeError, "boolean"):
                pool(Path(directory) / "invalid", retain_seed=1)

    def test_snapshot_ignores_nonleaf_distribution_and_freezes_independent_weights(self):
        source = TinyPolicy()
        obs = {"policy": torch.ones(2, 3), "action_mask": torch.tensor([[False, True, True]] * 2)}
        source.act_inference(obs)
        self.assertIsNotNone(source.distribution.logits.grad_fn)
        with self.assertRaises(RuntimeError):
            deepcopy(source)  # The real CTS distribution has the same issue.
        with tempfile.TemporaryDirectory() as directory:
            history = pool(directory)
            rng_before = torch.get_rng_state().clone()
            snapshot = history.publish(source, iteration=-1, source={"kind": "initial_weights"})
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertTrue(snapshot.metadata["seed"])
            self.assertFalse(snapshot.model.training)
            self.assertTrue(all(not p.requires_grad for p in snapshot.model.parameters()))
            self.assertIsNone(snapshot.model.distribution)
            before = {key: value.clone() for key, value in snapshot.model.state_dict().items()}
            with torch.no_grad():
                for parameter in source.parameters():
                    parameter.add_(10)
                source.clock.add_(20)
            for key, value in snapshot.model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))
                self.assertNotEqual(value.data_ptr(), source.state_dict()[key].data_ptr())
            with torch.inference_mode():
                actions = snapshot.model.act_inference(obs)
            self.assertTrue(obs["action_mask"].gather(1, actions).all())
            payload = torch.load(Path(directory) / snapshot.metadata["filename"], weights_only=True)
            self.assertEqual(set(payload), {"schema_version", "snapshot_id", "iteration", "seed",
                                           "source", "policy_spec_sha256", "model_state_dict"})
            self.assertFalse(any(value.requires_grad for value in payload["model_state_dict"].values()))

    def test_pruning_retains_seed_and_active_handles_without_sampling_evicted_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            history = pool(directory, max_size=3, retain_seed=True)
            source = TinyPolicy()
            snapshots = [history.publish(source, iteration=i - 1) for i in range(3)]
            active = snapshots[1]
            frozen_weights = active.model.actor.weight.clone()
            with torch.no_grad():
                source.actor.weight.add_(1)
            newest = history.publish(source, iteration=3)
            self.assertEqual(history.ids, (0, 2, 3))
            self.assertFalse((Path(directory) / active.metadata["filename"]).exists())
            self.assertTrue(torch.equal(active.model.actor.weight, frozen_weights))
            self.assertIs(history.get(newest.snapshot_id), newest)
            assigned = history.sample(100, generator=torch.Generator().manual_seed(11))
            self.assertEqual({value.snapshot_id for value in assigned}, {0, 2, 3})
            history.publish(source, iteration=4)
            self.assertEqual(history.ids, (0, 3, 4))
            self.assertEqual(len(list(Path(directory).glob("snapshot_*.pt"))), 3)
            with self.assertRaises(ValueError):
                history.publish(source, iteration=5, seed=True)

    def test_sampling_uses_only_explicit_generator_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            history = pool(directory)
            for i in range(4):
                history.publish(TinyPolicy(), iteration=i)
            before = torch.get_rng_state().clone()
            first = history.sample(100, generator=torch.Generator().manual_seed(999))
            second = history.sample(100, generator=torch.Generator().manual_seed(999))
            self.assertEqual([h.snapshot_id for h in first], [h.snapshot_id for h in second])
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            self.assertEqual(history.sample(0, generator=torch.Generator()), [])
            with self.assertRaises(ValueError):
                history.sample(-1, generator=torch.Generator())

    def test_read_only_refresh_reuses_models_without_writes_and_retains_active_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = pool(directory, max_size=3, retain_seed=True)
            source = TinyPolicy()
            for iteration in (-1, 0, 1):
                writer.publish(source, iteration=iteration)
            before = file_hashes(directory)
            with patch.object(HistoricalPolicyPool, "_write_manifest", side_effect=AssertionError("reader wrote")), \
                    patch.object(Path, "mkdir", side_effect=AssertionError("reader created directory")):
                reader = pool(directory, read_only=True, max_size=2, retain_seed=True)
            self.assertEqual(file_hashes(directory), before)
            self.assertEqual(reader.ids, writer.ids)  # Readers never prune the shared pool.
            seed, active, kept = (reader.get(snapshot_id) for snapshot_id in reader.ids)
            frozen = active.model.actor.weight.clone()
            with torch.no_grad():
                source.actor.weight.add_(1)
            writer.publish(source, iteration=2)
            self.assertEqual(reader.ids, (0, 1, 2))  # Refresh only after publication synchronization.
            before = file_hashes(directory)
            rng = torch.get_rng_state().clone()
            with patch.object(reader, "_make_model", wraps=reader._make_model) as construct:
                reader.refresh()
                self.assertEqual(construct.call_count, 1)
                reader.refresh()
                self.assertEqual(construct.call_count, 1)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(file_hashes(directory), before)
            self.assertEqual(reader.ids, (0, 2, 3))
            self.assertIs(reader.get(0), seed)
            self.assertIs(reader.get(2), kept)
            self.assertTrue(torch.equal(active.model.actor.weight, frozen))
            self.assertFalse((Path(directory) / active.metadata["filename"]).exists())
            self.assertEqual(reader.describe(), writer.describe())
            sampled = reader.sample(100, generator=torch.Generator().manual_seed(99))
            self.assertEqual({snapshot.snapshot_id for snapshot in sampled}, {0, 2, 3})
            with self.assertRaisesRegex(PermissionError, "read-only"):
                reader.publish(source, iteration=3)
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                writer.refresh()
            self.assertEqual(file_hashes(directory), before)

    def test_read_only_open_and_failed_refresh_do_not_mutate_existing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(FileNotFoundError):
                pool(missing, read_only=True)
            self.assertFalse(missing.exists())
            writer = pool(Path(directory) / "writer")
            writer.publish(TinyPolicy(), iteration=-1)
            with self.assertRaisesRegex(ValueError, "only the writer resumes"):
                pool(missing, read_only=True, resume_from=writer.directory)
            with self.assertRaisesRegex(TypeError, "boolean"):
                pool(missing, read_only=1)
            reader = pool(writer.directory, read_only=True)
            original = reader.get(0)
            manifest_path = writer.directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["snapshots"][0]["iteration"] = 8
            manifest_path.write_text(json.dumps(manifest))
            before = file_hashes(writer.directory)
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                reader.refresh()
            self.assertEqual(reader.ids, (0,))
            self.assertIs(reader.get(0), original)
            self.assertEqual(file_hashes(writer.directory), before)

    def test_resume_copies_hash_identical_weights_and_filters_future_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "old"
            new_path = Path(directory) / "new"
            history = pool(old_path, retain_seed=True)
            source = TinyPolicy()
            for iteration in (-1, 10, 20, 30):
                with torch.no_grad():
                    source.clock.add_(1)
                history.publish(source, iteration=iteration,
                                source={"checkpoint_sha256": str(iteration), "version": 1})
            original_hashes = file_hashes(old_path)
            rng_before = torch.get_rng_state().clone()
            resumed = pool(new_path, resume_from=old_path, resume_iteration=20, max_size=2,
                           retain_seed=True)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertEqual(resumed.ids, (0, 2))
            self.assertEqual(resumed.describe()["next_id"], 4)
            self.assertEqual(resumed.describe()["resume_source"]["iteration_cutoff"], 20)
            for snapshot_id in resumed.ids:
                self.assertEqual(dict(resumed.get(snapshot_id).metadata), dict(history.get(snapshot_id).metadata))
                for key, value in resumed.get(snapshot_id).model.state_dict().items():
                    self.assertTrue(torch.equal(value, history.get(snapshot_id).model.state_dict()[key]))
            newer = resumed.publish(source, iteration=25)
            self.assertEqual(newer.snapshot_id, 4)
            self.assertEqual(file_hashes(old_path), original_hashes)
            reloaded = pool(new_path, max_size=2, retain_seed=True)
            self.assertEqual(reloaded.ids, (0, 4))
            self.assertEqual(reloaded.describe(), resumed.describe())
            manifest = json.loads((new_path / "manifest.json").read_text())
            for metadata in manifest["snapshots"]:
                self.assertEqual(_hash_file(new_path / metadata["filename"]), metadata["sha256"])

    def test_integrity_architecture_version_and_factory_are_checked_on_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            history = pool(source)
            snapshot = history.publish(TinyPolicy(), iteration=-1)
            before = file_hashes(source)
            with self.assertRaisesRegex(ValueError, "policy_spec"):
                HistoricalPolicyPool(Path(directory) / "wrong_spec", policy_factory=TinyPolicy,
                                     policy_spec={**SPEC, "features": 4}, resume_from=source)
            with self.assertRaisesRegex(ValueError, "factory"):
                HistoricalPolicyPool(Path(directory) / "wrong_factory", policy_factory=lambda: TinyPolicy(4),
                                     policy_spec=SPEC, resume_from=source)
            with self.assertRaisesRegex(ValueError, "new destination"):
                pool(source, resume_from=source)
            self.assertEqual(file_hashes(source), before)
            payload = source / snapshot.metadata["filename"]
            payload.write_bytes(payload.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                pool(Path(directory) / "corrupt", resume_from=source)

    def test_empty_pool_and_invalid_capacity_and_cutoff_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                pool(directory, max_size=1)
            history = pool(directory)
            with self.assertRaisesRegex(RuntimeError, "initial snapshot"):
                history.sample(1, generator=torch.Generator())
            history.publish(TinyPolicy(), iteration=10)
            with self.assertRaisesRegex(ValueError, "predates"):
                pool(Path(directory) / "before_seed", resume_from=directory, resume_iteration=9)
            with self.assertRaises(ValueError):
                pool(Path(directory) / "cutoff_only", resume_iteration=9)


def _hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
