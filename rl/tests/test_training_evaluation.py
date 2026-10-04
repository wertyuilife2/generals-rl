"""Real subprocess evaluations, snapshot eviction and learner RNG isolation."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from generals.controllers import AI_KINDS
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.train import train
from generals_rl.training_evaluation import PeriodicEvaluator, TrainingEvalConfig
from rl_test_utils import assert_nested_equal, small_training_config


def small_config():
    config = small_training_config(runner_overrides={"learning_epochs": 1})
    config["self_play"] = {"pool_size": 2, "snapshot_interval": 1}
    # Retain the default cadence, but keep the actual match workload tiny.
    config["evaluation"] = {"maps_per_ai": 1, "max_ticks": 3, "batch_size": 8}
    return config


class TrainingEvaluationTests(unittest.TestCase):
    def test_default_is_enabled_every_ten_learned_snapshots_and_validates_early(self):
        settings = TrainingEvalConfig()
        self.assertTrue(settings.enabled)
        self.assertEqual(settings.every_n_snapshots, 10)
        self.assertEqual(settings.maps_per_ai, 50)
        cases = ({"enabled": "yes"}, {"every_n_snapshots": 0}, {"maps_per_ai": True},
                 {"batch_size": 0}, {"max_ticks": -1}, {"seed": -1}, {"cpu_threads": 0})
        with tempfile.TemporaryDirectory() as directory:
            for values in cases:
                config = small_config()
                config["evaluation"].update(values)
                output = Path(directory) / "run"
                with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                    train(config, output, iterations=1)
                self.assertFalse(output.exists())

    def test_real_eval_survives_eviction_records_four_ai_in_tensorboard_and_preserves_training(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "enabled"
            config = small_config()
            original = deepcopy(config)
            summary = train(config, output, iterations=13)
            learner_rng_after_eval = torch.get_rng_state().clone()
            self.assertEqual(config, original)
            evidence = summary["evaluation"]
            self.assertEqual((evidence["scheduled"], evidence["completed"], evidence["failed"]), (1, 1, 0))
            point = evidence["latest_completed"]
            self.assertEqual((point["snapshot_id"], point["iteration"], point["publication_index"]), (10, 9, 10))
            self.assertEqual(point["steps_per_environment"], 40)
            self.assertEqual(point["environment_steps"], 160)
            frozen = output / "evaluations/snapshot_00000010/snapshot.pt"
            self.assertTrue(frozen.exists())
            self.assertFalse((output / "history_pool/snapshot_00000010.pt").exists())
            self.assertEqual(sha256(frozen.read_bytes()).hexdigest(), point["checkpoint_sha256"])
            report = json.loads(Path(point["report"]).read_text())
            self.assertEqual(report["training_opponent"], "self_play")
            self.assertEqual(report["device"], "cpu")
            self.assertEqual(report["policy"], "student_local")
            self.assertEqual(report["completed_games"], 8)
            self.assertFalse(report["stochastic"])
            self.assertEqual(set(report["by_ai"]), set(AI_KINDS))
            for metrics in report["by_ai"].values():
                self.assertEqual(metrics["games"], 2)
                self.assertEqual(metrics["wins"] + metrics["losses"] + metrics["draws"], 2)
            events = EventAccumulator(str(output)).Reload()
            for kind in (*AI_KINDS, "overall"):
                logged = events.Scalars(f"Eval/{kind}/win_rate")
                self.assertEqual(len(logged), 1)
                self.assertEqual(logged[0].step, 9)
                metrics = report["overall"] if kind == "overall" else report["by_ai"][kind]
                self.assertAlmostEqual(logged[0].value, metrics["win_rate"])
            jobs = json.loads((output / "evaluations/index.json").read_text())["jobs"]
            self.assertEqual([job["status"] for job in jobs], ["completed"])
            # The independent process must not consume learner RNG, update its
            # policy, alter either optimizer, or change self-play assignments.
            disabled = Path(directory) / "disabled"
            config["evaluation"]["enabled"] = False
            disabled_summary = train(config, disabled, iterations=13)
            torch.testing.assert_close(torch.get_rng_state(), learner_rng_after_eval, rtol=0, atol=0)
            self.assertFalse((disabled / "evaluations").exists())
            actual = torch.load(output / "checkpoint.pt", weights_only=False, map_location="cpu")
            expected = torch.load(disabled / "checkpoint.pt", weights_only=False, map_location="cpu")
            for key in ("model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"):
                assert_nested_equal(self, actual[key], expected[key])
            for key in ("episode_assignments", "active_snapshot_counts"):
                self.assertEqual(summary["self_play"][key], disabled_summary["self_play"][key])

    def test_all_due_evaluations_are_queued_and_reported_in_snapshot_order(self):
        config = small_config()
        config["evaluation"]["every_n_snapshots"] = 2
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            summary = train(config, output, iterations=6)
            self.assertEqual(summary["evaluation"]["completed"], 3)
            self.assertEqual(summary["evaluation"]["queued"], 0)
            self.assertEqual(summary["evaluation"]["running"], 0)
            points = json.loads((output / "evaluations/index.json").read_text())["jobs"]
            self.assertTrue(all(point["status"] == "completed" for point in points))
            self.assertEqual([p["publication_index"] for p in points], [2, 4, 6])
            self.assertEqual([p["iteration"] for p in points], [1, 3, 5])
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual([p.step for p in events.Scalars("Eval/overall/win_rate")], [1, 3, 5])

    def test_retention_keeps_recent_weights_best_and_all_reports(self):
        config = small_config()
        config["evaluation"].update(every_n_snapshots=2, keep_snapshots=1)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            summary = train(config, output, iterations=6)
            evaluations = output / "evaluations"
            jobs = json.loads((evaluations / "index.json").read_text())["jobs"]
            self.assertEqual([Path(job["checkpoint"]).exists() for job in jobs], [False, False, True])
            self.assertTrue(all(Path(job["report"]).is_file() for job in jobs))
            best = summary["evaluation"]["best_completed"]
            self.assertEqual(sha256((evaluations / "best.pt").read_bytes()).hexdigest(), best["checkpoint_sha256"])
            self.assertEqual(json.loads((evaluations / "run_config.json").read_text()), config)
            best_report = json.loads((evaluations / "best_report.json").read_text())
            self.assertEqual(best_report["checkpoint_sha256"], best["checkpoint_sha256"])
            deployed = evaluate_builtin(evaluations / "best.pt", maps_per_ai=1, batch_size=8, max_ticks=3)
            self.assertEqual(deployed["completed_games"], 8)
            self.assertEqual(deployed["checkpoint_sha256"], best["checkpoint_sha256"])
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual([point.step for point in events.Scalars("Eval/worst_ai/win_rate")], [1, 3, 5])

    def test_best_prioritizes_worst_ai_then_aggressive_and_retention_does_not_delete_pending(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, keep_snapshots=1), {}, directory, None)
            queued_checkpoint = evaluator.directory / "queued.pt"
            queued_checkpoint.write_bytes(b"still needed by worker")
            evaluator.jobs.append({"status": "queued", "checkpoint": str(queued_checkpoint)})
            # Higher overall/ aggressive scores cannot compensate for a weaker
            # worst opponent. Exact ties use aggressive, then overall.
            cases = ((0.4, 0.4, 0.8, 1), (0.5, 0.5, 0.6, 2),
                     (0.45, 0.7, 0.9, 2), (0.5, 0.55, 0.55, 4), (0.5, 0.55, 0.7, 5))
            for index, (worst, aggressive, overall, expected_best) in enumerate(cases, 1):
                checkpoint = evaluator.directory / f"candidate_{index}.pt"
                checkpoint.write_bytes(f"weights {index}".encode())
                report_path = evaluator.directory / f"report_{index}.json"
                report = {"checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                          "scheduled_games": 8, "completed_games": 8, "policy": "student_local",
                          "device": "cpu", "stochastic": False, "overall": {"win_rate": overall},
                          "by_ai": {kind: {"games": 2, "win_rate": aggressive if kind == "aggressive" else worst}
                                    for kind in AI_KINDS}}
                report_path.write_text(json.dumps(report))
                job = {"status": "running", "snapshot_id": index, "iteration": index,
                       "publication_index": index, "steps_per_environment": index * 4,
                       "environment_steps": index * 16, "checkpoint": str(checkpoint),
                       "checkpoint_sha256": report["checkpoint_sha256"],
                       "report": str(report_path), "worker_seconds": 0.1}
                evaluator.jobs.append(job)
                evaluator._complete(job)
                self.assertEqual(evaluator.best_completed["snapshot_id"], expected_best)
                self.assertEqual((evaluator.directory / "best.pt").read_bytes(), f"weights {expected_best}".encode())
                self.assertTrue(queued_checkpoint.exists())
            self.assertEqual(len(list(evaluator.directory.glob("candidate_*.pt"))), 1)

    def test_seed_is_not_counted_and_worker_failure_is_visible_without_a_false_win_rate(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "pool"
            source.mkdir()
            path = source / "snapshot_00000010.pt"
            path.write_bytes(b"test frozen snapshot")
            digest = sha256(path.read_bytes()).hexdigest()
            snapshot = SimpleNamespace(snapshot_id=10, metadata={
                "iteration": 499, "filename": path.name, "sha256": digest, "seed": False})
            writer = SummaryWriter(root / "run")
            evaluator = PeriodicEvaluator(TrainingEvalConfig(), {}, root / "run", writer)
            seed = SimpleNamespace(snapshot_id=0, metadata={"seed": True})
            kwargs = {"steps_per_environment": 16000, "environment_steps": 262144000}
            self.assertFalse(evaluator.on_snapshot(seed, source, publication_index=0, **kwargs))
            self.assertFalse(evaluator.on_snapshot(snapshot, source, publication_index=9, **kwargs))
            with patch("generals_rl.training_evaluation.subprocess.Popen", side_effect=OSError("test spawn failure")):
                self.assertTrue(evaluator.on_snapshot(snapshot, source, publication_index=10, **kwargs))
            self.assertFalse(evaluator.on_snapshot(snapshot, source, publication_index=10, **kwargs))
            evidence = evaluator.describe()
            self.assertEqual((evidence["scheduled"], evidence["failed"], evidence["completed"]), (1, 1, 0))
            self.assertIsNone(evidence["latest_completed"])
            self.assertIn("test spawn failure", evaluator.jobs[0]["error"])
            evaluator.finish()
            evaluator.close()
            writer.close()
            events = EventAccumulator(str(root / "run")).Reload()
            self.assertEqual(events.Scalars("Eval/failed_jobs")[0].value, 1)
            self.assertNotIn("Eval/overall/win_rate", events.Tags()["scalars"])


if __name__ == "__main__":
    unittest.main()
