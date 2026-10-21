"""Real subprocess evaluations, snapshot eviction and learner RNG isolation."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, PropertyMock

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from generals.controllers import AI_KINDS
from generals_rl.evaluate_builtin import _inference_metadata, evaluate_builtin
from generals_rl.train import train
from generals_rl.training_evaluation import PeriodicEvaluator, TrainingEvalConfig, _stop_worker
from rl_test_utils import assert_nested_equal, small_training_config


def small_config():
    config = small_training_config(runner_overrides={"learning_epochs": 1})
    config["self_play"] = {"pool_size": 2, "snapshot_interval": 1}
    # Retain the default cadence, but keep the actual match workload tiny.
    config["evaluation"] = {"maps_per_ai": 1, "max_ticks": 3, "batch_size": 8}
    return config


def completion_fixture(evaluator):
    """A complete worker report stub; never launches inference or a process."""
    checkpoint = evaluator.directory / "snapshot.pt"
    checkpoint.write_bytes(b"immutable stub weights")
    report_path = evaluator.directory / "report.json"
    report = {"checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
              "completed_games": 8, "scheduled_games": 8, "policy": "student_local",
              "device": evaluator.settings.device, "stochastic": True,
              "matmul_precision": "highest", "cudnn_allow_tf32": False, "cudnn_benchmark": False,
              "parameter_dtype": "float32", "inference_dtype": "float32",
              "overall": {"win_rate": 0.0},
              "by_ai": {kind: {"games": 2, "win_rate": 0.0} for kind in AI_KINDS},
              "games": [{"game_id": index, "opponent": kind, "outcome": "draw"}
                        for index, kind in enumerate(kind for kind in AI_KINDS for _ in range(2))],
              "selected_game_traces": {"protocol": "recorded_joint_actions_core_v1", "cases": [],
                  "missing": [{"opponent": kind, "outcome": outcome}
                              for kind in AI_KINDS for outcome in ("loss", "win")]}}
    job = {"status": "running", "snapshot_id": 1, "iteration": 2, "publication_index": 1,
           "steps_per_environment": 32, "environment_steps": 128,
           "checkpoint": str(checkpoint), "checkpoint_sha256": report["checkpoint_sha256"],
           "report": str(report_path), "worker_seconds": 0.1}
    evaluator.jobs.append(job)
    return report_path, report, job


class TrainingEvaluationTests(unittest.TestCase):
    def test_report_must_match_requested_parallel_baseline_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, baseline_workers=4),
                                          {}, directory, None)
            path, report, job = completion_fixture(evaluator)
            for workers in (None, 1, 2):
                with self.subTest(workers=workers):
                    report.pop("baseline_workers", None)
                    if workers is not None:
                        report["baseline_workers"] = workers
                    path.write_text(json.dumps(report))
                    with self.assertRaisesRegex(RuntimeError, "baseline workers"):
                        evaluator._complete(job)
                    self.assertIsNone(evaluator.best_completed)

    def test_cuda_report_rejects_missing_or_wrong_explicit_cudnn_precision_before_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, device="cuda:0"),
                                          {"cudnn_allow_tf32": False}, directory, None)
            report_path = evaluator.directory / "report.json"
            report = {"checkpoint_sha256": "snapshot", "completed_games": 8, "scheduled_games": 8,
                      "policy": "student_local", "device": "cuda:0", "stochastic": True,
                      "by_ai": {kind: {"games": 2} for kind in AI_KINDS}}
            for actual in (None, True):
                if actual is None:
                    report.pop("cudnn_allow_tf32", None)
                else:
                    report["cudnn_allow_tf32"] = actual
                report_path.write_text(json.dumps(report))
                with self.subTest(actual=actual), self.assertRaisesRegex(RuntimeError, "cuDNN TF32"):
                    evaluator._complete({"report": str(report_path), "checkpoint_sha256": "snapshot"})
                self.assertIsNone(evaluator.best_completed)
                self.assertFalse((evaluator.directory / "best.pt").exists())

    def test_cuda_report_rejects_missing_or_wrong_explicit_matmul_precision(self):
        for actual in (None, "high"):
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as directory:
                evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, device="cuda:0"),
                                              {"matmul_precision": "highest"}, directory, None)
                path, report, job = completion_fixture(evaluator)
                if actual is None:
                    report.pop("matmul_precision")
                else:
                    report["matmul_precision"] = actual
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "matmul precision"):
                    evaluator._complete(job)
                self.assertIsNone(evaluator.best_completed)
                self.assertFalse((evaluator.directory / "best.pt").exists())

    def test_cuda_report_rejects_missing_or_wrong_explicit_cudnn_benchmark(self):
        for actual in (None, False):
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as directory:
                evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, device="cuda:0"),
                                              {"cudnn_benchmark": True}, directory, None)
                path, report, job = completion_fixture(evaluator)
                if actual is None:
                    report.pop("cudnn_benchmark")
                else:
                    report["cudnn_benchmark"] = actual
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "cuDNN benchmark"):
                    evaluator._complete(job)
                self.assertIsNone(evaluator.best_completed)
                self.assertFalse((evaluator.directory / "best.pt").exists())

    def test_cuda_report_rejects_missing_or_wrong_policy_inference_dtype(self):
        for amp, actual in ((False, None), (False, "bfloat16"), (True, None), (True, "float32")):
            with self.subTest(amp=amp, actual=actual), tempfile.TemporaryDirectory() as directory:
                evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, device="cuda:0"),
                    {"runner": {"policy": {"amp_bfloat16": amp}}}, directory, None)
                path, report, job = completion_fixture(evaluator)
                if actual is None:
                    report.pop("inference_dtype")
                else:
                    report["inference_dtype"] = actual
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(RuntimeError, "inference dtype"):
                    evaluator._complete(job)
                self.assertIsNone(evaluator.best_completed)
                self.assertFalse((evaluator.directory / "best.pt").exists())

    def test_cuda_report_accepts_matching_precision_and_default_fp32_metadata(self):
        cases = (({}, "high", "float32"),
                 ({"matmul_precision": None}, "medium", "float32"),
                 ({"matmul_precision": "highest", "cudnn_allow_tf32": False,
                   "runner": {"policy": {"amp_bfloat16": False}}}, "highest", "float32"),
                 ({"matmul_precision": "highest", "cudnn_allow_tf32": False, "cudnn_benchmark": True,
                   "runner": {"policy": {"amp_bfloat16": True}}}, "highest", "bfloat16"))
        for config, matmul, dtype in cases:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()):
                evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1, device="cuda:0"),
                                              config, directory, None)
                path, report, job = completion_fixture(evaluator)
                report.update(matmul_precision=matmul, inference_dtype=dtype,
                              cudnn_benchmark=config.get("cudnn_benchmark", False))
                path.write_text(json.dumps(report))
                evaluator._complete(job)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(evaluator.latest_completed["matmul_precision"], matmul)
                self.assertEqual(evaluator.latest_completed["inference_dtype"], dtype)
                self.assertEqual(evaluator.latest_completed["cudnn_benchmark"], report["cudnn_benchmark"])
                self.assertEqual((evaluator.directory / "best.pt").read_bytes(), b"immutable stub weights")

    def test_cpu_report_keeps_precision_validation_disabled(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            evaluator = PeriodicEvaluator(TrainingEvalConfig(maps_per_ai=1),
                {"matmul_precision": "highest", "cudnn_allow_tf32": False, "cudnn_benchmark": True,
                 "runner": {"policy": {"amp_bfloat16": True}}}, directory, None)
            path, report, job = completion_fixture(evaluator)
            for key in ("matmul_precision", "cudnn_allow_tf32", "cudnn_benchmark", "inference_dtype"):
                report.pop(key)
            path.write_text(json.dumps(report))
            evaluator._complete(job)
            self.assertEqual(job["status"], "completed")

    def test_precision_metadata_distinguishes_cuda_autocast_from_cpu_parameters(self):
        policy = torch.nn.Linear(2, 1)
        policy.amp_bfloat16 = True
        self.assertEqual(_inference_metadata(policy, "cpu"),
                         {"parameter_dtype": "float32", "inference_dtype": "float32"})
        # Device strings inspect the contract without creating a CUDA context.
        self.assertEqual(_inference_metadata(policy, "cuda:0"),
                         {"parameter_dtype": "float32", "inference_dtype": "bfloat16"})
        policy.amp_bfloat16 = False
        self.assertEqual(_inference_metadata(policy, "cuda:0"),
                         {"parameter_dtype": "float32", "inference_dtype": "float32"})

    def test_default_is_enabled_every_ten_learned_snapshots_and_validates_early(self):
        settings = TrainingEvalConfig()
        self.assertTrue(settings.enabled)
        self.assertTrue(settings.stochastic)
        self.assertEqual(settings.every_n_snapshots, 10)
        self.assertEqual(settings.maps_per_ai, 50)
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.baseline_workers, 1)
        self.assertFalse(settings.synchronous)
        self.assertIsNone(settings.worker_timeout_seconds)
        self.assertTrue(TrainingEvalConfig(device="cuda:0").synchronous)
        self.assertEqual(TrainingEvalConfig(device="cuda").worker_timeout_seconds, 1800)
        self.assertEqual(TrainingEvalConfig(device="cuda", timeout_seconds=15).worker_timeout_seconds, 15)
        cases = ({"enabled": "yes"}, {"stochastic": "yes"}, {"every_n_snapshots": 0}, {"maps_per_ai": True},
                 {"batch_size": 0}, {"max_ticks": -1}, {"seed": -1}, {"cpu_threads": 0},
                 {"baseline_workers": 0}, {"baseline_workers": True},
                 {"device": None}, {"device": "invalid"}, {"device": "mps"},
                 {"timeout_seconds": 0}, {"timeout_seconds": True}, {"timeout_seconds": float("inf")})
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
            self.assertEqual(report["parameter_dtype"], "float32")
            self.assertEqual(report["inference_dtype"], "float32")
            self.assertIn(report["matmul_precision"], ("highest", "high", "medium"))
            self.assertEqual(report["policy"], "student_local")
            self.assertEqual(report["completed_games"], 8)
            self.assertTrue(report["stochastic"])
            self.assertEqual(report["sampling_seed"], report["seed"])
            self.assertEqual(report["sampling_protocol"], "per_game_numpy_uniform_cdf_v1")
            self.assertEqual(set(report["by_ai"]), set(AI_KINDS))
            for metrics in report["by_ai"].values():
                self.assertEqual(metrics["games"], 2)
                self.assertEqual(metrics["wins"] + metrics["losses"] + metrics["draws"], 2)
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual(events.Scalars("Eval/stochastic")[0].value, 1)
            for kind in (*AI_KINDS, "overall"):
                logged = events.Scalars(f"Eval/{kind}/win_rate")
                self.assertEqual(len(logged), 1)
                self.assertEqual(logged[0].step, 9)
                metrics = report["overall"] if kind == "overall" else report["by_ai"][kind]
                self.assertAlmostEqual(logged[0].value, metrics["win_rate"])
                for key, value in metrics["action_metrics"].items():
                    action = events.Scalars(f"Eval/{kind}/action/{key}")
                    self.assertEqual(action[0].step, 9)
                    self.assertAlmostEqual(action[0].value, value, places=5)
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

    def test_synchronous_path_completes_at_update_boundary_and_preserves_learner(self):
        """Exercise real workers on CPU; CUDA device routing is tested separately."""
        config = small_config()
        config["evaluation"].update(every_n_snapshots=2, keep_snapshots=1)
        observed = []
        original_on_snapshot = PeriodicEvaluator.on_snapshot

        def on_snapshot(evaluator, *args, **kwargs):
            scheduled = original_on_snapshot(evaluator, *args, **kwargs)
            if scheduled:
                observed.append(evaluator.jobs[-1]["iteration"])
                self.assertEqual(evaluator.jobs[-1]["status"], "completed")
                self.assertIsNone(evaluator.running)
                self.assertFalse(evaluator.pending)
            return scheduled

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "sync"
            with patch.object(TrainingEvalConfig, "synchronous", new_callable=PropertyMock, return_value=True), \
                    patch.object(PeriodicEvaluator, "on_snapshot", on_snapshot):
                summary = train(config, output, iterations=6)
            learner_rng = torch.get_rng_state().clone()
            self.assertEqual(observed, [1, 3, 5])
            self.assertEqual(summary["evaluation"]["completed"], 3)
            points = json.loads((output / "evaluations/index.json").read_text())["jobs"]
            self.assertTrue(all(point["synchronous"] for point in points))
            self.assertEqual(sum(Path(point["checkpoint"]).exists() for point in points), 1)
            events = EventAccumulator(str(output)).Reload()
            self.assertEqual([point.step for point in events.Scalars("Eval/overall/win_rate")], [1, 3, 5])
            wall_times = events.Scalars("Perf/wall_seconds")
            self.assertGreaterEqual(wall_times[-1].value, sum(point["worker_seconds"] for point in points))
            disabled = Path(directory) / "disabled"
            config["evaluation"]["enabled"] = False
            train(config, disabled, iterations=6)
            torch.testing.assert_close(torch.get_rng_state(), learner_rng, rtol=0, atol=0)
            actual = torch.load(output / "checkpoint.pt", weights_only=False, map_location="cpu")
            expected = torch.load(disabled / "checkpoint.pt", weights_only=False, map_location="cpu")
            for key in ("model_state_dict", "optimizer_state_dict", "optimizer_stu_enc_state_dict"):
                assert_nested_equal(self, actual[key], expected[key])

    def test_cuda_worker_inherits_devices_and_releases_only_unused_cache(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            evaluator = PeriodicEvaluator(TrainingEvalConfig(device="cuda:0", baseline_workers=4), {}, directory, None)
            job = {"checkpoint": "snapshot.pt", "report": "report.json",
                   "worker_log": str(evaluator.directory / "worker.log"), "status": "queued"}
            evaluator.jobs.append(job)
            evaluator.pending.append(job)
            process = SimpleNamespace(pid=123, poll=lambda: None)
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "test-device", "PYTHONPATH": "frozen-source"}), \
                    patch("generals_rl.training_evaluation.subprocess.Popen", return_value=process) as popen, \
                    patch("generals_rl.training_evaluation.torch.cuda.is_initialized", return_value=True), \
                    patch("generals_rl.training_evaluation.torch.cuda.synchronize") as synchronize, \
                    patch("generals_rl.training_evaluation.torch.cuda.empty_cache") as empty_cache:
                evaluator.poll()
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")
            self.assertEqual(command[command.index("--baseline-workers") + 1], "4")
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "test-device")
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONPATH"], "frozen-source")
            self.assertEqual(popen.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
            synchronize.assert_called_once_with()
            empty_cache.assert_called_once_with()
            # This is a fake routing worker; no CUDA context or process exists.
            evaluator.running = None

    def test_worker_group_cleanup_kills_a_child_that_ignores_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "child_pid"
            child_code = ("import os, signal, sys, time; from pathlib import Path; "
                          "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                          "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)")
            parent_code = ("import subprocess, sys, time; "
                           "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); time.sleep(60)")
            process = subprocess.Popen([sys.executable, "-c", parent_code, child_code, str(ready)],
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists())
                child_pid = int(ready.read_text())
                _stop_worker(process)
                self.assertIsNotNone(process.poll())
                state_path = Path(f"/proc/{child_pid}/stat")
                deadline = time.monotonic() + 2
                while state_path.exists():
                    state = state_path.read_text().rsplit(") ", 1)[1].split()[0]
                    if state in ("Z", "X"):
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def test_nonzero_worker_exit_poll_reaps_surviving_sigterm_ignoring_child(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            ready = root / "child_pid"
            child_code = ("import os, signal, sys, time; from pathlib import Path; "
                          "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                          "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)")
            parent_code = ("import subprocess, sys, time\nfrom pathlib import Path\n"
                           "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
                           "while not Path(sys.argv[2]).exists():\n    time.sleep(.01)\n"
                           "sys.exit(7)\n")
            process = subprocess.Popen([sys.executable, "-c", parent_code, child_code, str(ready)],
                                       start_new_session=True)
            evaluator = PeriodicEvaluator(TrainingEvalConfig(baseline_workers=4, keep_snapshots=1),
                                          {}, root / "run", None)
            checkpoint = root / "snapshot.pt"
            checkpoint.write_bytes(b"immutable test stub; no model")
            job = {"status": "running", "snapshot_id": 1, "iteration": 0,
                   "publication_index": 1, "checkpoint": str(checkpoint),
                   "report": str(root / "report.json"), "worker_log": str(root / "worker.log")}
            evaluator.jobs.append(job)
            try:
                self.assertEqual(process.wait(timeout=5), 7)
                self.assertTrue(ready.exists())
                child_pid = int(ready.read_text())
                state_path = Path(f"/proc/{child_pid}/stat")
                self.assertTrue(state_path.exists())
                self.assertNotIn(state_path.read_text().rsplit(") ", 1)[1].split()[0], ("Z", "X"))
                evaluator.running = job, process, time.perf_counter()
                evaluator.poll()
                self.assertIsNone(evaluator.running)
                self.assertEqual(job["status"], "failed")
                self.assertEqual(job["returncode"], 7)
                self.assertIn("exited with code 7", job["error"])
                self.assertEqual(evaluator.describe()["failed"], 1)
                self.assertIsNone(evaluator.latest_completed)
                self.assertIsNone(evaluator.best_completed)
                index = json.loads((evaluator.directory / "index.json").read_text())
                self.assertEqual(index["jobs"][0]["status"], "failed")
                self.assertEqual(index["jobs"][0]["returncode"], 7)
                deadline = time.monotonic() + 2
                while state_path.exists():
                    try:
                        state = state_path.read_text().rsplit(") ", 1)[1].split()[0]
                    except FileNotFoundError:
                        break
                    if state in ("Z", "X"):
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
            finally:
                evaluator.close()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def test_timeout_and_exception_cleanup_reap_workers_and_bound_failed_weights(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            pool = root / "pool"
            pool.mkdir()
            checkpoint = pool / "snapshot.pt"
            checkpoint.write_bytes(b"test immutable weights")
            snapshot = lambda sid: SimpleNamespace(snapshot_id=sid, metadata={
                "iteration": sid, "filename": checkpoint.name,
                "sha256": sha256(checkpoint.read_bytes()).hexdigest(), "seed": False})
            evaluator = PeriodicEvaluator(TrainingEvalConfig(every_n_snapshots=1, keep_snapshots=1,
                                                             timeout_seconds=0.02), {}, root / "run", None)
            real_popen = subprocess.Popen
            workers = []

            def sleeping_worker(*args, **kwargs):
                process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
                workers.append(process)
                return process

            kwargs = {"steps_per_environment": 4, "environment_steps": 16}
            try:
                with patch("generals_rl.training_evaluation.subprocess.Popen", side_effect=sleeping_worker):
                    evaluator.on_snapshot(snapshot(1), pool, publication_index=1, **kwargs)
                    evaluator.finish()
                    self.assertEqual(evaluator.jobs[0]["status"], "failed")
                    self.assertIn("TimeoutError", evaluator.jobs[0]["error"])
                    self.assertIsNotNone(workers[0].poll())
                    self.assertIsNone(evaluator.latest_completed)
                    self.assertFalse((evaluator.directory / "best.pt").exists())
                    evaluator.settings = TrainingEvalConfig(every_n_snapshots=1, keep_snapshots=1)
                    evaluator.on_snapshot(snapshot(2), pool, publication_index=2, **kwargs)
                    evaluator.on_snapshot(snapshot(3), pool, publication_index=3, **kwargs)
                    evaluator.close()
                self.assertTrue(all(process.poll() is not None for process in workers))
                self.assertFalse(evaluator.pending)
                self.assertIsNone(evaluator.running)
                self.assertEqual([job["status"] for job in evaluator.jobs], ["failed", "cancelled", "cancelled"])
                self.assertEqual(sum(Path(job["checkpoint"]).exists() for job in evaluator.jobs), 1)
                index = json.loads((evaluator.directory / "index.json").read_text())
                self.assertEqual([job["status"] for job in index["jobs"]], ["failed", "cancelled", "cancelled"])
            finally:
                evaluator.close()

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
                          "device": "cpu", "stochastic": True, "overall": {"win_rate": overall},
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

    def test_trace_retention_preserves_best_and_recent_finished_without_deleting_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = PeriodicEvaluator(TrainingEvalConfig(keep_snapshots=1), {}, directory, None)
            for index, status in enumerate(("completed", "failed", "cancelled", "completed", "queued"), 1):
                target = evaluator.directory / f"snapshot_{index:08d}"
                target.mkdir()
                (target / "snapshot.pt").write_bytes(f"weights {index}".encode())
                trace = target / "demo_traces"
                trace.mkdir()
                (trace / "loss.jsonl").write_text(f"actions {index}\n")
                (target / "report.json").write_text(f"metrics {index}\n")
                (target / "worker.log").write_text(f"log {index}\n")
                evaluator.jobs.append({"status": status, "checkpoint": str(target / "snapshot.pt"),
                                       "report": str(target / "report.json")})
            best = evaluator.jobs[0]
            evaluator.best_completed = {"report": best["report"]}
            (evaluator.directory / "best.pt").write_bytes(b"protected best weights")
            best_report = {"selected_game_traces": {"cases": [{"path": str(
                Path(best["report"]).parent / "demo_traces/loss.jsonl")} ]}}
            (evaluator.directory / "best_report.json").write_text(json.dumps(best_report))
            evaluator._prune_snapshots()
            evaluator._save_index()
            self.assertEqual([Path(job["checkpoint"]).exists() for job in evaluator.jobs],
                             [False, False, False, True, True])
            self.assertEqual([(Path(job["report"]).parent / "demo_traces").exists()
                              for job in evaluator.jobs], [True, False, False, True, True])
            self.assertEqual([job.get("trace_retained") for job in evaluator.jobs],
                             [True, False, False, True, None])
            self.assertTrue(Path(best_report["selected_game_traces"]["cases"][0]["path"]).exists())
            self.assertEqual((evaluator.directory / "best.pt").read_bytes(), b"protected best weights")
            for index, job in enumerate(evaluator.jobs, 1):
                self.assertEqual(Path(job["report"]).read_text(), f"metrics {index}\n")
                self.assertEqual((Path(job["report"]).parent / "worker.log").read_text(), f"log {index}\n")
            self.assertFalse(json.loads((evaluator.directory / "index.json").read_text())["jobs"][1]["trace_retained"])
            # Once a newer best replaces it, the old best's traces can expire.
            evaluator.best_completed = {"report": evaluator.jobs[3]["report"]}
            evaluator._prune_snapshots()
            self.assertFalse((Path(best["report"]).parent / "demo_traces").exists())
            self.assertFalse(best["trace_retained"])
            # Historical CPU jobs without traces keep their index shape.
            cpu = evaluator.directory / "cpu"
            cpu.mkdir()
            (cpu / "snapshot.pt").write_bytes(b"cpu weights")
            evaluator.jobs.append({"status": "completed", "checkpoint": str(cpu / "snapshot.pt"),
                                   "report": str(cpu / "report.json")})
            evaluator._prune_snapshots()
            self.assertNotIn("trace_retained", evaluator.jobs[-1])

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
