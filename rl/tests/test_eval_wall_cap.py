"""Pure CPU mocks for CUDA eval deadlines; no worker or CUDA context starts."""
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
import io
from pathlib import Path
import shutil
import subprocess
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from generals_rl.training_evaluation import PeriodicEvaluator, TrainingEvalConfig


class EvalWallCapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.source = self.pool / "snapshot.pt"
        self.source.write_bytes(b"immutable synthetic snapshot, not a model")
        self.clock = 100.0
        self.copy_seconds = 0.0
        self.cache_seconds = 0.0
        self.writer = Mock()
        self.process = Mock(pid=123)
        self.process.returncode = None
        self.process.poll.return_value = None
        real_copy = shutil.copyfile

        def timed_copy(source, target):
            result = real_copy(source, target)
            self.clock += self.copy_seconds
            return result

        def timed_cache():
            self.clock += self.cache_seconds

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch("generals_rl.training_evaluation.perf_counter",
                                      side_effect=lambda: self.clock))
        self.copy = self.stack.enter_context(patch("generals_rl.training_evaluation.shutil.copyfile",
                                                  side_effect=timed_copy))
        self.popen = self.stack.enter_context(patch("generals_rl.training_evaluation.subprocess.Popen",
                                                   return_value=self.process))
        self.killpg = self.stack.enter_context(patch("generals_rl.training_evaluation.os.killpg"))
        self.stack.enter_context(patch("generals_rl.training_evaluation.torch.cuda.is_initialized",
                                      return_value=True))
        self.synchronize = self.stack.enter_context(
            patch("generals_rl.training_evaluation.torch.cuda.synchronize"))
        self.empty_cache = self.stack.enter_context(
            patch("generals_rl.training_evaluation.torch.cuda.empty_cache", side_effect=timed_cache))

    def make(self, name="run", completed_seconds=None, **kwargs):
        settings = TrainingEvalConfig(every_n_snapshots=1, **kwargs)
        evaluator = PeriodicEvaluator(settings, {}, self.root / name, self.writer)
        if completed_seconds is not None:
            evaluator.jobs.append({"status": "completed", "publication_index": 0,
                                   "worker_seconds": completed_seconds,
                                   "checkpoint": str(evaluator.directory / "prior.pt")})
        return evaluator

    def snapshot(self, sid=1):
        return SimpleNamespace(snapshot_id=sid, metadata={
            "iteration": sid - 1, "filename": self.source.name, "seed": False,
            "sha256": sha256(self.source.read_bytes()).hexdigest()})

    def schedule(self, evaluator, remaining, sid=1):
        return evaluator.on_snapshot(self.snapshot(sid), self.pool, publication_index=sid,
                                     steps_per_environment=32 * sid,
                                     environment_steps=1024 * sid,
                                     remaining_seconds=remaining)

    def assert_no_winrate(self):
        tags = [call.args[0] for call in self.writer.add_scalar.call_args_list]
        self.assertFalse(any(tag.endswith("/win_rate") for tag in tags), tags)

    def test_cuda_insufficient_wall_budget_creates_no_job_or_copy_or_worker(self):
        evaluator = self.make(device="cuda:0")
        index_before = (evaluator.directory / "index.json").read_bytes()
        self.assertFalse(self.schedule(evaluator, remaining=10))
        self.assertTrue(evaluator.budget_exhausted)
        self.assertEqual(evaluator.jobs, [])
        self.assertFalse((evaluator.directory / "snapshot_00000001").exists())
        self.assertEqual((evaluator.directory / "index.json").read_bytes(), index_before)
        self.assertEqual((evaluator.describe()["failed"], evaluator.describe()["cancelled"]), (0, 0))
        self.assertFalse(evaluator.pending)
        self.assertIsNone(evaluator.running)
        self.copy.assert_not_called()
        self.popen.assert_not_called()
        self.synchronize.assert_not_called()
        self.empty_cache.assert_not_called()
        self.assert_no_winrate()

    def test_cuda_deadline_starts_at_entry_and_respects_the_shorter_limit(self):
        cases = ((30., None, 80., 20., 3., 130., 30., "evaluation_timeout"),
                 (100., 4., 50., 3., 2., 140., 40., "wall_budget"))
        for index, (configured, prior, remaining, copy_time, cache_time, deadline, timeout, reason) in enumerate(cases):
            with self.subTest(remaining=remaining):
                self.clock = 100.
                self.copy_seconds, self.cache_seconds = copy_time, cache_time
                evaluator = self.make(f"run{index}", device="cuda:0", timeout_seconds=configured,
                                      completed_seconds=prior)
                # Inspect the scheduled worker without any real wait/worker.
                with patch.object(evaluator, "finish") as finish:
                    self.assertTrue(self.schedule(evaluator, remaining))
                job, process, started = evaluator.running
                self.assertEqual(job["deadline"], deadline)
                self.assertEqual(job["timeout_seconds"], timeout)
                self.assertEqual(job["deadline_reason"], reason)
                self.assertFalse(evaluator.budget_exhausted)
                self.assertEqual(started, 100. + copy_time + cache_time)
                self.assertLess(started, deadline)
                self.assertIs(process, self.process)
                finish.assert_called_once_with()
                evaluator.running = None
        self.assertEqual(self.copy.call_count, 2)
        self.assertEqual(self.popen.call_count, 2)
        self.assertEqual(self.synchronize.call_count, 2)
        self.assertEqual(self.empty_cache.call_count, 2)
        self.assert_no_winrate()

    def test_copy_or_cache_exhaustion_cancels_and_bounds_retained_snapshots(self):
        cases = (("copy", 100., 4., 50., 41., 0., "wall_budget"),
                 ("cache", 100., 4., 50., 2., 39., "wall_budget"),
                 ("config", 5., None, 30., 6., 0., "evaluation_timeout"))
        for stage, timeout, prior, remaining, copy_time, cache_time, reason in cases:
            with self.subTest(stage=stage):
                self.copy.reset_mock()
                self.empty_cache.reset_mock()
                self.copy_seconds, self.cache_seconds = copy_time, cache_time
                evaluator = self.make(stage, device="cuda:0", timeout_seconds=timeout,
                                      keep_snapshots=1, completed_seconds=prior)
                for sid in (1, 2):
                    self.assertTrue(self.schedule(evaluator, remaining, sid))
                    job = evaluator.jobs[-1]
                    self.assertEqual((job["status"], job["reason"]), ("cancelled", reason))
                    self.assertGreaterEqual(self.clock, job["deadline"])
                first, second = evaluator.jobs[-2:]
                self.assertFalse(Path(first["checkpoint"]).exists())
                self.assertFalse(first["checkpoint_retained"])
                self.assertTrue(Path(second["checkpoint"]).exists())
                self.assertTrue(second["checkpoint_retained"])
                self.assertFalse(evaluator.pending)
                self.assertIsNone(evaluator.running)
                self.assertEqual(self.copy.call_count, 2)
                self.assertEqual(self.empty_cache.call_count, 2 if stage == "cache" else 0)
        self.popen.assert_not_called()
        self.assert_no_winrate()

    def test_cpu_evaluation_ignores_remaining_budget_and_keeps_async_routing(self):
        evaluator = self.make(device="cpu")
        with patch.object(evaluator, "finish") as finish:
            self.assertTrue(self.schedule(evaluator, remaining=0))
        job, process, _ = evaluator.running
        self.assertEqual(job["status"], "running")
        self.assertIsNone(job["deadline"])
        self.assertIsNone(job["timeout_seconds"])
        self.assertIs(process, self.process)
        self.assertFalse(evaluator.budget_exhausted)
        self.assertEqual(self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        finish.assert_not_called()
        self.copy.assert_called_once()
        self.popen.assert_called_once()
        self.synchronize.assert_not_called()
        self.empty_cache.assert_not_called()
        self.assert_no_winrate()
        evaluator.running = None

    def test_running_worker_uses_absolute_deadline_and_is_reaped(self):
        self.copy_seconds, self.cache_seconds = 38., 1.
        evaluator = self.make(device="cuda:0", timeout_seconds=100., completed_seconds=4.)
        with patch.object(evaluator, "finish"):
            self.assertTrue(self.schedule(evaluator, remaining=50))
        job, _, started = evaluator.running
        self.assertEqual((job["deadline"], started), (140., 139.))
        self.clock = 141.
        # The worker's own two-second runtime is below its forty-second
        # timeout; only the absolute entry deadline has been exhausted.
        self.assertLess(self.clock - started, job["timeout_seconds"])

        def wait(timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("mock-worker", timeout)
            self.process.returncode = -9
            return -9

        self.process.wait.side_effect = wait
        evaluator.poll()
        self.assertIsNone(evaluator.running)
        self.assertEqual(job["status"], "failed")
        self.assertIn("TimeoutError", job["error"])
        self.assertEqual((job["returncode"], job["worker_seconds"]), (-9, 2.))
        self.assertEqual(self.killpg.call_args_list,
                         [call(self.process.pid, signal.SIGTERM), call(self.process.pid, signal.SIGKILL)])
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertEqual(self.process.wait.call_args_list[0].kwargs, {"timeout": 5})
        self.assertIsNone(evaluator.latest_completed)
        self.assertIsNone(evaluator.best_completed)
        self.assert_no_winrate()

    def test_actual_17_completed_points_keep_best_15_and_reject_37_second_tail(self):
        # Admission-only replay of RN1 N32768: actual22 worker metadata and the
        # closed-training receipt c312e2b948c3f96b record the failed 18th window.
        seconds = (510.0311540076509, 504.7947943219915, 481.14257365837693,
                   398.9889707034454, 322.0510884160176, 286.8880187748,
                   276.48873458616436, 203.67661800514907, 220.01480725873262,
                   241.0790263870731, 205.8772769952193, 210.22702806442976,
                   198.2763826576993, 203.42833570484072, 168.8251058710739,
                   231.72910583019257, 231.52866885252297)
        evaluator = self.make(device="cuda:0", maps_per_ai=100)
        evaluator.jobs = [{"status": "completed", "publication_index": sid,
                           "snapshot_id": sid, "worker_seconds": duration}
                          for sid, duration in enumerate(seconds[:14], 1)]
        scores = ([.865, .865, .9475], [.84, .84, .93625], [.85, .85, .935])
        windows = (1600.4589794529602, 1119.151905644685, 577.6296828845516)
        for sid, available, score in zip((15, 16, 17), windows, scores):
            with patch.object(evaluator, "finish"):
                self.assertTrue(self.schedule(evaluator, available + 10., sid))
            job, _, _ = evaluator.running
            self.assertAlmostEqual(job["timeout_seconds"], available)
            job.update(status="completed", worker_seconds=seconds[sid - 1], selection_score=score)
            evaluator.running = None
        evaluator.latest_completed = {"snapshot_id": 17, "selection_score": scores[-1]}
        evaluator.best_completed = {"snapshot_id": 15, "selection_score": scores[0]}
        evaluator.on_complete = Mock()
        evaluator._save_index()
        latest, best = evaluator.latest_completed, evaluator.best_completed
        index_before = (evaluator.directory / "index.json").read_bytes()
        effects_before = [mock.call_count for mock in
                          (self.copy, self.popen, self.synchronize, self.empty_cache)]

        self.assertFalse(self.schedule(evaluator, remaining=46.8885, sid=18))
        self.assertTrue(evaluator.budget_exhausted)
        self.assertEqual(evaluator.describe()["completed"], 17)
        self.assertEqual((evaluator.describe()["failed"], evaluator.describe()["cancelled"]), (0, 0))
        self.assertIs(evaluator.latest_completed, latest)
        self.assertIs(evaluator.best_completed, best)
        self.assertEqual((latest["snapshot_id"], best["snapshot_id"]), (17, 15))
        self.assertEqual(best["selection_score"], [.865, .865, .9475])
        self.assertEqual((evaluator.directory / "index.json").read_bytes(), index_before)
        self.assertEqual([mock.call_count for mock in
                          (self.copy, self.popen, self.synchronize, self.empty_cache)], effects_before)
        self.assertFalse((evaluator.directory / "snapshot_00000018").exists())
        self.assertFalse(evaluator.pending)
        self.assertIsNone(evaluator.running)
        evaluator.on_complete.assert_not_called()
        self.assert_no_winrate()


if __name__ == "__main__":
    unittest.main()
