"""Pure CPU mocks for CUDA eval deadlines; no worker or CUDA context starts."""
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
        self.stack.enter_context(patch("generals_rl.training_evaluation.torch.cuda.is_initialized",
                                      return_value=True))
        self.synchronize = self.stack.enter_context(
            patch("generals_rl.training_evaluation.torch.cuda.synchronize"))
        self.empty_cache = self.stack.enter_context(
            patch("generals_rl.training_evaluation.torch.cuda.empty_cache", side_effect=timed_cache))

    def make(self, name="run", **kwargs):
        settings = TrainingEvalConfig(every_n_snapshots=1, **kwargs)
        return PeriodicEvaluator(settings, {}, self.root / name, self.writer)

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

    def test_cuda_insufficient_wall_budget_cancels_before_copy_or_spawn(self):
        evaluator = self.make(device="cuda:0")
        self.assertFalse(self.schedule(evaluator, remaining=10))
        job = evaluator.jobs[0]
        self.assertEqual((job["status"], job["reason"]), ("cancelled", "wall_budget"))
        self.assertEqual((job["deadline"], job["timeout_seconds"]), (100., 0.))
        self.assertFalse(Path(job["checkpoint"]).parent.exists())
        self.assertFalse(job["checkpoint_retained"])
        self.assertFalse(evaluator.pending)
        self.assertIsNone(evaluator.running)
        self.copy.assert_not_called()
        self.popen.assert_not_called()
        self.synchronize.assert_not_called()
        self.empty_cache.assert_not_called()
        self.assert_no_winrate()

    def test_cuda_deadline_starts_at_entry_and_respects_the_shorter_limit(self):
        cases = ((80., 20., 3., 130., 30., "evaluation_timeout"),
                 (25., 3., 2., 115., 15., "wall_budget"))
        for index, (remaining, copy_time, cache_time, deadline, timeout, reason) in enumerate(cases):
            with self.subTest(remaining=remaining):
                self.clock = 100.
                self.copy_seconds, self.cache_seconds = copy_time, cache_time
                evaluator = self.make(f"run{index}", device="cuda:0", timeout_seconds=30.)
                # Inspect the scheduled worker without any real wait/worker.
                with patch.object(evaluator, "finish") as finish:
                    self.assertTrue(self.schedule(evaluator, remaining))
                job, process, started = evaluator.running
                self.assertEqual(job["deadline"], deadline)
                self.assertEqual(job["timeout_seconds"], timeout)
                self.assertEqual(job["deadline_reason"], reason)
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
        cases = (("copy", 30., 20., 11., 0., "wall_budget"),
                 ("cache", 30., 20., 2., 9., "wall_budget"),
                 ("config", 5., 30., 6., 0., "evaluation_timeout"))
        for stage, timeout, remaining, copy_time, cache_time, reason in cases:
            with self.subTest(stage=stage):
                self.copy.reset_mock()
                self.empty_cache.reset_mock()
                self.copy_seconds, self.cache_seconds = copy_time, cache_time
                evaluator = self.make(stage, device="cuda:0", timeout_seconds=timeout, keep_snapshots=1)
                for sid in (1, 2):
                    self.assertTrue(self.schedule(evaluator, remaining, sid))
                    job = evaluator.jobs[-1]
                    self.assertEqual((job["status"], job["reason"]), ("cancelled", reason))
                    self.assertGreaterEqual(self.clock, job["deadline"])
                first, second = evaluator.jobs
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
        self.assertEqual(self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        finish.assert_not_called()
        self.copy.assert_called_once()
        self.popen.assert_called_once()
        self.synchronize.assert_not_called()
        self.empty_cache.assert_not_called()
        self.assert_no_winrate()
        evaluator.running = None

    def test_running_worker_uses_absolute_deadline_and_is_reaped(self):
        self.copy_seconds, self.cache_seconds = 18., 1.
        evaluator = self.make(device="cuda:0", timeout_seconds=100.)
        with patch.object(evaluator, "finish"):
            self.assertTrue(self.schedule(evaluator, remaining=30))
        job, _, started = evaluator.running
        self.assertEqual((job["deadline"], started), (120., 119.))
        self.clock = 121.
        # The worker's own two-second runtime is below its twenty-second
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
        self.process.terminate.assert_called_once_with()
        self.process.kill.assert_called_once_with()
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertEqual(self.process.wait.call_args_list[0].kwargs, {"timeout": 5})
        self.assertIsNone(evaluator.latest_completed)
        self.assertIsNone(evaluator.best_completed)
        self.assert_no_winrate()


if __name__ == "__main__":
    unittest.main()
