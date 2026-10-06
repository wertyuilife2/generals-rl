"""Evaluate immutable learner snapshots in isolated CPU subprocesses.

Scheduling and reporting live outside the vendored runner, policy pool and MDP.
The evaluator reuses the complete, paired built-in-controller match protocol.
Only the parent writes TensorBoard, with the evaluated iteration as its x axis.
"""
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from generals.controllers import AI_KINDS


@dataclass(frozen=True)
class TrainingEvalConfig:
    enabled: bool = True
    stochastic: bool = True
    every_n_snapshots: int = 10
    maps_per_ai: int = 50
    batch_size: int = 128
    max_ticks: int = 2000
    seed: int = 1000042
    cpu_threads: int = 1
    keep_snapshots: int | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("evaluation.enabled must be boolean")
        if not isinstance(self.stochastic, bool):
            raise TypeError("evaluation.stochastic must be boolean")
        for name in ("every_n_snapshots", "maps_per_ai", "batch_size", "max_ticks", "cpu_threads", "seed"):
            minimum = 0 if name == "seed" else 1
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"evaluation.{name} must be an integer >= {minimum}")
        if self.keep_snapshots is not None and (not isinstance(self.keep_snapshots, int)
                or isinstance(self.keep_snapshots, bool) or self.keep_snapshots < 1):
            raise ValueError("evaluation.keep_snapshots must be a positive integer or null")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class PeriodicEvaluator:
    """One CPU worker at a time; all due snapshots are queued without dropping.

    An evaluation owns an exact copy of its snapshot before pool eviction can
    delete it. Neither policy inference nor RNG operations run in the training
    process. Worker failures are explicit in the index, log and TensorBoard;
    they do not discard learner progress or fabricate zero-win measurements.
    """

    def __init__(self, settings, run_config, output_dir, writer, *, on_complete=None):
        self.settings = settings
        self.run_config = run_config
        self.directory = Path(output_dir).resolve() / "evaluations"
        self.writer = writer
        self.jobs = []
        self.pending = deque()
        self.running = None
        self.latest_completed = None
        self.best_completed = None
        self.on_complete = on_complete
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_json(self.directory / "run_config.json", run_config)
        self._save_index()

    def on_snapshot(self, snapshot, pool_directory, *, publication_index,
                    steps_per_environment, environment_steps):
        """Count periodic learned snapshots; the initial random seed is excluded.

        The publication ordinal is derived from the absolute learner iteration,
        so resuming does not shift the ten-snapshot cadence.
        """
        if (not self.settings.enabled or publication_index < 1 or snapshot.metadata["seed"]
                or publication_index % self.settings.every_n_snapshots):
            return False
        if any(j["publication_index"] == publication_index for j in self.jobs):
            return False
        target = self.directory / f"snapshot_{snapshot.snapshot_id:08d}"
        target.mkdir()
        checkpoint = target / "snapshot.pt"
        shutil.copyfile(Path(pool_directory) / snapshot.metadata["filename"], checkpoint)
        digest = sha256(checkpoint.read_bytes()).hexdigest()
        if digest != snapshot.metadata["sha256"]:
            raise RuntimeError("evaluation snapshot differs from the published pool snapshot")
        _write_json(target / "run_config.json", self.run_config)
        job = {
            "snapshot_id": snapshot.snapshot_id,
            "iteration": snapshot.metadata["iteration"],
            "publication_index": publication_index,
            "steps_per_environment": steps_per_environment,
            "environment_steps": environment_steps,
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
            "report": str(target / "report.json"),
            "worker_log": str(target / "worker.log"),
            "stochastic": self.settings.stochastic,
            "status": "queued",
        }
        self.jobs.append(job)
        self.pending.append(job)
        self._save_index()
        print(f"[eval] queued snapshot={job['snapshot_id']} iteration={job['iteration']} "
              f"games={8 * self.settings.maps_per_ai}", flush=True)
        self.poll()
        return True

    def poll(self):
        if self.running is not None:
            job, process, started = self.running
            returncode = process.poll()
            if returncode is not None:
                self.running = None
                job["worker_seconds"] = perf_counter() - started
                job["returncode"] = returncode
                try:
                    if returncode != 0:
                        raise RuntimeError(f"evaluation worker exited with code {returncode}")
                    self._complete(job)
                except Exception as exc:
                    self._fail(job, exc)
        while self.running is None and self.pending:
            job = self.pending.popleft()
            settings = self.settings
            command = [sys.executable, "-u", "-m", "generals_rl.evaluate_builtin",
                       job["checkpoint"], "--maps-per-ai", str(settings.maps_per_ai),
                       "--batch-size", str(settings.batch_size), "--max-ticks", str(settings.max_ticks),
                       "--seed", str(settings.seed), "--cpu-threads", str(settings.cpu_threads),
                       "--device", "cpu", "--output", job["report"]]
            command.append("--stochastic" if settings.stochastic else "--greedy")
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            try:
                with Path(job["worker_log"]).open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=environment)
                job["status"] = "running"
                job["pid"] = process.pid
                self.running = job, process, perf_counter()
                self._save_index()
            except OSError as exc:
                self._fail(job, exc)

    def _complete(self, job):
        report = json.loads(Path(job["report"]).read_text())
        expected = 8 * self.settings.maps_per_ai
        if (report["checkpoint_sha256"] != job["checkpoint_sha256"]
                or report["completed_games"] != expected
                or report["scheduled_games"] != expected
                or set(report["by_ai"]) != set(AI_KINDS)
                or report["policy"] != "student_local"
                or report["device"] != "cpu" or report["stochastic"] != self.settings.stochastic):
            raise RuntimeError("evaluation report does not match the requested snapshot/protocol")
        for kind in AI_KINDS:
            if report["by_ai"][kind]["games"] != 2 * self.settings.maps_per_ai:
                raise RuntimeError(f"evaluation did not finish all scheduled {kind} games")
        report["training_evaluation"] = {key: job[key] for key in
            ("snapshot_id", "iteration", "publication_index", "steps_per_environment", "environment_steps")}
        _write_json(Path(job["report"]), report)
        point = {key: job[key] for key in
                 ("snapshot_id", "iteration", "publication_index", "steps_per_environment",
                  "environment_steps", "checkpoint_sha256", "report", "worker_seconds")}
        score = (min(report["by_ai"][kind]["win_rate"] for kind in AI_KINDS),
                 report["by_ai"]["aggressive"]["win_rate"], report["overall"]["win_rate"])
        point["selection_score"] = list(score)
        point["stochastic"] = report["stochastic"]
        if self.best_completed is None or score > tuple(self.best_completed["selection_score"]):
            temporary = self.directory / "best.pt.tmp"
            shutil.copyfile(job["checkpoint"], temporary)
            temporary.replace(self.directory / "best.pt")
            _write_json(self.directory / "best_report.json", report)
            self.best_completed = dict(point, checkpoint=str(self.directory / "best.pt"))
        if self.writer is not None:
            for name, metrics in {**report["by_ai"], "overall": report["overall"]}.items():
                for key in ("win_rate", "decisive_win_rate", "draw_rate", "wins", "losses", "draws", "mean_ticks"):
                    if metrics[key] is not None:
                        self.writer.add_scalar(f"Eval/{name}/{key}", metrics[key], job["iteration"])
                for key, value in metrics.get("action_metrics", {}).items():
                    self.writer.add_scalar(f"Eval/{name}/action/{key}", value, job["iteration"])
            self.writer.add_scalar("Eval/environment_steps", job["environment_steps"], job["iteration"])
            self.writer.add_scalar("Eval/snapshot_id", job["snapshot_id"], job["iteration"])
            self.writer.add_scalar("Eval/worker_seconds", job["worker_seconds"], job["iteration"])
            self.writer.add_scalar("Eval/stochastic", int(report["stochastic"]), job["iteration"])
            self.writer.add_scalar("Eval/worst_ai/win_rate", score[0], job["iteration"])
            self.writer.add_scalar("Eval/best_worst_ai/win_rate", self.best_completed["selection_score"][0], job["iteration"])
            self.writer.flush()
        job["status"] = "completed"
        job["selection_score"] = list(score)
        self.latest_completed = point
        if self.on_complete is not None:
            self.on_complete(point)
        self._prune_snapshots()
        self._save_index()
        rates = " ".join(f"{kind}={report['by_ai'][kind]['win_rate']:.1%}" for kind in AI_KINDS)
        print(f"[eval] snapshot={job['snapshot_id']} iteration={job['iteration']} {rates} "
              f"overall={report['overall']['win_rate']:.2%} report={job['report']}", flush=True)

    def _fail(self, job, exc):
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        self._save_index()
        if self.writer is not None:
            self.writer.add_scalar("Eval/failed_jobs", self.describe()["failed"], job["iteration"])
            self.writer.flush()
        print(f"[eval] FAILED snapshot={job['snapshot_id']}: {job['error']}; "
              f"log={job['worker_log']}", flush=True)

    def _prune_snapshots(self):
        if self.settings.keep_snapshots is None:
            return
        completed = [job for job in self.jobs if job["status"] == "completed"]
        for job in completed[:-self.settings.keep_snapshots]:
            Path(job["checkpoint"]).unlink(missing_ok=True)
            job["checkpoint_retained"] = False

    def _save_index(self):
        _write_json(self.directory / "index.json", {"jobs": self.jobs, "best": self.best_completed})

    def describe(self):
        return {"scheduled": len(self.jobs),
                **{status: sum(j["status"] == status for j in self.jobs)
                   for status in ("queued", "running", "completed", "failed", "cancelled")},
                "latest_completed": self.latest_completed, "best_completed": self.best_completed}

    def finish(self):
        """Drain every scheduled evaluation before the final training summary."""
        self.poll()
        while self.running is not None or self.pending:
            if self.running is not None:
                try:
                    self.running[1].wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            self.poll()

    def close(self):
        """Reap workers on a training exception; retain cancelled requests."""
        if self.running is not None:
            job, process, _ = self.running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            self.running = None
            job["status"] = "cancelled"
            self._save_index()
        while self.pending:
            job = self.pending.popleft()
            job["status"] = "cancelled"
            self._save_index()
