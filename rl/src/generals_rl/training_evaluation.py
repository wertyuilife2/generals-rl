"""Evaluate immutable learner snapshots in isolated subprocesses.

Scheduling and reporting live outside the vendored runner, policy pool and MDP.
The evaluator reuses the complete, paired built-in-controller match protocol.
Only the parent writes TensorBoard, with the evaluated iteration as its x axis.
"""
from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import perf_counter

import torch

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
    baseline_workers: int = 1
    keep_snapshots: int | None = None
    device: str = "cpu"
    timeout_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("evaluation.enabled must be boolean")
        if not isinstance(self.stochastic, bool):
            raise TypeError("evaluation.stochastic must be boolean")
        if not isinstance(self.device, str):
            raise TypeError("evaluation.device must be a CPU or CUDA device string")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("evaluation.device must be a CPU or CUDA device string") from exc
        if device.type not in ("cpu", "cuda"):
            raise ValueError("evaluation.device must be a CPU or CUDA device string")
        object.__setattr__(self, "device", str(device))
        if self.timeout_seconds is not None and (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("evaluation.timeout_seconds must be positive and finite or null")
        for name in ("every_n_snapshots", "maps_per_ai", "batch_size", "max_ticks", "cpu_threads", "baseline_workers", "seed"):
            minimum = 0 if name == "seed" else 1
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"evaluation.{name} must be an integer >= {minimum}")
        if self.keep_snapshots is not None and (not isinstance(self.keep_snapshots, int)
                or isinstance(self.keep_snapshots, bool) or self.keep_snapshots < 1):
            raise ValueError("evaluation.keep_snapshots must be a positive integer or null")

    @property
    def synchronous(self):
        """A CUDA worker shares the learner's device only while it is paused."""
        return torch.device(self.device).type == "cuda"

    @property
    def worker_timeout_seconds(self):
        if self.timeout_seconds is not None:
            return self.timeout_seconds
        return 1800.0 if self.synchronous else None


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _stop_worker(process):
    # The evaluator owns a separate session, including its spawned controllers.
    # Reap the leader and kill surviving children even if the leader exits first.
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class PeriodicEvaluator:
    """One worker at a time; CPU evaluation queues and CUDA evaluation blocks.

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
        self.budget_exhausted = False
        self.on_complete = on_complete
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_json(self.directory / "run_config.json", run_config)
        self._save_index()

    def _estimated_worker_budget_seconds(self):
        # Admission estimate only; the actual worker timeout remains unchanged.
        completed = [job["worker_seconds"] for job in self.jobs
                     if job["status"] == "completed"
                     and isinstance(job.get("worker_seconds"), (int, float))
                     and not isinstance(job["worker_seconds"], bool)
                     and math.isfinite(job["worker_seconds"]) and job["worker_seconds"] > 0]
        if completed:
            return 1.25 * max(completed[-3:]) + 30.0
        return self.settings.worker_timeout_seconds

    def on_snapshot(self, snapshot, pool_directory, *, publication_index,
                    steps_per_environment, environment_steps, remaining_seconds=None):
        """Count periodic learned snapshots; the initial random seed is excluded.

        The publication ordinal is derived from the absolute learner iteration,
        so resuming does not shift the ten-snapshot cadence.
        """
        entered = perf_counter()
        if (not self.settings.enabled or publication_index < 1 or snapshot.metadata["seed"]
                or publication_index % self.settings.every_n_snapshots):
            return False
        if any(j["publication_index"] == publication_index for j in self.jobs):
            return False
        timeout = self.settings.worker_timeout_seconds
        deadline_reason = "evaluation_timeout"
        if self.settings.synchronous and remaining_seconds is not None:
            if (isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds, (int, float))
                    or not math.isfinite(remaining_seconds)):
                raise ValueError("remaining_seconds must be finite or null")
            available = max(0.0, remaining_seconds - 10.0)
            estimated = self._estimated_worker_budget_seconds()
            if available < estimated:
                self.budget_exhausted = True
                print(f"[eval] budget_reserve snapshot={snapshot.snapshot_id} "
                      f"available={available:g} estimated={estimated:g}; no job queued", flush=True)
                return False
            if available <= timeout:
                timeout = available
                deadline_reason = "wall_budget"
        deadline = entered + timeout if self.settings.synchronous else None
        target = self.directory / f"snapshot_{snapshot.snapshot_id:08d}"
        checkpoint = target / "snapshot.pt"
        job = {
            "snapshot_id": snapshot.snapshot_id,
            "iteration": snapshot.metadata["iteration"],
            "publication_index": publication_index,
            "steps_per_environment": steps_per_environment,
            "environment_steps": environment_steps,
            "checkpoint": str(checkpoint), "checkpoint_sha256": snapshot.metadata["sha256"],
            "report": str(target / "report.json"),
            "worker_log": str(target / "worker.log"),
            "stochastic": self.settings.stochastic,
            "device": self.settings.device,
            "synchronous": self.settings.synchronous,
            "timeout_seconds": timeout, "deadline": deadline, "deadline_reason": deadline_reason,
            "status": "queued",
        }
        if deadline is not None and perf_counter() >= deadline:
            self.jobs.append(job)
            self._cancel(job, deadline_reason)
            return False
        target.mkdir()
        shutil.copyfile(Path(pool_directory) / snapshot.metadata["filename"], checkpoint)
        if sha256(checkpoint.read_bytes()).hexdigest() != job["checkpoint_sha256"]:
            raise RuntimeError("evaluation snapshot differs from the published pool snapshot")
        _write_json(target / "run_config.json", self.run_config)
        self.jobs.append(job)
        self.pending.append(job)
        self._save_index()
        print(f"[eval] queued snapshot={job['snapshot_id']} iteration={job['iteration']} "
              f"games={8 * self.settings.maps_per_ai}", flush=True)
        self.poll()
        if self.settings.synchronous:
            # Called by train.on_update after a complete PPO update. Wait before
            # the learner can collect another rollout or publish another job.
            self.finish()
        return True

    def poll(self):
        if self.running is not None:
            job, process, started = self.running
            returncode = process.poll()
            timeout = job.get("timeout_seconds", self.settings.worker_timeout_seconds)
            deadline = job.get("deadline")
            expired = ((deadline is not None and perf_counter() >= deadline)
                       or (deadline is None and timeout is not None and perf_counter() - started >= timeout))
            if returncode is None and expired:
                _stop_worker(process)
                self.running = None
                job["worker_seconds"] = perf_counter() - started
                job["returncode"] = process.returncode
                self._fail(job, TimeoutError(f"evaluation worker exceeded {timeout:g} seconds"))
            if returncode is not None:
                self.running = None
                job["worker_seconds"] = perf_counter() - started
                job["returncode"] = returncode
                try:
                    if returncode != 0:
                        _stop_worker(process)
                        raise RuntimeError(f"evaluation worker exited with code {returncode}")
                    self._complete(job)
                except Exception as exc:
                    self._fail(job, exc)
        while self.running is None and self.pending:
            job = self.pending.popleft()
            settings = self.settings
            if job.get("deadline") is not None and perf_counter() >= job["deadline"]:
                self._cancel(job, job["deadline_reason"])
                continue
            command = [sys.executable, "-u", "-m", "generals_rl.evaluate_builtin",
                       job["checkpoint"], "--maps-per-ai", str(settings.maps_per_ai),
                       "--batch-size", str(settings.batch_size), "--max-ticks", str(settings.max_ticks),
                       "--seed", str(settings.seed), "--cpu-threads", str(settings.cpu_threads),
                       "--baseline-workers", str(settings.baseline_workers),
                       "--device", settings.device, "--output", job["report"]]
            command.append("--stochastic" if settings.stochastic else "--greedy")
            if torch.device(settings.device).type == "cuda":
                command.extend(["--trace-directory", str(Path(job["report"]).parent / "demo_traces")])
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            if torch.device(settings.device).type == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            elif torch.cuda.is_initialized():
                # Release only unused allocator blocks. Live learner, optimizer
                # and historical opponent tensors remain on their devices.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if job.get("deadline") is not None and perf_counter() >= job["deadline"]:
                self._cancel(job, job["deadline_reason"])
                continue
            try:
                with Path(job["worker_log"]).open("w") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                               env=environment, start_new_session=True)
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
                or report["device"] != self.settings.device or report["stochastic"] != self.settings.stochastic):
            raise RuntimeError("evaluation report does not match the requested snapshot/protocol")
        if report.get("baseline_workers", 1) != self.settings.baseline_workers:
            raise RuntimeError("evaluation baseline workers differ from the requested run")
        for kind in AI_KINDS:
            if report["by_ai"][kind]["games"] != 2 * self.settings.maps_per_ai:
                raise RuntimeError(f"evaluation did not finish all scheduled {kind} games")
        if torch.device(self.settings.device).type == "cuda":
            requested_cudnn = self.run_config.get("cudnn_allow_tf32")
            if requested_cudnn is not None and report.get("cudnn_allow_tf32") != requested_cudnn:
                raise RuntimeError("evaluation cuDNN TF32 setting differs from the requested run")
            requested_benchmark = self.run_config.get("cudnn_benchmark")
            if requested_benchmark is not None and report.get("cudnn_benchmark") != requested_benchmark:
                raise RuntimeError("evaluation cuDNN benchmark setting differs from the requested run")
            requested_matmul = self.run_config.get("matmul_precision")
            if requested_matmul is not None and report.get("matmul_precision") != requested_matmul:
                raise RuntimeError("evaluation matmul precision differs from the requested run")
            policy_options = self.run_config.get("runner", {}).get("policy", {})
            expected_dtype = "bfloat16" if policy_options.get("amp_bfloat16", False) else "float32"
            if report.get("inference_dtype") != expected_dtype:
                raise RuntimeError("evaluation inference dtype differs from the requested policy AMP setting")
            traces = report["selected_game_traces"]
            if traces["protocol"] != "recorded_joint_actions_core_v1":
                raise RuntimeError("evaluation did not record the requested action trace protocol")
            games = {game["game_id"]: game for game in report["games"]}
            expected_pairs = {(kind, outcome) for kind in AI_KINDS for outcome in ("loss", "win")}
            covered = []
            for trace in traces["cases"]:
                pair = trace["opponent"], trace["outcome"]
                game = games[trace["game_id"]]
                if (pair != (game["opponent"], game["outcome"])
                        or trace["checkpoint_sha256"] != job["checkpoint_sha256"]
                        or sha256(Path(trace["path"]).read_bytes()).hexdigest() != trace["sha256"]):
                    raise RuntimeError("evaluation action trace differs from its report/checkpoint")
                covered.append(pair)
            for missing in traces["missing"]:
                pair = missing["opponent"], missing["outcome"]
                if any((game["opponent"], game["outcome"]) == pair for game in games.values()):
                    raise RuntimeError("evaluation omitted an available win/loss trace")
                covered.append(pair)
            if len(covered) != 8 or set(covered) != expected_pairs:
                raise RuntimeError("evaluation did not account for all eight demo outcomes")
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
        for key in ("device", "matmul_precision", "cudnn_allow_tf32", "cudnn_benchmark",
                    "parameter_dtype", "inference_dtype", "baseline_workers"):
            if key in report:
                point[key] = job[key] = report[key]
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
        self._prune_snapshots()
        self._save_index()
        if self.writer is not None:
            self.writer.add_scalar("Eval/failed_jobs", self.describe()["failed"], job["iteration"])
            self.writer.flush()
        print(f"[eval] FAILED snapshot={job['snapshot_id']}: {job['error']}; "
              f"log={job['worker_log']}", flush=True)

    def _cancel(self, job, reason):
        job.update(status="cancelled", reason=reason,
                   checkpoint_retained=Path(job["checkpoint"]).exists())
        self._prune_snapshots()
        self._save_index()
        if self.writer is not None:
            self.writer.add_scalar("Eval/cancelled_jobs", self.describe()["cancelled"], job["iteration"])
            self.writer.flush()
        print(f"[eval] cancelled snapshot={job['snapshot_id']} reason={reason}", flush=True)

    def _prune_snapshots(self):
        if self.settings.keep_snapshots is None:
            return
        finished = [job for job in self.jobs if job["status"] in ("completed", "failed", "cancelled")]
        for job in finished[:-self.settings.keep_snapshots]:
            Path(job["checkpoint"]).unlink(missing_ok=True)
            job["checkpoint_retained"] = False
        recent_reports = {job.get("report") for job in finished[-self.settings.keep_snapshots:]}
        best_report = self.best_completed["report"] if self.best_completed else None
        for job in finished:
            if not job.get("report"):
                continue
            traces = Path(job["report"]).parent / "demo_traces"
            if traces.is_dir():
                # best_report.json keeps paths into the original report's
                # directory, even after its snapshot weights are pruned.
                retained = job["report"] == best_report or job["report"] in recent_reports
                if not retained:
                    shutil.rmtree(traces)
                job["trace_retained"] = retained

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
                    deadline = self.running[0].get("deadline")
                    timeout = 1.0 if deadline is None else min(1.0, max(0.0, deadline - perf_counter()))
                    self.running[1].wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
            self.poll()

    def close(self):
        """Reap workers on a training exception; retain cancelled requests."""
        if self.running is not None:
            job, process, _ = self.running
            _stop_worker(process)
            self.running = None
            job["status"] = "cancelled"
        while self.pending:
            job = self.pending.popleft()
            job["status"] = "cancelled"
        self._prune_snapshots()
        self._save_index()
