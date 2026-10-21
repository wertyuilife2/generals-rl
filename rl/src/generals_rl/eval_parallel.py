"""Split one fixed evaluation schedule across isolated CUDA subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

import torch
from generals_baselines import AI_KINDS

from .evaluate_builtin import SelectedGameTraces, aggregate_games, evaluate_builtin, make_schedule


def merge_reports(reports, *, devices, elapsed, worker_reports):
    """Validate complete, disjoint shards and recompute global statistics."""
    if len(reports) != len(devices) or not reports:
        raise ValueError("one report is required per evaluation device")
    first = reports[0]
    variable = {"device", "shard_index", "parallel_games", "scheduled_games", "completed_games",
                "overall", "by_ai", "by_seat", "by_ai_and_seat", "timing", "games", "selected_game_traces",
                "baseline_workers"}
    for index, report in enumerate(reports):
        if (report["shard_index"] != index or report["num_shards"] != len(reports)
                or report["device"] != devices[index]):
            raise RuntimeError("evaluation worker returned the wrong shard/device")
        if {k: v for k, v in report.items() if k not in variable} != {k: v for k, v in first.items() if k not in variable}:
            raise RuntimeError("evaluation shards disagree on checkpoint/protocol")
    games = sorted((game for report in reports for game in report["games"]), key=lambda game: game["game_id"])
    opponents = tuple(first["by_ai"])
    schedule = make_schedule(maps_per_ai=first["maps_per_ai"], seed=first["seed"], opponents=opponents)
    if [game["game_id"] for game in games] != [spec.game_id for spec in schedule]:
        raise RuntimeError("evaluation shards contain missing or repeated games")
    for index, report in enumerate(reports):
        expected = {spec.game_id for spec in schedule[index::len(reports)]}
        if (report["scheduled_games"] != len(expected) or report["completed_games"] != len(expected)
                or {game["game_id"] for game in report["games"]} != expected):
            raise RuntimeError("evaluation worker did not finish its exact schedule shard")
    worker_sum = {key: sum(report["timing"][key] for report in reports) for key in first["timing"]
                  if not key.endswith("per_second")}
    ticks, decisions = worker_sum["environment_ticks"], worker_sum["student_decisions"]
    result = {**first, "device": devices[0], "devices": devices, "num_shards": len(reports),
              "shard_index": None, "scheduled_games": len(games), "completed_games": len(games),
              "parallel_games": sum(report["parallel_games"] for report in reports), "games": games,
              "overall": aggregate_games(games),
              "by_ai": {kind: aggregate_games([g for g in games if g["opponent"] == kind]) for kind in opponents},
              "by_seat": {str(seat): aggregate_games([g for g in games if g["learner_seat"] == seat]) for seat in (0, 1)},
              "by_ai_and_seat": {kind: {str(seat): aggregate_games([
                  g for g in games if g["opponent"] == kind and g["learner_seat"] == seat])
                  for seat in (0, 1)} for kind in opponents},
              "worker_reports": worker_reports,
              "timing": {"wall_seconds": elapsed, "total_seconds_including_setup": elapsed,
                         "environment_ticks": ticks, "student_decisions": decisions,
                         "environment_ticks_per_second": ticks / elapsed,
                         "student_decisions_per_second": decisions / elapsed,
                         "games_per_second": len(games) / elapsed,
                         "scope": "parallel wall including worker startup; worker sums are not wall time",
                         "worker_sum": worker_sum}}
    result.pop("baseline_workers", None)  # Older reports may still contain this no-op field.
    if "selected_game_traces" in first:
        candidates = [case for report in reports for case in report["selected_game_traces"]["cases"]]
        selected = {}
        for case in sorted(candidates, key=lambda case: case["game_id"]):
            selected.setdefault((case["opponent"], case["outcome"]), case)
        pairs = [(kind, outcome) for kind in opponents for outcome in ("loss", "win")]
        result["selected_game_traces"] = {
            "protocol": SelectedGameTraces.protocol, "selection": "first_game_id_win_and_loss_per_ai",
            "cases": [selected[pair] for pair in pairs if pair in selected],
            "missing": [{"opponent": kind, "outcome": outcome} for kind, outcome in pairs if (kind, outcome) not in selected]}
    return result


def evaluate_parallel(checkpoint, *, devices, output=None, trace_directory=None, **options):
    """Launch one CUDA worker per device; no learner state is accessed."""
    devices = [str(torch.device(device)) for device in devices]
    if (not devices or len(set(devices)) != len(devices)
            or any(torch.device(device).type != "cuda" or torch.device(device).index is None for device in devices)):
        raise ValueError("devices must be unique indexed CUDA devices")
    schedule = make_schedule(maps_per_ai=options.get("maps_per_ai", 50), seed=options.get("seed", 1000042),
                             opponents=options.get("opponents", AI_KINDS))
    if len(devices) > len(schedule):
        raise ValueError("cannot use more devices than scheduled games")
    options.pop("device", None)
    options.pop("baseline_workers", None)
    if len(devices) == 1:
        return evaluate_builtin(checkpoint, device=devices[0], output=output, trace_directory=trace_directory, **options)
    # Reports/logs survive alongside the merged report. No pipe can fill while
    # another worker is being waited on, and all workers die with the owning
    # PeriodicEvaluator process group on a timeout or training exception.
    with tempfile.TemporaryDirectory(prefix="generals-eval-") as temporary:
        directory = Path(output).resolve().with_suffix(".workers") if output is not None else Path(temporary)
        if output is not None:
            directory.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        processes, paths = [], []
        try:
            for index, device in enumerate(devices):
                path = directory / f"shard_{index:02d}.json"
                command = [sys.executable, "-u", "-m", "generals_rl.evaluate_builtin", str(checkpoint),
                           "--device", device, "--output", str(path),
                           "--shard-index", str(index), "--num-shards", str(len(devices))]
                for name, value in options.items():
                    if name == "device" or value is None:
                        continue
                    flag = "--" + name.replace("_", "-")
                    if name == "stochastic":
                        command.append("--stochastic" if value else "--greedy")
                    elif name == "reference_inference":
                        if value:
                            command.append(flag)
                    else:
                        command.extend([flag, *map(str, value)] if isinstance(value, (tuple, list)) else [flag, str(value)])
                if trace_directory is not None:
                    command.extend(["--trace-directory", str(Path(trace_directory) / f"shard_{index:02d}")])
                environment = os.environ.copy()
                environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                with path.with_suffix(".log").open("w") as stream:
                    processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment))
                paths.append(path)
            pending = set(range(len(processes)))
            while pending:
                for index in tuple(pending):
                    code = processes[index].poll()
                    if code is not None:
                        if code != 0:
                            raise RuntimeError(f"evaluation shard {index} exited {code}; see {paths[index].with_suffix('.log')}")
                        pending.remove(index)
                if pending:
                    sleep(.05)
            reports = [json.loads(path.read_text()) for path in paths]
            report = merge_reports(reports, devices=devices, elapsed=perf_counter() - started,
                                   worker_reports=[str(path) for path in paths] if output is not None else [])
            if trace_directory is not None:
                retained = {case["path"] for case in report["selected_game_traces"]["cases"]}
                for path in Path(trace_directory).glob("shard_*/*.trace.jsonl"):
                    if str(path.resolve()) not in retained:
                        path.unlink()
            if output is not None:
                Path(output).write_text(json.dumps(report, indent=2) + "\n")
            return report
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
