"""CPU/GPU comparison of one public step(action) + completion at a time."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform

from generals.protocol.config import MapConfig, _integer
from .contracts import BenchmarkCase
from .measurement import (create_step_env, compile_step, warm_steps, measure_sample,
                          summarize, by_tick_kind)
from .workload import WORKLOAD, prepare_workload

TIMING_SCOPE = {
    "included": ["public env.step(action) and completion synchronization EVERY tick",
                 "Python validation, launch, game rules, statistics and automatic reset"],
    "excluded": ["decisions, action views and upload", "map generation, construction and compilation",
                 "warmup and reset before samples", "readback and validation",
                 "observations, reward features, policy inference and training"],
    "units": "batch step/s = 1 / mean latency; env step/s = batch size / mean latency",
}


def measure_step_case(case, *, backend, steps=1000,
                      warmup=1000, repeats=3, gpu_group_threads=None,
                      gpu_envs_per_block=None, mps_threads_per_group=128, prepared=None):
    steps, warmup, repeats = (_integer("steps", steps, 1), _integer("warmup", warmup, 0),
                              _integer("repeats", repeats, 1))
    if backend not in ("cpu", "gpu", "mps"):
        raise ValueError("unsupported step backend")
    prepared = prepared or prepare_workload(case, steps, warmup)
    if prepared.steps != steps or len(prepared.history) < max(steps, warmup):
        raise ValueError("prepared workload length does not match measurement")
    env, actions, metadata = create_step_env(case, prepared.initial, prepared.history,
        backend=backend, gpu_group_threads=gpu_group_threads,
        gpu_envs_per_block=gpu_envs_per_block, mps_threads_per_group=mps_threads_per_group)
    try:
        compile_step(env, actions[0])
        samples = []
        for repeat in range(repeats):
            warm_steps(env, actions[:warmup])
            sample = measure_sample(env, actions[:steps], case,
                prepared.fingerprints["expected_final_state_sha256"], repeat=repeat)
            sample["by_tick_kind"] = by_tick_kind(sample["step_latency_ns"], prepared.tick_kinds)
            samples.append(sample)
        if backend in ("gpu", "mps"):
            metadata.update(env.unwrapped.step_metadata())
        return {"case": asdict(case), "backend": {**metadata, "device_kind": backend},
                "fingerprints": prepared.fingerprints, "samples": samples,
                "summary": summarize(samples)}
    finally:
        env.close()


def size(value):
    try:
        dimensions = tuple(int(x) for x in value.lower().split("x"))
        if len(dimensions) == 1:
            dimensions *= 2
        if len(dimensions) != 2 or min(dimensions) < 1:
            raise ValueError
        return dimensions
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be positive N or WIDTHxHEIGHT") from exc


def add_arguments(parser):
    parser.add_argument("--backend", choices=("both", "cpu", "gpu", "mps"), default="both")
    parser.add_argument("--sizes", type=size, nargs="+", default=[(25, 25)])
    parser.add_argument("--players", type=int, nargs="+", default=[4])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[10000])
    parser.add_argument("--threads", type=int, default=0, help="CPU OpenMP threads; 0 = available processors")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-group-threads", type=int, default=None)
    parser.add_argument("--gpu-envs-per-block", type=int, default=None)
    parser.add_argument("--mps-threads-per-group", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--episode-ticks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)


def cases(args):
    for width, height in dict.fromkeys(args.sizes):
        for players in dict.fromkeys(args.players):
            for batch in dict.fromkeys(args.batch_sizes):
                yield BenchmarkCase(MapConfig(width, height, players), batch_size=batch,
                    threads=args.threads, episode_ticks=args.episode_ticks,
                    seed=args.seed, device="mps" if args.backend == "mps" else args.device)


def backends(args):
    return ("cpu", "mps" if args.device.startswith("mps") else "gpu") if args.backend == "both" else (args.backend,)


def options(args):
    return dict(steps=args.steps, warmup=args.warmup, repeats=args.repeats,
                gpu_group_threads=args.gpu_group_threads, gpu_envs_per_block=args.gpu_envs_per_block,
                mps_threads_per_group=args.mps_threads_per_group)


def new_report(args, mode):
    for name, minimum in (("steps", 1), ("warmup", 0), ("repeats", 1)):
        _integer(name, getattr(args, name), minimum)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    cpu_name = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu_name = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), cpu_name)
    source = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(source)): sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob("*")) if path.suffix in (".py", ".cpp", ".cu", ".metal")}
    return {"schema_version": 2, "workload": WORKLOAD, "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "system": {"platform": platform.platform(), "python": platform.python_version(),
                       "cpu": cpu_name, "logical_cpus": os.cpu_count(),
                       "openmp": {k: os.environ.get(k) for k in
                                  ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY")}},
            "source_sha256": hashes, "parameters": {**options(args), "backend": args.backend},
            "timing_scope": TIMING_SCOPE, "results": []}


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def print_result(case, backend, result):
    summary = result["summary"]
    print(f"{backend:3} {case.map_config.width}x{case.map_config.height}/{case.map_config.players} "
          f"B={case.batch_size}: {summary['median_mean_step_us']:.3f} us/batch step; "
          f"{summary['median_env_steps_per_second']/1e6:.3f} M env step/s", flush=True)


def run_from_args(args):
    report = new_report(args, "step")
    for case in cases(args):
        print(f"Preparing {case.map_config.width}x{case.map_config.height}, "
              f"{case.map_config.players} players, B={case.batch_size} (excluded from timing)...", flush=True)
        prepared = prepare_workload(case, args.steps, args.warmup)
        paired = []
        for backend in backends(args):
            result = measure_step_case(case, backend=backend, prepared=prepared, **options(args))
            report["results"].append(result)
            paired.append(result)
            print_result(case, backend, result)
        if len(paired) == 2:
            ratio = paired[0]["summary"]["median_mean_step_us"] / paired[1]["summary"]["median_mean_step_us"]
            label = "mps_speedup" if paired[1]["backend"]["device_kind"] == "mps" else "gpu_speedup"
            report.setdefault("comparisons", []).append({"case": asdict(case), label: ratio})
            print(f"{label}: {ratio:.2f}x", flush=True)
    write_report(args.output, report)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
