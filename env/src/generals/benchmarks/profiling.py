"""Step-oriented diagnosis: unprofiled baseline, native CPU phases, CUDA timeline.

Instrumentation is a separate pass, with identical initial state/actions. It is
never a throughput result. CPU phase totals are sampled worker time; CUDA API
calls overlap device execution and must not be added to kernel durations.
"""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from generals.protocol.config import _integer

from .measurement import (create_step_env, compile_step, warm_steps, timed_steps,
                          verify_final, by_tick_kind)
from .step import (add_arguments, backends, cases, options, new_report, write_report,
                   measure_step_case, print_result)
from .workload import native_tools, prepare_workload

CPU_PHASES = ("setup_cache", "growth", "actions_elimination", "outputs_counters",
              "autoreset", "cache_writeback", "early_return")
CPU_COLUMNS = ("native_wall_ns", "worker_max_ns", "worker_mean_ns", "worker_min_ns",
               "sampled_lanes", *[name + "_ns" for name in CPU_PHASES],
               "team_entry_ns", "worker_start_spread_ns", "team_join_tail_ns")


def cpu_profile(env, actions, expected, kinds, sample_stride):
    raw = env.unwrapped
    library = native_tools()
    pointer = library.generals_profile_begin(len(actions), raw.threads, sample_stride)
    original = raw._lib
    try:
        raw._lib = library
        durations = timed_steps(env, actions)
        rows = np.ctypeslib.as_array(pointer, shape=(len(actions), len(CPU_COLUMNS))).copy()
    finally:
        raw._lib = original
    verification = verify_final(env, expected)
    phases = rows[:, 5:12].sum(axis=0)
    phase_total = phases.sum()
    mean_native = float(rows[:, 0].mean()) / 1000
    mean_step = float(np.mean(durations)) / 1000
    return {
        "method": "compile-time hooks in production step.cpp; rotating 1/N lane sample",
        "sample_stride": sample_stride, "sampled_lanes": int(rows[:, 4].sum()),
        "mean_profiled_step_us": mean_step, "mean_native_region_us": mean_native,
        "mean_outside_native_us": mean_step - mean_native,
        "mean_max_worker_span_us": float(rows[:, 1].mean()) / 1000,
        "mean_mean_worker_span_us": float(rows[:, 2].mean()) / 1000,
        "mean_min_worker_span_us": float(rows[:, 3].mean()) / 1000,
        "mean_native_minus_max_worker_us": float((rows[:, 0] - rows[:, 1]).mean()) / 1000,
        "mean_team_entry_us": float(rows[:, 12].mean()) / 1000,
        "mean_worker_start_spread_us": float(rows[:, 13].mean()) / 1000,
        "mean_team_join_tail_us": float(rows[:, 14].mean()) / 1000,
        "sampled_worker_phases": {name: {
            "total_us": float(value) / 1000,
            "fraction": float(value / phase_total) if phase_total else 0.0}
            for name, value in zip(CPU_PHASES, phases)},
        "step_latency_ns": durations, "native_columns": CPU_COLUMNS, "native_rows": rows.tolist(),
        "by_tick_kind": by_tick_kind(durations, kinds), "verification": verification,
        "limitations": [
            "Phase clocks and sample branches perturb execution; small phases include clock overhead.",
            "Worker phase sums are not wall time or additive parts of batch latency.",
            "Worker spans include OS preemption. Native minus max worker includes team entry/join and scheduling.",
            "Outside-native time includes Python/ctypes and profiler aggregation; it is not pure Python overhead.",
            "No hardware counters: this does not establish cache-miss, bandwidth or instruction-stall causes.",
        ],
    }


def union_us(intervals):
    total, end = 0.0, float("-inf")
    for left, right in sorted(intervals):
        total += max(0.0, right - max(left, end))
        end = max(end, right)
    return total


def parse_cuda_trace(path, steps):
    events = [event for event in json.loads(Path(path).read_text())["traceEvents"]
              if event.get("ph") == "X" and event.get("dur", 0) >= 0]
    markers = sorted((event for event in events if event.get("cat") == "user_annotation" and event.get("name", "").startswith("env_step/")),
                     key=lambda event: event["ts"])
    if [event["name"] for event in markers] != [f"env_step/{i}" for i in range(steps)]:
        raise ValueError("trace must contain exactly one ordered marker per measured step")
    starts = [event["ts"] for event in markers]
    spans = [[] for _ in markers]
    kernels, apis, transfers = {}, {}, []
    for event in events:
        category = str(event.get("cat", ""))
        if category not in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver"):
            continue
        index = bisect_right(starts, event["ts"]) - 1
        if index < 0:
            continue
        marker_end = markers[index]["ts"] + markers[index]["dur"]
        if event["ts"] + event["dur"] > marker_end + .1:
            continue
        name, duration = event["name"], event["dur"]
        if category == "kernel":
            spans[index].append((event["ts"], event["ts"] + duration))
            entry = kernels.setdefault(name, {"calls": 0, "total_us": 0.0, "trace_args": event.get("args", {})})
        elif category in ("gpu_memcpy", "gpu_memset"):
            transfers.append({"name": name, "category": category, "us": duration, "step": index})
            continue
        else:
            entry = apis.setdefault(name, {"calls": 0, "total_us": 0.0})
        entry["calls"] += 1
        entry["total_us"] += duration
    if not kernels:
        raise RuntimeError("CUDA trace contains no measured kernels; check CUPTI availability")
    if any(not intervals for intervals in spans):
        raise RuntimeError("CUDA trace is incomplete: a measured step has no kernel")
    kernel_us = [union_us(intervals) for intervals in spans]
    marker_us = [event["dur"] for event in markers]
    return {"kernel_count": sum(entry["calls"] for entry in kernels.values()),
            "kernels": kernels, "cuda_api_calls": apis, "transfers": transfers,
            "kernel_union_us_per_step": kernel_us, "marker_us_per_step": marker_us,
            "mean_kernel_us": float(np.mean(kernel_us)),
            "mean_marker_us": float(np.mean(marker_us)),
            "gpu_busy_fraction_of_marked_steps": sum(kernel_us) / sum(marker_us),
            "mean_marker_outside_kernel_us": float(np.mean(marker_us) - np.mean(kernel_us))}


def gpu_profile(env, actions, expected, kinds, trace_path):
    import torch
    markers = [f"env_step/{i}" for i in range(len(actions))]
    durations = []
    step, complete = env.step, env.synchronize
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as capture:
        # Prime capture callbacks outside measured markers, then restore the same
        # cold initial state used by the baseline. Capture setup is not a tick cost.
        step(actions[0])
        complete()
        env.reset()
        complete()
        for marker, action in zip(markers, actions):
            with torch.profiler.record_function(marker):
                start = perf_counter_ns()
                step(action)
                complete()
                durations.append(perf_counter_ns() - start)
    verification = verify_final(env, expected)
    capture.export_chrome_trace(str(trace_path))
    timeline = parse_cuda_trace(trace_path, len(actions))
    return {"method": "PyTorch/CUPTI trace of the production CUDA public step",
            "mean_profiled_step_us": float(np.mean(durations)) / 1000,
            "step_latency_ns": durations, "by_tick_kind": by_tick_kind(durations, kinds),
            "kernel_by_tick_kind": by_tick_kind(np.array(timeline["kernel_union_us_per_step"]) * 1000, kinds),
            "trace_path": str(trace_path), "timeline": timeline, "verification": verification,
            "limitations": [
                "Throughput comes from the separate unprofiled baseline, not this trace.",
                "CUDA API/synchronize intervals overlap kernels; their durations are not additive.",
                "Marker time outside kernels includes host work, gaps and capture overhead.",
                "Trace occupancy/register metadata are estimates, not achieved hardware utilization.",
                "No hardware counters: this cannot prove memory bandwidth or warp-stall bottlenecks.",
            ]}


def profile_case(case, *, backend, prepared=None, trace_path=None, cpu_sample_stride=32, **kwargs):
    cpu_sample_stride = _integer("cpu_sample_stride", cpu_sample_stride, 1)
    for name, default, minimum in (("steps", 1000, 1), ("warmup", 1000, 0), ("repeats", 3, 1)):
        _integer(name, kwargs.get(name, default), minimum)
    if trace_path is not None:
        trace_path = Path(trace_path)
        if trace_path.exists():
            raise FileExistsError(f"trace already exists: {trace_path}")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "gpu" and trace_path is None:
        raise ValueError("GPU profile needs a new trace path")
    steps, warmup = kwargs.get("steps", 1000), kwargs.get("warmup", 1000)
    prepared = prepared or prepare_workload(case, steps, warmup)
    baseline = measure_step_case(case, backend=backend, prepared=prepared, **kwargs)
    env, actions, _ = create_step_env(case, prepared.initial, prepared.history, backend=backend,
        **{name: kwargs[name] for name in ("gpu_group_threads", "gpu_envs_per_block") if name in kwargs})
    try:
        compile_step(env, actions[0])
        warm_steps(env, actions[:warmup])
        env.reset()
        env.synchronize()
        expected = prepared.fingerprints["expected_final_state_sha256"]
        if backend == "cpu":
            profile = cpu_profile(env, actions[:steps], expected, prepared.tick_kinds, cpu_sample_stride)
        else:
            profile = gpu_profile(env, actions[:steps], expected, prepared.tick_kinds, trace_path)
        profile["profiled_over_baseline_latency_ratio"] = (
            profile["mean_profiled_step_us"] / baseline["summary"]["median_mean_step_us"])
        baseline["profile"] = profile
        return baseline
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--cpu-sample-stride", type=int, default=32,
                        help="profile one rotating lane in N; clocks are absent from the normal build")
    args = parser.parse_args(argv)
    try:
        report = new_report(args, "step_profile")
        report["parameters"]["cpu_sample_stride"] = args.cpu_sample_stride
        for case in cases(args):
            print(f"Preparing B={case.batch_size} (outside timing)...", flush=True)
            prepared = prepare_workload(case, args.steps, args.warmup)
            for backend in backends(args):
                suffix = f"{case.map_config.width}x{case.map_config.height}_p{case.map_config.players}_b{case.batch_size}"
                trace = args.output.with_name(f"{args.output.stem}_{backend}_{suffix}.cuda.trace.json") if backend == "gpu" else None
                result = profile_case(case, backend=backend, prepared=prepared, trace_path=trace,
                                      cpu_sample_stride=args.cpu_sample_stride, **options(args))
                report["results"].append(result)
                print_result(case, backend, result)
                print(f"  profile/baseline latency: {result['profile']['profiled_over_baseline_latency_ratio']:.2f}x", flush=True)
        write_report(args.output, report)
        return 0
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
