"""Shared prepared-action step measurements; only step + completion are timed."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from generals.benchmarks.contracts import Counters
from .workload import WORKLOAD, array_digest, host, snapshot


def timed_steps(env, actions):
    """Every action is already a device-local [B,P] view, ready to consume."""
    durations = []
    step, complete = env.step, env.synchronize
    for action in actions:
        start = perf_counter_ns()
        step(action)
        complete()
        durations.append(perf_counter_ns() - start)
    return durations


def actions_on_device(history, device):
    if str(device) == "cpu":
        return list(history)
    import torch
    return list(torch.from_numpy(history).to(device).unbind(0))


def create_step_env(case, initial, history, *, backend, gpu_group_threads=None,
                    gpu_envs_per_block=None, mps_threads_per_group=128):
    env, metadata = None, {}
    try:
        cfg = case.map_config
        if backend == "cpu":
            from generals.vector import VecEnv
            env = VecEnv(initial, device="cpu", width=cfg.width, height=cfg.height, threads=case.threads,
                         auto_reset=True,
                         episode_ticks=case.episode_ticks, seed=case.seed)
            actions = actions_on_device(history, "cpu")
            metadata.update(name="cpu-step", threads=env.unwrapped.threads, **env.unwrapped.metadata)
            metadata["wrapper_sha256"] = sha256(Path(__import__("generals.backends.cpu.env", fromlist=["x"]).__file__).read_bytes()).hexdigest()
        elif backend in ("gpu", "mps"):
            import torch
            from generals.vector import VecEnv
            device = "mps" if backend == "mps" else case.device
            env = VecEnv({name: torch.from_numpy(value) for name, value in initial.items()},
                         width=cfg.width, height=cfg.height, device=device, seed=case.seed,
                         backend=backend, auto_reset=True, episode_ticks=case.episode_ticks)
            if backend == "gpu":
                env.unwrapped.configure_step(group_threads=gpu_group_threads,
                                             envs_per_block=gpu_envs_per_block)
            else:
                env.unwrapped.configure_step(threads_per_group=mps_threads_per_group)
            actions = actions_on_device(history, env.device)
            metadata.update(name=f"{backend}-step", torch=torch.__version__)
            if backend == "gpu":
                metadata.update(gpu=torch.cuda.get_device_name(env.device), cuda_runtime=torch.version.cuda)
        else:
            raise ValueError("step benchmark supports cpu, gpu and mps")
        return env, actions, metadata
    except BaseException:
        if env is not None:
            env.close()
        raise


def verify_final(env, expected):
    env.check_errors()
    if array_digest(snapshot(env)) != expected:
        raise AssertionError("step trajectory differs from prepared CPU trajectory")
    raw = getattr(env, "unwrapped", env)
    if not np.array_equal(host(raw.rng_state), host(raw._initial_rng)):
        raise AssertionError("step generated random decisions")
    return {"state_statistics_counters_match": True, "rng_unchanged": True}


def compile_step(env, action):
    env.step(action)
    env.synchronize()
    env.check_errors()
    env.reset()
    env.synchronize()


def warm_steps(env, actions):
    env.reset()
    env.synchronize()
    if len(actions):
        timed_steps(env, actions)


def measure_sample(env, actions, case, expected, *, repeat):
    env.reset()
    env.synchronize()
    durations = timed_steps(env, actions)
    elapsed = sum(durations) * 1e-9
    verification = verify_final(env, expected)
    raw = getattr(env, "unwrapped", env)
    counter_array = host(raw.counters)
    counters = Counters(*[sum(map(int, counter_array[:, col])) for col in range(9)])
    if counters.env_ticks != case.batch_size * len(actions):
        raise AssertionError("not every environment advanced one tick per step")
    if counters.player_actions != counters.move_actions + counters.wait_actions:
        raise AssertionError("action counters do not add up")
    return {"repeat": repeat, "elapsed_seconds": elapsed,
            "counters": asdict(counters), "env_steps_per_second": counters.env_ticks / elapsed,
            "batch_steps_per_second": len(actions) / elapsed,
            "mean_step_us": elapsed / len(actions) * 1e6,
            "p50_step_us": float(np.percentile(durations, 50)) / 1000,
            "p95_step_us": float(np.percentile(durations, 95)) / 1000,
            "step_latency_ns": durations, "verification": verification}


def summarize(samples):
    keys = ("mean_step_us", "p50_step_us", "p95_step_us", "env_steps_per_second",
            "batch_steps_per_second", "elapsed_seconds")
    return {**{f"median_{key}": median(s[key] for s in samples) for key in keys},
            "min_mean_step_us": min(s["mean_step_us"] for s in samples),
            "max_mean_step_us": max(s["mean_step_us"] for s in samples),
            "max_env_steps_per_second": max(s["env_steps_per_second"] for s in samples)}


def by_tick_kind(durations_ns, kinds):
    values = np.asarray(durations_ns)
    if len(values) != len(kinds):
        raise ValueError("latencies and tick labels must have equal length")
    return {kind: {"steps": int(len(group)), "mean_us": float(group.mean()) / 1000,
                   "p50_us": float(np.percentile(group, 50)) / 1000,
                   "p95_us": float(np.percentile(group, 95)) / 1000,
                   "wall_fraction": float(group.sum() / values.sum())}
            for kind in dict.fromkeys(kinds)
            if len(group := values[np.asarray(kinds) == kind])}
