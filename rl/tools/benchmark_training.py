"""Benchmark complete fresh rollouts, GAE and StudentPPO updates, without logger I/O.

``task.num_envs`` and --num-envs are PER RANK. Under torchrun, global N is
N_per_rank * WORLD_SIZE. Throughput counts fresh N*T transitions once; PPO
epochs and opponent NN decisions are included in time, never multiplied into
the transition numerator. Profile work is a separate additional full update.
"""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import argparse
import io
import json
import math
import os
from pathlib import Path
import statistics
from time import perf_counter

import torch
from rsl_rl.runners import OnPolicyRunnerCTS

from generals_rl.config import merge_config, seed_everything, task_config
from generals_rl.cts import GeneralsCTSVecEnv, runner_config
from generals_rl.curriculum import CurriculumConfig, RolloutCurriculum
from generals_rl.distributed_training import DistributedConfig
from generals_rl.self_play_training import SelfPlayConfig, prepare_self_play
from generals_rl.task import GeneralsTask, TaskConfig
from generals_rl.train import _matmul_precision


STAGES = ("collection_seconds", "gae_seconds", "update_seconds", "total_seconds")


def topology(environ=None):
    env = os.environ if environ is None else environ
    world, rank, local = (int(env.get(key, default)) for key, default in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world < 1 or not 0 <= rank < world or not 0 <= local < world:
        raise ValueError("invalid WORLD_SIZE/RANK/LOCAL_RANK")
    return {"world_size": world, "rank": rank, "local_rank": local}


class BenchmarkRunner(OnPolicyRunnerCTS):
    def _configure_multi_gpu(self):
        # The upstream runner initializes a standalone torchrun group. Reuse an
        # already initialized caller-owned group without initializing it twice.
        if not torch.distributed.is_initialized():
            return super()._configure_multi_gpu()
        context = topology()
        if (torch.distributed.get_world_size(), torch.distributed.get_rank()) != (
                context["world_size"], context["rank"]):
            raise ValueError("existing distributed group disagrees with torchrun environment")
        self.gpu_world_size, self.gpu_global_rank = context["world_size"], context["rank"]
        self.gpu_local_rank = context["local_rank"]
        self.is_distributed = self.gpu_world_size > 1
        self.multi_gpu_cfg = ({"world_size": self.gpu_world_size, "global_rank": self.gpu_global_rank,
                               "local_rank": self.gpu_local_rank} if self.is_distributed else None)
        if self.is_distributed and self.device not in ("cpu", f"cuda:{self.gpu_local_rank}"):
            raise ValueError("distributed runner device must match LOCAL_RANK")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def estimate_storage(observations, num_envs, rollout_steps, keys=None):
    selected = list(observations.keys()) if keys is None else list(keys)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("storage observation keys must be nonempty and unique")
    fields = {}
    for key in selected:
        value = observations[key]
        if value.shape[0] != num_envs:
            raise ValueError("storage observation batch does not match per-rank environments")
        fields[key] = {"shape_per_environment": list(value.shape[1:]), "dtype": str(value.dtype),
                       "bytes_per_transition": value.numel() // num_envs * value.element_size()}
    obs_bytes = sum(field["bytes_per_transition"] for field in fields.values())
    # Student categorical storage: five fp32 scalar buffers, one int64 action,
    # one uint8 done. No old649-class mean/std or teacher distillation buffers.
    scalar_bytes = 5 * 4 + 8 + 1
    return {"observation_fields": fields, "observation_bytes_per_transition": obs_bytes,
            "auxiliary_bytes_per_transition": scalar_bytes,
            "bytes_per_transition": obs_bytes + scalar_bytes,
            "observation_storage_bytes": obs_bytes * num_envs * rollout_steps,
            "total_storage_bytes": (obs_bytes + scalar_bytes) * num_envs * rollout_steps,
            "scope": "rollout storage only; excludes live env, optimizer, gradients, indices, probes and activations"}


def actual_storage_bytes(storage):
    obs = sum(value.numel() * value.element_size() for value in storage.observations.values())
    auxiliary = sum(value.numel() * value.element_size() for name in
                    ("rewards", "actions", "dones", "values", "actions_log_prob", "returns", "advantages", "mu", "sigma")
                    if isinstance(value := getattr(storage, name, None), torch.Tensor))
    return {"observation_storage_bytes": obs, "auxiliary_storage_bytes": auxiliary,
            "total_storage_bytes": obs + auxiliary}


def memory(device):
    if torch.device(device).type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None, "peak_allocated_bytes": None,
                "peak_reserved_bytes": None, "total_device_bytes": None}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "total_device_bytes": torch.cuda.get_device_properties(device).total_memory}


class BoundedProfile:
    """Profile an extra update's first ticks, full GAE, and first logical batch.

    Limiting the profiled window bounds the profiler's event memory. The extra
    update still collects T ticks and completes all PPO epochs/minibatches.
    """
    def __init__(self, output, device, ticks, minibatches):
        self.output, self.device = output, torch.device(device)
        self.ticks, self.minibatches = ticks, minibatches
        self.phases, self.current, self.current_phase = {}, None, None

    def start(self, phase):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.current = torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False)
        self.current_phase = phase
        self.current.start()

    def stop(self):
        if self.current is not None:
            synchronize(self.device)
            self.current.stop()
            # Export before starting another Kineto session. In torch 2.13,
            # deferred export loses earlier sessions' CUDA timestamps.
            profiler, phase = self.current, self.current_phase
            self.current = None
            self.current_phase = None
            rows = []
            for event in profiler.key_averages():
                rows.append({"operator": event.key, "calls": event.count,
                             "self_cpu_time_us": event.self_cpu_time_total,
                             "self_device_time_us": getattr(event, "self_device_time_total", 0.0),
                             "device_time_us": getattr(event, "device_time_total", 0.0)})
            rows.sort(key=lambda row: (row["self_device_time_us"], row["self_cpu_time_us"]), reverse=True)
            trace = self.output / f"profile_{phase}.json"
            profiler.export_chrome_trace(str(trace))
            self.phases[phase] = {"trace": str(trace), "top_operators": rows[:50],
                                  "operator_prefixes": sorted({row["operator"].split("::")[0] for row in rows})}

    def finish(self):
        self.stop()
        return {"timed_measurements_include_profile": False, "rollout_ticks_profiled": self.ticks,
                "logical_minibatches_profiled": self.minibatches, "full_gae_profiled": True,
                "phases": self.phases}


def complete_update(runner, obs, *, profile=None):
    device, alg, steps = runner.device, runner.alg, runner.cfg["num_steps_per_env"]
    synchronize(device)
    started = perf_counter()
    if profile:
        profile.start("collection")
    with torch.inference_mode():
        for step in range(steps):
            marked = profile is not None and step < profile.ticks
            with torch.profiler.record_function("Benchmark/learner_decision") if marked else nullcontext():
                actions = alg.act(obs)
            with torch.profiler.record_function("Benchmark/environment_and_opponent") if marked else nullcontext():
                obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
            obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
            with torch.profiler.record_function("Benchmark/store_and_timeout_bootstrap") if marked else nullcontext():
                alg.process_env_step(obs, rewards, dones, extras)
            if profile and step + 1 == profile.ticks:
                profile.stop()
        synchronize(device)
        collected = perf_counter()
        if profile:
            profile.start("gae")
        with torch.profiler.record_function("Benchmark/gae") if profile else nullcontext():
            alg.compute_returns(obs)
        synchronize(device)
        returned = perf_counter()
        if profile:
            profile.stop()
    original_hook = getattr(alg, "_on_ppo_minibatch", None)
    if profile:
        profile.start("update")
        observed = 0
        def observer(*args, **kwargs):
            nonlocal observed
            result = original_hook(*args, **kwargs)
            observed += 1
            if observed == profile.minibatches:
                profile.stop()
            return result
        alg._on_ppo_minibatch = observer
    try:
        losses = alg.update()
        synchronize(device)
        updated = perf_counter()
    finally:
        if profile:
            alg._on_ppo_minibatch = original_hook
            profile.stop()
    if not all(math.isfinite(float(value)) for value in losses.values()):
        raise FloatingPointError(f"nonfinite benchmark PPO losses: {losses}")
    return obs, {"collection_seconds": collected - started, "gae_seconds": returned - collected,
                 "update_seconds": updated - returned, "total_seconds": updated - started,
                 "losses": {key: float(value) for key, value in losses.items()}}


def aggregate_ranks(ranks, num_envs, steps, epochs, minibatches):
    if not ranks or not ranks[0]["iterations"]:
        raise ValueError("need nonempty rank measurements")
    count, world = len(ranks[0]["iterations"]), len(ranks)
    if any(len(rank["iterations"]) != count for rank in ranks):
        raise ValueError("ranks measured different numbers of complete updates")
    critical = [{stage: max(rank["iterations"][index][stage] for rank in ranks) for stage in STAGES}
                for index in range(count)]
    local_samples = num_envs * steps
    seconds = sum(row["total_seconds"] for row in critical)
    if seconds <= 0 or any(not math.isfinite(row[stage]) or row[stage] < 0 for row in critical for stage in STAGES):
        raise ValueError("invalid synchronized rank timings")
    global_samples = local_samples * world
    throughput = count * global_samples / seconds
    return {"num_envs_semantics": "per_rank", "num_envs_per_rank": num_envs,
            "global_num_envs": num_envs * world, "world_size": world, "rollout_steps": steps,
            "measured_updates": count, "fresh_transitions_per_update_per_rank": local_samples,
            "fresh_transitions_per_update_global": global_samples,
            "fresh_transitions_measured_global": count * global_samples,
            "ppo_epochs": epochs, "logical_minibatches_per_epoch_per_rank": minibatches,
            "optimizer_steps_per_update_per_rank": epochs * minibatches,
            "ppo_sample_visits_per_update_global": global_samples * epochs,
            "critical_path_seconds": seconds, "transitions_per_second": throughput,
            "median_iteration_transitions_per_second": statistics.median(global_samples / row["total_seconds"] for row in critical),
            "max_cumulative_rank_seconds": max(sum(row["total_seconds"] for row in rank["iterations"]) for rank in ranks),
            "stage_seconds_sum_of_iteration_rank_maxima": {stage: sum(row[stage] for row in critical) for stage in STAGES},
            "stage_medians_of_iteration_rank_maxima": {stage: statistics.median(row[stage] for row in critical) for stage in STAGES},
            "critical_path_iterations": critical,
            "timing_rule": "CUDA synchronized stage boundaries; fresh global NT divided by sum of per-update maximum full-rank elapsed times"}


def prepare_historical_benchmark(task, runner, config, output, distributed, *, fill=False):
    """Prepare one shared frozen pool entirely outside the measured updates."""
    session, error = None, None
    if distributed.is_distributed:
        runner.alg.broadcast_parameters()
    if distributed.is_main:
        try:
            session = prepare_self_play(task, runner, config, output)
            if fill:
                for _ in range(1, session.config.pool_size):
                    session.pool.publish(runner.alg.policy, iteration=-1,
                                         source={"kind": "benchmark_synthetic_initial_policy_copy",
                                                 "trained_snapshot": False})
                task.reset()
        except Exception as caught:
            error = caught
    ready = distributed.broadcast_control({"error": str(error) if error is not None else None})
    if ready["error"] is not None:
        if error is not None:
            raise error
        raise RuntimeError(f"rank-zero benchmark history preparation failed: {ready['error']}")
    if not distributed.is_main:
        try:
            settings = SelfPlayConfig(**config.get("self_play", {}))
            session = prepare_self_play(task, runner, config, output, read_only=True,
                                       opponent_seed=distributed.rank_seed(settings.seed))
            if fill:
                task.reset()
        except Exception as caught:
            error = caught
    if not distributed.all_true(error is None):
        if error is not None:
            raise error
        raise RuntimeError("benchmark historical reader preparation failed on another rank")
    return session


def resolve_config(args, context):
    overrides = json.loads(args.config.read_text())
    config = merge_config({"task": asdict(TaskConfig(opponent="current_self_play")),
                           "runner": runner_config(), "torch_threads": 1}, overrides)
    # Model constructors accept complete policy subtrees; do not carry unused
    # dense defaults into a different ResNet/ViT constructor.
    if "policy" in overrides.get("runner", {}):
        config["runner"]["policy"] = deepcopy(overrides["runner"]["policy"])
    task, run = config["task"], config["runner"]
    for name in ("global_num_envs", "num_envs", "rollout_steps", "logical_batch_size",
                 "physical_minibatch_size", "inference_chunk_size"):
        if (value := getattr(args, name)) is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if args.global_num_envs is not None:
        if args.global_num_envs % context["world_size"]:
            raise ValueError("global environment count must divide WORLD_SIZE")
        task["num_envs"] = args.global_num_envs // context["world_size"]
    elif args.num_envs is not None:
        task["num_envs"] = args.num_envs
    for key, value in (("backend", args.backend), ("device", args.device), ("opponent", args.opponent)):
        if value is not None:
            task[key] = value
    if args.backend is not None and args.device is None:
        task["device"] = f"cuda:{context['local_rank']}" if args.backend == "gpu" else "cpu"
    if context["world_size"] > 1:
        if task["backend"] == "gpu":
            task["device"] = f"cuda:{context['local_rank']}"
        elif task["backend"] != "cpu":
            raise ValueError("torchrun benchmark requires CPU or GPU environments")
        if task["opponent"] not in ("current_self_play", "self_play"):
            raise ValueError("distributed benchmark requires historical or current self-play")
    if args.historical_pool_fill and task["opponent"] != "self_play":
        raise ValueError("historical_pool_fill requires historical self_play")
    if args.rollout_steps is not None:
        run["num_steps_per_env"] = args.rollout_steps
    total = task["num_envs"] * run["num_steps_per_env"]
    if args.logical_batch_size is not None:
        if total % args.logical_batch_size:
            raise ValueError("per-rank NT must divide logical_batch_size exactly")
        run["algorithm"]["num_mini_batches"] = total // args.logical_batch_size
    if args.physical_minibatch_size is not None:
        run["algorithm"]["physical_minibatch_size"] = args.physical_minibatch_size
    if args.inference_chunk_size is not None:
        run["policy"]["inference_chunk_size"] = args.inference_chunk_size
    if args.amp_bfloat16 is not None:
        run["policy"]["amp_bfloat16"] = run["algorithm"]["amp_bfloat16"] = args.amp_bfloat16
    if (run["algorithm"].get("teacher_env_ratio") != 0
            or not run["algorithm"]["class_name"].endswith("StudentPPO")
            or run["algorithm"]["num_learning_epochs"] != 4):
        raise ValueError("benchmark requires student-only PPO with four complete epochs")
    if run["algorithm"]["num_mini_batches"] < 1 or total % run["algorithm"]["num_mini_batches"]:
        raise ValueError("NT must divide the number of PPO minibatches exactly")
    settings = task_config(task)
    if settings.reward.potential_gamma is not None and settings.reward.potential_gamma != run["algorithm"].get("gamma"):
        raise ValueError("reward potential_gamma and algorithm.gamma must match")
    config["evaluation"] = {"enabled": False}
    run["robogauge"] = {"enabled": False}
    return config


def benchmark(args):
    context = topology()
    config = resolve_config(args, context)
    if args.iterations < 2 or min(args.profile_ticks, args.profile_minibatches) < 1 or args.warmup_iterations < 0:
        raise ValueError("at least two measured updates, positive profile counts and nonnegative warmup required")
    if not 0 < args.max_storage_fraction < 1:
        raise ValueError("max_storage_fraction must be between zero and one")
    if args.minimum_tps is not None and (not math.isfinite(args.minimum_tps) or args.minimum_tps < 0):
        raise ValueError("minimum_tps must be finite and nonnegative")
    output = args.output.resolve()
    if context["rank"] == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rank_config = deepcopy(config)
    distributed = DistributedConfig(device=config["task"]["device"], **context)
    if config["task"]["opponent"] == "self_play":
        rank_config["task"]["seed"] = distributed.rank_seed(rank_config["task"]["seed"])
    else:
        rank_config["task"]["seed"] += context["rank"]
    settings = task_config(rank_config["task"])
    if settings.device.startswith("cuda"):
        torch.cuda.set_device(settings.device)
    seed_everything(settings.seed)
    torch.set_num_threads(config.get("torch_threads", 1))
    had_group = torch.distributed.is_initialized()
    profile_summary = None
    try:
        with _matmul_precision(config.get("matmul_precision"), config.get("cudnn_allow_tf32"), config.get("cudnn_benchmark")), \
                GeneralsTask(settings) as task:
            env = GeneralsCTSVecEnv(task)
            observations = env.get_observations()
            estimated = estimate_storage(observations, task.num_envs, config["runner"]["num_steps_per_env"],
                                         config["runner"].get("storage_obs_keys"))
            if task.device.type == "cuda":
                free, total_device = torch.cuda.mem_get_info(task.device)
                estimated.update(cuda_free_before_runner_bytes=free, total_device_bytes=total_device)
            if context["rank"] == 0:
                (output / "storage_estimate.json").write_text(json.dumps(
                    {**context, "num_envs_per_rank": task.num_envs,
                     "global_num_envs": task.num_envs * context["world_size"], **estimated}, indent=2) + "\n")
            if task.device.type == "cuda":
                if estimated["total_storage_bytes"] > free * args.max_storage_fraction:
                    raise MemoryError(f"rollout storage alone needs {estimated['total_storage_bytes']} bytes, "
                                      f"exceeding {args.max_storage_fraction:.0%} of currently free CUDA memory ({free})")
            with redirect_stdout(io.StringIO()) if context["rank"] else nullcontext():
                runner = BenchmarkRunner(env, deepcopy(config["runner"]), log_dir=None, device=str(task.device))
                historical = None
                if settings.opponent == "self_play":
                    historical = prepare_historical_benchmark(
                        task, runner, rank_config, output, distributed, fill=args.historical_pool_fill)
                elif context["rank"] == 0:
                    prepare_self_play(task, runner, rank_config, output)
                elif settings.opponent == "current_self_play":
                    selfplay = SelfPlayConfig(**config.get("self_play", {}))
                    task.opponent.attach_policy(runner.alg.policy, stochastic=selfplay.stochastic,
                                                seed=selfplay.seed + context["rank"])
            if runner.is_distributed and historical is None:
                runner.alg.broadcast_parameters()
            runner.train_mode()
            curriculum = RolloutCurriculum(CurriculumConfig(**config.get("curriculum", {})), runner.alg, task.reward_model)
            obs = runner.env.get_observations().to(runner.device)
            measured, warmup = [], []
            for iteration in range(args.warmup_iterations + args.iterations):
                curriculum.apply(iteration)
                if runner.is_distributed:
                    torch.distributed.barrier()
                if iteration == args.warmup_iterations:
                    setup_memory = memory(task.device)
                    if task.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(task.device)
                obs, row = complete_update(runner, obs)
                row["iteration"] = iteration
                (warmup if iteration < args.warmup_iterations else measured).append(row)
                runner.current_learning_iteration = iteration
            measured_memory = memory(task.device)
            if args.profile:
                iteration += 1
                curriculum.apply(iteration)
                profiler = (BoundedProfile(output, task.device, min(args.profile_ticks, runner.cfg["num_steps_per_env"]),
                                           min(args.profile_minibatches, runner.alg.num_mini_batches * runner.alg.num_learning_epochs))
                            if context["rank"] == 0 else None)
                obs, _ = complete_update(runner, obs, profile=profiler)
                if profiler:
                    profile_summary = profiler.finish()
            actor = getattr(runner.alg.policy, "actor", None)
            critic = getattr(runner.alg.policy, "critic", None)
            parameters = lambda module: sum(p.numel() for p in module.parameters()) if module is not None else None
            local = {**context, "device": str(task.device), "seed": settings.seed,
                     "device_name": torch.cuda.get_device_name(task.device) if task.device.type == "cuda" else "cpu",
                     "iterations": measured, "warmup_iterations": warmup,
                     "setup_and_warmup_memory": setup_memory, "measured_memory": measured_memory,
                     "including_profile_memory": memory(task.device), "estimated_storage": estimated,
                     "actual_storage": actual_storage_bytes(runner.alg.storage),
                     "parameter_counts": {"actor": parameters(actor), "critic": parameters(critic),
                                          "total": parameters(runner.alg.policy)},
                     "episodes": {"completed": task.completed_episodes, "terminated": task.terminated_episodes,
                                  "truncated": task.truncated_episodes}}
            if historical is not None:
                local["historical_pool"] = {
                    "snapshot_ids": list(historical.pool.ids), "snapshot_count": len(historical.pool),
                    "read_only": historical.pool.read_only, "opponent_seed": task.opponent.generator.initial_seed(),
                    "active_snapshot_counts": task.opponent.snapshot_counts(), "counts_scope": "per_rank"}
            if runner.is_distributed:
                ranks = [None] * context["world_size"]
                torch.distributed.all_gather_object(ranks, local)
            else:
                ranks = [local]
            if context["rank"]:
                return None
            algorithm = config["runner"]["algorithm"]
            performance = aggregate_ranks(ranks, task.num_envs, runner.cfg["num_steps_per_env"],
                                          algorithm["num_learning_epochs"], algorithm["num_mini_batches"])
            minimum = args.minimum_tps if args.minimum_tps is not None else (
                20000 if config["runner"]["policy"].get("model_size", "").lower() == "100m" else 50000)
            report = {"kind": "complete_student_ppo_benchmark", "config": config, "ranks": ranks,
                      "performance": performance, "profile": profile_summary,
                      "gate": {"minimum_transitions_per_second": minimum, "target_transitions_per_second": 200000,
                               "passed": performance["transitions_per_second"] >= minimum},
                      "runtime": {"torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                                  "benchmark_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()},
                      "protocol": {"fresh_initialization": True, "cuda_synchronized": task.device.type == "cuda",
                                   "timed_updates": args.iterations, "warmup_updates_excluded": args.warmup_iterations,
                                   "profile_updates_excluded": int(args.profile), "gae_included": True,
                                   "learner_action_nn_included": True,
                                   "opponent_action_nn_included": settings.opponent in ("current_self_play", "self_play"),
                                   "learner_and_opponent_action_nn_included": settings.opponent in ("current_self_play", "self_play"),
                                   "logging_checkpoint_evaluation_and_snapshot_io_in_timing": False,
                                   "selfplay_mode": settings.opponent,
                                   "snapshot_publication": "initial preparation only; periodic publications disabled",
                                   "gpu_utilization_telemetry": "external nvidia-smi CSV recommended; no telemetry subprocess overhead in timed stages"}}
            if historical is not None:
                report["protocol"]["historical_pool"] = {
                    "synthetic_pool_fill": args.historical_pool_fill, "snapshot_count": len(historical.pool),
                    "trained_snapshots": False,
                    "weights_source": "identical fresh initial learner weights in separate frozen models",
                    "initialization_and_pool_fill_excluded_from_timing": True,
                    "assignment": "uniform sampling at episode reset; each snapshot has an independent ID"}
            with (output / "report.json").open("x") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            return report
    finally:
        if not had_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing outputs are refused")
    parser.add_argument("--iterations", "--updates", type=int, default=12)
    parser.add_argument("--warmup-iterations", "--warmup", type=int, default=3)
    envs = parser.add_mutually_exclusive_group()
    envs.add_argument("--num-envs-per-rank", "--num-envs", dest="num_envs", type=int)
    envs.add_argument("--global-num-envs", type=int, help="explicit alternative total count, divisible by WORLD_SIZE")
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument("--device")
    parser.add_argument("--opponent", choices=("current_self_play", "self_play", "random", "wait"))
    parser.add_argument("--historical-pool-fill", action="store_true",
                        help="fill historical pool_size outside timing with separate frozen initial-weight copies; synthetic, not trained history")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--logical-batch-size", type=int, help="per-rank optimizer batch; sets num_mini_batches=NT/B")
    parser.add_argument("--physical-minibatch-size", type=int)
    parser.add_argument("--inference-chunk-size", type=int)
    parser.add_argument("--amp-bfloat16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-storage-fraction", type=float, default=0.85)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-ticks", type=int, default=1)
    parser.add_argument("--profile-minibatches", type=int, default=1)
    parser.add_argument("--minimum-tps", type=float)
    parser.add_argument("--require-throughput", action="store_true")
    args = parser.parse_args(argv)
    report = benchmark(args)
    if report is not None:
        print(json.dumps({"output": str(args.output.resolve()), "performance": report["performance"], "gate": report["gate"]}))
        return int(args.require_throughput and not report["gate"]["passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
