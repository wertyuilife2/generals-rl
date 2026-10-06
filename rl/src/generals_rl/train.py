"""Train the vendored MoE CTS baseline with historical self-play opponents."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import resource
from time import perf_counter

import torch
from rsl_rl.runners import OnPolicyRunnerCTS

from .config import merge_config, seed_everything, task_config
from .cts import GeneralsCTSVecEnv, runner_config
from .self_play_training import SelfPlayConfig, prepare_self_play
from .task import GeneralsTask, TaskConfig
from .training_evaluation import PeriodicEvaluator, TrainingEvalConfig
from .training_control import TrainingControlConfig, TrainingController, TrainingStopped, prune_checkpoints


@contextmanager
def _matmul_precision(precision):
    """Change this process's precision only for an explicitly configured run."""
    if precision is None:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def train(config, output_dir, *, iterations=None, steps_per_env=None, resume=None, warm_start=None):
    started = perf_counter()
    if "baseline_league" in config:
        raise ValueError("baseline_league training has been removed; use pure historical self-play. "
                         "Replay legacy v6-v8 experiments with their frozen source.")
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")
    precision = config.get("matmul_precision")
    if precision is not None and precision not in ("highest", "high", "medium"):
        raise ValueError("matmul_precision must be highest, high, medium or null")
    if (iterations is None) == (steps_per_env is None):
        raise ValueError("specify exactly one of iterations or steps_per_env")
    rollout_steps = config["runner"]["num_steps_per_env"]
    settings = task_config(config["task"])
    potential_gamma = settings.reward.potential_gamma
    if potential_gamma is not None:
        algorithm_gamma = config["runner"]["algorithm"].get("gamma")
        if isinstance(algorithm_gamma, bool) or algorithm_gamma != potential_gamma:
            raise ValueError("reward potential_gamma requires an explicit matching runner.algorithm.gamma")
    value_offset = config["runner"]["policy"].get("value_potential_offset", False)
    if not isinstance(value_offset, bool):
        raise TypeError("value_potential_offset must be boolean")
    if value_offset:
        from rsl_rl.utils import resolve_callable
        from .student_ppo import StudentPPOPolicy
        policy_type = resolve_callable(config["runner"]["policy"]["class_name"])
        if (not settings.critic_potential or potential_gamma is None
                or not isinstance(policy_type, type) or not issubclass(policy_type, StudentPPOPolicy)):
            raise ValueError("value_potential_offset requires StudentPPOPolicy, task.critic_potential and potential reward")
    evaluation_settings = TrainingEvalConfig(**config.get("evaluation", {}))
    control_settings = TrainingControlConfig(**config.get("training_control", {}))
    controller = TrainingController(control_settings)
    if settings.opponent == "self_play":
        SelfPlayConfig(**config.get("self_play", {}))
        if evaluation_settings.enabled and settings.players != 2:
            raise ValueError("periodic built-in evaluation requires players=2; use --no-eval for multiplayer training")
    if rollout_steps < 1:
        raise ValueError("rollout steps must be positive")
    if steps_per_env is not None:
        if steps_per_env < 1:
            raise ValueError("steps_per_env must be positive")
        iterations = math.ceil(steps_per_env / rollout_steps)
    elif iterations < 1:
        raise ValueError("iterations must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("this baseline supports single-device training; student gradient synchronization is not implemented upstream")
    seed_everything(settings.seed)
    torch.set_num_threads(config.get("torch_threads", 1))
    output = Path(output_dir).resolve()
    if (output / "checkpoint.pt").exists() or (output / "run_config.json").exists():
        raise FileExistsError(f"training output already exists: {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    # Save before the upstream runner mutates configuration dictionaries.
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    sampled_ticks = 0
    completed_updates = 0
    collection_seconds = update_seconds = 0.0
    last_losses = None
    stop_reason = None
    warm_source = None
    with _matmul_precision(precision), GeneralsTask(settings) as task:
        runner = OnPolicyRunnerCTS(GeneralsCTSVecEnv(task), deepcopy(config["runner"]),
                                   str(output), str(task.device))
        if resume:
            runner.load(str(Path(resume).resolve()), map_location=str(task.device))
            # Restore the adaptive scheduler scalar and continue after the saved index.
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            runner.current_learning_iteration += 1
        elif warm_start is not None:
            checkpoint = Path(warm_start).resolve()
            saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
            # Preserve fresh optimizer states, configured LRs and iteration zero.
            runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
            del saved
            warm_source = {"kind": "warm_start_weights", "path": str(checkpoint),
                           "sha256": sha256(checkpoint.read_bytes()).hexdigest()}
        self_play = prepare_self_play(task, runner, config, output, resume=resume,
                                      initial_iteration=runner.current_learning_iteration - 1,
                                      initial_source=warm_source)
        evaluator = (PeriodicEvaluator(evaluation_settings, config, output, runner.logger.writer,
                                       on_complete=controller.on_evaluation)
                     if self_play is not None and evaluation_settings.enabled else None)
        original_save = runner.save
        final_iteration = runner.current_learning_iteration + iterations - 1

        def save_checkpoint(path, it, last_model, infos=None):
            # Route upstream's final save to one canonical checkpoint. A periodic
            # save on that same iteration would otherwise duplicate its weights.
            if it == final_iteration and not last_model:
                return
            if last_model:
                path = str(output / "checkpoint.pt")
            result = original_save(path, it=it, last_model=last_model, infos=infos)
            prune_checkpoints(output, control_settings.keep_checkpoints)
            return result

        runner.save = save_checkpoint
        if steps_per_env is not None and steps_per_env < rollout_steps:
            runner.cfg["num_steps_per_env"] = steps_per_env
            runner.alg.storage.num_transitions_per_env = steps_per_env
        if task.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(task.device)
        original_log = runner.logger.log

        def on_update(**kwargs):
            nonlocal sampled_ticks, completed_updates, collection_seconds, update_seconds, last_losses, stop_reason
            sampled_ticks += runner.cfg["num_steps_per_env"]
            completed_updates += 1
            collection_seconds += kwargs["collect_time"]
            update_seconds += kwargs["learn_time"]
            last_losses = kwargs["loss_dict"]
            if not all(math.isfinite(value) for value in last_losses.values()):
                raise FloatingPointError(f"non-finite training loss: {last_losses}")
            update_metrics = getattr(runner.alg, "update_metrics", {})
            if not all(math.isfinite(value) for value in update_metrics.values()):
                raise FloatingPointError(f"non-finite PPO update metric: {update_metrics}")
            snapshot = self_play.after_update(kwargs["it"]) if self_play is not None else None
            original_log(**kwargs)
            if evaluator is not None:
                evaluator.poll()
                if snapshot is not None:
                    evaluator.on_snapshot(
                        snapshot, self_play.pool.directory,
                        publication_index=(kwargs["it"] + 1) // self_play.config.snapshot_interval,
                        steps_per_environment=sampled_ticks,
                        environment_steps=sampled_ticks * task.num_envs,
                    )
            elapsed = perf_counter() - started
            if runner.logger.writer is not None:
                writer = runner.logger.writer
                for name, value in update_metrics.items():
                    writer.add_scalar(f"PPO/{name}", value, kwargs["it"])
                writer.add_scalar("Perf/environment_steps", sampled_ticks * task.num_envs, kwargs["it"])
                writer.add_scalar("Perf/wall_seconds", elapsed, kwargs["it"])
                writer.add_scalar("Perf/wall_fps", sampled_ticks * task.num_envs / elapsed, kwargs["it"])
                writer.add_scalar("Resource/peak_process_rss_bytes", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, kwargs["it"])
                if task.device.type == "cuda":
                    writer.add_scalar("Resource/gpu_allocated_bytes", torch.cuda.memory_allocated(task.device), kwargs["it"])
                    writer.add_scalar("Resource/gpu_reserved_bytes", torch.cuda.memory_reserved(task.device), kwargs["it"])
                if evaluator is not None:
                    writer.add_scalar("Eval/queued_jobs", len(evaluator.pending), kwargs["it"])
            stop_reason = ("requested_stop" if (output / "STOP").is_file() else
                           controller.stop_reason(elapsed, evaluation_healthy=(evaluator is not None and not evaluator.describe()["failed"])))
            if stop_reason is not None:
                raise TrainingStopped(stop_reason)
            if steps_per_env is not None:
                remaining = steps_per_env - sampled_ticks
                if 0 < remaining < runner.cfg["num_steps_per_env"]:
                    # Storage retains its allocation; only this prefix is active.
                    runner.cfg["num_steps_per_env"] = remaining
                    runner.alg.storage.num_transitions_per_env = remaining

        runner.logger.log = on_update
        try:
            try:
                runner.learn(iterations)
            except TrainingStopped:
                runner.save(str(output / "checkpoint.pt"), it=runner.current_learning_iteration, last_model=True)
            if steps_per_env is not None and stop_reason is None and sampled_ticks != steps_per_env:
                raise RuntimeError(f"sampled {sampled_ticks} ticks instead of {steps_per_env}")
            if not all(bool(torch.isfinite(p).all()) for p in runner.alg.policy.parameters()):
                raise FloatingPointError("non-finite policy parameters")
            if self_play is not None:
                self_play.publish(runner.current_learning_iteration, source={"kind": "final_learner"})
            if evaluator is not None:
                evaluator.finish()
            transitions = sampled_ticks * task.num_envs
            wall_seconds = perf_counter() - started
            performance = {
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "environment_transitions_per_second": transitions / (collection_seconds + update_seconds),
                "wall_seconds": wall_seconds,
                "wall_environment_transitions_per_second": transitions / wall_seconds,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(task.device) if task.device.type == "cuda" else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(task.device) if task.device.type == "cuda" else None,
            }
            summary = {
                "iterations": completed_updates, "steps_per_environment": sampled_ticks,
                "environment_steps": transitions, "performance": performance,
                "completed_episodes": task.completed_episodes,
                "terminated_episodes": task.terminated_episodes,
                "truncated_episodes": task.truncated_episodes,
                "teacher_envs": runner.alg.teacher_num_envs,
                "student_envs": runner.alg.student_num_envs,
                "last_losses": last_losses, "resume": resume,
                "stop_reason": stop_reason or "budget",
                "evaluation": evaluator.describe() if evaluator is not None else None,
            }
            if warm_source is not None:
                summary["warm_start"] = warm_source
            if self_play is not None:
                summary["self_play"] = {
                    "pool_ids": list(self_play.pool.ids), "new_publications": self_play.publications,
                    "episode_assignments": task.opponent.episode_assignments,
                    "active_snapshot_counts": {str(k): v for k, v in task.opponent.snapshot_counts().items()},
                }
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(json.dumps(summary, indent=2))
            return summary
        finally:
            if evaluator is not None:
                evaluator.close()
            if runner.logger.writer is not None:
                runner.logger.writer.flush()
                runner.logger.writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for task, runner, self_play and evaluation")
    parser.add_argument("--replace-policy", action="store_true",
                        help="replace runner.policy with the complete JSON subtree; preserve other recursive overrides")
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument("--device")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--opponent", choices=("self_play", "random", "wait"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-eval", action="store_true", help="disable periodic evaluation (enabled by default)")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--iterations", type=int)
    budget.add_argument("--steps-per-env", type=int, help="exact vector ticks per lane")
    parser.add_argument("--output", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="restore policy, both optimizers and the available historical pool")
    initialization.add_argument("--warm-start", type=Path, help="load model weights into a fresh run with new optimizers and pool")
    args = parser.parse_args(argv)
    if args.resume:
        config = json.loads(args.resume.with_name("run_config.json").read_text())
    else:
        config = {"task": asdict(TaskConfig(opponent="self_play")), "runner": runner_config(),
                  "self_play": asdict(SelfPlayConfig()), "evaluation": asdict(TrainingEvalConfig()),
                  "torch_threads": 1}
    if args.config:
        overrides = json.loads(args.config.read_text())
        config = merge_config(config, overrides)
        if args.replace_policy:
            policy = overrides.get("runner", {}).get("policy")
            if not isinstance(policy, dict) or "class_name" not in policy:
                parser.error("--replace-policy requires --config with a complete runner.policy and class_name")
            config["runner"]["policy"] = deepcopy(policy)
    elif args.replace_policy:
        parser.error("--replace-policy requires --config")
    for field in ("num_envs", "opponent", "seed", "backend", "device"):
        value = getattr(args, field)
        if value is not None:
            config["task"][field] = value
    if args.backend == "gpu" and args.device is None:
        config["task"]["device"] = "cuda:0"
    if args.no_eval:
        config.setdefault("evaluation", {})["enabled"] = False
    iterations = None if args.steps_per_env is not None else (args.iterations if args.iterations is not None else 1000)
    destination = args.output or Path("outputs/moe_cts") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    train(config, destination, iterations=iterations, steps_per_env=args.steps_per_env,
          resume=str(args.resume) if args.resume else None,
          warm_start=str(args.warm_start) if args.warm_start else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
