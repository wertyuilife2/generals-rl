"""Train the vendored MoE CTS baseline with historical self-play opponents."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch
from rsl_rl.runners import OnPolicyRunnerCTS

from .config import merge_config, seed_everything, task_config
from .cts import GeneralsCTSVecEnv, runner_config
from .self_play_training import SelfPlayConfig, prepare_self_play
from .task import GeneralsTask, TaskConfig
from .training_evaluation import PeriodicEvaluator, TrainingEvalConfig


def train(config, output_dir, *, iterations=None, steps_per_env=None, resume=None):
    started = perf_counter()
    if (iterations is None) == (steps_per_env is None):
        raise ValueError("specify exactly one of iterations or steps_per_env")
    rollout_steps = config["runner"]["num_steps_per_env"]
    settings = task_config(config["task"])
    evaluation_settings = TrainingEvalConfig(**config.get("evaluation", {}))
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
    with GeneralsTask(settings) as task:
        runner = OnPolicyRunnerCTS(GeneralsCTSVecEnv(task), deepcopy(config["runner"]),
                                   str(output), str(task.device))
        if resume:
            runner.load(str(Path(resume).resolve()), map_location=str(task.device))
            # Restore the adaptive scheduler scalar and continue after the saved index.
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            runner.current_learning_iteration += 1
        self_play = prepare_self_play(task, runner, config, output, resume=resume,
                                      initial_iteration=runner.current_learning_iteration - 1)
        evaluator = (PeriodicEvaluator(evaluation_settings, config, output, runner.logger.writer)
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
            return original_save(path, it=it, last_model=last_model, infos=infos)

        runner.save = save_checkpoint
        if steps_per_env is not None and steps_per_env < rollout_steps:
            runner.cfg["num_steps_per_env"] = steps_per_env
            runner.alg.storage.num_transitions_per_env = steps_per_env
        if task.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(task.device)
        original_log = runner.logger.log

        def on_update(**kwargs):
            nonlocal sampled_ticks, completed_updates, collection_seconds, update_seconds, last_losses
            sampled_ticks += runner.cfg["num_steps_per_env"]
            completed_updates += 1
            collection_seconds += kwargs["collect_time"]
            update_seconds += kwargs["learn_time"]
            last_losses = kwargs["loss_dict"]
            if not all(math.isfinite(value) for value in last_losses.values()):
                raise FloatingPointError(f"non-finite training loss: {last_losses}")
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
            if steps_per_env is not None:
                remaining = steps_per_env - sampled_ticks
                if 0 < remaining < runner.cfg["num_steps_per_env"]:
                    # Storage retains its allocation; only this prefix is active.
                    runner.cfg["num_steps_per_env"] = remaining
                    runner.alg.storage.num_transitions_per_env = remaining

        runner.logger.log = on_update
        try:
            runner.learn(iterations)
            if steps_per_env is not None and sampled_ticks != steps_per_env:
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
                "evaluation": evaluator.describe() if evaluator is not None else None,
            }
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
    parser.add_argument("--resume", type=Path, help="restore policy, both optimizers and the available historical pool")
    args = parser.parse_args(argv)
    if args.resume:
        config = json.loads(args.resume.with_name("run_config.json").read_text())
    else:
        config = {"task": asdict(TaskConfig(opponent="self_play")), "runner": runner_config(),
                  "self_play": asdict(SelfPlayConfig()), "evaluation": asdict(TrainingEvalConfig()),
                  "torch_threads": 1}
    if args.config:
        config = merge_config(config, json.loads(args.config.read_text()))
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
          resume=str(args.resume) if args.resume else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
