"""Train local-information PPO with historical or current-policy self-play."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import torch
import torch.distributed as dist
from rsl_rl.runners import OnPolicyRunnerCTS

from .config import merge_config, seed_everything, task_config
from .cts import GeneralsCTSVecEnv, runner_config
from .curriculum import CurriculumConfig, RolloutCurriculum
from .distributed_training import DistributedConfig, distributed_process_group
from .self_play_training import SelfPlayConfig, prepare_self_play
from .task import GeneralsTask, TaskConfig
from .training_evaluation import PeriodicEvaluator, TrainingEvalConfig
from .training_control import TrainingControlConfig, TrainingController, TrainingStopped, prune_checkpoints


@contextmanager
def _matmul_precision(precision, cudnn_allow_tf32=None, cudnn_benchmark=None):
    """Change this process's precision only for an explicitly configured run."""
    if precision is None and cudnn_allow_tf32 is None and cudnn_benchmark is None:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        if precision is not None:
            torch.set_float32_matmul_precision(precision)
        if cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = cudnn_benchmark
        yield
    finally:
        if cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = previous_cudnn
        if precision is not None:
            torch.set_float32_matmul_precision(previous)
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = previous_benchmark


def train(config, output_dir, *, iterations=None, steps_per_env=None, resume=None, warm_start=None):
    """Use equal torchrun environment shards; only rank zero writes artifacts."""
    distributed = DistributedConfig.from_env(config.get("task", {}).get("device", "cpu"))
    with distributed_process_group(distributed):
        return _train(config, output_dir, iterations=iterations, steps_per_env=steps_per_env,
                      resume=resume, warm_start=warm_start, distributed=distributed)


def _publish_shared_history(distributed, session, publish):
    """Finish rank zero's disk publication before readers refresh candidates."""
    snapshot, error = None, None
    if distributed.is_main:
        try:
            snapshot = publish()
        except Exception as caught:
            error = caught
    control = distributed.broadcast_control({
        "error": str(error) if error is not None else None,
        "snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
        "publications": session.publications,
        "last_published_iteration": session.last_published_iteration,
    })
    if control["error"] is not None:
        if error is not None:
            raise error
        raise RuntimeError(f"rank-zero historical publication failed: {control['error']}")
    refresh_error = None
    if not distributed.is_main:
        try:
            if control["snapshot_id"] is not None:
                session.pool.refresh()
                session.pool.get(control["snapshot_id"])
            session.publications = control["publications"]
            session.last_published_iteration = control["last_published_iteration"]
        except Exception as caught:
            refresh_error = caught
    if control["snapshot_id"] is not None and not distributed.all_true(refresh_error is None):
        if refresh_error is not None:
            raise refresh_error
        raise RuntimeError("historical pool refresh failed on another rank")
    return snapshot


def _train(config, output_dir, *, iterations, steps_per_env, resume, warm_start, distributed):
    started = perf_counter()
    if "baseline_league" in config:
        raise ValueError("baseline_league training has been removed; use pure historical self-play. "
                         "Replay legacy v6-v8 experiments with their frozen source.")
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")
    precision = config.get("matmul_precision")
    if precision is not None and precision not in ("highest", "high", "medium"):
        raise ValueError("matmul_precision must be highest, high, medium or null")
    cudnn_allow_tf32 = config.get("cudnn_allow_tf32")
    if cudnn_allow_tf32 is not None and not isinstance(cudnn_allow_tf32, bool):
        raise TypeError("cudnn_allow_tf32 must be boolean or null")
    cudnn_benchmark = config.get("cudnn_benchmark")
    if cudnn_benchmark is not None and not isinstance(cudnn_benchmark, bool):
        raise TypeError("cudnn_benchmark must be boolean or null")
    if (iterations is None) == (steps_per_env is None):
        raise ValueError("specify exactly one of iterations or steps_per_env")
    rollout_steps = config["runner"]["num_steps_per_env"]
    settings = task_config(config["task"])
    curriculum_settings = CurriculumConfig(**config.get("curriculum", {}))
    potential_gamma = settings.reward.potential_gamma
    if potential_gamma is not None:
        algorithm_gamma = config["runner"]["algorithm"].get("gamma")
        if isinstance(algorithm_gamma, bool) or algorithm_gamma != potential_gamma:
            raise ValueError("reward potential_gamma requires an explicit matching runner.algorithm.gamma")
    if config["runner"]["policy"].get("value_potential_offset", False) is not False:
        raise ValueError("analytic value_potential_offset is retired")
    evaluation_settings = TrainingEvalConfig(**config.get("evaluation", {}))
    evaluation_settings.validate_training_devices(distributed.device)
    control_settings = TrainingControlConfig(**config.get("training_control", {}))
    controller = TrainingController(control_settings)
    if settings.opponent in ("self_play", "current_self_play"):
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
    if distributed.is_distributed:
        if settings.opponent not in ("self_play", "current_self_play"):
            raise ValueError("distributed training requires historical or current self-play")
        if (config["runner"]["algorithm"]["class_name"] != "generals_rl.student_ppo.StudentPPO"
                or config["runner"]["algorithm"].get("teacher_env_ratio", 0) != 0):
            raise ValueError("multi-GPU training requires student-only StudentPPO")
        settings = replace(settings, device=str(distributed.device), seed=distributed.rank_seed(settings.seed))
    seed_everything(settings.seed)
    torch.set_num_threads(config.get("torch_threads", 1))
    output = Path(output_dir).resolve()
    setup_error = None
    if distributed.is_main:
        try:
            if (output / "checkpoint.pt").exists() or (output / "run_config.json").exists():
                raise FileExistsError(f"training output already exists: {output}; choose a new output directory")
            output.mkdir(parents=True, exist_ok=True)
            # Save before the upstream runner mutates configuration dictionaries.
            (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        except Exception as error:
            setup_error = error
    setup = distributed.broadcast_control({"error": str(setup_error) if setup_error is not None else None})
    if setup["error"] is not None:
        if setup_error is not None:
            raise setup_error
        raise RuntimeError(f"rank-zero output setup failed: {setup['error']}")
    sampled_ticks = 0
    completed_updates = 0
    collection_seconds = update_seconds = 0.0
    last_losses = None
    stop_reason = None
    warm_source = None
    with _matmul_precision(precision, cudnn_allow_tf32, cudnn_benchmark), GeneralsTask(settings) as task:
        runner = OnPolicyRunnerCTS(GeneralsCTSVecEnv(task), deepcopy(config["runner"]),
                                   str(output) if distributed.is_main else None, str(task.device))
        if resume:
            runner.load(str(Path(resume).resolve()), map_location=str(task.device))
            # Restore the adaptive scheduler scalar and continue after the saved index.
            if hasattr(runner.alg, "restore_learning_rates"):
                runner.alg.restore_learning_rates()
            else:
                runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            runner.current_learning_iteration += 1
        elif warm_start is not None:
            checkpoint = Path(warm_start).resolve()
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            # Preserve fresh optimizer states, configured LRs and iteration zero.
            runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
            del saved
            warm_source = {"kind": "warm_start_weights", "path": str(checkpoint),
                           "sha256": sha256(checkpoint.read_bytes()).hexdigest()}
        curriculum = RolloutCurriculum(curriculum_settings, runner.alg, task.reward_model)
        curriculum_metrics = curriculum.apply(runner.current_learning_iteration)
        shared_history = distributed.is_distributed and settings.opponent == "self_play"
        if shared_history:
            # The runner broadcasts in learn(), after pool preparation. Freeze
            # the same initial learner on every rank before publishing its seed.
            runner.alg.broadcast_parameters()
            self_play, history_error = None, None
            opponent_settings = SelfPlayConfig(**config.get("self_play", {}))
            if distributed.is_main:
                try:
                    self_play = prepare_self_play(
                        task, runner, config, output, resume=resume,
                        initial_iteration=runner.current_learning_iteration - 1, initial_source=warm_source)
                except Exception as error:
                    history_error = error
            ready = distributed.broadcast_control({
                "error": str(history_error) if history_error is not None else None,
                "publications": self_play.publications if self_play is not None else None,
                "last_published_iteration": self_play.last_published_iteration if self_play is not None else None,
            })
            if ready["error"] is not None:
                if history_error is not None:
                    raise history_error
                raise RuntimeError(f"rank-zero historical preparation failed: {ready['error']}")
            reader_error = None
            if not distributed.is_main:
                try:
                    self_play = prepare_self_play(
                        task, runner, config, output, read_only=True,
                        opponent_seed=distributed.rank_seed(opponent_settings.seed))
                    self_play.publications = ready["publications"]
                    self_play.last_published_iteration = ready["last_published_iteration"]
                except Exception as error:
                    reader_error = error
            if not distributed.all_true(reader_error is None):
                if reader_error is not None:
                    raise reader_error
                raise RuntimeError("historical pool preparation failed on another rank")
        else:
            self_play = (prepare_self_play(task, runner, config, output, resume=resume,
                                           initial_iteration=runner.current_learning_iteration - 1,
                                           initial_source=warm_source) if distributed.is_main else None)
            if not distributed.is_main:
                opponent_settings = SelfPlayConfig(**config.get("self_play", {}))
                task.opponent.attach_policy(runner.alg.policy, stochastic=opponent_settings.stochastic,
                                            seed=distributed.rank_seed(opponent_settings.seed))
        evaluator = (PeriodicEvaluator(evaluation_settings, config, output, runner.logger.writer,
                                       on_complete=controller.on_evaluation)
                     if distributed.is_main and self_play is not None and evaluation_settings.enabled else None)
        original_save = runner.save
        final_iteration = runner.current_learning_iteration + iterations - 1

        def save_checkpoint(path, it, last_model, infos=None):
            if not distributed.is_main:
                return
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
            nonlocal curriculum_metrics
            sampled_ticks += runner.cfg["num_steps_per_env"]
            completed_updates += 1
            if task.device.type == "cuda" and distributed.is_distributed:
                torch.cuda.synchronize(task.device)
            kwargs["collect_time"], kwargs["learn_time"] = distributed.max_timings(
                kwargs["collect_time"], kwargs["learn_time"])
            collection_seconds += kwargs["collect_time"]
            update_seconds += kwargs["learn_time"]
            last_losses = kwargs["loss_dict"]
            if not distributed.all_true(all(math.isfinite(value) for value in last_losses.values())):
                raise FloatingPointError(f"non-finite training loss: {last_losses}")
            update_metrics = getattr(runner.alg, "update_metrics", {})
            if not distributed.all_true(all(math.isfinite(value) for value in update_metrics.values())):
                raise FloatingPointError(f"non-finite PPO update metric: {update_metrics}")
            if shared_history:
                snapshot = _publish_shared_history(
                    distributed, self_play, lambda: self_play.after_update(kwargs["it"]))
            else:
                snapshot = self_play.after_update(kwargs["it"]) if self_play is not None else None
            original_log(**kwargs)
            if evaluator is not None:
                evaluator.poll()
                if snapshot is not None:
                    evaluation_budget = {}
                    if evaluation_settings.synchronous and control_settings.max_seconds is not None:
                        evaluation_budget["remaining_seconds"] = max(
                            0.0, control_settings.max_seconds - (perf_counter() - started))
                    evaluator.on_snapshot(
                        snapshot, self_play.pool.directory,
                        publication_index=(kwargs["it"] + 1) // self_play.config.snapshot_interval,
                        steps_per_environment=sampled_ticks,
                        environment_steps=distributed.global_transitions(task.num_envs, sampled_ticks),
                        **evaluation_budget,
                    )
            elapsed = perf_counter() - started
            if runner.logger.writer is not None:
                writer = runner.logger.writer
                for name, value in update_metrics.items():
                    writer.add_scalar(f"PPO/{name}", value, kwargs["it"])
                for name, value in curriculum_metrics.items():
                    writer.add_scalar(f"Curriculum/{name}", value, kwargs["it"])
                global_transitions = distributed.global_transitions(task.num_envs, sampled_ticks)
                writer.add_scalar("Perf/environment_steps", global_transitions, kwargs["it"])
                writer.add_scalar("Perf/wall_seconds", elapsed, kwargs["it"])
                writer.add_scalar("Perf/wall_fps", global_transitions / elapsed, kwargs["it"])
                writer.add_scalar("Resource/peak_process_rss_bytes", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, kwargs["it"])
                if task.device.type == "cuda":
                    writer.add_scalar("Resource/gpu_allocated_bytes", torch.cuda.memory_allocated(task.device), kwargs["it"])
                    writer.add_scalar("Resource/gpu_reserved_bytes", torch.cuda.memory_reserved(task.device), kwargs["it"])
                if evaluator is not None:
                    writer.add_scalar("Eval/queued_jobs", len(evaluator.pending), kwargs["it"])
            control = None
            if distributed.is_main:
                stop_reason = ("requested_stop" if (output / "STOP").is_file() else
                               controller.stop_reason(
                                   elapsed, completed_updates=kwargs["it"] + 1,
                                   evaluation_healthy=(evaluator is not None and not evaluator.describe()["failed"])))
                if (stop_reason is None and evaluator is not None
                        and evaluator.budget_exhausted is True
                        and elapsed >= control_settings.min_seconds):
                    stop_reason = "budget_reserve"
                control = {"stop_reason": stop_reason}
            stop_reason = distributed.broadcast_control(control)["stop_reason"]
            if stop_reason is not None:
                raise TrainingStopped(stop_reason)
            if steps_per_env is not None:
                remaining = steps_per_env - sampled_ticks
                if 0 < remaining < runner.cfg["num_steps_per_env"]:
                    # Storage retains its allocation; only this prefix is active.
                    runner.cfg["num_steps_per_env"] = remaining
                    runner.alg.storage.num_transitions_per_env = remaining
            if kwargs["it"] < final_iteration:
                # This rollout's returns/update already used one fixed setting.
                # Advance only before collecting the next rollout.
                curriculum_metrics = curriculum.apply(kwargs["it"] + 1)

        runner.logger.log = on_update
        try:
            try:
                runner.learn(iterations)
            except TrainingStopped:
                runner.save(str(output / "checkpoint.pt"), it=runner.current_learning_iteration, last_model=True)
            if steps_per_env is not None and stop_reason is None and sampled_ticks != steps_per_env:
                raise RuntimeError(f"sampled {sampled_ticks} ticks instead of {steps_per_env}")
            if not distributed.all_true(all(bool(torch.isfinite(p).all()) for p in runner.alg.policy.parameters())):
                raise FloatingPointError("non-finite policy parameters")
            if shared_history:
                _publish_shared_history(distributed, self_play, lambda: self_play.publish(
                    runner.current_learning_iteration, source={"kind": "final_learner"}))
            elif self_play is not None:
                self_play.publish(runner.current_learning_iteration, source={"kind": "final_learner"})
            if evaluator is not None:
                evaluator.finish()
            if evaluation_settings.enabled and not evaluation_settings.synchronous:
                distributed.wait_on_cpu()
            transitions = distributed.global_transitions(task.num_envs, sampled_ticks)
            wall_seconds, = distributed.max_timings(perf_counter() - started)
            completed_episodes, terminated_episodes, truncated_episodes = distributed.sum_counts(
                task.completed_episodes, task.terminated_episodes, task.truncated_episodes)
            peak_allocated, peak_reserved = distributed.max_timings(
                torch.cuda.max_memory_allocated(task.device) if task.device.type == "cuda" else 0,
                torch.cuda.max_memory_reserved(task.device) if task.device.type == "cuda" else 0)
            performance = {
                "collection_seconds": collection_seconds,
                "update_seconds": update_seconds,
                "environment_transitions_per_second": transitions / (collection_seconds + update_seconds),
                "wall_seconds": wall_seconds,
                "wall_environment_transitions_per_second": transitions / wall_seconds,
                "gpu_peak_allocated_bytes": int(peak_allocated) if task.device.type == "cuda" else None,
                "gpu_peak_reserved_bytes": int(peak_reserved) if task.device.type == "cuda" else None,
            }
            summary = {
                "iterations": completed_updates, "steps_per_environment": sampled_ticks,
                "environment_steps": transitions, "performance": performance,
                "completed_episodes": completed_episodes,
                "terminated_episodes": terminated_episodes,
                "truncated_episodes": truncated_episodes,
                "teacher_envs": runner.alg.teacher_num_envs * distributed.world_size,
                "student_envs": runner.alg.student_num_envs * distributed.world_size,
                "last_losses": last_losses, "resume": resume,
                "stop_reason": stop_reason or "budget",
                "evaluation": evaluator.describe() if evaluator is not None else None,
            }
            if warm_source is not None:
                summary["warm_start"] = warm_source
            if self_play is not None:
                summary["self_play"] = {
                    "pool_ids": list(self_play.pool.ids), "new_publications": self_play.publications,
                }
                if settings.opponent == "self_play":
                    assignments = task.opponent.episode_assignments
                    counts = {str(k): v for k, v in task.opponent.snapshot_counts().items()}
                    if shared_history:
                        ranks = [None] * distributed.world_size
                        dist.all_gather_object(ranks, {"assignments": assignments, "counts": counts})
                        assignments, counts = 0, {}
                        for rank in ranks:
                            assignments += rank["assignments"]
                            for key, value in rank["counts"].items():
                                counts[key] = counts.get(key, 0) + value
                        summary["self_play"]["counts_scope"] = "global"
                    summary["self_play"].update(
                        episode_assignments=assignments,
                        active_snapshot_counts=counts,
                    )
                else:
                    summary["self_play"].update(mode="current_self_play", training_opponent="live_learner",
                                                snapshots_used_for_training=False)
            if distributed.is_distributed:
                summary["distributed"] = {"world_size": distributed.world_size,
                                          "num_envs_per_rank": task.num_envs,
                                          "global_num_envs": task.num_envs * distributed.world_size,
                                          "timing": "sum of per-update max-rank phases",
                                          "gpu_memory": "largest peak of any rank"}
            if distributed.is_main:
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
    parser.add_argument("--opponent", choices=("self_play", "current_self_play", "random", "wait"))
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
        config = {"task": asdict(TaskConfig(opponent="current_self_play", productive_actions=True,
                                                  combat_features=True, king_threat_distance=True,
                                                  persistent_knowledge=True)), "runner": runner_config(),
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
