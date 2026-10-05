"""Stream old deployed-student labels into a new CNN actor/privileged critic.

No PPO storage or observation dataset is created. This is weights-only supervised
initialization, not exact warm start or resume. Run the CLI only when authorized.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import torch
from torch.nn import functional as F
from tensordict import TensorDict

from generals_rl.config import task_config
from generals_rl.self_play import HistoricalPolicyPool, _json_hash
from generals_rl.self_play_training import SelfPlayConfig
from generals_rl.task import GeneralsTask
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.utils import resolve_callable

from generals_rl.spatial_ppo import (
    FRAME_DIM, HISTORY, NUM_ACTIONS, policy_config,
)


MAX_ENVS, MAX_BATCH, MAX_MODELS, MAX_SECONDS, CHUNK_TICKS = 4096, 1024, 4, 600, 32
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_TICKS, MAX_CONFIG_BYTES = 1000000, 1024 * 1024


def pool_signature(task, policy, options, observations):
    """Prototype equivalent of the existing fresh self-play factory signature."""
    config = deepcopy(options)
    config.pop("class_name", None)
    signature = {
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy_config": config,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "observation_shapes": {key: list(value.shape[1:]) for key, value in observations.items()},
        "mdp": {key: getattr(task.config, key) for key in
                ("width", "height", "players", "history_length", "max_episode_steps")},
        "encoder": f"{type(task.observation_encoder).__module__}.{type(task.observation_encoder).__qualname__}",
        "action_space": f"{type(task.action_space).__module__}.{type(task.action_space).__qualname__}",
    }
    if task.config.productive_actions:
        signature["mdp"]["productive_actions"] = True
    if hasattr(task.observation_encoder, "semantics"):
        signature["encoder_semantics"] = task.observation_encoder.semantics
    return signature


def shape_example(observations):
    # No true map value is needed for construction, including the critic.
    return observations[:1].clone().to("cpu").zero_()


def teacher_factory(config, observations):
    options = deepcopy(config["runner"]["policy"])
    cls = resolve_callable(options.pop("class_name"))
    if not isinstance(cls, type) or not issubclass(cls, ActorCriticMoECTS):
        raise ValueError("teacher must expose the existing deployed-student categorical interface")
    if options.get("action_distribution") != "categorical":
        raise ValueError("local distillation requires categorical teacher actions")
    if config["runner"]["obs_groups"] != {"policy": ["policy"], "critic": ["critic"]}:
        raise ValueError("teacher requires exact local policy and privileged critic groups")
    example = shape_example(observations)

    def construct():
        # CPU module initialization must not consume collection/global RNG.
        with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
            return cls(example, deepcopy(config["runner"]["obs_groups"]), NUM_ACTIONS, **deepcopy(options))
    return construct


def source_bytes(path, limit):
    """Bounded immutable bytes, also used for the exact loaded-content SHA."""
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("source file exceeds its byte limit")
    return content


def load_teacher(checkpoint, factory, device, *, content=None):
    content = source_bytes(checkpoint, MAX_SOURCE_BYTES) if content is None else content
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "model_state_dict" not in saved:
        raise ValueError("source checkpoint must contain model_state_dict")
    policy = factory()
    policy.load_state_dict(saved["model_state_dict"], strict=True)
    del saved  # Source optimizer data, if present, is never used or moved to GPU.
    return policy.to(device).eval().requires_grad_(False)


class ReadOnlyTeacherPool:
    """Validate only a bounded seed/latest subset without mutating source files.

    The normal HistoricalPolicyPool constructor can rewrite/prune an existing
    pool. Instead its reader sees an ephemeral metadata/symlink view containing
    at most four snapshots. No source weight or manifest is copied or written.
    """
    def __init__(self, directory, *, factory, signature, device, max_models, scratch):
        directory = Path(directory).resolve()
        manifest_path = directory / "manifest.json"
        manifest_content = source_bytes(manifest_path, MAX_CONFIG_BYTES)
        manifest = json.loads(manifest_content)
        entries = manifest.get("snapshots")
        if not isinstance(entries, list) or not entries or not entries[0].get("seed"):
            raise ValueError("teacher pool must contain its seed and frozen historical snapshots")
        selected = [entries[0]] + (entries[-(max_models - 1):] if max_models > 1 else [])
        selected = list({entry["snapshot_id"]: entry for entry in selected}.values())
        loader = HistoricalPolicyPool.__new__(HistoricalPolicyPool)
        loader.policy_factory, loader.policy_spec = factory, deepcopy(signature)
        loader.policy_spec_sha256, loader.device = _json_hash(signature), torch.device(device)
        self.source = {"directory": str(directory), "manifest_sha256": sha256(manifest_content).hexdigest(),
                       "selected": [], "policy_spec_sha256": loader.policy_spec_sha256}
        with TemporaryDirectory(dir=scratch) as temporary:
            view = Path(temporary)
            for entry in selected:
                # Reject traversal before creating a symlink or reading a weight.
                filename = f"snapshot_{entry['snapshot_id']:08d}.pt"
                if entry.get("filename") != filename:
                    raise ValueError("invalid teacher pool snapshot filename")
                source = directory / filename
                if source.stat().st_size > MAX_SOURCE_BYTES:
                    raise ValueError("teacher snapshot exceeds the prototype's 64 MiB limit")
                (view / filename).symlink_to(source)
                self.source["selected"].append({"id": entry["snapshot_id"],
                                               "iteration": entry["iteration"], "sha256": entry["sha256"]})
            bounded = deepcopy(manifest)
            bounded["snapshots"] = selected
            (view / "manifest.json").write_text(json.dumps(bounded))
            _, snapshots = loader._read_pool(view)  # Existing schema/SHA/layout/finite checks.
        self._snapshots = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
        self.ids = tuple(self._snapshots)

    def sample(self, count, *, generator):
        indices = torch.randint(len(self.ids), (count,), generator=generator,
                                device=generator.device).tolist()
        return [self._snapshots[self.ids[index]] for index in indices]

    def get(self, snapshot_id):
        return self._snapshots[int(snapshot_id)]


@torch.no_grad()
def teacher_labels(teacher, observations, *, fit_value=True):
    # The old deployed student sees local history and the public action mask.
    # The privileged teacher encoder/inference branch is never selected.
    local = observations.select("policy", "single_obs", "action_mask")
    probabilities = teacher.action_probabilities(local, is_teacher=False).detach().clone()
    values = teacher.evaluate(observations, is_teacher=False).detach().clone() if fit_value else None
    return probabilities, values


def distillation_losses(student, observations, probabilities, values=None):
    if probabilities.requires_grad or (values is not None and values.requires_grad):
        raise ValueError("distillation labels must be detached")
    student.action_probabilities(observations.select("policy", "single_obs", "action_mask"), is_teacher=False)
    # Categorical.logits are globally normalized log probabilities. Illegal
    # targets are zero, so kl_div handles 0*log0 without clamping valid gradients.
    kl = F.kl_div(student.distribution.logits, probabilities, reduction="batchmean")
    mse = F.mse_loss(student.evaluate(observations), values) if values is not None else kl.new_zeros(())
    return kl, mse


def fresh_student(observations, device, seed, *, value_potential_offset=False, light_cnn=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        options = policy_config(value_potential_offset=value_potential_offset, light_cnn=light_cnn)
        policy_class = resolve_callable(options.pop("class_name"))
        policy = policy_class(shape_example(observations),
                                  {"policy": ["policy"], "critic": ["critic"]}, NUM_ACTIONS, **options)
    return policy.to(device).train()


def update_stream_batch(student, optimizer, observations, labels, *, batch_size, generator, value_coef,
                        deadline=None):
    count = observations.batch_size[0]
    order = torch.randperm(count, generator=generator, device=observations.device)
    sums = torch.zeros(2, device=observations.device)
    processed, updates = 0, 0
    for rows in order.split(batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            break
        kl, mse = distillation_losses(student, observations[rows], labels[0][rows],
                                      labels[1][rows] if labels[1] is not None else None)
        optimizer.zero_grad(set_to_none=True)
        (kl + value_coef * mse).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        sums += torch.stack((kl.detach(), mse.detach())) * len(rows)
        processed += len(rows)
        updates += 1
    return sums / max(1, processed), processed, updates


def new_run_config(source, settings, *, light_cnn=False):
    config = deepcopy(source)
    # Collection's reduced lane/device/seed settings belong only to distillation.
    # Keep the formal task budget and league proportions from the source run.
    config.setdefault("task", asdict(settings))
    config["runner"]["policy"] = policy_config(
        value_potential_offset=source["runner"]["policy"].get("value_potential_offset", False), light_cnn=light_cnn)
    algorithm = config["runner"]["algorithm"]
    algorithm.update(class_name="generals_rl.student_ppo.StudentPPO", teacher_env_ratio=0,
                     load_balance_coef=0, schedule="fixed", desired_kl=None,
                     student_encoder_learning_rate=algorithm.get("learning_rate", 3e-4))
    # These are formal PPO settings, independent of distillation's bounded pool.
    config.setdefault("self_play", asdict(SelfPlayConfig()))
    return config


def validate_limits(num_envs, batch_size, max_models, time_budget, ticks, value_coef, learning_rate):
    for name, value, maximum in (("num_envs", num_envs, MAX_ENVS), ("batch_size", batch_size, MAX_BATCH),
                                 ("max_teacher_models", max_models, MAX_MODELS)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be in [1,{maximum}]")
    if not 0 < time_budget <= MAX_SECONDS or not isinstance(ticks, int) or isinstance(ticks, bool) or not 1 <= ticks <= MAX_TICKS:
        raise ValueError("time budget must be (0,600] seconds and ticks must be in [1,1000000]")
    if not 0 <= value_coef <= 100 or not 0 < learning_rate <= .1:
        raise ValueError("invalid value coefficient or learning rate")


def distill(checkpoint, output_dir, *, config_path=None, teacher_pool=None, num_envs=1024,
            batch_size=512, max_teacher_models=4, time_budget=540., ticks=1000000,
            device="cpu", seed=42, learning_rate=3e-4, value_coef=1., tensorboard=True,
            benchmark=False, light_cnn=False):
    if not isinstance(light_cnn, bool):
        raise TypeError("light_cnn must be boolean")
    validate_limits(num_envs, batch_size, max_teacher_models, time_budget, ticks, value_coef, learning_rate)
    checkpoint = Path(checkpoint).resolve()
    config_path = Path(config_path).resolve() if config_path else checkpoint.with_name("run_config.json")
    pool_path = Path(teacher_pool).resolve() if teacher_pool else checkpoint.parent / "history_pool"
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("distillation requires a new output directory and a fresh CNN pool")
    if output.is_relative_to(checkpoint.parent) or output.is_relative_to(pool_path):
        raise ValueError("output must be outside the read-only teacher run/pool")
    config_content = source_bytes(config_path, MAX_CONFIG_BYTES)
    source_config = json.loads(config_content)
    config_sha256 = sha256(config_content).hexdigest()
    checkpoint_content = source_bytes(checkpoint, MAX_SOURCE_BYTES)
    checkpoint_sha256 = sha256(checkpoint_content).hexdigest()
    run_id = f"cnn_kl_{uuid4().hex[:12]}"
    source_settings = task_config(source_config["task"])
    if (source_settings.width, source_settings.height, source_settings.players, source_settings.history_length) != (9, 9, 2, 4):
        raise ValueError("CNN distillation supports only P2 9x9 history4")
    device = torch.device(device)
    settings = replace(source_settings, num_envs=num_envs, seed=seed, device=str(device),
                       backend="gpu" if device.type == "cuda" else "cpu", opponent="self_play", threads=1)
    output.mkdir(parents=True)
    source_provenance = {"run_id": run_id, "source": {
        "checkpoint": str(checkpoint), "sha256": checkpoint_sha256,
        "run_config": str(config_path), "config_sha256": config_sha256,
        "content_frozen_before_collection": True}}
    (output / "source_provenance.json").write_text(json.dumps(source_provenance, indent=2) + "\n")
    task, writer = None, None
    try:
        task = GeneralsTask(settings)
        observations = TensorDict(task.get_observations(), batch_size=[num_envs], device=device)
        if (observations["policy"].shape[-1] != HISTORY * FRAME_DIM or
                observations["single_obs"].shape[-1] != FRAME_DIM or
                observations["critic"].shape[-1] != FRAME_DIM or
                observations["action_mask"].shape[-1] != NUM_ACTIONS):
            raise ValueError("distillation supports only 985D frames / 12 maps+13 public, H4 and649 actions")
        factory = teacher_factory(source_config, observations)
        teacher = load_teacher(checkpoint, factory, device, content=checkpoint_content)
        del checkpoint_content, config_content
        signature = pool_signature(task, teacher, source_config["runner"]["policy"], observations)
        pool = ReadOnlyTeacherPool(pool_path, factory=factory, signature=signature, device=device,
                                   max_models=max_teacher_models, scratch=output)
        task.opponent.attach(pool, stochastic=True, seed=seed + 7001)
        student = fresh_student(observations, device, seed + 17,
                                value_potential_offset=source_config["runner"]["policy"].get("value_potential_offset", False),
                                light_cnn=light_cnn)
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
        assert not optimizer.state  # Never load a source optimizer or CNN resume.
        action_rng = torch.Generator(device=device).manual_seed(seed + 101)
        batch_rng = torch.Generator(device=device).manual_seed(seed + 202)
        # Productive-action MDPs commonly permit only WAIT at tick0. Defer the
        # probe until at least one row has multiple legal choices, so its KL is
        # informative; it is an optimization probe, not held-out validation.
        probe = probe_labels = initial_probe = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output / "tensorboard"))
        baseline_memory = peak_memory = baseline_reserved = peak_reserved = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if benchmark:
                baseline_memory = torch.cuda.memory_allocated(device)
                baseline_reserved = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        deadline = started + time_budget
        completed, samples, updates, label_batches = 0, 0, 0, 0
        sums = torch.zeros(2, device=device)
        for tick in range(ticks):
            if time.monotonic() >= deadline:
                break
            labels = teacher_labels(teacher, observations, fit_value=value_coef > 0)
            label_batches += 1
            if probe is None:
                rows = (observations["action_mask"].sum(-1) > 1).nonzero(as_tuple=True)[0][:64]
                if rows.numel():
                    probe = observations[rows].clone()
                    probe_labels = (labels[0][rows].clone(), labels[1][rows].clone() if labels[1] is not None else None)
                    with torch.no_grad():
                        initial_probe = torch.stack(distillation_losses(student, probe, *probe_labels)).tolist()
            actions = torch.multinomial(labels[0], 1, generator=action_rng).squeeze(1)
            losses, processed, steps = update_stream_batch(
                student, optimizer, observations, labels, batch_size=batch_size,
                generator=batch_rng, value_coef=value_coef, deadline=deadline)
            samples += processed
            updates += steps
            if processed != num_envs or time.monotonic() >= deadline:
                del labels, actions
                break
            sums += losses
            result = task.step(actions)  # Teacher actions drive both seats, CNN never drives collection.
            observations = TensorDict(result.observations, batch_size=[num_envs], device=device)
            completed += 1
            if completed % CHUNK_TICKS == 0:
                last_metrics = (sums / CHUNK_TICKS).tolist()
                sums.zero_()
                if writer:
                    for name, value in zip(("actor_kl", "value_mse"), last_metrics):
                        writer.add_scalar(f"Distill/train_{name}", value, samples)
                    if probe is not None:
                        with torch.no_grad():
                            probe_metrics = torch.stack(distillation_losses(student, probe, *probe_labels)).tolist()
                        for name, value in zip(("actor_kl", "value_mse"), probe_metrics):
                            writer.add_scalar(f"Distill/probe_{name}", value, samples)
            # Neither labels nor the previous input batch survive into a dataset/storage.
            del labels, actions, result
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            if benchmark:
                peak_memory = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
        elapsed = time.monotonic() - started
        final_probe = None
        if probe is not None:
            with torch.no_grad():
                final_probe = torch.stack(distillation_losses(student, probe, *probe_labels)).tolist()
        metadata = {
            "schema_version": 1, "run_id": run_id, "strategy": "local_masked_kl_distillation", "exact_warm_start": False,
            "value_potential_offset": student.value_potential_offset,
            "production_training_cli": "use --replace-policy for a new run; --warm-start weights only",
            "input_layout": "P2_9x9_H4_frame985_maps12_public13",
            "observation_semantics": signature.get("encoder_semantics", "local_p2_985_v1"),
            "actor_inputs": ["policy", "single_obs", "action_mask"],
            "collection_opponent": "bounded_read_only_historical_pool_no_baseline_overlay",
            "formal_task_config": "preserved_source_task_not_distillation_execution_settings",
            "resume_supported": False, "future_initialization": "warm_start_weights_fresh_Adam_and_CNN_pool",
            "teacher_branch": "deployed_student_local_only_is_teacher_False", "teacher_frozen": True,
            "old_normalizers": "frozen_eval_never_updated", "kl_direction": "teacher || CNN",
            "value_labels": "old_student_critic_returned_value_same_offset_semantics_no_grad" if value_coef > 0 else "none",
            "value_coef": value_coef, "distill_optimizer": "fresh_single_Adam", "learning_rate": learning_rate,
            "source": {**source_provenance["source"],
                       "policy_class": f"{type(teacher).__module__}.{type(teacher).__qualname__}",
                       "historical_pool": pool.source},
            "limits": {"envs": num_envs, "max_envs": MAX_ENVS, "minibatch": batch_size,
                       "max_minibatch": MAX_BATCH, "teacher_models": len(pool.ids), "max_teacher_models": MAX_MODELS,
                       "resident_old_models": 1 + len(pool.ids), "max_resident_old_models": 1 + MAX_MODELS,
                       "chunk_ticks": CHUNK_TICKS, "time_budget_seconds": time_budget,
                       "max_ticks": MAX_TICKS, "max_scalar_chunk_reports": MAX_TICKS // CHUNK_TICKS,
                       "source_file_byte_limit": MAX_SOURCE_BYTES, "source_config_manifest_byte_limit": MAX_CONFIG_BYTES,
                       "observation_storage": "one_tick_plus_fixed_at_most64_row_probe", "saved_datasets": 0,
                       "CNN_weight_files": 2, "rolling_training_snapshots": 0},
            "seeds": {"maps": seed, "CNN_initialization": seed + 17, "teacher_actions": seed + 101,
                      "minibatch_order": seed + 202, "opponent": seed + 7001},
            "ticks": completed, "samples": samples, "Adam_steps": updates,
            "teacher_label_samples": label_batches * num_envs,
            "stop_reason": "tick_limit" if completed == ticks else "time_budget",
            "collection_update_seconds": elapsed, "source_setup_and_disk_save_excluded": True,
            "deadline_contract": "soft loop budget checked per tick/minibatch; one in-flight operation may overrun; setup/save excluded",
            "probe_contract": "fixed first state with multiple legal choices, included in training; not held-out validation",
            "initial_probe": {"actor_kl": initial_probe[0], "value_mse": initial_probe[1]} if initial_probe else None,
            "final_probe": {"actor_kl": final_probe[0], "value_mse": final_probe[1]} if final_probe else None,
            "benchmark": {"enabled": benchmark, "device": str(device), "samples_per_second": samples / elapsed,
                          "baseline_allocated_bytes": baseline_memory, "peak_allocated_bytes": peak_memory,
                          "baseline_reserved_bytes": baseline_reserved, "peak_reserved_bytes": peak_reserved,
                          "tensor_dtype": "float32", "autocast": False,
                          "scope": "teacher_labels+CNN_update+native_selfplay_step, FP32; no PPO rollout storage"},
        }
        new_config = new_run_config(source_config, settings, light_cnn=light_cnn)
        new_config["run_id"] = run_id
        new_config["initialization"] = {"kind": "local_masked_kl_distillation", "exact_warm_start": False}
        torch.save({"model_state_dict": student.state_dict(), "migration": metadata}, output / "checkpoint.pt")
        (output / "run_config.json").write_text(json.dumps(new_config, indent=2) + "\n")
        cnn_options = deepcopy(new_config["runner"]["policy"])
        cnn_class = resolve_callable(cnn_options.pop("class_name"))
        def cnn_factory():
            return cnn_class(shape_example(observations),
                                    {"policy": ["policy"], "critic": ["critic"]}, NUM_ACTIONS, **cnn_options)
        fresh_pool = HistoricalPolicyPool(output / "history_pool", policy_factory=cnn_factory,
                                          policy_spec=pool_signature(task, student, new_config["runner"]["policy"], observations),
                                          device=device, max_size=2)
        fresh_pool.publish(student, iteration=-1,
                           source={"kind": "local_masked_kl_distillation", "teacher_checkpoint_sha256": metadata["source"]["sha256"]})
        (output / "distillation_report.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if writer and metadata["final_probe"]:
            for name, value in metadata["final_probe"].items():
                writer.add_scalar(f"Distill/final_{name}", value, samples)
        return metadata
    finally:
        if writer:
            writer.close()
        if task:
            task.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--teacher-pool", type=Path, help="frozen source history_pool; default beside checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new CNN output directory outside teacher run/pool")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-teacher-models", type=int, default=4)
    parser.add_argument("--time-budget", type=float, default=540., help="supervised/collection budget, <=600s; default leaves setup/save margin")
    parser.add_argument("--ticks", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=1.)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--light-cnn", action="store_true", help="opt in to the two-layer 16-channel standard CNN")
    parser.add_argument("--benchmark-seconds", type=float, help="short full-pipeline benchmark, <=600s; still fits CNN weights")
    arguments = parser.parse_args(argv)
    result = distill(arguments.checkpoint, arguments.output, config_path=arguments.config,
                     teacher_pool=arguments.teacher_pool, num_envs=arguments.num_envs,
                     batch_size=arguments.batch_size, max_teacher_models=arguments.max_teacher_models,
                     time_budget=arguments.benchmark_seconds if arguments.benchmark_seconds is not None else arguments.time_budget,
                     ticks=arguments.ticks, device=arguments.device, seed=arguments.seed,
                     learning_rate=arguments.learning_rate, value_coef=arguments.value_coef,
                     tensorboard=not arguments.no_tensorboard, benchmark=arguments.benchmark_seconds is not None,
                     light_cnn=arguments.light_cnn)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
