"""Weights-only migration from duplicate P2 identity planes to combat planes.

The result is for ``train --warm-start``: optimizers, iteration counters and
historical opponents must be initialized by a fresh training run.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path

import torch

from .mdp.combat import CombatObservationEncoder


COMBAT_SEMANTICS = CombatObservationEncoder.semantics
FRAME_DIM = 985
CELLS = 81
INPUT_LAYERS = {
    "teacher_encoder.0.network.0.weight": (False, False),
    "student_moe_encoder.moe.experts.backbone.network.0.weight": (True, False),
    "student_moe_encoder.moe.gating_network.0.network.0.weight": (True, False),
    "actor.network.0.weight": (False, True),
    "critic.network.0.weight": (False, True),
}
SUPPORTED_CLASSES = {
    "ActorCriticMoECTS", "rsl_rl.modules.ActorCriticMoECTS",
    "rsl_rl.modules:ActorCriticMoECTS",
    "rsl_rl.modules.actor_critic_moe_cts.ActorCriticMoECTS",
    "rsl_rl.modules.actor_critic_moe_cts:ActorCriticMoECTS",
    "StudentPPOPolicy", "generals_rl.student_ppo.StudentPPOPolicy",
    "generals_rl.student_ppo:StudentPPOPolicy",
}


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dense_input_specs(state_dict, policy_config, obs_groups, *,
                       history_length=4, players=2, width=9, height=9):
    """Shared verified dense first-layer layout; never transform source tensors."""
    dimensions = tuple(_positive_int(value, name) for value, name in
                       ((players, "players"), (width, "width"), (height, "height")))
    if dimensions != (2, 9, 9):
        raise ValueError("migration supports only P2 9x9 / 985D frames")
    history_length = _positive_int(history_length, "history_length")
    if obs_groups != {"policy": ["policy"], "critic": ["critic"]}:
        raise ValueError("migration requires exact dense policy/critic obs groups")
    policy_class = policy_config.get("class_name", "ActorCriticMoECTS")
    if not isinstance(policy_class, str) or policy_class not in SUPPORTED_CLASSES:
        raise ValueError("migration supports only verified dense CTS/StudentPPO policy classes")
    if policy_config.get("action_distribution") != "categorical":
        raise ValueError("migration requires categorical actions")
    if policy_config.get("state_dependent_std", False) is not False:
        raise ValueError("migration does not support state-dependent Gaussian std")
    for option in ("actor_obs_normalization", "critic_obs_normalization"):
        if policy_config.get(option, False) is not False:
            raise ValueError("migration requires both observation normalizers disabled")
    if not isinstance(state_dict, dict) or not state_dict or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state_dict.items()):
        raise ValueError("model_state_dict must contain only named tensors")
    if any("obs_normalizer" in key for key in state_dict):
        raise ValueError("normalizer state is unsupported")
    if any(value.layout != torch.strided or not bool(torch.isfinite(value).all())
           for value in state_dict.values()):
        raise ValueError("model_state_dict must contain finite dense tensors")
    latent_dim = _positive_int(policy_config.get("latent_dim", 32), "latent_dim")
    specs = {}
    for key, (history, latent) in INPUT_LAYERS.items():
        if key not in state_dict:
            raise ValueError(f"unsupported dense layout: missing {key}")
        frames, prefix = (history_length if history else 1), (latent_dim if latent else 0)
        expected = prefix + frames * FRAME_DIM
        weight = state_dict[key]
        if (weight.ndim != 2 or weight.shape[1] != expected or
                not weight.is_floating_point() or weight.layout != torch.strided):
            raise ValueError(f"unsupported input shape for {key}: expected width {expected}")
        specs[key] = (frames, prefix)
    return specs, latent_dim


def fold_identity_input_weights(state_dict, policy_config, obs_groups, *,
                                history_length=4, players=2, width=9, height=9):
    """Fold duplicate identity planes W10/W11 into W5/W6, then zero W10/W11."""
    specs, latent_dim = _dense_input_specs(
        state_dict, policy_config, obs_groups, history_length=history_length,
        players=players, width=width, height=height)
    result = OrderedDict((key, value.detach().clone()) for key, value in state_dict.items())
    if hasattr(state_dict, "_metadata"):
        result._metadata = deepcopy(state_dict._metadata)
    report = {
        "schema_version": 1, "observation_semantics": COMBAT_SEMANTICS,
        "frame_dim": FRAME_DIM, "history_length": history_length, "latent_dim": latent_dim,
        "initialization": "warm_start_weights", "resume_supported": False,
        "optimizer": "fresh", "historical_pool": "fresh_reseed", "changed_layers": [],
    }
    for key, (frames, prefix) in specs.items():
        weight = result[key]
        for frame in range(frames):
            offset = prefix + frame * FRAME_DIM
            for old_plane, existing_plane in ((10, 5), (11, 6)):
                old = slice(offset + old_plane * CELLS, offset + (old_plane + 1) * CELLS)
                existing = slice(offset + existing_plane * CELLS, offset + (existing_plane + 1) * CELLS)
                weight[:, existing] += weight[:, old]
                weight[:, old] = 0
        if not bool(torch.isfinite(weight).all()):
            raise ValueError(f"identity folding overflowed {key}")
        report["changed_layers"].append({"key": key, "shape": list(weight.shape),
                                         "frames": frames, "latent_prefix": prefix})
    return result, report


def _validate_policy_state(weights, policy_config, history_length):
    """Check remaining layer keys/shapes against the known CPU policy class."""
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCriticMoECTS
    from .student_ppo import StudentPPOPolicy

    kwargs = deepcopy(policy_config)
    name = kwargs.pop("class_name", "ActorCriticMoECTS").replace(":", ".")
    policy_class = StudentPPOPolicy if name.split(".")[-1] == "StudentPPOPolicy" else ActorCriticMoECTS
    observations = TensorDict({"policy": torch.zeros(1, history_length * FRAME_DIM),
                               "single_obs": torch.zeros(1, FRAME_DIM),
                               "critic": torch.zeros(1, FRAME_DIM)}, batch_size=[1])
    with redirect_stdout(io.StringIO()):
        policy = policy_class(observations, {"policy": ["policy"], "critic": ["critic"]},
                              1 + CELLS * 8, **kwargs)
    try:
        policy.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint weights do not match the configured dense policy") from exc
    if any(not bool(torch.isfinite(value).all()) for value in policy.state_dict().values()):
        raise ValueError("checkpoint weights overflow the configured policy dtype")


def migrate_checkpoint(checkpoint, output_dir, *, config_path=None, king_threat_distance=False):
    """Write a new weights-only checkpoint and matching fresh-run configuration.

    Source training checkpoints and frozen snapshots are accepted. Neither
    source files nor optimizer/history metadata are modified or copied.
    """
    checkpoint = Path(checkpoint).resolve()
    config_path = Path(config_path).resolve() if config_path else checkpoint.with_name("run_config.json")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"migration output already exists: {output}; choose a new directory")
    config = json.loads(config_path.read_text())
    task, runner = config["task"], config["runner"]
    if king_threat_distance and task.get("king_threat_distance", False) is not False:
        raise ValueError("source already uses king threat distance")
    base_semantics = COMBAT_SEMANTICS if task.get("combat_features", False) else "local_p2_985_v1"
    if king_threat_distance and config.get("observation_semantics") not in (None, base_semantics):
        raise ValueError("unsupported source observation semantics for king threat migration")
    if not king_threat_distance and (task.get("combat_features", False) is not False or config.get("observation_semantics") is not None):
        raise ValueError("source must use the original identity observations; already-versioned input cannot be migrated")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "model_state_dict" not in saved:
        raise ValueError("checkpoint must contain model_state_dict")
    if "migration" in saved:
        if not king_threat_distance:
            raise ValueError("checkpoint has already been migrated")
        migration = saved["migration"]
        if not isinstance(migration, dict) or migration.get("observation_semantics") != base_semantics:
            raise ValueError("checkpoint migration observation semantics must match source base observations")
    if king_threat_distance:
        from .king_threat_migration import fold_king_threat_input_weights
        fold = fold_king_threat_input_weights
    else:
        fold = fold_identity_input_weights
    weights, report = fold(
        saved["model_state_dict"], runner["policy"], runner["obs_groups"],
        history_length=task.get("history_length", 4), players=task.get("players", 2),
        width=task.get("width", 9), height=task.get("height", 9))
    _validate_policy_state(weights, runner["policy"], report["history_length"])
    report["source"] = {"checkpoint": str(checkpoint),
                         "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                         "run_config": str(config_path),
                         "run_config_sha256": sha256(config_path.read_bytes()).hexdigest()}
    migrated_config = deepcopy(config)
    if king_threat_distance:
        migrated_config["task"]["king_threat_distance"] = True
        from .config import task_config
        from .mdp.observations import make_observation_encoder
        report["observation_semantics"] = make_observation_encoder(task_config(migrated_config["task"])).semantics
    else:
        migrated_config["task"]["combat_features"] = True
    migrated_config["observation_semantics"] = report["observation_semantics"]
    migrated_config["runner"]["policy"].setdefault("class_name", "ActorCriticMoECTS")
    # RSL resolves simple names only within rsl_rl; the student adapter lives here.
    if migrated_config["runner"]["policy"].get("class_name") == "StudentPPOPolicy":
        migrated_config["runner"]["policy"]["class_name"] = "generals_rl.student_ppo:StudentPPOPolicy"
    config_json = json.dumps(migrated_config, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"model_state_dict": weights, "migration": report}, output / "checkpoint.pt")
    (output / "run_config.json").write_text(config_json)
    return report
