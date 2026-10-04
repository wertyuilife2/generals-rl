"""CPU model preservation and fresh warm-start migration artifacts."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from tensordict import TensorDict

from generals_rl.combat_migration import (
    CELLS, COMBAT_SEMANTICS, FRAME_DIM, INPUT_LAYERS, fold_identity_input_weights, migrate_checkpoint,
)
from generals_rl.cts import runner_config
from generals_rl.student_ppo import StudentPPOPolicy
from rsl_rl.modules import ActorCriticMoECTS


def config_for(policy_class="ActorCriticMoECTS"):
    config = {"task": {"players": 2, "width": 9, "height": 9, "history_length": 4},
              "runner": runner_config(hidden_dim=16, expert_num=2)}
    config["runner"]["policy"].update(class_name=policy_class, latent_dim=8)
    return config


def observations(count=6):
    frames = torch.randn(count, 4, FRAME_DIM)
    frames[..., 10 * CELLS:12 * CELLS] = frames[..., 5 * CELLS:7 * CELLS]
    old = TensorDict({"policy": frames.flatten(1), "single_obs": frames[:, -1].clone(),
                      "critic": frames[:, 0].clone(),
                      "action_mask": torch.ones(count, 1 + CELLS * 8, dtype=torch.bool)}, batch_size=[count])
    new = old.clone()
    for key in ("policy", "single_obs", "critic"):
        changed = new[key].reshape(count, -1, FRAME_DIM)
        changed[..., 10 * CELLS:12 * CELLS] = torch.randn_like(changed[..., 10 * CELLS:12 * CELLS])
    return old, new


def make_policy(config, obs):
    kwargs = deepcopy(config["runner"]["policy"])
    name = kwargs.pop("class_name").replace(":", ".")
    cls = StudentPPOPolicy if name.split(".")[-1] == "StudentPPOPolicy" else ActorCriticMoECTS
    with redirect_stdout(io.StringIO()):
        return cls(obs[:1], config["runner"]["obs_groups"], 1 + CELLS * 8, **kwargs).eval()


def fold(state, config):
    return fold_identity_input_weights(state, config["runner"]["policy"],
                                       config["runner"]["obs_groups"], **config["task"])


@pytest.mark.parametrize("policy_class", ["rsl_rl.modules:ActorCriticMoECTS", "generals_rl.student_ppo:StudentPPOPolicy"])
def test_h4_model_function_and_unmodified_source(policy_class):
    torch.manual_seed(27)
    config, (old_obs, new_obs) = config_for(policy_class), observations()
    old_model, new_model = make_policy(config, old_obs), make_policy(config, new_obs)
    source, before = old_model.state_dict(), deepcopy(old_model.state_dict())
    migrated, report = fold(source, config)
    new_model.load_state_dict(migrated, strict=True)
    with torch.inference_mode():
        old_latent, old_gate = old_model.student_moe_encoder(old_obs["policy"])
        new_latent, new_gate = new_model.student_moe_encoder(new_obs["policy"])
        for old, new in ((old_latent, new_latent), (old_gate, new_gate),
                         (old_model.teacher_encoder(old_obs["critic"]), new_model.teacher_encoder(new_obs["critic"]))):
            torch.testing.assert_close(old, new, atol=1e-5, rtol=1e-5)
        branches = (False,) if isinstance(old_model, StudentPPOPolicy) else (False, True)
        for teacher in branches:
            for method in ("action_probabilities", "evaluate"):
                torch.testing.assert_close(getattr(old_model, method)(old_obs, is_teacher=teacher),
                                           getattr(new_model, method)(new_obs, is_teacher=teacher), atol=5e-5, rtol=1e-5)
    for key, original in source.items():
        torch.testing.assert_close(original, before[key], atol=0, rtol=0)
        assert migrated[key].data_ptr() != original.data_ptr()
        if key not in INPUT_LAYERS:
            torch.testing.assert_close(migrated[key], original, atol=0, rtol=0)
            continue
        layer = next(entry for entry in report["changed_layers"] if entry["key"] == key)
        unchanged = torch.ones(original.shape[1], dtype=torch.bool)
        for frame in range(layer["frames"]):
            offset = layer["latent_prefix"] + frame * FRAME_DIM
            for plane in (5, 6, 10, 11):
                unchanged[offset + plane * CELLS:offset + (plane + 1) * CELLS] = False
            assert torch.count_nonzero(migrated[key][:, offset + 10 * CELLS:offset + 12 * CELLS]) == 0
        torch.testing.assert_close(migrated[key][:, unchanged], original[:, unchanged], atol=0, rtol=0)


@pytest.fixture
def source_state():
    config = config_for()
    obs, _ = observations()
    return make_policy(config, obs).state_dict(), config


@pytest.mark.parametrize("invalid", ["normalizer", "history_width", "class"])
def test_rejects_unsupported_layout_without_mutating_weights(source_state, invalid):
    state, config = source_state
    before = deepcopy(state)
    if invalid == "normalizer":
        config["runner"]["policy"]["actor_obs_normalization"] = True
    elif invalid == "history_width":
        config["task"]["history_length"] = 1
    else:
        config["runner"]["policy"]["class_name"] = "generals_rl:student_ppo.StudentPPOPolicy"
    with pytest.raises(ValueError):
        fold(state, config)
    for key in state:
        torch.testing.assert_close(state[key], before[key], atol=0, rtol=0)


def write_source(path, config, weights):
    path.mkdir()
    (path / "run_config.json").write_text(json.dumps(config))
    torch.save({"model_state_dict": weights, "optimizer_state_dict": {"old": "Adam"},
                "optimizer_stu_enc_state_dict": {"old": "Adam"}, "iter": 31299,
                "source": {"snapshot_id": 42}}, path / "snapshot.pt")
    return path / "snapshot.pt"


def test_cli_preserves_source_and_emits_fresh_warm_start(tmp_path, source_state):
    state, config = source_state
    source = write_source(tmp_path / "source", config, state)
    checkpoint_bytes, config_bytes = source.read_bytes(), source.with_name("run_config.json").read_bytes()
    output = tmp_path / "migrated"
    script = Path(__file__).resolve().parents[1] / "migrate_combat_observations.py"
    result = subprocess.run([sys.executable, str(script), "--checkpoint", str(source), "--output", str(output)],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    saved = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
    migrated_config = json.loads((output / "run_config.json").read_text())
    assert set(saved) == {"model_state_dict", "migration"} and saved["migration"] == report
    assert report["optimizer"] == "fresh" and report["historical_pool"] == "fresh_reseed"
    assert report["initialization"] == "warm_start_weights" and report["resume_supported"] is False
    assert report["source"]["checkpoint_sha256"] == sha256(checkpoint_bytes).hexdigest()
    assert report["source"]["run_config_sha256"] == sha256(config_bytes).hexdigest()
    assert migrated_config["task"]["combat_features"] is True
    assert migrated_config["observation_semantics"] == COMBAT_SEMANTICS
    assert source.read_bytes() == checkpoint_bytes and source.with_name("run_config.json").read_bytes() == config_bytes
    model = make_policy(migrated_config, observations()[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.load_state_dict(saved["model_state_dict"], strict=True)  # train(warm_start=...) uses this operation.
    assert optimizer.state_dict()["state"] == {}
    with pytest.raises(FileExistsError):
        migrate_checkpoint(source, output)
    with pytest.raises(ValueError, match="original identity"):
        migrate_checkpoint(output / "checkpoint.pt", tmp_path / "double")
    assert not (tmp_path / "double").exists()


@pytest.mark.parametrize("invalid", ["other_shape", "dtype_overflow"])
def test_invalid_checkpoint_creates_no_output(tmp_path, source_state, invalid):
    state, config = source_state
    if invalid == "other_shape":
        state["actor.network.4.weight"] = torch.zeros(1, 16)
    else:
        state["actor.network.4.bias"] = torch.full_like(state["actor.network.4.bias"], 1e100, dtype=torch.float64)
    source = write_source(tmp_path / "source", config, state)
    with pytest.raises(ValueError):
        migrate_checkpoint(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
