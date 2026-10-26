"""The selected recipe works as a CLI default and a small real CPU run."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from unittest.mock import patch

import torch

from generals_rl.config import DEFAULT_ITERATIONS, default_training_config
from generals_rl.train import main, train


def test_default_cli_uses_the_selected_recipe_and_explicit_budget_overrides():
    expected = default_training_config()
    with patch("generals_rl.train.train") as execute:
        main(["--output", "unused-default"])
    assert execute.call_args.args[0] == expected
    assert execute.call_args.kwargs["iterations"] == DEFAULT_ITERATIONS == 224
    assert execute.call_args.kwargs["steps_per_env"] is None
    assert 8 * expected["task"]["num_envs"] * expected["runner"]["num_steps_per_env"] * 224 == 939_524_096
    with patch("generals_rl.train.train") as execute:
        main(["--output", "unused-ticks", "--steps-per-env", "64"])
    assert execute.call_args.kwargs["iterations"] is None
    assert execute.call_args.kwargs["steps_per_env"] == 64
    expected["runner"]["policy"]["backbone"] = "changed-by-caller"
    assert default_training_config()["runner"]["policy"]["backbone"] == "vit"


def test_other_model_config_replaces_the_policy_without_vit_fields(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("v32.json", "v32-opt.json"):
        old = json.loads((root / "configs" / name).read_text())
        config = tmp_path / name
        config.write_text(json.dumps(old))
        with patch("generals_rl.train.train") as execute:
            main(["--config", str(config), "--iterations", "1", "--output", str(tmp_path / "unused")])
        actual = execute.call_args.args[0]
        assert actual == old
        assert "backbone" not in actual["runner"]["policy"]
        assert "amp_bfloat16" not in actual["runner"]["algorithm"]
        assert "gamma_final" not in actual["curriculum"]
        assert actual["evaluation"]["every_n_updates"] == 100
        assert actual["evaluation"]["maps_per_opponent"] == 200
        assert actual["evaluation"].get("backend", "cpu") == "cpu"
        assert actual["inference"] == {"mode": "stochastic", "precision": "from_config", "execution": "eager"}
        assert execute.call_args.kwargs["iterations"] == 1


def test_sparse_model_override_replaces_the_whole_policy(tmp_path):
    policy = {"class_name": "custom.Model", "custom_option": 7}
    config = tmp_path / "custom.json"
    config.write_text(json.dumps({"runner": {"policy": policy}}))
    with patch("generals_rl.train.train") as execute:
        main(["--config", str(config), "--output", str(tmp_path / "unused")])
    actual = execute.call_args.args[0]
    assert actual["runner"]["policy"] == policy
    assert actual["task"] == default_training_config()["task"]


def test_partial_override_and_resume_preserve_their_configuration_boundaries(tmp_path):
    config = tmp_path / "override.json"
    config.write_text(json.dumps({"runner": {"policy": {"head_channels": 64}}}))
    with patch("generals_rl.train.train") as execute:
        main(["--config", str(config), "--iterations", "1", "--output", str(tmp_path / "override")])
    policy = execute.call_args.args[0]["runner"]["policy"]
    assert policy["backbone"] == "vit" and policy["head_channels"] == 64
    original = {"task": {"seed": 73}, "runner": {"policy": {"class_name": "saved.Policy"}}}
    (tmp_path / "run_config.json").write_text(json.dumps(original))
    with patch("generals_rl.train.train") as execute:
        main(["--resume", str(tmp_path / "checkpoint.pt"), "--iterations", "2"])
    assert execute.call_args.args[0] == original
    assert execute.call_args.kwargs["iterations"] == 2


def test_selected_vit_recipe_completes_cpu_updates_and_resumes(tmp_path):
    config = default_training_config()
    config["task"].update(backend="cpu", device="cpu", num_envs=2, map_pool_size=0,
                          max_episode_steps=3, mountain_density=0, city_density=0)
    config["runner"].update(num_steps_per_env=4, save_interval=1)
    config["runner"]["policy"].update(amp_bfloat16=False, compile_forward=False)
    config["runner"]["algorithm"].update(amp_bfloat16=False, num_learning_epochs=1,
                                           num_mini_batches=1, physical_minibatch_size=4,
                                           ppo_probe_samples=8, ppo_probe_batch_size=4)
    config["evaluation"]["enabled"] = False
    original = deepcopy(config)
    with redirect_stdout(io.StringIO()):
        summary = train(config, tmp_path / "first", iterations=2)
        checkpoint = tmp_path / "first/checkpoint.pt"
        continued = train(config, tmp_path / "resumed", iterations=1, resume=str(checkpoint))
    saved = json.loads((tmp_path / "first/run_config.json").read_text())
    assert config == original == saved
    assert saved["self_play"]["pool_size"] == 8
    assert saved["self_play"]["snapshot_interval"] == 1
    assert summary["iterations"] == 2
    assert summary["self_play"]["episode_assignments"] > 0
    assert saved["runner"]["algorithm"]["adaptive_lr_max"] == 1e-3
    weights = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=False, map_location="cpu")
    assert weights["iter"] == 2
    assert len(weights["optimizer_state_dict"]["param_groups"]) == 1
    assert 1e-5 <= weights["optimizer_state_dict"]["param_groups"][0]["lr"] <= 1e-3
    assert continued["iterations"] == 1
    assert max(continued["self_play"]["pool_ids"]) > max(summary["self_play"]["pool_ids"])
