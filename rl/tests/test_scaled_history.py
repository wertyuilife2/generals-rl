"""Eight-frame information flow and deployment, with the H4 computation intact."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile

import pytest
import torch

from generals_rl.config import task_config
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.train import train
from test_scaled_ppo import GROUPS, construct, observations


@pytest.mark.parametrize("family", ("resnet", "vit"))
@pytest.mark.parametrize("spatial_mode", ("dense", "pyramid"))
@pytest.mark.parametrize("projection", (0, 32))
def test_h8_preserves_masking_and_backpropagates_through_every_frame(family, spatial_mode, projection):
    torch.set_num_threads(1)
    obs = observations(history_length=8)
    for key in ("policy", "knowledge", "critic"):
        obs[key].requires_grad_()
    policy = construct(obs, family, history_length=8, spatial_mode=spatial_mode,
                       critic_projection_channels=projection)
    assert "hist8" in policy.semantics
    assert policy.num_actor_obs == 8 * 985 + 891
    probabilities = policy.action_probabilities(obs)
    assert probabilities.shape == (2, 649)
    assert probabilities[~obs["action_mask"]].count_nonzero() == 0
    policy.actor(obs["policy"], obs["knowledge"])[:, [0, 1, 17]].square().sum().backward()
    frame_grad = obs["policy"].grad.reshape(2, 8, 985)[..., :972].abs().sum((0, 2))
    assert (frame_grad > 0).all()
    assert obs["critic"].grad is None
    assert all(parameter.grad is None for parameter in policy.critic.parameters())
    obs["policy"].grad = None
    obs["knowledge"].grad = None
    policy.zero_grad(set_to_none=True)
    policy.evaluate(obs).sum().backward()
    assert (obs["policy"].grad.reshape(2, 8, 985)[..., :972].abs().sum((0, 2)) > 0).all()
    assert obs["critic"].grad.abs().sum() > 0
    assert obs["knowledge"].grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.parameters())


@pytest.mark.parametrize("projection", (0, 32))
def test_h8_zero_extended_inputs_reproduce_h4_actor_and_critic(projection):
    """New frames cannot silently move knowledge, latest-public, or XY channels."""
    torch.set_num_threads(1)
    obs8 = observations(history_length=8)
    obs4 = obs8.clone()
    obs4["policy"] = obs8["policy"].reshape(2, 8, 985)[:, -4:].flatten(1)
    # ResNet's weight-standardized input convolution intentionally also changes
    # its normalization domain when channels grow; this embedding tests ViT.
    policy4 = construct(obs4, "vit", critic_projection_channels=projection)
    policy8 = construct(obs8, "vit", history_length=8, critic_projection_channels=projection)
    state4, state8 = policy4.state_dict(), policy8.state_dict()
    changed = []
    for key, value in state4.items():
        if state8[key].shape == value.shape:
            state8[key] = value
        else:
            changed.append(key)
            # In each expanded layer the oldest four history maps are inserted
            # before the old history and knowledge inputs. The unprojected
            # critic also has a privileged-map prefix which stays in place.
            prefix = 12 if key == "critic.spatial_stem.0.weight" else 0
            expanded = torch.zeros_like(state8[key])
            expanded[:, :prefix] = value[:, :prefix]
            expanded[:, prefix + 48:] = value[:, prefix:]
            state8[key] = expanded
    assert len(changed) == 2
    policy8.load_state_dict(state8, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(policy8.actor(obs8["policy"], obs8["knowledge"]),
                                   policy4.actor(obs4["policy"], obs4["knowledge"]), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(policy8.evaluate(obs8), policy4.evaluate(obs4), rtol=2e-5, atol=2e-6)
        changed_old = obs8.clone()
        changed_old["policy"][:, :4 * 985].add_(17)
        torch.testing.assert_close(policy8.evaluate(changed_old), policy8.evaluate(obs8), rtol=0, atol=0)
        torch.testing.assert_close(policy8.action_probabilities(changed_old), policy8.action_probabilities(obs8),
                                   rtol=0, atol=0)


def test_h4_explicit_default_keeps_checkpoint_and_semantics_identical():
    obs = observations()
    torch.manual_seed(981)
    default = construct(obs, "vit")
    torch.manual_seed(981)
    explicit = construct(obs, "vit", history_length=4)
    assert default.semantics == explicit.semantics == (
        "scaled_v32_p2_9x9_hist4_knowledge11_independent_critic_v1_vit_1m_projection32_head128")
    for key, value in default.state_dict().items():
        torch.testing.assert_close(explicit.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(explicit.action_probabilities(obs), default.action_probabilities(obs), rtol=0, atol=0)
        torch.testing.assert_close(explicit.evaluate(obs), default.evaluate(obs), rtol=0, atol=0)


@pytest.mark.parametrize("opponent", ("current_self_play", "self_play"))
def test_h8_real_training_checkpoint_and_builtin_evaluation(opponent):
    torch.set_num_threads(1)
    config = json.loads((Path(__file__).parents[1] / "configs/v32-opt.json").read_text())
    config["task"].update(history_length=8, num_envs=2, backend="cpu", device="cpu",
                          opponent=opponent, max_episode_steps=3, map_pool_size=0)
    config["runner"]["policy"] = {
        "class_name": "generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy",
        "action_distribution": "categorical", "history_length": 8, "backbone": "vit",
        "model_size": "1m", "critic_projection_channels": 0}
    config["runner"].update(num_steps_per_env=2, save_interval=1)
    config["runner"]["algorithm"].update(num_learning_epochs=1, num_mini_batches=1,
                                         desired_kl=None, schedule="fixed", ppo_monitor=False)
    config["self_play"].update(pool_size=2, snapshot_interval=1, retain_seed=False)
    config["evaluation"] = {"enabled": False}
    config["training_control"] = {}
    config["curriculum"] = {}
    original = deepcopy(config)
    with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
        output = Path(directory) / "run"
        summary = train(config, output, iterations=2)
        assert summary["environment_steps"] == 8
        assert config == original
        manifest = json.loads((output / "history_pool/manifest.json").read_text())
        assert manifest["policy_spec"]["mdp"]["history_length"] == 8
        assert not manifest["retain_seed"]
        assert [entry["snapshot_id"] for entry in manifest["snapshots"]] == [1, 2]
        report = evaluate_builtin(output / "checkpoint.pt", maps_per_ai=1, opponents=("random",),
                                  max_ticks=4, device="cpu", env_backend="cpu", output=output / "eval.json")
        assert report["completed_games"] == report["overall"]["games"] == 2
        assert task_config(config["task"]).history_length == 8
