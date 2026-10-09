"""Shared local-student inputs and checkpoint loading for evaluation adapters."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticMoECTS


def local_inputs(batch, player_ids, encoder, history, action_space):
    # No privileged critic key is ever passed into the deployed student.
    return TensorDict({
        "policy": history.flattened(),
        "single_obs": encoder.encode(batch, player_ids),
        "action_mask": action_space.mask(batch, player_ids),
    }, batch_size=[batch.num_envs], device=batch.device)


def load_student(checkpoint, config, example_obs, device):
    policy_config = deepcopy(config["runner"]["policy"])
    class_name = policy_config.pop("class_name")
    if class_name not in ("ActorCriticMoECTS", "rsl_rl.modules.ActorCriticMoECTS"):
        raise ValueError("built-in evaluation currently requires ActorCriticMoECTS")
    if policy_config.get("action_distribution") != "categorical":
        raise ValueError("Generals built-in evaluation requires categorical encoded actions")
    # Construction only inspects feature dimensions. Use an all-zero placeholder
    # for the critic shape; no true privileged map is needed to load the student.
    shape_obs = example_obs.clone()
    shape_obs["critic"] = torch.zeros_like(example_obs["single_obs"])
    # The vendored constructor prints entire networks; evaluation reports only
    # match progress and results, keeping its CLI usable with many versions.
    with redirect_stdout(StringIO()):
        policy = ActorCriticMoECTS(
            shape_obs, deepcopy(config["runner"]["obs_groups"]),
            example_obs["action_mask"].shape[-1], **policy_config,
        ).to(device)
    saved = torch.load(checkpoint, weights_only=False, map_location=device)
    policy.load_state_dict(saved["model_state_dict"])
    policy.eval()
    return policy
