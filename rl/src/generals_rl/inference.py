"""Shared observation-only inputs and retained-checkpoint loading."""
from copy import deepcopy

import torch
from tensordict import TensorDict
from rsl_rl.utils import resolve_callable


def local_inputs(batch, player_ids, encoder, history, action_space, knowledge=None, *, current=None):
    # No privileged critic key is ever passed into the deployed student.
    observations = {
        "policy": history.flattened(),
        "single_obs": encoder.encode(batch, player_ids) if current is None else current,
        "action_mask": action_space.mask(batch, player_ids),
    }
    if knowledge is not None:
        observations["knowledge"] = knowledge.encode(batch.state["tick"], player_ids).flatten(1)
    return TensorDict(observations, batch_size=[batch.num_envs], device=batch.device)


def load_student(checkpoint, config, example_obs, device):
    policy_config = deepcopy(config["runner"]["policy"])
    class_name = policy_config.pop("class_name")
    supported = {
        "generals_rl.information_ppo.InformationSpatialPPOPolicy",
        "generals_rl.optimized_ppo.OptimizedInformationSpatialPPOPolicy",
        "generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy",
    }
    if class_name not in supported:
        raise ValueError("checkpoint must use a v32, v32-opt or F03 policy")
    policy_class = resolve_callable(class_name)
    if policy_config.get("action_distribution") != "categorical":
        raise ValueError("Generals built-in evaluation requires categorical encoded actions")
    # Construction only inspects feature dimensions. Use an all-zero placeholder
    # for the critic shape; no true privileged map is needed to load the student.
    shape_obs = example_obs.clone()
    shape_obs["critic"] = torch.zeros_like(example_obs["single_obs"])
    policy = policy_class(
        shape_obs, deepcopy(config["runner"]["obs_groups"]),
        example_obs["action_mask"].shape[-1], **policy_config,
    ).to(device)
    # Do not allocate a second full checkpoint on the GPU during loading.
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    policy.load_state_dict(saved["model_state_dict"])
    policy.eval()
    return policy

