"""Synthetic training checkpoints and self-contained deployment bundles."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from generals_rl.config import task_config
from generals_rl.export import export_policy, prepare_policy_source
from generals_rl.task import GeneralsTask
from rl_test_utils import small_training_config


def make_checkpoint(directory, seed=7, filename="checkpoint.pt", family="information", history_length=4):
    config = small_training_config(task_overrides={"opponent": "wait", "persistent_knowledge": True,
                                                   "max_episode_steps": 32, "history_length": history_length})
    config["runner"]["obs_groups"] = {"policy": ["policy", "knowledge"],
                                      "critic": ["critic", "policy", "knowledge"]}
    options = config["runner"]["policy"]
    options.update(history_length=history_length, knowledge_channels=11, critic_actor_information=True)
    if family == "optimized":
        options["class_name"] = "generals_rl.optimized_ppo.OptimizedInformationSpatialPPOPolicy"
    elif family == "scaled":
        options.pop("spatial_channels")
        options.update(class_name="generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy",
                       backbone="vit", model_size="1m", spatial_mode="dense", critic_projection_channels=0)
    policy_options = deepcopy(options)
    policy_type = resolve_callable(policy_options.pop("class_name"))
    with torch.random.fork_rng(devices=[]), GeneralsTask(task_config(config["task"])) as task:
        torch.manual_seed(seed)
        inputs = TensorDict(task.get_observations(), [task.num_envs])
        policy = policy_type(inputs, config["runner"]["obs_groups"], 649, **policy_options)
    checkpoint = directory / filename
    torch.save({"model_state_dict": policy.state_dict()}, checkpoint)
    (directory / "run_config.json").write_text(json.dumps(config))
    return checkpoint, config


def make_submission(directory, *, seed=7, codename='test-ViT', updates=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, training = make_checkpoint(directory, seed=seed, family='scaled')
    training['model'] = {'codename': codename}
    source = prepare_policy_source(training, directory)
    bundle = export_policy(checkpoint, directory/'bundle', run_config=training,
                           source=source['source_path'], completed_updates=updates)
    return bundle, training
