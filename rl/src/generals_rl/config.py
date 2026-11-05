"""Shared configuration and seeding helpers for training and evaluation."""
from copy import deepcopy
from importlib.resources import files
import json
import random

import numpy as np
import torch

from .mdp import RewardConfig
from .task import TaskConfig


DEFAULT_ITERATIONS = 224


def default_training_config():
    """Load the selected F03 recipe from package data as a fresh dictionary."""
    return json.loads(files("generals_rl").joinpath("default.json").read_text())


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def task_config(values):
    values = deepcopy(values)
    if isinstance(values.get("reward"), dict):
        values["reward"] = RewardConfig(**values["reward"])
    return TaskConfig(**values)


def configure_observations(config):
    """Resolve the scaled policy's single privilege switch into task/storage."""
    config = deepcopy(config)
    runner = config['runner']
    policy = runner['policy']
    if policy['class_name'] == 'generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy':
        privileged = policy.get('critic_privileged', True)
        if type(privileged) is not bool:
            raise TypeError('critic_privileged must be boolean')
        config['task']['critic_privileged'] = privileged
        local = ['policy', 'knowledge', 'valid_cells']
        runner['obs_groups'] = {'policy': local, 'critic': (['critic'] if privileged else []) + local}
        keys = runner.get('storage_obs_keys')
        if keys is not None:
            keys = [key for key in keys if key != 'critic']
            runner['storage_obs_keys'] = keys + (['critic'] if privileged else [])
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
