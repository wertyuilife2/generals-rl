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
    """Load temporal-paths v32 with terminal draws as a fresh dictionary."""
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
    if isinstance(values.get('map_curriculum'), dict):
        from .map_curriculum import MapCurriculumConfig
        values['map_curriculum'] = MapCurriculumConfig(**values['map_curriculum'])
    return TaskConfig(**values)


def configure_observations(config):
    """Resolve feature recipes and privilege into matching task/model/storage.

    Recipes own history and memory widths. Built-in codecs follow the recipe;
    explicit custom codecs or disabled compression/history remain caller-owned.
    """
    config = deepcopy(config)
    runner = config['runner']
    policy = runner['policy']
    if policy['class_name'] == 'generals_rl.scaled_ppo.ScaledInformationSpatialPPOPolicy':
        from .policy.features import feature_recipe
        feature_set = config['task'].get('feature_set', 'stacked')
        recipe = feature_recipe(feature_set)
        policy['feature_set'] = feature_set
        config['task']['history_length'] = policy['history_length'] = recipe.history_length
        policy['knowledge_channels'] = recipe.knowledge_channels
        if 'storage_history' not in runner:
            runner['storage_history'] = {'policy': recipe.history_length}
        elif runner['storage_history'] and 'policy' in runner['storage_history']:
            runner['storage_history']['policy'] = recipe.history_length
        codecs = ('generals_rl.rollout_codec.GeneralsObservationCodec',
                  'generals_rl.rollout_codec.CompactObservationCodec',
                  'generals_rl.rollout_codec.DynamicObservationCodec',
                  'generals_rl.rollout_codec.PathObservationCodec')
        if 'storage_codec' not in runner or runner['storage_codec'] in codecs:
            runner['storage_codec'] = codecs[3 if recipe.terrain_distances else
                                             2 if recipe.dynamic_memory else int(recipe.half_precision)]
        if policy.get('value_distribution', 'scalar') != 'scalar':
            reward = config['task']['reward']
            if any(reward.get(k, 0) != 0 for k in ('territory', 'cities', 'army', 'tick')) or (reward.get('win', 1), reward.get('loss', -1)) != (1, -1):
                raise ValueError('bounded value heads require pure terminal +/-1 rewards')
            if any(config.get('curriculum', {}).get(k) for k in ('tick_decay', 'potential_decay', 'gamma_final')):
                raise ValueError('bounded value experiments require fixed sparse rewards/discount')
            if runner['algorithm'].get('use_clipped_value_loss', True):
                raise ValueError('classification values replace clipped value regression; set use_clipped_value_loss=False')
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
