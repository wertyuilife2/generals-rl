"""Small fixtures shared by the training integration tests."""
from dataclasses import asdict

import torch

from generals_rl.cts import runner_config
from generals_rl.task import TaskConfig


GROUPS = {"policy": ["policy"], "critic": ["critic"]}
MAP_GROUPS = {"policy": ["policy", "knowledge"], "critic": ["critic", "policy", "knowledge"]}


def light_policy_config():
    return {"class_name": "generals_rl.spatial_ppo.LightSpatialPPOPolicy",
            "action_distribution": "categorical", "spatial_channels": 16,
            "history_length": 4, "players": 2, "map_width": 9, "map_height": 9,
            "map_planes": 12, "public_dim": 13,
            "actor_obs_normalization": False, "critic_obs_normalization": False}


def spatial_observations(count=4, *, width=9, height=9, history_length=4, memory=False):
    from tensordict import TensorDict
    from generals_rl.mdp.layout import ObservationLayout
    layout = ObservationLayout(width, height, history_length=history_length,
                               knowledge_channels=11 if memory else 0)
    frames = torch.randn(count, history_length, layout.frame_dim)
    maps = frames[..., :12 * layout.cells].reshape(count, history_length, 12, layout.cells)
    maps[:, :, 4:6].zero_()
    maps[:, :, 4:6, width + 1] = 1
    values = {"policy": frames.flatten(1), "single_obs": frames[:, -1].clone(),
              "critic": frames[:, -1].clone(),
              "action_mask": torch.ones(count, layout.num_actions, dtype=torch.bool)}
    if memory:
        values["knowledge"] = torch.randn(count, layout.knowledge_dim)
    return TensorDict(values, [count])


def small_training_config(*, task_overrides=None, runner_overrides=None):
    """Bounded CPU fixture for the supported v32 information-policy lifecycle."""
    task = dict(width=9, height=9, num_envs=4, max_episode_steps=3,
                history_length=4, opponent="self_play", seed=19,
                mountain_density=0, city_density=0, threads=1,
                combat_features=True, king_threat_distance=True)
    task.update(task_overrides or {})
    options = dict(rollout_steps=4, learning_epochs=2, mini_batches=2)
    options.update(runner_overrides or {})
    runner = runner_config(**options)
    runner["obs_groups"] = GROUPS.copy()
    runner.pop("storage_obs_keys", None)
    runner["policy"] = light_policy_config()
    runner["policy"].update(class_name="generals_rl.information_ppo.InformationSpatialPPOPolicy",
                            own_king_geometry=True, wait_move_factorization=True)
    runner["algorithm"].update(class_name="generals_rl.student_ppo.StudentPPO",
                                teacher_env_ratio=0, load_balance_coef=0,
                                schedule="fixed", desired_kl=None)
    runner["algorithm"].pop("student_encoder_learning_rate", None)
    return {"task": asdict(TaskConfig(**task)), "runner": runner, "torch_threads": 1}


def assert_nested_equal(test, actual, expected):
    """Compare full optimizer/model states, including non-tensor metadata."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_nested_equal(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            assert_nested_equal(test, a, e)
    else:
        test.assertEqual(actual, expected)
