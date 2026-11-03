"""The only training-environment bridge that imports RSL/TensorDict."""
from tensordict import TensorDict

from rsl_rl.env import VecEnv


class GeneralsCTSVecEnv(VecEnv):
    def __init__(self, task):
        self.task = task
        self.device = task.device
        self.num_envs = task.num_envs
        self.num_actions = task.action_space.size
        self.max_episode_length = task.config.max_episode_steps
        self.episode_length_buf = task.episode_length_buf
        self.cfg = task.cfg

    def _tensor_dict(self, values):
        return TensorDict(values, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        return self._tensor_dict(self.task.get_observations())

    def step(self, actions):
        result = self.task.step(actions)
        log = {key: value for key, value in result.metrics.items() if value.numel()}
        extras = {"time_outs": result.truncated,
                  "terminal_observations": self._tensor_dict(result.terminal_observations),
                  "log": log}
        return (self._tensor_dict(result.observations), result.reward,
                result.terminated | result.truncated, extras)


def runner_config(*, rollout_steps=32, learning_epochs=4, mini_batches=4,
                  spatial_channels=16):
    """LightCNN student runner for official maps, H4 combat history and K11."""
    return {
        "num_steps_per_env": rollout_steps,
        "save_interval": 2000,
        "logger": "tensorboard",
        "obs_groups": {"policy": ["policy", "knowledge"],
                       "critic": ["critic", "policy", "knowledge"]},
        "robogauge": {"enabled": False},
        "policy": {
            "class_name": "generals_rl.optimized_ppo.OptimizedInformationSpatialPPOPolicy",
            "action_distribution": "categorical", "spatial_channels": spatial_channels,
            "history_length": 4, "players": 2, "map_width": 23, "map_height": 23,
            "map_planes": 14, "public_dim": 13, "knowledge_channels": 11,
            "actor_obs_normalization": False, "critic_obs_normalization": False,
            "own_king_geometry": True, "wait_move_factorization": True,
            "critic_actor_information": True, "input_channel_alignment": 4,
            "critic_projection_channels": 0, "share_history_unpack": True,
        },
        "algorithm": {
            "class_name": "generals_rl.student_ppo.StudentPPO", "teacher_env_ratio": 0,
            "num_learning_epochs": learning_epochs, "num_mini_batches": mini_batches,
            "learning_rate": 1e-4,
            "schedule": "adaptive", "desired_kl": 0.01,
            "gamma": 0.995, "lam": 0.95, "entropy_coef": 0.01,
            "rnd_cfg": None,
        },
        "storage_obs_keys": ["policy", "critic", "action_mask", "knowledge"],
    }
