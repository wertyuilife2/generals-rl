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
                  teacher_ratio=0.75, hidden_dim=128, expert_num=4):
    return {
        "num_steps_per_env": rollout_steps,
        "save_interval": 2000,
        "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "robogauge": {"enabled": False},
        "policy": {
            "class_name": "ActorCriticMoECTS", "action_distribution": "categorical",
            "actor_hidden_dims": [hidden_dim, hidden_dim],
            "critic_hidden_dims": [hidden_dim, hidden_dim],
            "teacher_encoder_hidden_dims": [hidden_dim, hidden_dim],
            "student_encoder_hidden_dims": [hidden_dim, hidden_dim],
            "latent_dim": 32, "expert_num": expert_num,
            "actor_obs_normalization": False, "critic_obs_normalization": False,
        },
        "algorithm": {
            "class_name": "MoECTS", "teacher_env_ratio": teacher_ratio,
            "num_learning_epochs": learning_epochs, "num_mini_batches": mini_batches,
            "learning_rate": 3e-4, "student_encoder_learning_rate": 3e-4,
            "schedule": "adaptive", "desired_kl": 0.02,
            "gamma": 0.99, "lam": 0.95, "entropy_coef": 0.01,
            "load_balance_coef": 0.01, "rnd_cfg": None,
        },
    }
