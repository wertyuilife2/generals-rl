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
        result = self.task.step(actions, capture_all_terminals=False)
        log = {key: value for key, value in result.metrics.items() if value.numel()}
        observations = self._tensor_dict(result.observations)
        # RSL reads this field only for time_outs. Without timeouts, alias the
        # next observation instead of constructing/retaining unused snapshots.
        time_outs = (result.truncated if self.task.config.time_limit_mode == 'bootstrap'
                     else result.truncated.new_zeros(result.truncated.shape))
        extras = {"time_outs": time_outs,
                  "has_bootstrap_timeouts": result.terminal_observations is not None,
                  "terminal_observations": (observations if result.terminal_observations is None
                                            else self._tensor_dict(result.terminal_observations)),
                  "log": log}
        return (observations, result.reward,
                result.terminated | result.truncated, extras)
