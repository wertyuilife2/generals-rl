"""Feedforward categorical actor/critic lifecycle used by PPO training."""
from torch import nn


class CategoricalActorCritic(nn.Module):
    is_recurrent = False
    actor_obs_normalization = False
    critic_obs_normalization = False

    def action_probabilities(self, obs, is_teacher=False):
        raise NotImplementedError

    def evaluate(self, obs, is_teacher=False, **kwargs):
        raise NotImplementedError

    def act(self, obs, is_teacher=False, **kwargs):
        self.action_probabilities(obs, is_teacher=is_teacher)
        return self.distribution.sample().unsqueeze(-1)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.probs

    @property
    def action_std(self):
        # PPO's logger expects one scale value even for categorical actions.
        return self.distribution.logits.new_ones((*self.distribution.batch_shape, 1))

    @property
    def entropy(self):
        return self.distribution.entropy()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions.long().squeeze(-1))

    def update_normalization(self, obs):
        # RSL calls this hook; features already use fixed game-specific scales.
        pass

    def load_state_dict(self, state_dict, strict=True):
        # RSL's runner uses this boolean to restore optimizers and iteration.
        super().load_state_dict(state_dict, strict=strict)
        return True
