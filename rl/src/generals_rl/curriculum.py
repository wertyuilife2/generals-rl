"""Optional gamma/tick courses, applied before a rollout at its TB iteration.

Defaults do nothing. The original RewardConfig is the immutable reference;
absolute iteration reconstructs the course on resume without scheduler state.
"""
from dataclasses import dataclass, replace
import math
from numbers import Real

from .mdp.rewards import ShapedReward


@dataclass(frozen=True)
class CurriculumConfig:
    tick_decay: bool = False
    gamma_final: float | None = None
    gamma_end_iteration: int | None = None

    def __post_init__(self):
        if not isinstance(self.tick_decay, bool):
            raise TypeError("curriculum.tick_decay must be boolean")
        if (self.gamma_final is None) != (self.gamma_end_iteration is None):
            raise ValueError("gamma_final and gamma_end_iteration must be configured together")
        if self.gamma_final is not None:
            if (isinstance(self.gamma_final, bool) or not isinstance(self.gamma_final, Real)
                    or not math.isfinite(self.gamma_final) or not 0 <= self.gamma_final <= 1):
                raise ValueError("curriculum.gamma_final must be finite and in [0, 1]")
            if (isinstance(self.gamma_end_iteration, bool) or not isinstance(self.gamma_end_iteration, int)
                    or self.gamma_end_iteration < 1):
                raise ValueError("curriculum.gamma_end_iteration must be a positive integer")

    @property
    def enabled(self):
        return self.tick_decay or self.gamma_final is not None


class RolloutCurriculum:
    """Synchronize PPO/PBRS discount and scale only the original tick penalty.

Call apply(absolute_iteration) BEFORE collecting that rollout, not between its
steps or before computing returns for an already collected rollout. Gamma's
end iteration is deliberately required, not chosen by this module.
"""

    def __init__(self, settings, algorithm, reward_model):
        self.settings = settings
        self.algorithm = algorithm
        self.reward_model = reward_model
        if not settings.enabled:
            return
        if not isinstance(reward_model, ShapedReward):
            raise TypeError("enabled curriculum requires ShapedReward")
        gamma = algorithm.gamma
        if (isinstance(gamma, bool) or not isinstance(gamma, Real)
                or not math.isfinite(gamma) or not 0 <= gamma <= 1):
            raise ValueError("enabled curriculum requires finite algorithm.gamma in [0, 1]")
        self.initial_gamma = float(gamma)
        self.base_reward = reward_model.config
        if self.base_reward.potential_gamma is None or self.base_reward.potential_gamma != gamma:
            raise ValueError("enabled curriculum requires initial PPO gamma == reward potential_gamma")

    def apply(self, iteration):
        if not self.settings.enabled:
            return {}
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("curriculum iteration must be a nonnegative absolute TB iteration")
        gamma = self.initial_gamma
        if self.settings.gamma_final is not None:
            progress = min(iteration / self.settings.gamma_end_iteration, 1.0)
            gamma = (float(self.settings.gamma_final) if progress == 1.0 else
                     self.initial_gamma + (self.settings.gamma_final - self.initial_gamma) * progress)
        tick_scale = max(0.0, 1.0 - iteration / 1000) if self.settings.tick_decay else 1.0
        # Rebuild from the original coefficients, never from the last scaled
        # configuration. PBRS resource and terminal coefficients stay intact.
        self.reward_model.config = replace(self.base_reward, potential_gamma=gamma,
                                           tick=self.base_reward.tick * tick_scale)
        if self.settings.gamma_final is not None:
            self.algorithm.gamma = gamma
        return {"gamma": gamma, "tick_scale": tick_scale, "tick": self.reward_model.config.tick}
