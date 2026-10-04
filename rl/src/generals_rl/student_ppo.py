"""Student-only PPO using upstream PPO.update and corrected CTS rollout logic.

CTS remains the default. This adapter gives the deployed local student task
gradients with one optimizer and no privileged latent-distillation target.
"""
from __future__ import annotations

import torch

from rsl_rl.algorithms import MoECTS, PPO
from rsl_rl.modules import ActorCriticMoECTS


class StudentPPOPolicy(ActorCriticMoECTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_distribution != "categorical":
            raise ValueError("StudentPPOPolicy currently requires categorical actions")
        self.teacher_encoder.requires_grad_(False)

    def action_probabilities(self, obs, is_teacher=False):
        if is_teacher:
            raise ValueError("student-only policy has no teacher inference branch")
        return super().action_probabilities(obs, is_teacher=False)

    def act(self, obs, is_teacher=False, **kwargs):
        # action_probabilities computes the student latent with gradients. The
        # inherited act(False) explicitly disables them and cannot be reused.
        self.action_probabilities(obs, is_teacher=is_teacher)
        return self.distribution.sample().unsqueeze(-1)

    def evaluate(self, obs, is_teacher=False, **kwargs):
        # CTS GAE evaluates an empty teacher slice before the student slice.
        # Avoid zero-batch SimNorm/Conv1d forwards; no value is needed there.
        if obs.batch_size[0] == 0:
            return obs["single_obs"].new_empty((0, 1))
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        # Upstream critic input is cat(latent.detach(), critic_obs). Its encoder
        # graph is unused; keep critic gradients and recompute the current latent.
        with torch.no_grad():
            latent, _ = self.student_moe_encoder(
                self.actor_obs_normalizer(self.get_actor_obs(obs)))
        return self.critic(torch.cat((latent.detach(), critic_obs), dim=-1))


class StudentPPO(PPO):
    """One task-gradient optimizer, all student lanes, no distillation pass."""

    requires_old_distribution = False

    @staticmethod
    def teacher_environment_count(num_envs, teacher_env_ratio=0.0):
        if num_envs < 1 or teacher_env_ratio != 0:
            raise ValueError("student PPO requires positive env count and teacher_env_ratio=0")
        return 0

    def __init__(self, policy, storage, num_envs, *, teacher_env_ratio=0.0,
                 student_encoder_learning_rate=None, load_balance_coef=0.0,
                 schedule="fixed", desired_kl=None, learning_rate=3e-4,
                 betas=(0.9, 0.999), weight_decay=0.0, **kwargs):
        self.teacher_environment_count(num_envs, teacher_env_ratio)
        if not isinstance(policy, StudentPPOPolicy):
            raise TypeError("StudentPPO requires StudentPPOPolicy")
        if storage.num_envs != num_envs or storage.teacher_num_envs != 0:
            raise ValueError("student PPO requires matching all-student storage")
        if schedule != "fixed" or desired_kl is not None:
            raise ValueError("categorical PPO requires schedule=fixed and desired_kl=None")
        if load_balance_coef != 0:
            raise ValueError("student-only PPO has no distillation/load-balance loss")
        if student_encoder_learning_rate not in (None, learning_rate):
            raise ValueError("student PPO uses one learning rate for its single optimizer")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("student-only PPO does not support RND or symmetry")
        super().__init__(policy, storage, schedule=schedule, desired_kl=desired_kl,
                         learning_rate=learning_rate, **kwargs)
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1:
            raise ValueError("PPO epoch and minibatch counts must be positive")
        self._check_sample_count()
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=learning_rate, betas=betas, weight_decay=weight_decay)
        self.optimizer_stu_enc = None
        self.teacher_num_envs = 0
        self.student_num_envs = num_envs
        self.teacher_env_idxs = torch.empty(0, dtype=torch.long, device=self.device)
        self.student_env_idxs = torch.arange(num_envs, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        if extras.get("terminal_observations") is None:
            timeouts = extras.get("time_outs")
            if timeouts is not None and bool(timeouts.any()):
                raise ValueError("timeout bootstrap requires terminal_observations")
        # Correct terminal-next-state bootstrap, normalizer updates, transition
        # recording and reset are already covered by the CTS bridge tests.
        return MoECTS.process_env_step(self, obs, rewards, dones, extras)

    def _check_sample_count(self):
        # std() uses sample variance: global normalization needs >=2 samples;
        # per-minibatch normalization additionally needs >=2 in every minibatch.
        required = max(2, self.num_mini_batches *
                       (2 if self.normalize_advantage_per_mini_batch else 1))
        if self.storage.num_envs * self.storage.num_transitions_per_env < required:
            raise ValueError(f"active PPO rollout needs at least {required} samples")

    def compute_returns(self, obs):
        # Recheck after the runner shortens the final active rollout. Its GAE
        # uses only the active prefix; no return or normalization formula is copied.
        self._check_sample_count()
        return MoECTS.compute_returns(self, obs)
