"""Student-only PPO using upstream PPO losses and corrected CTS rollout logic.

CTS remains the default. This adapter gives the deployed local student task
gradients with one optimizer and no privileged latent-distillation target.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import torch
from torch.distributions import Categorical

from rsl_rl.algorithms import MoECTS, PPO
from rsl_rl.modules import ActorCriticMoECTS

from .ppo_monitor import PPOUpdateMonitor


class StudentPPOPolicy(ActorCriticMoECTS):
    def __init__(self, *args, value_potential_offset=False, **kwargs):
        if not isinstance(value_potential_offset, bool):
            raise TypeError("value_potential_offset must be boolean")
        super().__init__(*args, **kwargs)
        self.value_potential_offset = value_potential_offset
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
        potential = None
        if self.value_potential_offset:
            potential = obs.get("critic_potential")
            if potential is None or potential.shape != (obs.batch_size[0], 1):
                raise ValueError("value_potential_offset requires obs['critic_potential'] with shape [N, 1]")
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        # Upstream critic input is cat(latent.detach(), critic_obs). Its encoder
        # graph is unused; keep critic gradients and recompute the current latent.
        with torch.no_grad():
            latent, _ = self.student_moe_encoder(
                self.actor_obs_normalizer(self.get_actor_obs(obs)))
        value = self.critic(torch.cat((latent.detach(), critic_obs), dim=-1))
        return value if potential is None else value - potential.detach()


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
                 schedule="adaptive", desired_kl=0.02, learning_rate=3e-4,
                 ppo_monitor=True, ppo_probe_samples=8192, ppo_probe_batch_size=1024,
                 physical_minibatch_size=None, amp_bfloat16=False,
                 betas=(0.9, 0.999), weight_decay=0.0, **kwargs):
        self.teacher_environment_count(num_envs, teacher_env_ratio)
        if not isinstance(policy, StudentPPOPolicy):
            raise TypeError("StudentPPO requires StudentPPOPolicy")
        if storage.num_envs != num_envs or storage.teacher_num_envs != 0:
            raise ValueError("student PPO requires matching all-student storage")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("PPO schedule must be fixed or adaptive")
        if (desired_kl is not None and (isinstance(desired_kl, bool) or not isinstance(desired_kl, (int, float))
                                      or not math.isfinite(desired_kl) or desired_kl <= 0)
                or schedule == "adaptive" and desired_kl is None):
            raise ValueError("adaptive PPO requires a positive finite desired_kl")
        if load_balance_coef != 0:
            raise ValueError("student-only PPO has no distillation/load-balance loss")
        if student_encoder_learning_rate not in (None, learning_rate):
            raise ValueError("student PPO uses one learning rate for its single optimizer")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("student-only PPO does not support RND or symmetry")
        if (physical_minibatch_size is not None
                and (isinstance(physical_minibatch_size, bool)
                     or not isinstance(physical_minibatch_size, int)
                     or physical_minibatch_size < 1)):
            raise ValueError("physical_minibatch_size must be a positive integer or None")
        if not isinstance(amp_bfloat16, bool):
            raise TypeError("amp_bfloat16 must be boolean")
        # Map policies also autocast rollout/inference forwards. Updating a
        # float32 rollout in bf16 (or vice versa) introduces numerical policy
        # drift before any optimizer step. Legacy policies have no such flag.
        if hasattr(policy, "amp_bfloat16") and policy.amp_bfloat16 != amp_bfloat16:
            raise ValueError("policy and StudentPPO amp_bfloat16 must match for rollout/update precision")
        self.physical_minibatch_size = physical_minibatch_size
        self.amp_bfloat16 = amp_bfloat16
        self._chunked_update = physical_minibatch_size is not None or amp_bfloat16
        if self._chunked_update:
            if (getattr(policy, "action_distribution", None) != "categorical"
                    or policy.is_recurrent):
                raise ValueError("microbatch/AMP requires feedforward categorical student PPO")
            if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                   or isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0
                   or type(module).__name__ == "DropPath" and getattr(module, "drop_prob", 0) > 0
                   for module in policy.modules()):
                raise ValueError("microbatch/AMP requires sample-independent deterministic network layers")
            if amp_bfloat16:
                device = torch.device(kwargs.get("device", "cpu"))
                if device.type != "cuda" or not torch.cuda.is_bf16_supported():
                    raise ValueError("amp_bfloat16 requires a CUDA device with bfloat16 support")
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
        self.monitor = PPOUpdateMonitor(enabled=ppo_monitor, probe_samples=ppo_probe_samples,
                                        probe_batch_size=ppo_probe_batch_size)
        self.update_metrics = {}

    def _on_ppo_minibatch(self, log_ratio, entropy, grad_norm):
        self.monitor.minibatch(self, log_ratio, entropy, grad_norm)

    def update(self):
        self.update_metrics = {}
        try:
            self.monitor.begin(self)
            # The disabled path remains upstream PPO.update, including its RNG
            # draws, reduction order and observer lifecycle.
            # Distributed minibatch normalization uses the same adapter with
            # one physical chunk when memory splitting is disabled. Upstream
            # PPO would otherwise normalize each rank's advantages locally.
            use_adapter = self._chunked_update or (self.is_multi_gpu and self.normalize_advantage_per_mini_batch)
            losses = self._update_microbatched() if use_adapter else super().update()
            self.update_metrics = self.monitor.finish(self)
            if self._chunked_update:
                self.update_metrics.update(self.microbatch_metrics)
            return losses
        finally:
            self.monitor.release()

    def _update_microbatched(self):
        """Memory-bounded execution of existing logical categorical PPO batches.

        Losses/adaptive-KL mirror vendored algorithms/ppo.py:246-317; gradient
        clipping/step and observer mirror :367-409. This adapter supports only
        the StudentPPO subset already excluding RND, symmetry and recurrence.
        Gradients are mathematically identical in float32, with normal floating
        point differences from chunked kernels and reduction order.
        """
        mean_losses = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
        physical_batches = 0
        logical_batch_size = 0
        mask_validator = getattr(self.policy, "validate_action_mask", None)
        for (obs, actions, old_values, advantages, returns, old_log_probs,
             _old_mean, _old_std, _hidden, _masks) in self.storage.mini_batch_generator(
                 self.num_mini_batches, self.num_learning_epochs):
            count = obs.batch_size[0]
            logical_batch_size = count
            if self.normalize_advantage_per_mini_batch:
                # Normalize the full logical minibatch, never each physical
                # chunk, including all ranks in a distributed logical batch.
                with torch.no_grad():
                    advantages = self._normalize_advantages(advantages)
            chunk_size = self.physical_minibatch_size or count
            if mask_validator is not None:
                # The full logical batch must be valid before any physical
                # forward/backward; public inference keeps its own validation.
                mask_validator(obs)
            self.optimizer.zero_grad()
            batch_losses = {key: returns.new_zeros(()) for key in mean_losses}
            log_ratios, entropies, probabilities = [], [], []
            for start in range(0, count, chunk_size):
                end = min(count, start + chunk_size)
                sl = slice(start, end)
                context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if self.amp_bfloat16 else nullcontext())
                with context:
                    # Unlike act(), probability evaluation consumes no policy
                    # sampling RNG. One full-batch draw below preserves the
                    # upstream update's otherwise-unused sample call.
                    if mask_validator is None:
                        self.policy.action_probabilities(obs[sl])
                    else:
                        self.policy.action_probabilities(obs[sl], validate_mask=False)
                    values = self.policy.evaluate(obs[sl])
                if self.policy.distribution.logits.dtype != torch.float32:
                    self.policy.distribution = Categorical(
                        logits=self.policy.distribution.logits.float(), validate_args=False)
                action_log_probs = self.policy.get_actions_log_prob(actions[sl]).float()
                entropy = self.policy.entropy.float()
                log_ratio = action_log_probs - old_log_probs[sl].squeeze(-1).float()
                ratio = torch.exp(log_ratio)
                advantage = advantages[sl].squeeze(-1).float()
                surrogate = -advantage * ratio
                clipped_surrogate = -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.maximum(surrogate, clipped_surrogate).mean()
                values = values.float()
                target, old_value = returns[sl].float(), old_values[sl].float()
                value_losses = (values - target).pow(2)
                if self.use_clipped_value_loss:
                    clipped_value = old_value + (values - old_value).clamp(-self.clip_param, self.clip_param)
                    value_losses = torch.maximum(value_losses, (clipped_value - target).pow(2))
                value_loss = value_losses.mean()
                entropy_loss = entropy.mean()
                weight = (end - start) / count
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                (loss * weight).backward()
                for key, value in (("value", value_loss), ("surrogate", surrogate_loss),
                                   ("entropy", entropy_loss)):
                    batch_losses[key] += value.detach() * weight
                log_ratios.append(log_ratio.detach())
                entropies.append(entropy.detach())
                probabilities.append(self.policy.distribution.probs.detach())
                physical_batches += 1
            # Categorical.sample() uses precisely this one multinomial call on
            # its flattened probabilities. Sampling each chunk changes RNG
            # consumption and the subsequent rollout/mini-batch plan.
            torch.multinomial(torch.cat(probabilities), 1, replacement=True)
            log_ratio = torch.cat(log_ratios)
            entropy = torch.cat(entropies)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = (torch.expm1(log_ratio) - log_ratio).clamp_min(0).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            # bf16 keeps float32 master weights/gradients and needs no fp16
            # scaler. Clip and Adam step only after the complete logical batch.
            if self.is_multi_gpu:
                self.reduce_parameters()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._on_ppo_minibatch(log_ratio, entropy, grad_norm.detach())
            for key in mean_losses:
                mean_losses[key] += batch_losses[key].item()
        updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.microbatch_metrics = {
            "physical_minibatch_size": float(self.physical_minibatch_size or logical_batch_size),
            "logical_minibatch_size": float(logical_batch_size),
            "microbatches": float(physical_batches), "amp_bfloat16": float(self.amp_bfloat16),
        }
        return {key: value / updates for key, value in mean_losses.items()}

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
        if not self.is_multi_gpu or self.normalize_advantage_per_mini_batch:
            return MoECTS.compute_returns(self, obs)
        # Let CTS compute unchanged GAE, then normalize the active advantages
        # using moments from every rank rather than separate local populations.
        self.normalize_advantage_per_mini_batch = True
        try:
            MoECTS.compute_returns(self, obs)
        finally:
            self.normalize_advantage_per_mini_batch = False
        self.storage.advantages = self._normalize_advantages(self.storage.advantages)

    @torch.no_grad()
    def _normalize_advantages(self, advantages):
        if not self.is_multi_gpu:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Float64 moments avoid cancellation from summing large rollouts. Count
        # is reduced too, so the variance uses the global unbiased denominator.
        values = advantages.double()
        moments = torch.stack((values.sum(), values.square().sum(), values.new_tensor(values.numel())))
        torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
        total, squares, count = moments.unbind()
        mean = total / count
        variance = ((squares - total * mean) / (count - 1)).clamp_min(0)
        return (advantages - mean.to(advantages.dtype)) / (variance.sqrt().to(advantages.dtype) + 1e-8)
