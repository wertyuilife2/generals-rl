"""Student-only PPO using upstream PPO losses and corrected CTS rollout logic.

Current spatial students use one task-gradient optimizer and no privileged
latent-distillation target. CTS contributes the shared rollout lifecycle.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import torch
from torch.distributions import Categorical

from rsl_rl.algorithms import MoECTS, PPO

from .actor_critic import CategoricalActorCritic
from .ppo_monitor import PPOUpdateMonitor


class StudentPPO(PPO):
    """One task-gradient optimizer, all student lanes, no distillation pass."""

    requires_old_distribution = False

    @staticmethod
    def teacher_environment_count(num_envs, teacher_env_ratio=0.0):
        if num_envs < 1 or teacher_env_ratio != 0:
            raise ValueError("student PPO requires positive env count and teacher_env_ratio=0")
        return 0

    def __init__(self, policy, storage, num_envs, *, teacher_env_ratio=0.0,
                 schedule="adaptive", desired_kl=0.02, learning_rate=3e-4,
                 ppo_monitor=True, ppo_probe_samples=8192, ppo_probe_batch_size=1024,
                 physical_minibatch_size=None, amp_bfloat16=False,
                 adaptive_lr_min=1e-5, adaptive_lr_max=1e-2,
                 betas=(0.9, 0.999), weight_decay=0.0, advantage_top_fraction=1.0,
                 pair_minibatches=False, **kwargs):
        self.teacher_environment_count(num_envs, teacher_env_ratio)
        if not isinstance(policy, CategoricalActorCritic):
            raise TypeError("StudentPPO requires CategoricalActorCritic")
        if storage.num_envs != num_envs or storage.teacher_num_envs != 0:
            raise ValueError("student PPO requires matching all-student storage")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("PPO schedule must be fixed or adaptive")
        if (desired_kl is not None and (isinstance(desired_kl, bool) or not isinstance(desired_kl, (int, float))
                                      or not math.isfinite(desired_kl) or desired_kl <= 0)
                or schedule == "adaptive" and desired_kl is None):
            raise ValueError("adaptive PPO requires a positive finite desired_kl")
        for name, value in (("adaptive_lr_min", adaptive_lr_min),
                            ("adaptive_lr_max", adaptive_lr_max)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if adaptive_lr_min > adaptive_lr_max:
            raise ValueError("adaptive_lr_min must not exceed adaptive_lr_max")
        self.adaptive_lr_min, self.adaptive_lr_max = adaptive_lr_min, adaptive_lr_max
        self._custom_learning_rates = (adaptive_lr_min, adaptive_lr_max) != (1e-5, 1e-2)
        if self._custom_learning_rates and policy.is_recurrent:
            raise ValueError("custom adaptive LR bounds require feedforward student PPO")
        if (self._custom_learning_rates and schedule == "adaptive"
                and not adaptive_lr_min <= learning_rate <= adaptive_lr_max):
            raise ValueError("initial learning_rate must be within adaptive LR bounds")
        if kwargs.get("rnd_cfg") is not None or kwargs.get("symmetry_cfg") is not None:
            raise ValueError("student-only PPO does not support RND or symmetry")
        if (isinstance(advantage_top_fraction, bool) or not isinstance(advantage_top_fraction, (int, float))
                or not math.isfinite(advantage_top_fraction) or not 0 < advantage_top_fraction <= 1):
            raise ValueError("advantage_top_fraction must be finite and in (0, 1]")
        self.advantage_top_fraction = float(advantage_top_fraction)
        if type(pair_minibatches) is not bool:
            raise TypeError('pair_minibatches must be boolean')
        if pair_minibatches and (num_envs % 2 or policy.is_recurrent or advantage_top_fraction != 1):
            raise ValueError('paired minibatches require even feedforward lanes and all samples')
        if getattr(policy, 'joint_critic', False) and not pair_minibatches:
            raise ValueError('joint critic requires paired minibatches')
        self.pair_minibatches = pair_minibatches
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
        if pair_minibatches and physical_minibatch_size is not None and physical_minibatch_size % 2:
            raise ValueError('paired physical minibatches must contain complete two-seat games')
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
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        actor_ids = {id(p) for p in policy.actor.parameters()}
        critic_ids = {id(p) for p in policy.critic.parameters()}
        # Separate forwards can release actor activations before building the
        # critic graph, even when both branches accumulate into shared weights.
        # Policies opt in only when this avoids duplicating a shared backbone
        # forward. Both losses still precede one reduction, clip and Adam step.
        self._separate_backward = (
            getattr(policy, 'separate_actor_critic_backward', False)
            or (not getattr(policy, 'share_backbone', False)
                and not actor_ids.intersection(critic_ids)
                and {id(p) for p in parameters} <= actor_ids | critic_ids))
        # Keep the default single group, including parameter ordering and Adam
        # execution, identical to the original implementation.
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=betas, weight_decay=weight_decay)
        self.optimizer_stu_enc = None
        self.teacher_num_envs = 0
        self.student_num_envs = num_envs
        self.teacher_env_idxs = torch.empty(0, dtype=torch.long, device=self.device)
        self.student_env_idxs = torch.arange(num_envs, device=self.device)
        self.monitor = PPOUpdateMonitor(enabled=ppo_monitor, probe_samples=ppo_probe_samples,
                                        probe_batch_size=ppo_probe_batch_size)
        self.update_metrics = {}
        self.value_distribution = getattr(policy, "value_distribution", None)
        self._classification_value = self.value_distribution is not None and self.value_distribution.kind != "scalar"
        if self._classification_value and self.use_clipped_value_loss:
            raise ValueError("classification values require use_clipped_value_loss=False")

    def restore_learning_rates(self):
        """Restore the shared scheduler scalar from a single-group Adam state."""
        groups = self.optimizer.param_groups
        if len(groups) != 1 or groups[0].get("role") is not None:
            raise ValueError("resume requires one shared actor/critic optimizer group; use warm_start for retired LR groups")
        self.learning_rate = groups[0]["lr"]
        if (self._custom_learning_rates and self.schedule == "adaptive"
                and not self.adaptive_lr_min <= self.learning_rate <= self.adaptive_lr_max):
            raise ValueError("checkpoint actor LR is outside configured adaptive LR bounds")

    def act(self, obs):
        # Keep canonical student rows and obtain logits/value together. The
        # shared policy encodes each row once; CTS teacher gathers are avoided.
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        _, values = self.policy.action_probabilities_and_value(obs)
        self.transition.actions = self.policy.distribution.sample().unsqueeze(-1).detach()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def _adapt_learning_rate(self, kl_mean):
        """Actor KL drives the shared actor/critic learning rate."""
        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
        if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(self.adaptive_lr_min, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(self.adaptive_lr_max, self.learning_rate * 1.5)
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
        if self._custom_learning_rates:
            # The float32 broadcast can round an exact endpoint just outside
            # its bounds; enforce the same configured endpoints on every rank.
            self.learning_rate = min(self.adaptive_lr_max, max(self.adaptive_lr_min, self.learning_rate))
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def _on_ppo_minibatch(self, log_ratio, entropy, grad_norm):
        self.monitor.minibatch(self, log_ratio, entropy, grad_norm)

    def update(self):
        self.update_metrics = {}
        try:
            self.monitor.begin(self)
            indices, selection_metrics = self._advantage_sample_indices()
            # Independent, unchunked execution retains upstream PPO.update.
            # Shared encoders need the explicit actor/critic update adapter.
            # Distributed minibatch normalization uses the same adapter with
            # one physical chunk when memory splitting is disabled. Upstream
            # PPO would otherwise normalize each rank's advantages locally.
            use_adapter = (self._chunked_update or self._custom_learning_rates
                           or getattr(self.policy, 'share_backbone', False) or self._classification_value
                           or indices is not None
                           or (self.is_multi_gpu and self.normalize_advantage_per_mini_batch))
            losses = self._update_microbatched(indices=indices) if use_adapter else super().update()
            self.update_metrics = self.monitor.finish(self)
            self.update_metrics.update(selection_metrics)
            self.update_metrics.update(getattr(self, "value_metrics", {}))
            self.update_metrics.update(getattr(self, "pair_metrics", {}))
            if self.monitor.enabled:
                self.update_metrics.update(actor_learning_rate=self.learning_rate,
                                           critic_learning_rate=self.learning_rate)
            if self._chunked_update:
                self.update_metrics.update(self.microbatch_metrics)
            return losses
        finally:
            self.monitor.release()

    @torch.no_grad()
    def _advantage_sample_indices(self):
        """AverageJoe-style rank-local top-|global-standardized A| selection.

        Keep the configured number of minibatches/epochs. Thus filtering
        reduces their sample count, not the number of optimizer steps.
        """
        if self.pair_minibatches:
            steps = self.storage.num_transitions_per_env
            games = torch.randperm(steps * self.storage.num_envs // 2, device=self.device)
            # CTS indexes env-major, while adjacent returned samples must be
            # (same tick/game seat0, same tick/game seat1).
            first = (games // steps) * (2 * steps) + games % steps
            indices = torch.stack((first, first + steps), -1).flatten()
            return (indices.new_empty(0), indices), {'paired_minibatches': 1.0}
        if self.advantage_top_fraction == 1:
            return None, {}
        storage = self.storage
        steps, envs = storage.num_transitions_per_env, storage.num_envs
        advantages = storage.advantages[:steps]
        standardized = self._normalize_advantages(advantages)
        scores = standardized.transpose(0, 1).flatten()
        keep = int(scores.numel() * self.advantage_top_fraction)
        keep -= keep % self.num_mini_batches
        minimum = self.num_mini_batches * (2 if self.normalize_advantage_per_mini_batch else 1)
        if keep < minimum:
            raise ValueError("advantage filtering leaves too few samples for the configured minibatches")
        magnitude, selected = scores.abs().topk(keep, sorted=False)
        selected = selected[torch.randperm(keep, device=self.device)]
        selected_done = storage.dones[selected % steps, selected // steps].float()
        metrics = {"advantage_top_fraction": self.advantage_top_fraction,
                   "retained_fraction": keep / (steps * envs),
                   "available_samples_per_rank": float(steps * envs),
                   "retained_samples_per_rank": float(keep),
                   "selected_positive_fraction": float((scores[selected] > 0).float().mean()),
                   "selected_done_fraction": float(selected_done.mean()),
                   "selected_abs_advantage_mean": float(magnitude.mean()),
                   "selected_abs_advantage_min": float(magnitude.min())}
        return (selected.new_empty(0), selected), metrics

    def _update_microbatched(self, *, indices=None):
        """Memory-bounded execution of existing logical categorical PPO batches.

        Losses/adaptive-KL mirror vendored algorithms/ppo.py:246-317; gradient
        clipping/step and observer mirror :367-409. This adapter supports only
        the StudentPPO subset already excluding RND, symmetry and recurrence.
        Gradients are mathematically identical in float32, with normal floating
        point differences from chunked kernels and reduction order.
        """
        mean_losses = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
        physical_batches = 0
        self.value_metrics = {}
        logical_batch_size = 0
        mask_validator = getattr(self.policy, "validate_action_mask", None)
        for (obs, actions, old_values, advantages, returns, old_log_probs,
             _old_mean, _old_std, _hidden, _masks) in self.storage.mini_batch_generator(
                 self.num_mini_batches, self.num_learning_epochs,
                 **({"indices": indices} if indices is not None else {})):
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
                    if not self._separate_backward:
                        _, values = self.policy.action_probabilities_and_value(
                            obs[sl], validate_mask=mask_validator is None,
                            **({"raw_value": True} if self._classification_value else {}))
                    elif mask_validator is None:
                        self.policy.action_probabilities(obs[sl])
                    else:
                        self.policy.action_probabilities(obs[sl], validate_mask=False)
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
                entropy_loss = entropy.mean()
                weight = (end - start) / count
                actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
                if self._separate_backward:
                    if actor_loss.requires_grad:
                        (actor_loss * weight).backward()
                    if physical_batches == 0:
                        self.monitor.capture_actor_gradients(self.policy, end - start)
                    with (torch.autocast('cuda', dtype=torch.bfloat16)
                          if self.amp_bfloat16 else nullcontext()):
                        values = self.policy.evaluate(obs[sl], **({"raw_value": True} if self._classification_value else {}))
                values = values.float()
                target, old_value = returns[sl].float(), old_values[sl].float()
                if self._classification_value:
                    value_loss = self.value_distribution.loss(values, target)
                    if physical_batches == 0:
                        self.value_metrics.update({k: float(v) for k, v in
                            self.value_distribution.diagnostics(values, target).items()})
                else:
                    value_losses = (values - target).pow(2)
                    if self.use_clipped_value_loss:
                        clipped_value = old_value + (values - old_value).clamp(-self.clip_param, self.clip_param)
                        value_losses = torch.maximum(value_losses, (clipped_value - target).pow(2))
                    value_loss = value_losses.mean()
                if self._separate_backward:
                    if value_loss.requires_grad:
                        (self.value_loss_coef * value_loss * weight).backward()
                else:
                    ((actor_loss + self.value_loss_coef * value_loss) * weight).backward()
                if self._separate_backward and physical_batches == 0:
                    self.monitor.capture_critic_gradients()
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
                    self._adapt_learning_rate(kl_mean)
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
            "separate_actor_critic_backward": float(self._separate_backward),
        }
        return {key: value / updates for key, value in mean_losses.items()}

    def process_env_step(self, obs, rewards, dones, extras):
        if extras.get('has_bootstrap_timeouts') is False:
            # The bridge already resolved this on the host. Skip CTS's CUDA
            # any()/zero-multiplies without changing its actual timeout path.
            extras = {key: value for key, value in extras.items() if key != 'time_outs'}
        if getattr(self.policy, 'joint_critic', False):
            pairs = rewards.reshape(-1, 2)
            valid = (pairs.sum(-1) == 0).all() & (dones[::2] == dones[1::2]).all()
            if valid.is_cuda:
                torch._assert_async(valid, 'joint critic requires zero-sum rewards and paired episode ends')
            elif not bool(valid):
                raise ValueError('joint critic requires zero-sum rewards and paired episode ends')
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
        if self.pair_minibatches and (self.storage.num_envs * self.storage.num_transitions_per_env) % (2 * self.num_mini_batches):
            raise ValueError('paired rollout must divide into complete two-seat minibatches')

    def compute_returns(self, obs):
        # Recheck after the runner shortens the final active rollout. Its GAE
        # uses only the active prefix; no return or normalization formula is copied.
        self._check_sample_count()
        if not self.is_multi_gpu or self.normalize_advantage_per_mini_batch:
            result = MoECTS.compute_returns(self, obs)
            self._check_value_targets()
            return result
        # Let CTS compute unchanged GAE, then normalize the active advantages
        # using moments from every rank rather than separate local populations.
        self.normalize_advantage_per_mini_batch = True
        try:
            MoECTS.compute_returns(self, obs)
        finally:
            self.normalize_advantage_per_mini_batch = False
        self.storage.advantages = self._normalize_advantages(self.storage.advantages)
        self._check_value_targets()

    def _check_value_targets(self):
        if self._classification_value:
            target = self.storage.returns[:self.storage.num_transitions_per_env]
            valid = (torch.isfinite(target) & (target.abs() <= 1 + 1e-4)).all()
            if target.is_cuda:
                torch._assert_async(valid, "bounded value requires returns within [-1,1]")
            elif not bool(valid):
                raise ValueError("bounded value requires returns within [-1,1]")
            if getattr(self.policy, 'joint_critic', False):
                valid = (target.reshape(-1, 2).sum(-1).abs() <= 1e-4).all()
                if valid.is_cuda:
                    torch._assert_async(valid, 'joint critic GAE targets must be antisymmetric')
                elif not bool(valid):
                    raise ValueError('joint critic GAE targets must be antisymmetric')
        if self.pair_minibatches:
            steps = self.storage.num_transitions_per_env
            residuals = torch.stack([field[:steps].reshape(-1, 2).sum(-1).abs().mean()
                                    for field in (self.storage.values, self.storage.returns)])
            if self.is_multi_gpu:
                torch.distributed.all_reduce(residuals)
                residuals /= self.gpu_world_size
            self.pair_metrics = dict(zip(('paired_value_sum_abs_mean', 'paired_return_sum_abs_mean'),
                                         residuals.tolist()))

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
