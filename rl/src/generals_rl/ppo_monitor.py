"""Bounded, observation-only diagnostics for categorical PPO updates."""
from __future__ import annotations

from time import perf_counter

import torch


def categorical_changes(old_logits, new_logits):
    """Per-state KL(old||new), decomposed at Generals' WAIT=0/MOVE gate.

    Double precision is confined to the small probe. The move term is weighted
    by the old MOVE mass, so gate + conditional_move equals joint KL.
    """
    old = old_logits.double().log_softmax(-1)
    new = new_logits.double().log_softmax(-1)
    probability = old.exp()
    terms = torch.where(probability > 0, probability * (old - new), 0)
    joint = terms.sum(-1).clamp_min(0)
    old_gate = torch.stack((old[:, 0], old[:, 1:].logsumexp(-1)), -1)
    new_gate = torch.stack((new[:, 0], new[:, 1:].logsumexp(-1)), -1)
    gate_probability = old_gate.exp()
    gate = torch.where(gate_probability > 0,
                       gate_probability * (old_gate - new_gate), 0).sum(-1).clamp_min(0)
    # Subtraction avoids a conditional log_softmax over all-masked MOVE rows.
    move = (joint - gate).clamp_min(0)
    new_probability = new_gate.exp()
    gate_entropy = -torch.where(new_probability > 0, new_probability * new_gate, 0).sum(-1)
    return {"exact_update_kl": joint, "wait_move_gate_kl": gate,
            "weighted_move_kl": move, "wait_probability": new_probability[:, 0],
            "wait_move_gate_entropy": gate_entropy}


class PPOUpdateMonitor:
    """One fixed random subset per rollout; never uses the policy RNG stream."""

    def __init__(self, *, enabled=True, probe_samples=8192, probe_batch_size=1024):
        if not isinstance(enabled, bool):
            raise TypeError("ppo_monitor must be boolean")
        for name, value in (("ppo_probe_samples", probe_samples), ("ppo_probe_batch_size", probe_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.enabled = enabled
        self.probe_samples, self.probe_batch_size = probe_samples, probe_batch_size
        self.generator = torch.Generator().manual_seed(20261006)
        self.release()

    def release(self):
        self.observations = self.old_logits = None
        self.rollout_statistics = None

    def _logits(self, policy):
        distribution = policy.distribution
        device = next(policy.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Also preserve RNG if a future actor adds stochastic inference layers.
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                chunks = []
                for start in range(0, len(self.observations), self.probe_batch_size):
                    policy.action_probabilities(self.observations[start:start + self.probe_batch_size])
                    chunks.append(policy.distribution.logits.clone())
                return torch.cat(chunks)
        finally:
            policy.distribution = distribution

    def begin(self, algorithm):
        self.metrics = {}
        if not self.enabled:
            return
        storage = algorithm.storage
        steps = storage.num_transitions_per_env
        if storage.step != steps:
            raise ValueError("PPO monitoring requires a complete active rollout")
        self.started = perf_counter()
        with torch.inference_mode():
            returns = storage.returns[:steps].float()
            values = storage.values[:steps].float()
            raw_advantage = returns - values
            return_var, return_mean = torch.var_mean(returns, correction=0)
            raw_var, raw_mean = torch.var_mean(raw_advantage, correction=0)
            buffer_var, buffer_mean = torch.var_mean(storage.advantages[:steps].float(), correction=0)
            # Collection-time critic predictions against bootstrapped GAE
            # returns, before this update. Near-constant targets have no usable
            # explained variance; omit that scalar instead of logging NaN.
            explained_variance = 1 - raw_var / return_var.clamp_min(1e-8)
            valid = ((return_var > 1e-8) & torch.isfinite(return_var)
                     & torch.isfinite(explained_variance))
            self.rollout_statistics = torch.stack((
                return_mean, return_var.sqrt(), raw_mean, raw_var.sqrt(),
                buffer_mean, buffer_var.sqrt(), valid.float(), explained_variance))
        count = min(self.probe_samples, steps * storage.num_envs)
        indices = torch.randperm(steps * storage.num_envs, generator=self.generator)[:count].to(storage.actions.device)
        keys = list(dict.fromkeys([*algorithm.policy.obs_groups["policy"], "action_mask"] +
                                  (["single_obs"] if "single_obs" in storage.observations else [])))
        # Env-major indexing matches CTS, without copying or flattening the rollout.
        self.observations = storage.observations.select(*keys)[indices % steps, indices // steps]
        self.old_logits = self._logits(algorithm.policy)
        self.metrics["probe_samples"] = float(count)
        self.metrics["probe_cache_bytes"] = float(sum(t.numel() * t.element_size() for t in self.observations.values()) +
                                                  self.old_logits.numel() * self.old_logits.element_size())
        self.sums = torch.zeros(4, dtype=torch.float64, device=storage.actions.device)
        self.maximum_kl = self.sums.new_zeros(())
        self.batches = self.samples = self.lr_changes = 0
        self.last_lr = algorithm.learning_rate
        self.lr_min = self.lr_max = algorithm.learning_rate
        self.monitor_seconds = perf_counter() - self.started

    def minibatch(self, algorithm, log_ratio, entropy, grad_norm):
        if not self.enabled:
            return
        started = perf_counter()
        with torch.inference_mode():
            estimate = (torch.expm1(log_ratio) - log_ratio).clamp_min(0)
            clipped = (torch.expm1(log_ratio).abs() > algorithm.clip_param)
            self.sums += torch.stack((estimate.sum(), clipped.sum(), entropy.sum(), grad_norm * log_ratio.numel())).double()
            self.maximum_kl = torch.maximum(self.maximum_kl, estimate.mean().double())
        self.batches += 1
        self.samples += log_ratio.numel()
        learning_rate = algorithm.learning_rate
        self.lr_changes += learning_rate != self.last_lr
        self.last_lr = learning_rate
        self.lr_min, self.lr_max = min(self.lr_min, learning_rate), max(self.lr_max, learning_rate)
        if self.batches % algorithm.num_mini_batches == 0:
            epoch = self.batches // algorithm.num_mini_batches
            logits = self._logits(algorithm.policy)
            values = []
            for start in range(0, len(logits), self.probe_batch_size):
                values.append(categorical_changes(self.old_logits[start:start + self.probe_batch_size],
                                                  logits[start:start + self.probe_batch_size]))
            for name in values[0]:
                value = torch.cat([chunk[name] for chunk in values])
                self.metrics[f"epoch_{epoch}/{name}"] = value.mean().item()
                self.metrics[name] = self.metrics[f"epoch_{epoch}/{name}"]
                if name == "exact_update_kl":
                    self.metrics[f"epoch_{epoch}/exact_update_kl_p95"] = torch.quantile(value, .95).item()
                    self.metrics["exact_update_kl_p95"] = self.metrics[f"epoch_{epoch}/exact_update_kl_p95"]
        self.monitor_seconds += perf_counter() - started

    def finish(self, algorithm):
        if not self.enabled:
            return {}
        # Reuse the existing summary transfer, rather than synchronizing once
        # per rollout statistic. The validity branch consumes only CPU values.
        scalars = torch.cat((self.sums / self.samples, self.rollout_statistics.double())).tolist()
        for name, value in zip(("approx_kl", "clip_fraction", "entropy", "grad_norm"), scalars[:4]):
            self.metrics[name] = value
        for name, value in zip(("returns_mean", "returns_std", "raw_advantage_mean", "raw_advantage_std",
                                "buffer_advantage_mean", "buffer_advantage_std"), scalars[4:10]):
            self.metrics[name] = value
        self.metrics["critic_explained_variance_valid"] = scalars[10]
        if scalars[10]:
            self.metrics["critic_explained_variance"] = scalars[11]
        self.metrics.update(approx_kl_max=self.maximum_kl.item(), learning_rate=algorithm.learning_rate,
                            learning_rate_min=self.lr_min, learning_rate_max=self.lr_max,
                            learning_rate_changes=float(self.lr_changes), minibatches=float(self.batches),
                            monitor_seconds=self.monitor_seconds)
        return self.metrics.copy()
