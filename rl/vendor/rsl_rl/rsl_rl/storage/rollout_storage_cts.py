# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorageCTS:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition."""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.privileged_actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        teacher_num_envs: int, 
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
        distribution_shape: tuple[int] | list[int] | None = None,
        action_dtype: torch.dtype = torch.float32,
        store_action_std: bool = True,
        store_action_mean: bool = True,
    ) -> None:
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs
        self.distribution_shape = actions_shape if distribution_shape is None else distribution_shape

        # Core
        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, dtype=action_dtype, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        
        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = (
                torch.zeros(num_transitions_per_env, num_envs, *self.distribution_shape, device=self.device)
                if store_action_mean else None
            )
            self.sigma = (
                torch.zeros(num_transitions_per_env, num_envs, *self.distribution_shape, device=self.device)
                if store_action_std else None
            )
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if self.mu is not None:
                self.mu[self.step].copy_(transition.action_mean)
            if self.sigma is not None:
                self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    # For distillation
    def generator(self) -> Generator:
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    def mini_batch_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one fixed env-major index plan, shared by all epochs and CTS phases."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if not 0 < self.num_transitions_per_env <= self.observations.shape[0]:
            raise ValueError("Active rollout length must fit the allocated storage.")
        teacher_samples_num = self.teacher_num_envs * self.num_transitions_per_env
        student_samples_num = self.student_num_envs * self.num_transitions_per_env
        teacher_indices = torch.randperm(teacher_samples_num, requires_grad=False, device=self.device)
        student_indices = teacher_samples_num + torch.randperm(
            student_samples_num, requires_grad=False, device=self.device
        )
        return teacher_indices, student_indices

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
        indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        # This ordering matches transpose(0, 1).flatten(0, 1), without copying
        # the entire rollout. One shared plan preserves the original RNG draws
        # and identical minibatches across epochs and both CTS update phases.
        teacher_indices, student_indices = self.mini_batch_indices() if indices is None else indices
        teacher_mini_batch_size = len(teacher_indices) // num_mini_batches
        student_mini_batch_size = len(student_indices) // num_mini_batches
        data = (
            self.observations, self.actions, self.values, self.advantages,
            self.returns, self.actions_log_prob, self.mu, self.sigma,
        )
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                batch_indices = torch.cat((
                    teacher_indices[i * teacher_mini_batch_size : (i + 1) * teacher_mini_batch_size],
                    student_indices[i * student_mini_batch_size : (i + 1) * student_mini_batch_size],
                ))
                # T may be shorter than allocated storage for the final rollout.
                env_indices = batch_indices // self.num_transitions_per_env
                time_indices = batch_indices % self.num_transitions_per_env
                obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch = (
                        field[time_indices, env_indices].detach() if field is not None else None
                        for field in data
                    )
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),
                    None,
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        return NotImplementedError("CTS rollout storage does not support RNNs yet.")

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        return NotImplementedError("CTS rollout storage does not support RNNs yet.")