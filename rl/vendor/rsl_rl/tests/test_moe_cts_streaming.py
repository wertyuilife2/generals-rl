# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming must reproduce the old materialized CTS updates and RNG sequence."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MoECTS
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS


class LegacyMaterializedStorage(RolloutStorageCTS):
    """Reference: old env-major flatten, teacher/student cat, all-epoch list cache.

    The unchanged optimization equations run against the old cached batches in
    both phases; only the data delivery implementation differs from streaming.
    """

    def clear(self):
        super().clear()
        self._materialized = None

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, indices=None):
        if getattr(self, "_materialized", None) is None:
            teacher_indices, student_indices = self.mini_batch_indices() if indices is None else indices
            teacher_size = len(teacher_indices) // num_mini_batches
            student_size = len(student_indices) // num_mini_batches
            fields = (
                self.observations, self.actions, self.values, self.advantages,
                self.returns, self.actions_log_prob, self.mu, self.sigma,
            )
            # This is the former full-rollout copying implementation.
            fields = [field[: self.num_transitions_per_env].transpose(0, 1).flatten(0, 1) for field in fields]
            self._materialized = []
            for _ in range(num_epochs):
                for batch in range(num_mini_batches):
                    teacher = teacher_indices[batch * teacher_size : (batch + 1) * teacher_size]
                    student = student_indices[batch * student_size : (batch + 1) * student_size]
                    data = tuple(torch.cat((field[teacher], field[student]), 0).detach() for field in fields)
                    self._materialized.append((*data, (None, None), None))
        return iter(self._materialized)


def make_algorithm(distribution, storage_class):
    torch.manual_seed(42)
    num_envs, action_count, steps = 7, 7, 5
    mask = torch.rand(num_envs, action_count) > 0.3
    mask[:, 0] = True
    obs = TensorDict(
        {"policy": torch.randn(num_envs, 36), "critic": torch.randn(num_envs, 16),
         "single_obs": torch.randn(num_envs, 12), "action_mask": mask},
        batch_size=[num_envs],
    )
    policy = ActorCriticMoECTS(
        obs, {"policy": ["policy"], "critic": ["critic"]}, action_count,
        action_distribution=distribution, actor_hidden_dims=[16, 12], critic_hidden_dims=[16, 12],
        teacher_encoder_hidden_dims=[16], student_encoder_hidden_dims=[16, 8], expert_num=2, latent_dim=8,
    )
    categorical = distribution == "categorical"
    storage = storage_class(
        "rl", num_envs, 4, steps, obs, [1] if categorical else [action_count],
        distribution_shape=[action_count], action_dtype=torch.long if categorical else torch.float32,
        # The reference retains the former full sigma cache. Streaming omits it
        # for Categorical, so the equivalence test covers this optimization too.
        store_action_std=not categorical or storage_class is LegacyMaterializedStorage,
    )
    algorithm = MoECTS(
        policy, storage, num_envs, teacher_env_ratio=0.6,
        num_learning_epochs=3, num_mini_batches=3,
    )
    return algorithm, obs


def run_iteration(algorithm, obs, seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        for _ in range(algorithm.storage.num_transitions_per_env):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.randn(len(obs)), torch.zeros(len(obs), dtype=torch.bool), {})
        algorithm.compute_returns(obs)
    result = algorithm.update()
    return result, torch.get_rng_state().clone()


def assert_equal_state(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal_state(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for expected_item, actual_item in zip(expected, actual):
            assert_equal_state(expected_item, actual_item)
    else:
        assert actual == expected


@pytest.mark.parametrize("distribution", ["normal", "categorical"])
def test_two_streaming_iterations_match_materialized_losses_parameters_optimizers_and_rng(distribution):
    expected, obs = make_algorithm(distribution, LegacyMaterializedStorage)
    actual, _ = make_algorithm(distribution, RolloutStorageCTS)
    if distribution == "categorical":
        assert expected.storage.sigma.shape == (5, 7, 7)
        assert actual.storage.sigma is None
    else:
        assert_equal_state(expected.storage.sigma, actual.storage.sigma)
    assert_equal_state(expected.policy.state_dict(), actual.policy.state_dict())
    for iteration in range(2):
        expected_loss, expected_rng = run_iteration(expected, obs, 123 + iteration)
        actual_loss, actual_rng = run_iteration(actual, obs, 123 + iteration)
        assert actual_loss == pytest.approx(expected_loss, rel=0, abs=0)
        assert_equal_state(expected.policy.state_dict(), actual.policy.state_dict())
        assert_equal_state(expected.optimizer.state_dict(), actual.optimizer.state_dict())
        assert_equal_state(expected.optimizer_stu_enc.state_dict(), actual.optimizer_stu_enc.state_dict())
        assert_equal_state(expected_rng, actual_rng)


def test_partial_rollout_uses_active_time_stride_and_never_stale_allocated_slots():
    algorithm, obs = make_algorithm("categorical", RolloutStorageCTS)
    storage = algorithm.storage
    # Allocated five ticks, active final rollout only two ticks.
    storage.num_transitions_per_env = 2
    for time in range(5):
        for env in range(7):
            storage.observations["single_obs"][time, env].fill_(100 * time + env)
            storage.actions[time, env].fill_(100 * time + env)
    before_rng = torch.get_rng_state().clone()
    teacher = torch.arange(8)
    student = torch.arange(8, 14)
    batches = list(storage.mini_batch_generator(2, 2, indices=(teacher, student)))
    assert_equal_state(before_rng, torch.get_rng_state())
    assert len(batches) == 4
    for batch in batches:
        observations, actions, *_ = batch
        assert int(actions.max()) < 200
        assert torch.equal(observations["single_obs"][:, 0].long(), actions[:, 0])
        assert observations["action_mask"].dtype == torch.bool
    assert actions[:, 0].tolist() == [2, 102, 3, 103, 105, 6, 106]
    assert storage.sigma is None
    assert all(batch[7] is None for batch in batches)


def test_partial_returns_and_advantages_ignore_nan_in_unused_allocated_tail():
    algorithm, obs = make_algorithm("categorical", RolloutStorageCTS)
    storage = algorithm.storage
    storage.num_transitions_per_env = 2
    storage.values.fill_(float("nan"))
    storage.returns.fill_(float("nan"))
    storage.values[:2].copy_(torch.randn_like(storage.values[:2]))
    storage.rewards[:2].copy_(torch.randn_like(storage.rewards[:2]))
    with torch.no_grad():
        algorithm.compute_returns(obs)
    assert storage.advantages.shape == (2, 7, 1)
    assert bool(torch.isfinite(storage.advantages).all())
    assert bool(torch.isfinite(storage.returns[:2]).all())
    assert bool(torch.isnan(storage.returns[2:]).all())
    assert float(storage.advantages.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(storage.advantages.std()) == pytest.approx(1.0, abs=1e-7)
