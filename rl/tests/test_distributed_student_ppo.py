"""Two-rank CPU checks for global advantage moments and synchronized PPO."""
from datetime import timedelta
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributions import Categorical
from tensordict import TensorDict

from generals_rl.student_ppo import StudentPPO
from generals_rl.actor_critic import CategoricalActorCritic
from rsl_rl.storage import RolloutStorageCTS


class TinyStudentPolicy(CategoricalActorCritic):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.actor = torch.nn.Linear(3, 3)
        self.critic = torch.nn.Linear(3, 1)
        self.action_distribution = "categorical"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.distribution = None

    def validate_action_mask(self, obs):
        if not obs["action_mask"].any(-1).all():
            raise ValueError("no legal action")

    def action_probabilities(self, obs, is_teacher=False, *, validate_mask=True):
        if validate_mask:
            self.validate_action_mask(obs)
        logits = self.actor(obs["policy"]).masked_fill(~obs["action_mask"], -torch.inf)
        self.distribution = Categorical(logits=logits, validate_args=False)
        return self.distribution.probs

    def evaluate(self, obs, **kwargs):
        return self.critic(obs["critic"])


def observations(rank, step=0):
    if rank is None:
        return torch.cat([observations(index, step) for index in range(2)], dim=0)
    features = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 8 + rank / 2 + step / 10
    return TensorDict({"policy": features, "single_obs": features.clone(),
                       "critic": features.square(), "action_mask": torch.ones(4, 3, dtype=torch.bool)},
                      batch_size=[4])


def algorithm(rank, *, physical=None, per_minibatch=False, adaptive=False, optimizer="sgd"):
    # Different initial rank weights make the broadcast check meaningful.
    torch.manual_seed(106 if rank is None else 106 + rank)
    obs = observations(rank)
    model = TinyStudentPolicy()
    storage = RolloutStorageCTS("rl", len(obs), 0, 3, obs, [1], distribution_shape=[3],
                                action_dtype=torch.long, store_action_mean=False,
                                store_action_std=False)
    result = StudentPPO(model, storage, len(obs), num_learning_epochs=2, num_mini_batches=2,
                        physical_minibatch_size=physical, normalize_advantage_per_mini_batch=per_minibatch,
                        schedule="adaptive" if adaptive else "fixed", desired_kl=.02 if adaptive else None,
                        learning_rate=.003, max_grad_norm=.15, value_loss_coef=.6, entropy_coef=.01,
                        gamma=.93, lam=.81, ppo_monitor=False,
                        multi_gpu_cfg=None if rank is None else {"global_rank": rank, "world_size": 2})
    if optimizer == "sgd":
        result.optimizer = torch.optim.SGD(model.parameters(), lr=result.learning_rate)
    if rank is not None:
        result.broadcast_parameters()
    # Only two active steps, with a poisoned allocated tail: normalization and
    # GAE must use the active prefix rather than all allocated transitions.
    storage.num_transitions_per_env = 2
    for step in range(2):
        current = observations(rank, step)
        storage.observations[step] = current
        with torch.no_grad():
            model.action_probabilities(current)
            actions = (torch.arange(len(current)) % 4 + step) % 3
            storage.actions[step, :, 0] = actions
            storage.values[step] = model.evaluate(current)
            old_log_probs = model.get_actions_log_prob(actions[:, None])
            if adaptive:
                # Rank zero sees small KL; rank one alone sees large KL. The
                # shared scheduler must respond to their combined population.
                delta = (torch.arange(len(current)) // 4).float() * .7 if rank is None else rank * .7
                old_log_probs -= delta
            storage.actions_log_prob[step, :, 0] = old_log_probs
        identities = torch.arange(len(current))
        rank_offsets = identities // 4 if rank is None else torch.full_like(identities, rank)
        storage.rewards[step, :, 0] = (identities % 4).float() / 2 + 4 * rank_offsets + step / 3
        storage.dones[step] = step == 1
    storage.step = 2
    for field in (storage.rewards, storage.values, storage.returns, storage.advantages):
        field[2:] = float("nan")
    indices = torch.arange(len(obs) * 2)
    if rank is None:
        # The global logical batches are the concatenation of each rank's
        # local logical batch, rather than one rank's entire rollout first.
        indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15])
    storage.mini_batch_indices = lambda: (torch.empty(0, dtype=torch.long), indices)
    return result


def distributed_worker(rank, init_file, result_file):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    results = []
    try:
        for physical in (None, 3):
            for per_minibatch, adaptive in ((False, False), (True, False), (False, True)):
                options = dict(physical=physical, per_minibatch=per_minibatch, adaptive=adaptive)
                candidate = algorithm(rank, **options)
                reference = algorithm(None, **{**options, "physical": None})
                candidate.compute_returns(observations(rank, 2))
                reference.compute_returns(observations(None, 2))
                torch.testing.assert_close(candidate.storage.advantages,
                                           reference.storage.advantages[:, rank * 4:(rank + 1) * 4],
                                           rtol=2e-6, atol=2e-7)
                events = []
                original_reduce = candidate.reduce_parameters
                original_clip = torch.nn.utils.clip_grad_norm_
                original_step = candidate.optimizer.step

                def reduce():
                    events.append("reduce")
                    original_reduce()

                def clip(*args, **kwargs):
                    events.append("clip")
                    return original_clip(*args, **kwargs)

                def step(*args, **kwargs):
                    events.append("step")
                    return original_step(*args, **kwargs)

                with patch.object(candidate, "reduce_parameters", side_effect=reduce), \
                     patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=clip), \
                     patch.object(candidate.optimizer, "step", side_effect=step):
                    candidate.update()
                assert events == ["reduce", "clip", "step"] * 4, events
                reference.update()
                weights = torch.cat([parameter.detach().flatten() for parameter in candidate.policy.parameters()])
                gathered = [torch.empty_like(weights) for _ in range(2)]
                dist.all_gather(gathered, weights)
                torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
                for actual, expected in zip(candidate.policy.parameters(), reference.policy.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                learning_rate = torch.tensor(candidate.learning_rate, dtype=torch.float64)
                rank_rates = [torch.empty_like(learning_rate) for _ in range(2)]
                dist.all_gather(rank_rates, learning_rate)
                torch.testing.assert_close(rank_rates[0], rank_rates[1], rtol=0, atol=0)
                assert math.isclose(candidate.learning_rate, reference.learning_rate, rel_tol=2e-6)
                assert candidate.optimizer.param_groups[0]["lr"] == candidate.learning_rate
                results.append({"physical": physical, "per_minibatch": per_minibatch,
                                "adaptive": adaptive, "learning_rate": candidate.learning_rate})
        # Moments count actual samples, even for unequal normalization inputs.
        candidate = algorithm(rank, physical=3)
        local = torch.tensor([1., 2.]) if rank == 0 else torch.tensor([10., 20., 30.])
        global_values = torch.tensor([1., 2., 10., 20., 30.])
        expected = (global_values - global_values.mean()) / (global_values.std() + 1e-8)
        actual = candidate._normalize_advantages(local)
        torch.testing.assert_close(actual, expected[:2] if rank == 0 else expected[2:], rtol=1e-6, atol=1e-7)
        # Default Adam also uses the synchronized gradients after accumulation.
        candidate = algorithm(rank, physical=3, adaptive=True, optimizer="adam")
        reference = algorithm(None, adaptive=True, optimizer="adam")
        candidate.compute_returns(observations(rank, 2))
        reference.compute_returns(observations(None, 2))
        candidate.update()
        reference.update()
        for actual, expected in zip(candidate.policy.parameters(), reference.policy.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        assert not torch.cuda.is_initialized()
        if rank == 0:
            torch.save(results, result_file)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is unavailable")
class DistributedStudentPPOTests(unittest.TestCase):
    def test_two_ranks_match_global_advantages_and_sgd_update(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "",
                                                                       "OMP_NUM_THREADS": "1"}):
            init_file = str(Path(directory) / "process_group")
            result_file = str(Path(directory) / "results.pt")
            mp.spawn(distributed_worker, args=(init_file, result_file), nprocs=2, join=True)
            results = torch.load(result_file, weights_only=True)
        self.assertEqual(len(results), 6)
        self.assertEqual({result["physical"] for result in results}, {None, 3})
        for result in results:
            if result["adaptive"]:
                self.assertLess(result["learning_rate"], .003)


if __name__ == "__main__":
    unittest.main()
