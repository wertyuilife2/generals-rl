"""Information ablations preserve actor privacy and baseline initialization."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tensordict import TensorDict

from generals_rl.information_ppo import InformationSpatialPPOPolicy
from generals_rl.spatial_ppo import LightSpatialPPOPolicy
from generals_rl.train import train


ROOT = Path(__file__).resolve().parents[1]


def sample(n=4, memory=False):
    g = torch.Generator().manual_seed(27)
    frame = torch.rand(n, 985, generator=g)
    maps = frame[:, :972].reshape(n, 12, 81)
    maps[:, 4].zero_(); maps[:, 5].zero_()
    maps[:, 4, 40] = 1; maps[:, 5, 40] = 1
    values = {"policy": frame[:, None].expand(-1, 4, -1).reshape(n, 3940).clone(),
              "single_obs": frame.clone(), "critic": frame.clone(),
              "action_mask": torch.ones(n, 649, dtype=torch.bool)}
    if memory:
        values["knowledge"] = torch.rand(n, 891, generator=g)
    return TensorDict(values, batch_size=[n])


def options():
    return {"own_king_geometry": True, "wait_move_factorization": True}


def groups(memory=False, critic=False):
    actor = ["policy"] + (["knowledge"] if memory else [])
    return {"policy": actor, "critic": ["critic"] + (actor if critic else [])}


class InformationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_is_exact_baseline_including_rng_and_gradients(self):
        obs = sample()
        torch.manual_seed(99)
        old = LightSpatialPPOPolicy(obs, groups(), 649, **options())
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(99)
        new = InformationSpatialPPOPolicy(obs, groups(), 649, **options())
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
        for policy in (old, new):
            p = policy.action_probabilities(obs)
            (policy.evaluate(obs).square().mean() - p[:, 1].log().mean()).backward()
        torch.testing.assert_close(new.action_probabilities(obs), old.action_probabilities(obs), rtol=0, atol=0)
        torch.testing.assert_close(new.evaluate(obs), old.evaluate(obs), rtol=0, atol=0)
        for (_, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_critic_extension_preserves_actor_initialization_rng_and_privacy(self):
        obs = sample()
        torch.manual_seed(123)
        baseline = LightSpatialPPOPolicy(obs, groups(), 649, **options())
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        policy = InformationSpatialPPOPolicy(obs, groups(critic=True), 649,
                                            critic_actor_information=True, **options())
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key, value in baseline.actor.state_dict().items():
            torch.testing.assert_close(policy.actor.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(policy.action_probabilities(obs), baseline.action_probabilities(obs), rtol=0, atol=0)
        torch.testing.assert_close(policy.evaluate(obs), baseline.evaluate(obs), rtol=1e-6, atol=1e-6)
        before = policy.action_probabilities(obs).detach().clone()
        changed = obs.clone(); changed["critic"] += 1000
        torch.testing.assert_close(policy.action_probabilities(changed), before, rtol=0, atol=0)
        policy.action_probabilities(obs.exclude("critic", "single_obs"))
        policy.evaluate(obs).sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        self.assertGreater(float(policy.critic.trunk[0].weight.grad[:, 12:].abs().sum()), 0)
        self.assertGreater(float(policy.critic.context[0].weight.grad[:, -13:].abs().sum()), 0)
        with torch.no_grad():
            policy.critic.trunk[0].weight[:, 12:].fill_(0.01)
        changed = obs.clone(); changed["policy"] *= 2
        self.assertFalse(torch.equal(policy.evaluate(obs), policy.evaluate(changed)))

    def test_memory_is_non_stacked_trainable_and_inference_requires_no_critic(self):
        obs = sample(memory=True)
        torch.manual_seed(145)
        baseline = LightSpatialPPOPolicy(obs, groups(), 649, **options())
        torch.manual_seed(145)
        policy = InformationSpatialPPOPolicy(obs, groups(memory=True, critic=True), 649,
                    knowledge_channels=11, critic_actor_information=True, **options())
        self.assertEqual(policy.actor.trunk[1].in_channels, 61)
        self.assertEqual(policy.critic.trunk[0].in_channels, 71)
        torch.testing.assert_close(policy.action_probabilities(obs), baseline.action_probabilities(obs), rtol=1e-6, atol=1e-6)
        policy.actor(obs["policy"], obs["knowledge"]).sum().backward()
        self.assertGreater(float(policy.actor.trunk[1].weight.grad[:, 48:59].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        policy.zero_grad(set_to_none=True)
        policy.evaluate(obs).sum().backward()
        self.assertGreater(float(policy.critic.trunk[0].weight.grad[:, -11:].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        local = obs.exclude("critic", "single_obs")
        policy.action_probabilities(local)
        self.assertEqual(tuple(policy.evaluate(obs[:0], is_teacher=True).shape), (0, 1))
        with self.assertRaisesRegex(ValueError, "knowledge"):
            policy.action_probabilities(local.exclude("knowledge"))

    def test_information_groups_and_shapes_are_explicit(self):
        obs = sample()
        with self.assertRaisesRegex(ValueError, "obs_groups"):
            InformationSpatialPPOPolicy(obs, groups(), 649, critic_actor_information=True)
        with self.assertRaisesRegex(ValueError, "knowledge"):
            InformationSpatialPPOPolicy(obs, groups(memory=True), 649, knowledge_channels=11)
        with self.assertRaises(TypeError):
            InformationSpatialPPOPolicy(obs, groups(), 649, critic_actor_information=1)

    def test_real_ppo_uses_history_from_existing_storage_and_bootstraps_timeouts(self):
        config = json.loads((ROOT / "configs/v19.json").read_text())
        config["task"].update(num_envs=8, backend="cpu", device="cpu", max_episode_steps=4, map_pool_size=16)
        config["runner"].update(num_steps_per_env=4, save_interval=1, obs_groups=groups(critic=True))
        config["runner"]["policy"].update(class_name="generals_rl.information_ppo.InformationSpatialPPOPolicy",
                                            critic_actor_information=True)
        config["runner"]["algorithm"].update(num_learning_epochs=2, num_mini_batches=4,
                                               ppo_probe_samples=32, ppo_probe_batch_size=16)
        config["self_play"].update(pool_size=3, snapshot_interval=1)
        config["evaluation"]["enabled"] = False
        config["training_control"].update(max_seconds=None, min_seconds=0, plateau_evaluations=None)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            summary = train(config, directory, iterations=3)
        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(summary["truncated_episodes"], 24)
        self.assertEqual(summary["teacher_envs"], 0)
        self.assertEqual(config["runner"]["storage_obs_keys"], ["policy", "critic", "action_mask"])
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in summary["last_losses"].values()))


if __name__ == "__main__":
    unittest.main()
