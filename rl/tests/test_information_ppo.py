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


def sample(n=4, memory=False, history_length=4):
    g = torch.Generator().manual_seed(27)
    frame = torch.rand(n, 1147, generator=g)
    maps = frame[:, :1134].reshape(n, 14, 81)
    maps[:, 4].zero_(); maps[:, 5].zero_()
    maps[:, 4, 40] = 1; maps[:, 5, 40] = 1
    values = {"policy": frame[:, None].expand(-1, history_length, -1).reshape(n, history_length * 1147).clone(),
              "single_obs": frame.clone(), "critic": frame.clone(),
              "action_mask": torch.ones(n, 649, dtype=torch.bool)}
    if memory:
        values["knowledge"] = torch.rand(n, 891, generator=g)
    return TensorDict(values, batch_size=[n])


def options():
    return {"map_width": 9, "map_height": 9, "own_king_geometry": True, "wait_move_factorization": True}


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
        self.assertGreater(float(policy.critic.trunk[0].weight.grad[:, 14:].abs().sum()), 0)
        self.assertGreater(float(policy.critic.context[0].weight.grad[:, -13:].abs().sum()), 0)
        with torch.no_grad():
            policy.critic.trunk[0].weight[:, 14:].fill_(0.01)
        changed = obs.clone(); changed["policy"] *= 2
        self.assertFalse(torch.equal(policy.evaluate(obs), policy.evaluate(changed)))

    def test_memory_is_non_stacked_trainable_and_inference_requires_no_critic(self):
        obs = sample(memory=True)
        torch.manual_seed(145)
        baseline = LightSpatialPPOPolicy(obs, groups(), 649, **options())
        torch.manual_seed(145)
        policy = InformationSpatialPPOPolicy(obs, groups(memory=True, critic=True), 649,
                    knowledge_channels=11, critic_actor_information=True, **options())
        self.assertEqual(policy.actor.trunk[1].in_channels, 69)
        self.assertEqual(policy.critic.trunk[0].in_channels, 81)
        torch.testing.assert_close(policy.action_probabilities(obs), baseline.action_probabilities(obs), rtol=1e-6, atol=1e-6)
        policy.actor(obs["policy"], obs["knowledge"]).sum().backward()
        self.assertGreater(float(policy.actor.trunk[1].weight.grad[:, 56:67].abs().sum()), 0)
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
            InformationSpatialPPOPolicy(obs, groups(memory=True), 649, knowledge_channels=11, **options())
        with self.assertRaises(TypeError):
            InformationSpatialPPOPolicy(obs, groups(), 649, critic_actor_information=1)

    def test_h8_latest_public_geometry_context_and_critic_share_history(self):
        obs = sample(n=2, memory=True, history_length=8)
        frames = obs["policy"].reshape(2, 8, 1147)
        maps = frames[..., :1134].reshape(2, 8, 14, 9, 9)
        # Old generals deliberately disagree with the newest local marker.
        maps[:, :, 4:6].zero_()
        for index in range(8):
            maps[:, index, 0].fill_(index / 8)
            maps[:, index, 4:6, index, index] = 1
            frames[:, index, -13:] = index
        policy = InformationSpatialPPOPolicy(obs, groups(memory=True, critic=True), 649,
                    history_length=8, knowledge_channels=11, critic_actor_information=True,
                    **options())
        self.assertEqual(policy.actor.trunk[1].in_channels, 125)
        self.assertEqual(policy.critic.trunk[0].in_channels, 137)
        self.assertEqual((policy.num_actor_obs, policy.num_critic_obs), (10067, 11214))
        self.assertIn("hist8", policy.semantics)
        captured = {}
        hooks = [
            policy.actor.trunk[1].register_forward_pre_hook(
                lambda _module, args: captured.update(actor_maps=args[0].detach().clone())),
            policy.actor.trunk.register_forward_hook(
                lambda _module, _args, output: captured.update(spatial=output.detach().clone())),
            policy.actor.context.register_forward_pre_hook(
                lambda _module, args: captured.update(actor_context=args[0].detach().clone())),
            policy.critic.trunk[0].register_forward_pre_hook(
                lambda _module, args: captured.update(critic_maps=args[0].detach().clone())),
            policy.critic.context.register_forward_pre_hook(
                lambda _module, args: captured.update(critic_context=args[0].detach().clone())),
        ]
        try:
            probabilities = policy.action_probabilities(obs).detach().clone()
            policy.evaluate(obs)
        finally:
            for hook in hooks:
                hook.remove()
        local = maps.reshape(2, 112, 9, 9)
        memory = obs["knowledge"].reshape(2, 11, 9, 9)
        torch.testing.assert_close(captured["actor_maps"][:, :112], local, rtol=0, atol=0)
        torch.testing.assert_close(captured["actor_maps"][:, 112:123], memory, rtol=0, atol=0)
        xy = (torch.arange(9.) - 7) / 8
        torch.testing.assert_close(captured["actor_maps"][:, -2], xy[None, None].expand(2, 9, 9), rtol=0, atol=0)
        torch.testing.assert_close(captured["actor_maps"][:, -1], xy[None, :, None].expand(2, 9, 9), rtol=0, atol=0)
        torch.testing.assert_close(captured["actor_context"][:, -13:], frames[:, -1, -13:], rtol=0, atol=0)
        torch.testing.assert_close(captured["critic_maps"][:, 14:126], local, rtol=0, atol=0)
        torch.testing.assert_close(captured["critic_maps"][:, -11:], memory, rtol=0, atol=0)
        torch.testing.assert_close(captured["critic_context"][:, -13:], frames[:, -1, -13:], rtol=0, atol=0)
        changed = obs.clone()
        changed["policy"].reshape(2, 8, 1147)[:, :-1, -13:] += 100
        changed["critic"] += 1000
        changed["single_obs"] += 1000
        torch.testing.assert_close(policy.action_probabilities(changed), probabilities, rtol=0, atol=0)
        torch.testing.assert_close(policy.action_probabilities(obs.exclude("critic", "single_obs")), probabilities, rtol=0, atol=0)

    def test_h8_all_history_frames_and_memory_receive_separate_actor_critic_gradients(self):
        obs = sample(memory=True, history_length=8)
        obs["policy"].requires_grad_(True)
        obs["critic"].requires_grad_(True)
        policy = InformationSpatialPPOPolicy(obs, groups(memory=True, critic=True), 649,
                    history_length=8, knowledge_channels=11, critic_actor_information=True, **options())
        policy.actor(obs["policy"], obs["knowledge"]).square().mean().backward()
        gradients = obs["policy"].grad.reshape(4, 8, 1147)
        self.assertTrue(bool((gradients[..., :1134].abs().sum((0, 2)) > 0).all()))
        self.assertGreater(float(policy.actor.trunk[1].weight.grad[:, 112:123].abs().sum()), 0)
        self.assertIsNone(obs["critic"].grad)
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        policy.zero_grad(set_to_none=True)
        obs["policy"].grad = None
        policy.evaluate(obs).square().mean().backward()
        critic_inputs = policy.critic.trunk[0].weight.grad[:, 14:126].reshape(16, 8, 14, 3, 3)
        self.assertTrue(bool((critic_inputs.abs().sum((0, 2, 3, 4)) > 0).all()))
        self.assertGreater(float(policy.critic.trunk[0].weight.grad[:, -11:].abs().sum()), 0)
        self.assertGreater(float(policy.critic.context[0].weight.grad[:, -13:].abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        # Information extensions start at zero, but learned history weights must
        # subsequently make every frame part of the critic's input graph.
        with torch.no_grad():
            policy.critic.trunk[0].weight[:, 14:126].fill_(.001)
        obs["policy"].grad = None
        policy.evaluate(obs).sum().backward()
        self.assertTrue(bool((obs["policy"].grad.reshape(4, 8, 1147)[..., :1134].abs().sum((0, 2)) > 0).all()))

    def test_history_lengths_and_runtime_shapes_fail_explicitly(self):
        for invalid in (True, 0, -1, 4.0):
            with self.subTest(history_length=invalid), self.assertRaisesRegex(ValueError, "history_length"):
                InformationSpatialPPOPolicy(sample(), groups(), 649, history_length=invalid)
        with self.assertRaisesRegex(ValueError, "policy shape"):
            InformationSpatialPPOPolicy(sample(), groups(), 649, history_length=8, **options())
        obs = sample(memory=True, history_length=8)
        policy = InformationSpatialPPOPolicy(obs, groups(memory=True, critic=True), 649,
                    history_length=8, knowledge_channels=11, critic_actor_information=True, **options())
        changed = obs.clone(); changed["policy"] = sample(memory=True)["policy"]
        with self.assertRaisesRegex(ValueError, "policy history"):
            policy.action_probabilities(changed)
        with self.assertRaisesRegex(ValueError, "critic history"):
            policy.evaluate(changed)
        changed = obs.clone(); changed["knowledge"] = torch.zeros(4, 8 * 891)
        with self.assertRaisesRegex(ValueError, "knowledge"):
            policy.action_probabilities(changed)

    def test_real_ppo_uses_history_from_existing_storage_and_bootstraps_timeouts(self):
        config = json.loads((ROOT / "configs/v32.json").read_text())
        config["task"].update(num_envs=8, backend="cpu", device="cpu", max_episode_steps=4,
                             map_pool_size=16, history_length=8, persistent_knowledge=True)
        config["runner"].update(num_steps_per_env=4, save_interval=1,
                                obs_groups=groups(memory=True, critic=True),
                                storage_obs_keys=["policy", "critic", "action_mask", "knowledge"])
        config["runner"]["policy"].update(class_name="generals_rl.information_ppo.InformationSpatialPPOPolicy",
                                            history_length=8, knowledge_channels=11,
                                            critic_actor_information=True)
        config["runner"]["algorithm"].update(num_learning_epochs=2, num_mini_batches=4,
                                               ppo_probe_samples=32, ppo_probe_batch_size=16)
        config["self_play"].update(pool_size=3, snapshot_interval=1)
        config["evaluation"]["enabled"] = False
        config["training_control"].update(max_seconds=None, min_seconds=0, plateau_evaluations=None)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            summary = train(config, directory, iterations=3)
            trained = torch.load(Path(directory) / "checkpoint.pt", weights_only=True, map_location="cpu")
        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(summary["truncated_episodes"], 24)
        self.assertEqual(summary["teacher_envs"], 0)
        self.assertEqual(config["runner"]["storage_obs_keys"], ["policy", "critic", "action_mask", "knowledge"])
        weights = trained["model_state_dict"]
        self.assertEqual(weights["actor.trunk.1.weight"].shape[1], 125)
        self.assertEqual(weights["critic.trunk.0.weight"].shape[1], 137)
        self.assertGreater(float(weights["actor.trunk.1.weight"][:, 112:123].abs().sum()), 0)
        self.assertGreater(float(weights["critic.trunk.0.weight"][:, 14:126].abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in summary["last_losses"].values()))


if __name__ == "__main__":
    unittest.main()
