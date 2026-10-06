"""Map PPO information boundaries, dynamic geometry and bounded execution."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from generals_rl.cts import runner_config
from generals_rl.evaluate_builtin import evaluate_builtin
from generals_rl.map_ppo import MapPPOPolicy
from generals_rl.mdp.layout import ObservationLayout, own_king_coordinates
from generals_rl.task import RewardConfig, TaskConfig
from generals_rl.train import train


GROUPS = {"policy": ["policy", "knowledge"], "critic": ["critic", "policy", "knowledge"]}


def observations(n=2, width=9, height=9):
    layout = ObservationLayout(width, height, knowledge_channels=11)
    generator = torch.Generator().manual_seed(961)
    frames = torch.rand(n, 4, layout.frame_dim, generator=generator)
    maps = frames[..., :12 * layout.cells].reshape(n, 4, 12, layout.cells)
    maps[:, :, 4:6].zero_()
    maps[:, :, 4:6, width + 1] = 1
    return TensorDict({"policy": frames.flatten(1), "single_obs": frames[:, -1].clone(),
                      "critic": frames[:, -1].clone(),
                      "knowledge": torch.rand(n, layout.knowledge_dim, generator=generator),
                      "action_mask": torch.ones(n, layout.num_actions, dtype=torch.bool)}, [n])


def construct(obs, kind="resnet", *, width=9, height=9, size="1m", **kwargs):
    return MapPPOPolicy(obs, GROUPS, 1 + 8 * width * height, backbone=kind, model_size=size,
                        map_width=width, map_height=height, knowledge_channels=11, **kwargs)


class MapPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_layout_coordinate_guards_and_actor_privacy(self):
        obs = observations()
        layout = ObservationLayout(9, 9, knowledge_channels=11)
        geometry = own_king_coordinates(layout.current_maps(obs["single_obs"]))
        self.assertEqual(float(geometry[0, 0, 1, 1]), 0)
        self.assertEqual(float(geometry[0, 1, 8, 1]), 7 / 8)
        missing = layout.current_maps(obs["single_obs"]).clone()
        missing[:, 4].zero_()
        self.assertEqual(int(own_king_coordinates(missing).count_nonzero()), 0)
        missing[:, 4].fill_(1)
        missing[:, 5].fill_(1)
        self.assertEqual(int(own_king_coordinates(missing).count_nonzero()), 0)
        with self.assertRaises(ValueError):
            layout.frames(obs["policy"][:, :-1])
        with self.assertRaises(ValueError):
            layout.memory_maps(None, 2)
        invalid_groups = deepcopy(GROUPS)
        invalid_groups["policy"].append("critic")
        with self.assertRaisesRegex(ValueError, "obs_groups"):
            MapPPOPolicy(obs, invalid_groups, 649, knowledge_channels=11)
        for kind in ("resnet", "vit"):
            with self.subTest(backbone=kind), torch.no_grad():
                policy = construct(obs, kind)
                expected = policy.action_probabilities(obs).clone()
                changed = obs.clone()
                changed["critic"] += 1000
                changed["single_obs"] += 1000
                torch.testing.assert_close(policy.action_probabilities(changed), expected, rtol=0, atol=0)
                local = obs.exclude("critic", "single_obs")
                torch.testing.assert_close(policy.action_probabilities(local), expected, rtol=0, atol=0)
                self.assertEqual(tuple(policy.action_probabilities(local[:0]).shape), (0, 649))
                self.assertEqual(tuple(policy.evaluate(obs[:0], is_teacher=True).shape), (0, 1))
                with self.assertRaisesRegex(ValueError, "knowledge"):
                    policy.action_probabilities(local.exclude("knowledge"))

    def test_gate_mask_and_cell_direction_mode_order(self):
        obs = observations()
        for kind in ("resnet", "vit"):
            policy = construct(obs, kind)
            codes = torch.arange(648).float() / 32
            cell_hook = policy.actor.cell_head.register_forward_hook(
                lambda module, args, output: codes.reshape(9, 9, 8).permute(2, 0, 1)[None].expand(2, -1, -1, -1))
            wait_hook = policy.actor.wait_head.register_forward_hook(
                lambda module, args, output: output.new_full((2, 1), math.log(3)))
            with torch.no_grad():
                raw = policy.actor(obs["policy"], obs["knowledge"])
                torch.testing.assert_close(raw[:, 1:], codes[None].expand(2, -1), rtol=0, atol=0)
                selected = [2, 19, 648]
                obs["action_mask"].zero_(); obs["action_mask"][:, [0, *selected]] = True
                probabilities = policy.action_probabilities(obs)
                torch.testing.assert_close(probabilities[:, 0], torch.full((2,), .75), rtol=1e-6, atol=1e-7)
                conditional = probabilities[:, selected] / probabilities[:, selected].sum(1, keepdim=True)
                torch.testing.assert_close(conditional, raw[:, selected].softmax(-1))
                self.assertEqual(int(probabilities[~obs["action_mask"]].count_nonzero()), 0)
                obs["action_mask"].zero_(); obs["action_mask"][:, 0] = True
                self.assertTrue(bool((policy.act_inference(obs) == 0).all()))
                torch.testing.assert_close(policy.entropy, torch.zeros(2), rtol=0, atol=0)
            cell_hook.remove(); wait_hook.remove()
            # WAIT-only rows must also have a finite backward path.
            policy.action_probabilities(obs)
            policy.get_actions_log_prob(torch.zeros(2, 1, dtype=torch.long)).sum().backward()
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in policy.parameters()))

    def test_history_memory_and_independent_actor_critic_gradients(self):
        for kind in ("resnet", "vit"):
            obs = observations()
            for key in ("policy", "knowledge", "critic"):
                obs[key].requires_grad_(True)
            policy = construct(obs, kind)
            self.assertFalse({id(p) for p in policy.actor.parameters()}
                             & {id(p) for p in policy.critic.parameters()})
            policy.actor(obs["policy"], obs["knowledge"])[:, [0, 1, 17]].sum().backward()
            self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
            self.assertIsNone(obs["critic"].grad)
            self.assertGreater(float(obs["knowledge"].grad.abs().sum()), 0)
            self.assertTrue(bool((obs["policy"].grad.reshape(2, 4, -1).abs().sum((0, 2)) > 0).all()))
            policy.zero_grad(set_to_none=True)
            for key in ("policy", "knowledge"):
                obs[key].grad = None
            policy.evaluate(obs).sum().backward()
            self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
            for key in ("policy", "knowledge", "critic"):
                self.assertGreater(float(obs[key].grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in policy.parameters()))

    def test_same_weights_accept_rectangular_and_larger_geometry(self):
        for kind in ("resnet", "vit"):
            policy = construct(observations(), kind)
            parameters = {key: id(value) for key, value in policy.named_parameters()}
            original = {key: value.clone() for key, value in policy.state_dict().items()}
            for width, height in ((7, 11), (13, 13)):
                obs = observations(width=width, height=height)
                policy.set_geometry(width, height)
                self.assertEqual({key: id(value) for key, value in policy.named_parameters()}, parameters)
                restored = construct(obs, kind, width=width, height=height)
                restored.load_state_dict(original, strict=True)
                self.assertEqual(policy.num_actor_obs, policy.get_actor_obs(obs).shape[-1])
                self.assertEqual(policy.num_critic_obs, policy.get_critic_obs(obs).shape[-1])
                with torch.no_grad():
                    probabilities, value = policy.action_probabilities(obs), policy.evaluate(obs)
                    self.assertEqual(tuple(probabilities.shape), (2, 1 + 8 * width * height))
                    self.assertEqual(tuple(value.shape), (2, 1))
                    torch.testing.assert_close(probabilities, restored.action_probabilities(obs), rtol=0, atol=0)
                    torch.testing.assert_close(value, restored.evaluate(obs), rtol=0, atol=0)
                for key, expected in original.items():
                    torch.testing.assert_close(policy.state_dict()[key], expected, rtol=0, atol=0)

    def test_chunked_forward_preserves_sampling_rng_and_batch_independence(self):
        obs, permutation = observations(7), torch.tensor([6, 1, 4, 0, 2, 5, 3])
        for kind in ("resnet", "vit"):
            policy = construct(obs, kind)
            with torch.no_grad():
                rng = torch.get_rng_state().clone()
                expected_p, expected_v = policy.action_probabilities(obs).clone(), policy.evaluate(obs).clone()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                with patch("torch.multinomial", wraps=torch.multinomial) as sample:
                    expected_actions = policy.act(obs)
                    self.assertEqual(sample.call_count, 1)
                expected_rng = torch.get_rng_state().clone()
                policy.inference_chunk_size = 3
                chunks = []
                hook = policy.actor.register_forward_pre_hook(lambda module, args: chunks.append(len(args[0])))
                torch.set_rng_state(rng)
                probabilities = policy.action_probabilities(obs)
                self.assertEqual(chunks, [3, 3, 1])
                hook.remove()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                torch.testing.assert_close(probabilities, expected_p, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(policy.evaluate(obs), expected_v, rtol=2e-5, atol=2e-5)
                with patch("torch.multinomial", wraps=torch.multinomial) as sample:
                    actions = policy.act(obs)
                    self.assertEqual(sample.call_count, 1)
                torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                torch.testing.assert_close(policy.action_probabilities(obs[permutation]),
                                           probabilities[permutation], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(policy.evaluate(obs[permutation]),
                                           expected_v[permutation], rtol=2e-5, atol=2e-5)

    def test_all_capacities_have_shared_heads_and_fixed_critic_on_meta(self):
        obs = observations(1)
        head_counts, critic_counts = set(), set()
        for size, budget in (("1m", 1_000_000), ("10m", 10_000_000), ("100m", 100_000_000)):
            counts = []
            for kind in ("resnet", "vit"):
                with torch.device("meta"):
                    policy = construct(obs, kind, size=size)
                self.assertTrue(all(p.is_meta for p in policy.parameters()))
                actor_count = sum(p.numel() for p in policy.actor.parameters())
                self.assertLess(abs(actor_count / budget - 1), .15)
                counts.append(actor_count)
                head_counts.add(sum(p.numel() for module in
                                    (policy.actor.context, policy.actor.cell_head, policy.actor.wait_head)
                                    for p in module.parameters()))
                critic_counts.add(sum(p.numel() for p in policy.critic.parameters()))
                self.assertEqual(policy.actor.projection.out_channels, 128)
            self.assertLess(abs(counts[0] - counts[1]) / max(counts), .05)
        self.assertEqual(len(head_counts), 1)
        self.assertEqual(len(critic_counts), 1)

    def test_real_microbatch_ppo_timeout_pool_and_sampled_builtin_eval(self):
        task = TaskConfig(width=7, height=11, num_envs=2, opponent="self_play",
                          max_episode_steps=2, combat_features=True, king_threat_distance=True,
                          persistent_knowledge=True, productive_actions=True, map_pool_size=8,
                          reward=RewardConfig(potential_gamma=.995))
        runner = runner_config(rollout_steps=2, learning_epochs=1, mini_batches=1, teacher_ratio=0)
        runner["obs_groups"] = GROUPS
        runner["storage_obs_keys"] = ["policy", "critic", "knowledge", "action_mask"]
        runner["policy"] = {"class_name": "generals_rl.map_ppo.MapPPOPolicy", "backbone": "vit",
                            "action_distribution": "categorical",
                            "model_size": "1m", "map_width": 7, "map_height": 11,
                            "knowledge_channels": 11, "critic_actor_information": True,
                            "inference_chunk_size": 2}
        runner["algorithm"].update(class_name="generals_rl.student_ppo.StudentPPO", load_balance_coef=0,
                                   gamma=.995, physical_minibatch_size=2, learning_rate=1e-4,
                                   student_encoder_learning_rate=1e-4, schedule="fixed", desired_kl=None,
                                   ppo_probe_samples=4, ppo_probe_batch_size=2)
        config = {"task": asdict(task), "runner": runner, "torch_threads": 1,
                  "self_play": {"pool_size": 2, "snapshot_interval": 1}, "evaluation": {"enabled": False},
                  "training_control": {"max_seconds": None, "min_seconds": 0, "plateau_evaluations": None}}
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            summary = train(config, root, iterations=1)
            saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
            report = evaluate_builtin(root / "checkpoint.pt", maps_per_ai=1, batch_size=8,
                                      max_ticks=2, device="cpu", cpu_threads=1)
        self.assertEqual((summary["environment_steps"], summary["teacher_envs"], summary["student_envs"]), (4, 0, 2))
        self.assertEqual(summary["truncated_episodes"], 2)
        self.assertTrue(all(math.isfinite(value) for value in summary["last_losses"].values()))
        self.assertEqual({int(value["step"]) for value in saved["optimizer_state_dict"]["state"].values()}, {1})
        self.assertEqual(report["completed_games"], 8)
        self.assertTrue(report["stochastic"])
        self.assertTrue(report["persistent_knowledge"])
        self.assertEqual(set(report["by_ai"]), {"aggressive", "defensive", "expansion", "random"})


if __name__ == "__main__":
    unittest.main()
