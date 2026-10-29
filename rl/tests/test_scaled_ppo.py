"""Balanced parameter audits and CPU training/rollout information contracts."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer

from generals_rl.policy.scaled import RESNET_TOTAL_PRESETS, VIT_TOTAL_PRESETS
from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy
from generals_rl.student_ppo import StudentPPO
from generals_rl.actor_critic import CategoricalActorCritic
from rsl_rl.storage import RolloutStorageCTS


GROUPS = {"policy": ["policy", "knowledge"], "critic": ["critic", "policy", "knowledge"]}


def observations(count=2):
    generator = torch.Generator().manual_seed(108)
    frames = torch.rand(count, 4, 1147, generator=generator)
    maps = frames[..., :1134].reshape(count, 4, 14, 9, 9)
    maps[:, :, 4:6].zero_()
    for tick in range(4):
        maps[:, tick, 4:6, tick, tick] = 1
    masks = torch.ones(count, 649, dtype=torch.bool)
    masks[:, 2::3] = False
    values = {"policy": frames.flatten(1).clone(), "single_obs": frames[:, -1].clone(),
              "critic": torch.rand(count, 1147, generator=generator),
              "knowledge": torch.rand(count, 891, generator=generator), "action_mask": masks}
    return TensorDict(values, [count])


def construct(obs, family="resnet", size="1m", **kwargs):
    return ScaledInformationSpatialPPOPolicy(obs, GROUPS, 649, map_width=9, map_height=9, backbone=family, model_size=size, **kwargs)


class ScaledInformationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_six_total_budgets_dense_shapes_and_fp32_masters_on_meta(self):
        obs = observations(1)
        for family, size in ((f, s) for f in ("resnet", "vit") for s in ("1m", "10m", "100m")):
            with self.subTest(family=family, size=size), torch.device("meta"):
                policy = construct(obs, family, size, amp_bfloat16=True)
                actor_count = sum(p.numel() for p in policy.actor.parameters())
                critic_count = sum(p.numel() for p in policy.critic.parameters())
                self.assertIsInstance(policy, CategoricalActorCritic)
                self.assertEqual(policy.parameter_counts(), {"actor": actor_count, "critic": critic_count,
                                                             "total": actor_count + critic_count})
                budget = {"1m": 1_000_000, "10m": 10_000_000, "100m": 100_000_000}[size]
                self.assertLess(abs((actor_count + critic_count) / budget - 1), 0.1)
                self.assertTrue(all(parameter.is_meta and parameter.dtype == torch.float32
                                    for parameter in policy.parameters()))
                local = obs.to("meta")
                self.assertEqual(policy.actor(local["policy"], local["knowledge"]).shape, (1, 649))
                self.assertEqual(policy.critic(local["critic"], local["policy"], local["knowledge"]).shape, (1, 1))
                if family == "resnet":
                    encoder = policy.actor.backbone.encoder
                    self.assertIsInstance(encoder, ResNetV2)
                    self.assertEqual([len(stage.blocks) for stage in encoder.stages], [3, 4, 6, 3])
                    self.assertEqual(policy.actor.backbone.out_channels, RESNET_TOTAL_PRESETS[size][0][-1])
                    self.assertTrue(all(module.stride == (1, 1) for module in encoder.modules()
                                        if isinstance(module, torch.nn.Conv2d)))
                else:
                    encoder = policy.actor.backbone.transformer
                    self.assertIsInstance(encoder, VisionTransformer)
                    self.assertEqual(encoder.patch_embed.patch_size, (1, 1))
                    self.assertEqual((encoder.embed_dim, len(encoder.blocks), encoder.blocks[0].attn.num_heads),
                                     VIT_TOTAL_PRESETS[size])

    def test_projected_and_full_critics_keep_two_spatial_layers_and_all_information_gradients(self):
        for family in ("resnet", "vit"):
            for projection in (0, 32):
                with self.subTest(family=family, projection=projection):
                    obs = observations()
                    for key in ("policy", "knowledge", "critic"):
                        obs[key].requires_grad_(True)
                    policy = construct(obs, family, critic_projection_channels=projection)
                    stem_convolutions = [module for module in policy.critic.spatial_stem if isinstance(module, torch.nn.Conv2d)]
                    self.assertEqual([module.kernel_size for module in stem_convolutions], [(3, 3), (3, 3)])
                    self.assertEqual(stem_convolutions[0].in_channels, 46 if projection else 81)
                    self.assertFalse({id(parameter) for parameter in policy.actor.parameters()}
                                     & {id(parameter) for parameter in policy.critic.parameters()})
                    raw = policy.actor(obs["policy"], obs["knowledge"])
                    raw[:, [0, 1, 17]].square().sum().backward()
                    self.assertTrue(all(parameter.grad is None for parameter in policy.critic.parameters()))
                    self.assertIsNone(obs["critic"].grad)
                    for key in ("policy", "knowledge"):
                        self.assertGreater(float(obs[key].grad.abs().sum()), 0)
                        obs[key].grad = None
                    policy.zero_grad(set_to_none=True)
                    policy.evaluate(obs).sum().backward()
                    self.assertTrue(all(parameter.grad is None for parameter in policy.actor.parameters()))
                    history_grad = obs["policy"].grad.reshape(2, 4, 1147)[..., :1134].abs().sum((0, 2))
                    memory_grad = obs["knowledge"].grad.reshape(2, 11, 81).abs().sum((0, 2))
                    self.assertTrue(bool((history_grad > 0).all()))
                    self.assertTrue(bool((memory_grad > 0).all()))
                    self.assertGreater(float(obs["critic"].grad.abs().sum()), 0)
                    self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                                        for parameter in policy.parameters()))
                    if projection:
                        self.assertGreater(float(policy.critic.history_projection.weight.grad.abs().sum()), 0)
                    else:
                        self.assertGreater(policy.parameter_counts()["critic"], policy.parameter_counts()["actor"])

    def test_packed_actor_uses_local_history_knowledge_latest_xy_and_critic_is_private(self):
        obs = observations()
        policy = construct(obs)
        captured = {}
        hook = policy.actor.backbone.register_forward_pre_hook(
            lambda _module, args: captured.update(inputs=args[0].detach().clone()))
        with torch.no_grad():
            expected = policy.action_probabilities(obs)
            changed = obs.clone()
            changed["critic"].add_(1000)
            changed["single_obs"].add_(1000)
            torch.testing.assert_close(policy.action_probabilities(changed), expected, rtol=0, atol=0)
            torch.testing.assert_close(policy.action_probabilities(obs.exclude("critic", "single_obs")), expected, rtol=0, atol=0)
        hook.remove()
        local = obs["policy"].reshape(2, 4, 1147)[..., :1134].reshape(2, 56, 9, 9)
        torch.testing.assert_close(captured["inputs"][:, :56], local, rtol=0, atol=0)
        torch.testing.assert_close(captured["inputs"][:, 56:67], obs["knowledge"].reshape(2, 11, 9, 9), rtol=0, atol=0)
        xy = (torch.arange(9.) - 3) / 8
        torch.testing.assert_close(captured["inputs"][0, 67], xy[None].expand(9, 9), rtol=0, atol=0)
        torch.testing.assert_close(captured["inputs"][0, 68], xy[:, None].expand(9, 9), rtol=0, atol=0)
        self.assertEqual(float(captured["inputs"][:, 69:].abs().sum()), 0)

    def test_chunking_bounds_both_networks_preserves_rng_and_does_not_chunk_training_graphs(self):
        obs = observations(5)
        for family in ("resnet", "vit"):
            with self.subTest(family=family):
                policy = construct(obs, family, inference_chunk_size=100)
                self.assertFalse(any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                                     for module in policy.modules()))
                self.assertFalse(any(isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0
                                     or type(module).__name__ == "DropPath" and getattr(module, "drop_prob", 0) > 0
                                     for module in policy.modules()))
                with torch.no_grad():
                    rng = torch.get_rng_state().clone()
                    expected_p, expected_v = policy.action_probabilities(obs), policy.evaluate(obs)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    expected_actions = policy.act(obs)
                    expected_rng = torch.get_rng_state().clone()
                    policy.inference_chunk_size = 2
                    actor_chunks, critic_chunks = [], []
                    hooks = [policy.actor.register_forward_pre_hook(lambda _module, args: actor_chunks.append(len(args[0]))),
                             policy.critic.register_forward_pre_hook(lambda _module, args: critic_chunks.append(len(args[0])))]
                    probabilities, values = policy.action_probabilities(obs), policy.evaluate(obs)
                    self.assertEqual(actor_chunks, [2, 2, 1])
                    self.assertEqual(critic_chunks, [2, 2, 1])
                    for hook in hooks:
                        hook.remove()
                    torch.testing.assert_close(probabilities, expected_p, rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(values, expected_v, rtol=2e-5, atol=2e-5)
                    torch.set_rng_state(rng)
                    with patch("torch.multinomial", wraps=torch.multinomial) as sample:
                        actions = policy.act(obs)
                        self.assertEqual(sample.call_count, 1)
                    torch.testing.assert_close(actions, expected_actions, rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                chunks = []
                hook = policy.actor.register_forward_pre_hook(lambda _module, args: chunks.append(len(args[0])))
                policy.action_probabilities(obs)
                hook.remove()
                self.assertEqual(chunks, [5])

    def test_fp32_categorical_boundary_with_bf16_forwards_and_wait_only_masks(self):
        obs = observations()
        obs["action_mask"][0].zero_()
        obs["action_mask"][0, 0] = True
        for family in ("resnet", "vit"):
            with self.subTest(family=family):
                policy = construct(obs, family, amp_bfloat16=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    probabilities = policy.action_probabilities(obs)
                    values = policy.evaluate(obs)
                    log_probs = policy.get_actions_log_prob(torch.zeros(2, 1, dtype=torch.long))
                    entropy = policy.entropy
                    loss = values.square().mean() - log_probs.mean() - entropy.mean() * 0.01
                for tensor in (probabilities, values, log_probs, entropy):
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertTrue(bool(torch.isfinite(tensor).all()))
                self.assertEqual(float(probabilities[0, 0].detach()), 1)
                self.assertEqual(float(entropy[0].detach()), 0)
                self.assertEqual(int(probabilities[~obs["action_mask"]].count_nonzero()), 0)
                loss.backward()
                self.assertTrue(all(parameter.dtype == torch.float32 and
                                    (parameter.grad is None or parameter.grad.dtype == torch.float32
                                     and bool(torch.isfinite(parameter.grad).all())) for parameter in policy.parameters()))
                frozen = deepcopy(policy)
                self.assertIsNone(frozen.distribution)

    def test_real_student_microbatch_update_trains_both_branches(self):
        for family in ("resnet", "vit"):
            with self.subTest(family=family):
                obs = observations()
                policy = construct(obs, family, inference_chunk_size=1)
                storage = RolloutStorageCTS("rl", 2, 0, 2, obs, [1], distribution_shape=[649],
                                             action_dtype=torch.long, store_action_std=False, store_action_mean=False)
                algorithm = StudentPPO(policy, storage, 2, num_learning_epochs=1, num_mini_batches=1,
                                       physical_minibatch_size=1, amp_bfloat16=False,
                                       schedule="fixed", desired_kl=None, ppo_monitor=False)
                with torch.inference_mode():
                    for tick in range(2):
                        algorithm.act(obs)
                        done = torch.zeros(2, dtype=torch.bool)
                        algorithm.process_env_step(obs, torch.tensor([0.1, 0.3]) + tick,
                                                   done, {"time_outs": done, "terminal_observations": obs})
                    algorithm.compute_returns(obs)
                before_actor = policy.actor.cell_head.weight.detach().clone()
                before_critic = policy.critic.spatial_stem[0].weight.detach().clone()
                with patch.object(policy, "validate_action_mask", wraps=policy.validate_action_mask) as validate:
                    losses = algorithm.update()
                    self.assertEqual(validate.call_count, 1)
                self.assertTrue(all(math.isfinite(value) for value in losses.values()))
                self.assertFalse(torch.equal(before_actor, policy.actor.cell_head.weight))
                self.assertFalse(torch.equal(before_critic, policy.critic.spatial_stem[0].weight))
                self.assertEqual(algorithm.update_metrics["microbatches"], 4)
                self.assertEqual({int(value["step"]) for value in algorithm.optimizer.state.values()}, {1})

    def test_explicit_schema_options_empty_teacher_and_checkpoint_restore(self):
        obs = observations()
        policy = construct(obs, "vit")
        self.assertEqual(policy.evaluate(obs[:0], is_teacher=True).shape, (0, 1))
        self.assertEqual(policy.action_probabilities(obs[:0]).shape, (0, 649))
        restored = construct(obs, "vit")
        restored.load_state_dict(policy.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored.action_probabilities(obs), policy.action_probabilities(obs), rtol=0, atol=0)
        for kwargs in ({"backbone": "unknown"}, {"model_size": "5m"}, {"history_length": 2}, {"history_length": 8},
                       {"knowledge_channels": 0}, {"critic_projection_channels": 8}, {"inference_chunk_size": 0},
                       {"critic_actor_information": False}, {"own_king_geometry": False}, {"input_channel_alignment": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ScaledInformationSpatialPPOPolicy(obs, GROUPS, 649, **kwargs)
        with self.assertRaises(TypeError):
            construct(obs, amp_bfloat16=1)
        with self.assertRaisesRegex(ValueError, "teacher"):
            policy.action_probabilities(obs, is_teacher=True)
        with self.assertRaisesRegex(ValueError, "knowledge"):
            policy.action_probabilities(obs.exclude("knowledge"))


if __name__ == "__main__":
    unittest.main()
