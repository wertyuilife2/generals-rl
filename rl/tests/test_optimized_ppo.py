"""CPU FP32 equivalence, cache lifetime and independent projected-critic tests."""
from copy import deepcopy
from contextlib import contextmanager
import gc
import unittest
from unittest.mock import patch
import weakref

import torch
from tensordict import TensorDict

from generals_rl.information_ppo import InformationSpatialPPOPolicy
import generals_rl.optimized_ppo as optimized
from generals_rl.optimized_ppo import OptimizedInformationSpatialPPOPolicy
from generals_rl.policy.layout import ObservationLayout
TEST_LAYOUT = ObservationLayout(9, 9)
FRAME_DIM, MAP_PLANES, MAP_SIZE, PUBLIC_DIM = TEST_LAYOUT.frame_dim, TEST_LAYOUT.map_planes, 9, TEST_LAYOUT.public_dim


def observations(count=4, history_length=4, knowledge_channels=11):
    generator = torch.Generator().manual_seed(84)
    frames = torch.rand(count, history_length, FRAME_DIM, generator=generator)
    maps = frames[..., :-PUBLIC_DIM].reshape(count, history_length, MAP_PLANES, MAP_SIZE, MAP_SIZE)
    maps[:, :, 4:6].zero_()
    # Old generals disagree with the latest local marker. The final row has no
    # valid own general, exercising zero geometry/global general context.
    for row in range(count):
        for tick in range(history_length):
            if row != count - 1:
                maps[row, tick, 4:6, (row + tick) % MAP_SIZE, (2 * row + tick) % MAP_SIZE] = 1
    masks = torch.zeros(count, 649, dtype=torch.bool)
    masks[:, 0] = True
    masks[:, 1::7] = True
    masks[0, 1:] = False
    if count > 1:
        masks[1, 0] = False
    values = {"policy": frames.flatten(1).clone(), "single_obs": frames[:, -1].clone(),
              "critic": torch.rand(count, FRAME_DIM, generator=generator), "action_mask": masks}
    if knowledge_channels:
        values["knowledge"] = torch.rand(count, knowledge_channels * MAP_SIZE * MAP_SIZE, generator=generator)
    return TensorDict(values, batch_size=[count])


def groups(knowledge_channels=11, critic_information=True):
    actor = ["policy"] + (["knowledge"] if knowledge_channels else [])
    return {"policy": actor, "critic": ["critic"] + (actor if critic_information else [])}


def policy_options(history_length=4, knowledge_channels=11, **overrides):
    values = {"map_width": 9, "map_height": 9, "spatial_channels": 16, "history_length": history_length, "knowledge_channels": knowledge_channels,
              "critic_actor_information": True, "own_king_geometry": True, "wait_move_factorization": True}
    values.update(overrides)
    return values


def learned_weights(policy):
    """Activate every added information channel rather than test its zero init."""
    generator = torch.Generator().manual_seed(194)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.uniform_(-0.07, 0.07, generator=generator)


@contextmanager
def strict_fp32_convolutions():
    enabled = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = enabled


class OptimizedInformationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_policy(self, obs=None, **options):
        obs = observations() if obs is None else obs
        kwargs = policy_options(**options)
        return OptimizedInformationSpatialPPOPolicy(obs, groups(kwargs["knowledge_channels"], kwargs["critic_actor_information"]),
                                                    649, **kwargs)

    def test_equivalent_mode_preserves_checkpoint_parameters_and_initial_rng(self):
        obs = observations()
        torch.manual_seed(91)
        baseline = InformationSpatialPPOPolicy(obs, groups(), 649, **policy_options())
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(91)
        policy = self.make_policy(obs)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertEqual(baseline.state_dict().keys(), policy.state_dict().keys())
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in policy.parameters()))
        for name, parameter in baseline.state_dict().items():
            torch.testing.assert_close(policy.state_dict()[name], parameter, rtol=0, atol=0)
        learned_weights(baseline)
        policy.load_state_dict(baseline.state_dict(), strict=True)
        baseline.load_state_dict(policy.state_dict(), strict=True)
        self.assertEqual((policy.actor.trunk[1].in_channels, policy.actor.trunk[1].compute_in_channels), (69, 72))
        self.assertEqual((policy.critic.trunk[0].in_channels, policy.critic.trunk[0].compute_in_channels), (81, 84))

    def test_learned_fp32_logits_masks_values_and_parameter_gradients_are_equivalent(self):
        # Disable the CPU backend's lower-precision/reordered kernels so this
        # exercises ordinary FP32 arithmetic, never TF32 or mixed precision.
        with strict_fp32_convolutions():
            for history_length, geometry, alignment in (
                    (4, True, 4), (8, True, 8), (4, False, 0)):
                with self.subTest(history_length=history_length, geometry=geometry, alignment=alignment):
                    obs = observations(history_length=history_length)
                    options = policy_options(history_length=history_length, own_king_geometry=geometry)
                    baseline = InformationSpatialPPOPolicy(obs, groups(), 649, **options)
                    learned_weights(baseline)
                    policy = OptimizedInformationSpatialPPOPolicy(obs, groups(), 649,
                                input_channel_alignment=alignment, **options)
                    policy.load_state_dict(baseline.state_dict())
                    outputs = []
                    for model in (baseline, policy):
                        logits = model.actor(obs["policy"], obs["knowledge"])
                        probabilities = model.action_probabilities(obs)
                        values = model.evaluate(obs)
                        actions = obs["action_mask"].long().argmax(-1)
                        selected = probabilities.gather(1, actions[:, None])
                        loss = logits.square().mean() + values.square().mean() - selected.log().mean()
                        loss.backward()
                        outputs.append((logits, probabilities, values, loss))
                        self.assertTrue(bool((probabilities[~obs["action_mask"]] == 0).all()))
                    for actual, expected in zip(outputs[1], outputs[0]):
                        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                    actual_parameters = dict(policy.named_parameters())
                    for name, parameter in baseline.named_parameters():
                        actual = actual_parameters[name].grad
                        self.assertIsNotNone(actual, name)
                        torch.testing.assert_close(actual, parameter.grad, rtol=3e-6, atol=3e-7, msg=name)
                    first_actor = baseline.actor.trunk[int(geometry)].weight.grad
                    self.assertGreater(float(first_actor[:, history_length * MAP_PLANES:history_length * MAP_PLANES + 11].abs().sum()), 0)
                    self.assertGreater(float(baseline.critic.trunk[0].weight.grad[:, 14:].abs().sum()), 0)

    def test_learned_fp32_input_gradients_cover_each_history_and_knowledge_plane(self):
        obs = observations(history_length=8)
        baseline = InformationSpatialPPOPolicy(obs, groups(), 649, **policy_options(history_length=8))
        learned_weights(baseline)
        policy = self.make_policy(obs, history_length=8)
        policy.load_state_dict(baseline.state_dict())
        inputs = []
        with strict_fp32_convolutions():
            for model in (baseline, policy):
                sample = obs.clone()
                for key in ("policy", "knowledge", "critic"):
                    sample[key].requires_grad_(True)
                probabilities = model.action_probabilities(sample)
                values = model.evaluate(sample)
                (values.square().mean() - probabilities[:, 1].clamp_min(1e-8).log().mean()).backward()
                inputs.append(sample)
            for key in ("policy", "knowledge", "critic"):
                torch.testing.assert_close(inputs[1][key].grad, inputs[0][key].grad, rtol=3e-6, atol=3e-7)
            per_frame = inputs[1]["policy"].grad.reshape(4, 8, FRAME_DIM)[..., :-PUBLIC_DIM].abs().sum((0, 2))
            self.assertTrue(bool((per_frame > 0).all()))
            memory = inputs[1]["knowledge"].grad.reshape(4, 11, -1).abs().sum((0, 2))
            self.assertTrue(bool((memory > 0).all()))
            self.assertIsNone(policy._pending_history)

    def test_actor_and_critic_pack_once_in_legacy_order_with_zero_padding(self):
        obs = observations()
        policy = self.make_policy(obs)
        captured = {}
        hooks = [policy.actor.trunk[1].register_forward_pre_hook(
                    lambda _module, args: captured.update(actor=args[0].detach().clone())),
                 policy.critic.trunk[0].register_forward_pre_hook(
                    lambda _module, args: captured.update(critic=args[0].detach().clone()))]
        try:
            with patch.object(optimized.torch, "cat", wraps=torch.cat) as concatenate:
                policy.action_probabilities(obs)
                actor_spatial_cats = [call for call in concatenate.call_args_list if call.args[0][0].ndim == 4]
                self.assertEqual(len(actor_spatial_cats), 1)
            with patch.object(optimized.torch, "cat", wraps=torch.cat) as concatenate:
                policy.evaluate(obs)
                critic_spatial_cats = [call for call in concatenate.call_args_list if call.args[0][0].ndim == 4]
                self.assertEqual(len(critic_spatial_cats), 1)
        finally:
            for hook in hooks:
                hook.remove()
        local = obs["policy"].reshape(4, 4, FRAME_DIM)[..., :-PUBLIC_DIM].reshape(4, 56, 9, 9)
        memory = obs["knowledge"].reshape(4, 11, 9, 9)
        self.assertEqual(tuple(captured["actor"].shape), (4, 72, 9, 9))
        self.assertEqual(tuple(captured["critic"].shape), (4, 84, 9, 9))
        torch.testing.assert_close(captured["actor"][:, :56], local, rtol=0, atol=0)
        torch.testing.assert_close(captured["actor"][:, 56:67], memory, rtol=0, atol=0)
        torch.testing.assert_close(captured["critic"][:, 14:70], local, rtol=0, atol=0)
        torch.testing.assert_close(captured["critic"][:, 70:81], memory, rtol=0, atol=0)
        self.assertEqual(float(captured["actor"][:, 69:].abs().sum()), 0)
        self.assertEqual(float(captured["critic"][:, 81:].abs().sum()), 0)
        torch.testing.assert_close(captured["actor"][0, 67], ((torch.arange(9.) - 3) / 8)[None, :].expand(9, 9))
        torch.testing.assert_close(captured["actor"][0, 68], ((torch.arange(9.) - 3) / 8)[:, None].expand(9, 9))
        self.assertEqual(float(captured["actor"][-1, 67:69].abs().sum()), 0)

    def test_history_handoff_is_single_use_and_never_reuses_mutated_history(self):
        obs = observations()
        policy = self.make_policy(obs)
        learned_weights(policy)
        with patch.object(optimized, "_unpack_history", wraps=optimized._unpack_history) as unpack:
            policy.action_probabilities(obs)
            self.assertIsNotNone(policy._pending_history)
            cached_maps = weakref.ref(policy._pending_history.maps)
            policy.evaluate(obs)
            self.assertIsNone(policy._pending_history)
            self.assertIsNone(cached_maps())  # Release immediately, without GC.
            self.assertEqual(unpack.call_count, 1)
            policy.evaluate(obs)
            self.assertEqual(unpack.call_count, 2)
            policy.action_probabilities(obs)
            obs["policy"].add_(0.1)
            actual = policy.evaluate(obs)
            self.assertEqual(unpack.call_count, 4)
            expected = policy.critic(obs["critic"], obs["policy"], obs["knowledge"])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        policy = self.make_policy(obs, share_history_unpack=False)
        with patch.object(optimized, "_unpack_history", wraps=optimized._unpack_history) as unpack:
            policy.action_probabilities(obs)
            policy.evaluate(obs)
            self.assertEqual(unpack.call_count, 2)

    def test_handoff_is_identity_and_precision_scoped_and_knowledge_is_fresh(self):
        obs = observations()
        policy = self.make_policy(obs)
        learned_weights(policy)
        with patch.object(optimized, "_unpack_history", wraps=optimized._unpack_history) as unpack:
            policy.action_probabilities(obs)
            policy.evaluate(obs.clone())
            self.assertEqual(unpack.call_count, 2)
            policy.action_probabilities(obs)
            with torch.no_grad():
                policy.evaluate(obs)
            self.assertEqual(unpack.call_count, 4)
            policy.action_probabilities(obs)
            obs["knowledge"].add_(0.2)
            actual = policy.evaluate(obs)
            self.assertEqual(unpack.call_count, 5)
        expected = policy.critic(obs["critic"], obs["policy"], obs["knowledge"])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cache_does_not_retain_observations_or_survive_actor_only_failed_or_copied_calls(self):
        obs = observations()
        policy = self.make_policy(obs)
        policy.action_probabilities(obs)
        history = weakref.ref(obs["policy"])
        del obs
        gc.collect()
        self.assertIsNone(history())
        self.assertIsNone(policy._pending_history)
        obs = observations()
        policy.action_probabilities(obs)
        frozen = deepcopy(policy)
        self.assertIsNone(frozen._pending_history)
        policy.action_probabilities(obs.exclude("critic", "single_obs"))
        self.assertIsNone(policy._pending_history)
        policy.act_inference(obs)
        self.assertIsNone(policy._pending_history)
        policy.action_probabilities(obs)
        policy.to(dtype=torch.float64)
        self.assertIsNone(policy._pending_history)
        policy.float()
        invalid = obs.clone()
        invalid["action_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "legal action"):
            policy.action_probabilities(invalid)
        self.assertIsNone(policy._pending_history)

    def test_inference_tensors_and_separate_input_backwards_are_safe(self):
        obs = observations()
        policy = self.make_policy(obs)
        learned_weights(policy)
        with torch.inference_mode():
            sample = obs.clone()
            policy.action_probabilities(sample)
            self.assertIsNone(policy._pending_history)
            self.assertTrue(bool(torch.isfinite(policy.evaluate(sample)).all()))
        obs["policy"].requires_grad_(True)
        policy.action_probabilities(obs).square().sum().backward()
        self.assertIsNone(policy._pending_history)
        # Reusing the previous input-unpack graph here would require retain_graph.
        policy.evaluate(obs).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(obs["policy"].grad).all()))



    def test_actor_privacy_empty_teacher_and_optional_information_modes(self):
        obs = observations()
        policy = self.make_policy(obs)
        before = policy.action_probabilities(obs)
        changed = obs.clone()
        changed["critic"].add_(1000)
        torch.testing.assert_close(policy.action_probabilities(changed), before, rtol=0, atol=0)
        raw = policy.critic(obs["critic"], obs["policy"], obs["knowledge"])
        torch.testing.assert_close(policy.evaluate(obs), raw, rtol=0, atol=0)
        policy.evaluate(obs).sum().backward()
        self.assertEqual(policy.evaluate(obs[:0], is_teacher=True).shape, (0, 1))
        with self.assertRaisesRegex(ValueError, "teacher"):
            policy.evaluate(obs, is_teacher=True)
        plain = observations(knowledge_channels=0)
        policy = self.make_policy(plain, knowledge_channels=0, critic_actor_information=False)
        self.assertEqual(policy.action_probabilities(plain).shape, (4, 649))
        self.assertEqual(policy.evaluate(plain).shape, (4, 1))
        self.assertIsNone(policy._pending_history)

    def test_options_and_runtime_shapes_fail_explicitly(self):
        obs = observations()
        for invalid in (-1, True, 4.0):
            with self.subTest(alignment=invalid), self.assertRaisesRegex(ValueError, "input_channel_alignment"):
                self.make_policy(obs, input_channel_alignment=invalid)
        for invalid in (-1, 8, 32, True, 32.0):
            with self.subTest(projection=invalid), self.assertRaisesRegex(ValueError, "critic_projection_channels"):
                self.make_policy(obs, critic_projection_channels=invalid)
        with self.assertRaises(TypeError):
            self.make_policy(obs, share_history_unpack=1)
        with self.assertRaisesRegex(ValueError, "critic_projection_channels"):
            self.make_policy(obs, critic_actor_information=False, critic_projection_channels=32)
        policy = self.make_policy(obs)
        changed = obs.clone()
        changed["policy"] = changed["policy"][:, :FRAME_DIM]
        with self.assertRaisesRegex(ValueError, "policy history"):
            policy.action_probabilities(changed)
        with self.assertRaisesRegex(ValueError, "policy history"):
            policy.evaluate(changed)
        changed = obs.exclude("knowledge")
        with self.assertRaisesRegex(ValueError, "knowledge"):
            policy.action_probabilities(changed)
        with self.assertRaisesRegex(ValueError, "knowledge"):
            policy.evaluate(changed)


if __name__ == "__main__":
    unittest.main()
