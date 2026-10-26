"""Frozen inference must preserve the original Scaled ViT checkpoint numerics."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from tensordict import TensorDict

from generals_rl.policy.actor import build_actor
from generals_rl.policy.distributions import action_probabilities, select_action
from generals_rl.policy.layout import ObservationLayout, own_king_coordinates
from generals_rl.scaled_ppo import ScaledInformationSpatialPPOPolicy
from generals_rl.spatial_ppo import SpatialPPOPolicy, LightSpatialPPOPolicy
from generals_rl.information_ppo import InformationSpatialPPOPolicy
from generals_rl.optimized_ppo import OptimizedInformationSpatialPPOPolicy


NETWORK = {"family": "scaled", "backbone": "vit", "model_size": "1m", "head_channels": 128,
           "input_channel_alignment": 4, "amp_bfloat16": True, "spatial_mode": "dense"}
FEATURES = {"width": 9, "height": 9, "players": 2, "history_length": 4,
            "max_episode_steps": 800, "productive_actions": True,
            "combat_features": True, "king_threat_distance": True, "persistent_knowledge": True}
GROUPS = {"policy": ["policy", "knowledge"], "critic": ["critic", "policy", "knowledge"]}


def observations(count=3):
    generator = torch.Generator().manual_seed(108)
    frames = torch.rand(count, 4, 985, generator=generator)
    maps = frames[..., :972].reshape(count, 4, 12, 9, 9)
    maps[:, :, 4:6].zero_()
    for tick in range(4):
        maps[:, tick, 4:6, tick, tick] = 1
    if count > 1:
        maps[1, -1, 4:6].zero_()  # Missing general has zero relative coordinates.
    if count > 2:
        maps[2, -1, 4:6, 7, 8] = 1  # Ambiguous general markers also give zeros.
    mask = torch.ones(count, 649, dtype=torch.bool)
    mask[:, 2::3] = False
    if count > 1:
        mask[1, 1:] = False
    if count > 2:
        mask[2, 0] = False
    return TensorDict({"policy": frames.flatten(1).clone(), "single_obs": frames[:, -1].clone(),
                       "critic": torch.rand(count, 985, generator=generator),
                       "knowledge": torch.rand(count, 891, generator=generator),
                       "action_mask": mask}, [count])


def old_policy(obs, **options):
    return ScaledInformationSpatialPPOPolicy(
        obs, GROUPS, 649, backbone="vit", model_size="1m",
        critic_projection_channels=0, compile_forward=False, **options).eval()


class PolicyActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_synthetic_training_checkpoint_is_exact_on_cpu(self):
        obs = observations()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(117)
            original = old_policy(obs, amp_bfloat16=True)
            frozen = build_actor(NETWORK, FEATURES)
        self.assertIs(type(frozen), type(original.actor))
        self.assertTrue(frozen.training)
        self.assertTrue(all(parameter.requires_grad for parameter in frozen.parameters()))
        self.assertFalse(any("critic" in key for key in frozen.state_dict()))
        self.assertEqual(sum(value.numel() for value in frozen.parameters()), 526_985)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic_training_checkpoint.pt"
            torch.save({"model_state_dict": original.state_dict()}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        actor_state = {key[len("actor."):]: value for key, value in checkpoint["model_state_dict"].items()
                       if key.startswith("actor.")}
        self.assertEqual(set(actor_state), set(frozen.state_dict()))
        for key, value in frozen.state_dict().items():
            self.assertEqual(value.shape, actor_state[key].shape, key)
        frozen.load_state_dict(actor_state, strict=True)
        with torch.inference_mode():
            logits = frozen(obs["policy"], obs["knowledge"])
            torch.testing.assert_close(logits, original._forward(obs), rtol=0, atol=0)
            probabilities = action_probabilities(logits, obs["action_mask"])
            torch.testing.assert_close(probabilities, original.action_probabilities(obs), rtol=0, atol=0)
            torch.testing.assert_close(select_action(logits, obs["action_mask"]),
                                       original.act_inference(obs), rtol=0, atol=0)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(probabilities.dtype, torch.float32)

    def test_all_supported_actor_sizes_preserve_state_keys_and_shapes(self):
        obs = observations(1)
        for size in ("1m", "10m", "100m"):
            with self.subTest(size=size), torch.device("meta"):
                original = ScaledInformationSpatialPPOPolicy(
                    obs, GROUPS, 649, backbone="vit", model_size=size, critic_projection_channels=0)
                frozen = build_actor(dict(NETWORK, model_size=size), FEATURES)
                self.assertEqual({key: value.shape for key, value in original.actor.state_dict().items()},
                                 {key: value.shape for key, value in frozen.state_dict().items()})

    def test_retained_families_share_actor_keys_outputs_and_gradients(self):
        obs = observations(2)
        basic_groups = {"policy": ["policy"], "critic": ["critic"]}
        cases = [
            ("spatial", SpatialPPOPolicy, {}, {}, basic_groups),
            ("lightcnn", LightSpatialPPOPolicy, {"own_king_geometry": True},
             {"own_king_geometry": True}, basic_groups),
            ("information", InformationSpatialPPOPolicy,
             {"knowledge_channels": 11, "own_king_geometry": True, "critic_actor_information": True},
             {"knowledge_channels": 11, "own_king_geometry": True}, GROUPS),
            ("optimized", OptimizedInformationSpatialPPOPolicy,
             {"knowledge_channels": 11, "own_king_geometry": True, "critic_actor_information": True},
             {"knowledge_channels": 11, "own_king_geometry": True}, GROUPS),
        ]
        for kind in ("vit", "resnet"):
            for mode in ("dense", "pyramid"):
                options = {"backbone": kind, "spatial_mode": mode}
                cases.append(("scaled", ScaledInformationSpatialPPOPolicy, options, options, GROUPS))
        for family, policy_class, options, network, groups in cases:
            with self.subTest(family=family, network=network):
                original = policy_class(obs, groups, 649, **options)
                features = dict(FEATURES, persistent_knowledge=family in ("information", "optimized", "scaled"))
                shared = build_actor(dict(network, family=family), features)
                self.assertEqual(list(shared.state_dict()), list(original.actor.state_dict()))
                shared.load_state_dict(original.actor.state_dict(), strict=True)
                for actor in (original.actor, shared):
                    self.assertTrue(actor.training)
                    self.assertTrue(all(parameter.requires_grad for parameter in actor.parameters()))
                original_probabilities = original.action_probabilities(obs)
                logits = shared(obs["policy"], obs["knowledge"]) if family in ("information", "optimized", "scaled") else shared(obs["policy"])
                probabilities = action_probabilities(logits, obs["action_mask"], factorized=original.wait_move_factorization)
                torch.testing.assert_close(probabilities, original_probabilities, rtol=0, atol=0)
                (-original_probabilities[:, 0].log().mean()).backward()
                (-probabilities[:, 0].log().mean()).backward()
                for (name, parameter), (_, reference) in zip(shared.named_parameters(), original.actor.named_parameters()):
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0, msg=name)
        # The export's generic scaled family uses the training constructor default.
        default_actor = build_actor({"family": "scaled"}, FEATURES)
        original = ScaledInformationSpatialPPOPolicy(obs, GROUPS, 649)
        self.assertEqual({key: value.shape for key, value in default_actor.state_dict().items()},
                         {key: value.shape for key, value in original.actor.state_dict().items()})

    def test_config_rejects_unsupported_architectures_or_feature_schemas(self):
        for change in ({"family": "unsupported"}, {"model_size": "2m"},
                       {"head_channels": True}, {"input_channel_alignment": -1},
                       {"amp_bfloat16": 1}, {"compile_forward": True}):
            with self.subTest(network=change), self.assertRaises((ValueError, TypeError)):
                build_actor(dict(NETWORK, **change), FEATURES)
        for change in ({"players": 4}, {"width": 11}, {"height": 7}, {"history_length": 8},
                       {"knowledge_channels": 10}, {"persistent_knowledge": False}):
            with self.subTest(features=change), self.assertRaises(ValueError):
                build_actor(NETWORK, dict(FEATURES, **change))

    def test_unaligned_input_and_smaller_head_also_match_original_actor(self):
        obs = observations()
        original = old_policy(obs, head_channels=16, input_channel_alignment=0)
        frozen = build_actor(dict(NETWORK, head_channels=16, input_channel_alignment=0),
                             ObservationLayout(9, 9, knowledge_channels=11))
        frozen.load_state_dict(original.actor.state_dict(), strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(frozen(obs["policy"], obs["knowledge"]),
                                       original._forward(obs), rtol=0, atol=0)

    def test_factorized_greedy_wait_gate_is_not_flat_joint_argmax(self):
        logits = torch.tensor([[-0.1, 0.0, 0.0], [0.0, -100.0, 0.0], [3.0, 7.0, 1.0]])
        mask = torch.tensor([[True, True, True], [True, False, False], [False, True, False]])
        probabilities = action_probabilities(logits, mask)
        self.assertEqual(probabilities[0].argmax().item(), 0)
        self.assertEqual(select_action(logits, mask).flatten().tolist(), [1, 0, 1])
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask])))
        for bad_mask in (None, torch.zeros_like(mask), mask[:, :2]):
            with self.subTest(mask=bad_mask), self.assertRaises(ValueError):
                action_probabilities(logits, bad_mask)

    def test_layout_geometry_and_empty_batch(self):
        obs = observations()
        layout = ObservationLayout(9, 9, knowledge_channels=11)
        current = layout.history_maps(obs["policy"])[:, -12:]
        geometry = own_king_coordinates(current)
        torch.testing.assert_close(geometry[0, :, 3, 3], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(geometry[0, :, 0, 0], torch.tensor([-3 / 8, -3 / 8]), rtol=0, atol=0)
        self.assertTrue(geometry[1:].eq(0).all())
        with torch.inference_mode():
            actor = build_actor(NETWORK, FEATURES)
            output = actor(torch.empty(0, layout.history_dim), torch.empty(0, layout.knowledge_dim))
        self.assertEqual(output.shape, (0, 649))
        self.assertEqual(output.dtype, torch.float32)
        with self.assertRaises(ValueError):
            actor(obs["policy"][:, :-1], obs["knowledge"])
        with self.assertRaises(ValueError):
            actor(obs["policy"], None)

    def test_package_import_and_build_do_not_depend_on_training_packages(self):
        script = '''
import importlib.abc
import sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('generals_rl', 'rsl_rl'):
            raise AssertionError('training import: ' + fullname)
sys.meta_path.insert(0, RejectTraining())
from pathlib import Path
import importlib.util
root = Path('rl/src/generals_rl/policy').resolve()
spec = importlib.util.spec_from_file_location('_exported_policy', root/'__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules['_exported_policy'] = module
spec.loader.exec_module(module)
from _exported_policy.actor import build_actor
actor = build_actor(NETWORK, FEATURES)
assert not any(key.startswith('critic.') for key in actor.state_dict())
'''
        result = subprocess.run([sys.executable, "-c", "NETWORK=" + repr(NETWORK) + "\nFEATURES=" + repr(FEATURES)
                                 + "\n" + script], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA unavailable")
    def test_cuda_bfloat16_logits_probabilities_and_greedy_are_exact(self):
        obs = observations().cuda()
        original = old_policy(obs, amp_bfloat16=True).cuda()
        frozen = build_actor(NETWORK, FEATURES).cuda()
        frozen.load_state_dict(original.actor.state_dict(), strict=True)
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = frozen(obs["policy"], obs["knowledge"]).float()
            torch.testing.assert_close(logits, original._forward(obs), rtol=0, atol=0)
            torch.testing.assert_close(action_probabilities(logits, obs["action_mask"]),
                                       original.action_probabilities(obs), rtol=0, atol=0)
            torch.testing.assert_close(select_action(logits, obs["action_mask"]),
                                       original.act_inference(obs), rtol=0, atol=0)
        self.assertEqual(logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
