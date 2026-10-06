"""Mature upstream blocks, dense geometry, and PPO distribution consistency."""
import unittest

import torch
from torch import nn
from timm.models.resnetv2 import PreActBottleneck, ResNetV2
from timm.models.vision_transformer import VisionTransformer

from generals_rl.backbones import build_backbone


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_upstream_classes_and_runtime_rectangular_maps(self):
        for kind in ("resnet", "vit"):
            model = build_backbone(kind, "1m", 61)
            if kind == "resnet":
                self.assertIsInstance(model.encoder, ResNetV2)
                self.assertEqual([len(stage.blocks) for stage in model.encoder.stages], [3, 4, 6, 3])
                self.assertEqual(sum(isinstance(m, PreActBottleneck)
                                     for m in model.modules()), 16)
                self.assertTrue(all(m.stride == (1, 1) for m in model.modules()
                                    if isinstance(m, nn.Conv2d)))
                self.assertTrue(all(m.dilation == (1, 1) for m in model.modules()
                                    if isinstance(m, nn.Conv2d)))
                self.assertEqual([model.encoder.stem.conv1.kernel_size,
                                  model.encoder.stem.conv2.kernel_size,
                                  model.encoder.stem.conv3.kernel_size], [(3, 3)] * 3)
                self.assertIsInstance(model.encoder.stem.pool, nn.Identity)
                self.assertTrue(all(info["reduction"] == 1 for info in model.encoder.feature_info))
            else:
                self.assertIsInstance(model.transformer, VisionTransformer)
                self.assertEqual(model.transformer.patch_embed.patch_size, (1, 1))
                self.assertEqual(model.transformer.num_prefix_tokens, 0)
                for block in model.transformer.blocks:
                    torch.testing.assert_close(
                        block.ls1.gamma,
                        torch.full_like(block.ls1.gamma, 1e-4), rtol=0, atol=0)
            with torch.no_grad():
                for height, width in ((9, 9), (7, 11), (13, 13)):
                    with self.subTest(kind=kind, geometry=(height, width)):
                        features = model(torch.randn(2, 61, height, width))
                        self.assertEqual(features.shape,
                                         (2, model.out_channels, height, width))
                        self.assertTrue(bool(features.isfinite().all()))

    def test_same_policy_in_train_eval_and_different_batch_composition(self):
        for kind in ("resnet", "vit"):
            with self.subTest(kind=kind):
                model = build_backbone(kind, "1m", 7)
                self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                                     for m in model.modules()))
                x = torch.randn(1, 7, 5, 7)
                different = torch.randn(2, 7, 5, 7) * 100
                with torch.no_grad():
                    model.train()
                    training = model(x)
                    batched = model(torch.cat((x, different)))[0:1]
                    changed_composition = model(torch.cat((x, different / 100)))[0:1]
                    model.eval()
                    evaluation = model(x)
                torch.testing.assert_close(training, evaluation, rtol=0, atol=0)
                # Keeping the batch shape fixed excludes kernel-selection
                # differences: changing other observations cannot affect x.
                torch.testing.assert_close(batched, changed_composition, rtol=0, atol=0)
                # CPU kernels can change accumulation order with batch size;
                # this checks sample-independent normalization numerically.
                torch.testing.assert_close(training, batched, rtol=1e-4, atol=3e-4)

    def test_backpropagation_reaches_observations_and_upstream_weights(self):
        for kind in ("resnet", "vit"):
            with self.subTest(kind=kind):
                model = build_backbone(kind, "1m", 7)
                x = torch.randn(2, 7, 5, 7, requires_grad=True)
                features = model(x)
                # The full ResNetV2 ends with ReLU: a single selected channel
                # can legitimately be inactive. Probe a weighted feature vector
                # so the test does not accidentally choose only dead outputs.
                weights = torch.linspace(-1, 1, model.out_channels)
                (features[:, :, 2, 3] * weights).sum().backward()
                self.assertTrue(bool(x.grad.isfinite().all()))
                self.assertGreater(float(x.grad.abs().sum()), 0)
                block_weights = (model.encoder.stages[0].blocks[0].conv2.weight if kind == "resnet"
                                 else model.transformer.blocks[0].attn.qkv.weight)
                self.assertTrue(bool(block_weights.grad.isfinite().all()))
                self.assertGreater(float(block_weights.grad.abs().sum()), 0)

    def test_all_scales_match_parameter_budgets_without_large_allocations(self):
        expected = {
            "resnet": {"1m": 997384, "10m": 9740084, "100m": 96735392},
            "vit": {"1m": 1011200, "10m": 9915200, "100m": 96602240},
        }
        totals = {}
        for kind, scales in expected.items():
            for size, count in scales.items():
                with self.subTest(kind=kind, size=size), torch.device("meta"):
                    model = build_backbone(kind, size, 61)
                    actual = sum(p.numel() for p in model.parameters())
                    self.assertEqual(actual, count)
                    self.assertTrue(all(p.is_meta for p in model.parameters()))
                    totals[kind, size] = actual
        for size in ("1m", "10m", "100m"):
            self.assertLess(abs(totals["resnet", size] - totals["vit", size])
                            / totals["resnet", size], 0.05)

    def test_invalid_configuration_and_input(self):
        for invalid in (True, 0, -1, 7.0):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                build_backbone("resnet", "1m", invalid)
        with self.assertRaisesRegex(ValueError, "size"):
            build_backbone("vit", "5m", 7)
        with self.assertRaisesRegex(ValueError, "kind"):
            build_backbone("naive", "1m", 7)
        model = build_backbone("vit", "1M", 7)
        for invalid in (torch.empty(2, 6, 9, 9), torch.empty(2, 7, 9),
                        torch.empty(2, 7, 0, 9)):
            with self.assertRaises(ValueError):
                model(invalid)


if __name__ == "__main__":
    unittest.main()
