"""Verify fixed feature sizes and exact internal hidden-channel slicing."""

import unittest

import torch
import torch.nn.functional as F

from scalable_ablation.internal_mlp.model import (
    FixedFeatureDecoder, InternalMLPScalableResNet, InternalPrefixMLP,
)
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer, accumulate_ema_prefix_gradients,
)


def make_model():
    return InternalMLPScalableResNet(
        grid_levels=4, grid_feat_dim=2, base_resolution=4,
        finest_resolution=8, aspect_ratio=(1, 1), time_scale=1.0,
        pe_freq=3, hidden_dim=16, feature_dim=8, num_blocks=3,
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )


class InternalMLPTest(unittest.TestCase):
    def test_sliced_block_matches_dense_masked_calculation(self):
        block = InternalPrefixMLP(8, 16).double()
        features = torch.randn(7, 8, dtype=torch.float64)
        for width in (8, 12, 16):
            hidden = F.gelu(block.expand(features))
            mask = torch.arange(16) < width
            expected = features + block.project(hidden * mask)
            actual = block(features, width)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
            self.assertEqual(actual.shape, features.shape)

    def test_every_block_keeps_fixed_feature_dim(self):
        decoder = FixedFeatureDecoder(20, 16, 8, 3)
        shapes = []
        handles = [block.register_forward_hook(
            lambda module, inputs, output: shapes.append((inputs[0].shape, output.shape))
        ) for block in decoder.blocks]
        for quality_level in range(3):
            output = decoder(torch.rand(5, 20), quality_level)
            self.assertEqual(tuple(output.shape), (5, 3))
        for handle in handles:
            handle.remove()
        self.assertEqual(len(shapes), 9)
        self.assertTrue(all(tuple(first) == (5, 8) and first == second
                            for first, second in shapes))

    def test_grid_and_internal_channel_gradients_respect_prefix(self):
        model = make_model()
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        for quality_level, width in enumerate((8, 12, 16)):
            model.zero_grad(set_to_none=True)
            output = model.forward_prefix(coords, quality_level)
            self.assertEqual(output.hidden_width, width)
            F.mse_loss(output.reconstruction, target).backward()
            active = model.hierarchy.active_levels(quality_level)
            for index, layer in enumerate(model.grid_encoder.levels):
                self.assertEqual(layer.grid.grad is not None, index in active)
            for block in model.decoder.blocks:
                self.assertGreater(block.expand.weight.grad[:width].abs().sum().item(), 0)
                self.assertEqual(torch.count_nonzero(block.expand.weight.grad[width:]).item(), 0)
                self.assertEqual(torch.count_nonzero(block.project.weight.grad[:, width:]).item(), 0)

    def test_parameter_budget_and_sequential_step(self):
        model = make_model()
        budget = model.parameter_budget()
        self.assertLess(budget[0], budget[1])
        self.assertLess(budget[1], budget[2])
        self.assertEqual(budget[2], sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        metrics, objective = accumulate_ema_prefix_gradients(
            model, torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12),
            F.mse_loss, EMADistortionBalancer(), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0)
        before = model.decoder.blocks[-1].expand.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.decoder.blocks[-1].expand.weight))

    def test_bunny_decoder_parameter_budget(self):
        decoder = FixedFeatureDecoder(144, 240, 128, 3)
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 239315)
        self.assertEqual(decoder.active_parameter_count(2), 239315)


if __name__ == "__main__":
    unittest.main()
