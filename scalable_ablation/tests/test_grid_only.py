"""Check controlled Full equivalence and Grid-prefix gradient isolation."""

import unittest

import torch
import torch.nn.functional as F

from scalable_ablation.grid_only.model import GridOnlyScalableResNet
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer, accumulate_ema_prefix_gradients,
)
from scalable_prefix_resnet.model import PrefixScalableResNet


def make_model(model_class):
    return model_class(
        grid_levels=4, grid_feat_dim=2, base_resolution=4,
        finest_resolution=8, aspect_ratio=(1, 1), time_scale=1.0,
        pe_freq=3, hidden_dim=16,
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )


class GridOnlyTest(unittest.TestCase):
    def test_initialization_and_full_path_match_reference(self):
        torch.manual_seed(42)
        reference = make_model(PrefixScalableResNet)
        torch.manual_seed(42)
        model = make_model(GridOnlyScalableResNet)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            expected = reference.forward_prefix(coords, 2).reconstruction
            actual = model.forward_prefix(coords, 2).reconstruction
        self.assertTrue(torch.equal(expected, actual))

    def test_all_prefixes_run_full_decoder_and_exclude_future_grids(self):
        model = make_model(GridOnlyScalableResNet)
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        seen_decoder_levels = []
        original_forward = model.decoder.forward

        def record_forward(x, quality_level=2):
            seen_decoder_levels.append(quality_level)
            return original_forward(x, quality_level)

        model.decoder.forward = record_forward
        for quality_level in range(3):
            model.zero_grad(set_to_none=True)
            output = model.forward_prefix(coords, quality_level)
            self.assertEqual(output.hidden_width, 16)
            F.mse_loss(output.reconstruction, target).backward()
            active = model.hierarchy.active_levels(quality_level)
            for index, layer in enumerate(model.grid_encoder.levels):
                self.assertEqual(layer.grid.grad is not None, index in active)
        self.assertEqual(seen_decoder_levels, [2, 2, 2])

    def test_budget_and_sequential_training_step(self):
        model = make_model(GridOnlyScalableResNet)
        budgets = model.parameter_budget()
        self.assertLess(budgets[0], budgets[1])
        self.assertLess(budgets[1], budgets[2])
        self.assertEqual(budgets[2], sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        metrics, objective = accumulate_ema_prefix_gradients(
            model, torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12),
            F.mse_loss, EMADistortionBalancer(), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0)
        self.assertTrue(all(p.grad is not None for p in
                            (layer.grid for layer in model.grid_encoder.levels)))
        before = model.decoder.linear[0].weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.decoder.linear[0].weight))


if __name__ == "__main__":
    unittest.main()
