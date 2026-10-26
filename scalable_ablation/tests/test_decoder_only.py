"""Validate all-Grid decoding with nested Decoder widths."""

import unittest

import torch
import torch.nn.functional as F

from scalable_ablation.decoder_only.model import DecoderOnlyScalableResNet
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


class DecoderOnlyTest(unittest.TestCase):
    def test_initialization_and_full_output_match_reference(self):
        torch.manual_seed(42)
        reference = make_model(PrefixScalableResNet)
        torch.manual_seed(42)
        model = make_model(DecoderOnlyScalableResNet)
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            expected = reference.forward_prefix(coords, 2).reconstruction
            actual = model.forward_prefix(coords, 2).reconstruction
        self.assertTrue(torch.equal(expected, actual))

    def test_all_grids_receive_gradients_but_decoder_is_sliced(self):
        model = make_model(DecoderOnlyScalableResNet)
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        inputs = []
        handle = model.decoder.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().clone())
        )
        for quality_level, width in enumerate((8, 12, 16)):
            model.zero_grad(set_to_none=True)
            output = model.forward_prefix(coords, quality_level)
            self.assertEqual(output.active_levels, (0, 1, 2, 3))
            self.assertEqual(output.hidden_width, width)
            F.mse_loss(output.reconstruction, target).backward()
            for layer in model.grid_encoder.levels:
                self.assertIsNotNone(layer.grid.grad)
                self.assertGreater(layer.grid.grad.abs().sum().item(), 0)
            grad = model.decoder.linear[0].weight.grad
            self.assertEqual(torch.count_nonzero(grad[width:]).item(), 0)
            self.assertGreater(grad[:width].abs().sum().item(), 0)
        handle.remove()
        self.assertTrue(torch.equal(inputs[0], inputs[1]))
        self.assertTrue(torch.equal(inputs[1], inputs[2]))

    def test_budgets_and_training_step(self):
        model = make_model(DecoderOnlyScalableResNet)
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
        before = model.grid_encoder.levels[-1].grid.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.grid_encoder.levels[-1].grid))


if __name__ == "__main__":
    unittest.main()
