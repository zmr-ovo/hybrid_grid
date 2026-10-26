"""Check exact slicing, fixed outputs and shared-prefix gradient routing."""

import unittest

import torch
import torch.nn.functional as F

from scalable_ablation.internal_mlp.model import InternalMLPScalableResNet
from scalable_ablation.unified_internal.model import (
    PrefixGate, PrefixTemporalModulation, UnifiedInternalScalableResNet,
)
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer, accumulate_ema_prefix_gradients,
)


def make_model(model_class=UnifiedInternalScalableResNet):
    kwargs = dict(
        grid_levels=4, grid_feat_dim=2, base_resolution=4,
        finest_resolution=8, aspect_ratio=(1, 1), time_scale=1.0,
        pe_freq=3, hidden_dim=16, feature_dim=8, num_blocks=3,
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )
    if model_class is UnifiedInternalScalableResNet:
        kwargs.update(gate_hidden_dim=8, time_hidden_dim=64)
    return model_class(**kwargs)


class UnifiedInternalTest(unittest.TestCase):
    def test_gate_matches_dense_masked_calculation(self):
        gate = PrefixGate(12, 7, 8).double()
        features = torch.randn(1, 3, 5, 12, dtype=torch.float64)
        for level, width in enumerate((4, 6, 8)):
            hidden = F.gelu(gate.expand(features))
            expected = torch.sigmoid(gate.project(hidden * (torch.arange(8) < width)))
            actual = gate(features, level)
            self.assertEqual(tuple(actual.shape), (1, 3, 5, 7))
            self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_time_matches_dense_masked_calculation(self):
        module = PrefixTemporalModulation(6, 8).double()
        features = torch.randn(1, 6, 5, 7, dtype=torch.float64)
        coords = torch.rand(1, 3, 5, 7, dtype=torch.float64)
        time = coords[:, 2, 0, 0].unsqueeze(1)
        for level, width in enumerate((4, 6, 8)):
            hidden = F.relu(module.mlp[0](time))
            gamma_beta = module.mlp[2](hidden * (torch.arange(8) < width))
            gamma, beta = gamma_beta.chunk(2, dim=1)
            expected = module.norm(features) * gamma.view(1, 6, 1, 1) + beta.view(1, 6, 1, 1)
            actual = module(features, coords, level)
            self.assertEqual(actual.shape, features.shape)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_reference_grid_decoder_and_full_time_initialization_preserved(self):
        torch.manual_seed(42)
        reference = make_model(InternalMLPScalableResNet)
        torch.manual_seed(42)
        model = make_model()
        for name, value in reference.state_dict().items():
            if not name.startswith(("gate_grid.", "gate_pe.")):
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        coords = torch.rand(1, 3, 12, 12)
        features = torch.rand(1, model.decoder.input_dim, 12, 12)
        self.assertTrue(torch.allclose(reference.time_mod(features, coords),
                                       model.time_mod(features, coords, 2)))

    def test_inactive_grid_and_hidden_channels_have_no_gradient(self):
        model = make_model()
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        for level in range(3):
            model.zero_grad(set_to_none=True)
            output = model.forward_prefix(coords, level)
            F.mse_loss(output.reconstruction, target).backward()
            active = model.hierarchy.active_levels(level)
            for index, layer in enumerate(model.grid_encoder.levels):
                self.assertEqual(layer.grid.grad is not None, index in active)
            modules = [
                (model.gate_grid.expand, model.gate_grid.project, model.gate_grid.hidden_widths[level]),
                (model.gate_pe.expand, model.gate_pe.project, model.gate_pe.hidden_widths[level]),
                (model.time_mod.mlp[0], model.time_mod.mlp[2], model.time_mod.hidden_widths[level]),
            ]
            modules.extend((block.expand, block.project, model.decoder.hidden_widths[level])
                           for block in model.decoder.blocks)
            for expand, project, width in modules:
                self.assertGreater(expand.weight.grad[:width].abs().sum().item(), 0)
                self.assertEqual(torch.count_nonzero(expand.weight.grad[width:]).item(), 0)
                self.assertEqual(torch.count_nonzero(expand.bias.grad[width:]).item(), 0)
                self.assertEqual(torch.count_nonzero(project.weight.grad[:, width:]).item(), 0)

    def test_budget_and_joint_optimizer_step(self):
        model = make_model()
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
        before = model.gate_grid.expand.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.gate_grid.expand.weight))

    def test_gate_default_budget(self):
        gates = (PrefixGate(144, 60, 48), PrefixGate(144, 84, 48))
        self.assertEqual(sum(p.numel() for gate in gates for p in gate.parameters()), 20976)


if __name__ == "__main__":
    unittest.main()
