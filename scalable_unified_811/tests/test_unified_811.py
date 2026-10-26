"""CPU checks for the isolated grouping-only Unified experiment."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scalable_ablation.unified_internal.model import UnifiedInternalScalableResNet
from scalable_generation.hierarchy import default_hierarchy
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer, accumulate_ema_prefix_gradients,
)
from scalable_unified_811.train import build_model, build_parser, train, validate_resume


def make_args(small=True):
    args = build_parser().parse_args(["-d", "/unused"])
    if small:
        args.base_resolution = 2
        args.finest_resolution = 5
        args.aspect_ratio = [1, 1]
        args.time_scale = 1.0
        args.grid_feat_dim = 2
        args.pe_freq = 3
        args.hidden_dim = 16
        args.feature_dim = 8
        args.gate_hidden_dim = 8
        args.time_hidden_dim = 8
    return args


def checkpoint_metadata(model):
    return dict(
        architecture=model.architecture,
        training_strategy="unified_internal_ema_distortion_v1",
        hierarchy=model.hierarchy.groups,
        hidden_widths=model.decoder.hidden_widths,
        gate_hidden_widths=model.gate_grid.hidden_widths,
        time_hidden_widths=model.time_mod.hidden_widths,
    )


class Unified811Test(unittest.TestCase):
    def test_default_budget_and_module_widths(self):
        args = make_args(small=False)
        model = build_model(args)
        self.assertEqual(model.parameter_budget(), (1009325, 1666201, 2784309))
        self.assertEqual(model.decoder.hidden_widths, (120, 180, 240))
        self.assertEqual(model.gate_grid.hidden_widths, (24, 36, 48))
        self.assertEqual(model.time_mod.hidden_widths, (32, 48, 64))
        self.assertEqual(args.target_shares, [0.2, 0.3, 0.5])
        self.assertEqual(args.out_dir, "./output/scalable_unified_811")
        self.assertEqual((args.epochs, args.lr, args.warmup), (300, 0.005, 0.2))

    def test_initialization_and_mid_full_match_unified_721(self):
        args = make_args()
        torch.manual_seed(42)
        model = build_model(args)
        torch.manual_seed(42)
        reference = build_model(args)
        reference.hierarchy = default_hierarchy(10)
        self.assertIsInstance(model, UnifiedInternalScalableResNet)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[key]), key)
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            for quality in (1, 2):
                self.assertTrue(torch.equal(model(coords, quality).reconstruction,
                                            reference(coords, quality).reconstruction))
        self.assertEqual(model.hierarchy.active_levels(0), tuple(range(8)))
        self.assertEqual(reference.hierarchy.active_levels(0), tuple(range(7)))

    def test_gradient_routing_and_single_joint_optimizer_step(self):
        torch.manual_seed(42)
        model = build_model(make_args())
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        for quality, active in enumerate((8, 9, 10)):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, quality).reconstruction, target).backward()
            for index, grid in enumerate(model.grid_encoder.levels):
                self.assertEqual(grid.grid.grad is not None, index < active)
            modules = [
                (model.gate_grid.expand, model.gate_grid.project, model.gate_grid.hidden_widths[quality]),
                (model.gate_pe.expand, model.gate_pe.project, model.gate_pe.hidden_widths[quality]),
                (model.time_mod.mlp[0], model.time_mod.mlp[2], model.time_mod.hidden_widths[quality]),
            ]
            modules.extend((block.expand, block.project, model.decoder.hidden_widths[quality])
                           for block in model.decoder.blocks)
            for expand, project, width in modules:
                self.assertEqual(torch.count_nonzero(expand.weight.grad[width:]).item(), 0)
                self.assertEqual(torch.count_nonzero(project.weight.grad[:, width:]).item(), 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        metrics, objective = accumulate_ema_prefix_gradients(
            model, coords, target, F.mse_loss, EMADistortionBalancer(),
            device=torch.device("cpu"), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0)
        self.assertTrue(all(grid.grid.grad is not None for grid in model.grid_encoder.levels))
        optimizer.step()
        self.assertTrue(all(int(state["step"]) == 1 for state in optimizer.state.values()))

    def test_resume_accepts_811_and_rejects_721_or_wrong_model(self):
        model = build_model(make_args())
        checkpoint = checkpoint_metadata(model)
        validate_resume(checkpoint, model)
        with self.assertRaisesRegex(ValueError, "721"):
            validate_resume(dict(checkpoint, hierarchy=default_hierarchy(10).groups), model)
        with self.assertRaisesRegex(ValueError, "不是Unified模型"):
            validate_resume(dict(checkpoint, architecture="original_prefix"), model)
        with self.assertRaisesRegex(ValueError, "宽度不一致"):
            validate_resume(dict(checkpoint, gate_hidden_widths=(1, 2, 3)), model)

    def test_delegates_to_unchanged_reference_trainer(self):
        args = make_args()
        with patch("scalable_unified_811.train.train_reference") as reference:
            train(args)
        reference.assert_called_once_with(args)

    def test_rejects_other_grouping(self):
        args = make_args()
        args.level_group_sizes = [7, 2, 1]
        with self.assertRaises(ValueError):
            build_model(args)


if __name__ == "__main__":
    unittest.main()
