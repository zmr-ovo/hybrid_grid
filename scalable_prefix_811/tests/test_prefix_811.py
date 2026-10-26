"""Configuration, exact Full equivalence and gradient routing for 8/1/1."""

import unittest

import torch
import torch.nn.functional as F

from scalable_prefix_811.train import build_model, build_parser, validate_resume
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer, accumulate_ema_prefix_gradients,
)
from scalable_prefix_resnet.model import PrefixScalableResNet


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
    return args


class Prefix811Test(unittest.TestCase):
    def test_defaults_and_paper_parameter_budget(self):
        args = make_args(small=False)
        model = build_model(args)
        self.assertEqual(args.target_shares, [0.2, 0.3, 0.5])
        self.assertEqual(args.out_dir, "./output/scalable_prefix_811")
        self.assertEqual(model.parameter_budget(), (969045, 1637589, 2783749))
        self.assertEqual(tuple(len(model.hierarchy.active_levels(k)) for k in range(3)),
                         (8, 9, 10))

    def test_full_path_and_initial_parameters_match_721(self):
        args = make_args()
        torch.manual_seed(42)
        model = build_model(args)
        torch.manual_seed(42)
        reference = PrefixScalableResNet(
            grid_feat_dim=args.grid_feat_dim,
            base_resolution=args.base_resolution,
            finest_resolution=args.finest_resolution,
            aspect_ratio=tuple(args.aspect_ratio),
            time_scale=args.time_scale,
            pe_freq=args.pe_freq,
            hidden_dim=args.hidden_dim,
        )
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, reference.state_dict()[key]), key)
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(coords, 2).reconstruction,
                                        reference(coords, 2).reconstruction))

    def test_gradient_routing_and_one_joint_update(self):
        model = build_model(make_args())
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        for quality, active_count in enumerate((8, 9, 10)):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, quality).reconstruction, target).backward()
            for index, level in enumerate(model.grid_encoder.levels):
                self.assertEqual(level.grid.grad is not None, index < active_count)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        metrics, objective = accumulate_ema_prefix_gradients(
            model, coords, target, F.mse_loss, EMADistortionBalancer(),
            device=torch.device("cpu"), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0)
        self.assertTrue(all(level.grid.grad is not None
                            for level in model.grid_encoder.levels))
        optimizer.step()
        self.assertTrue(all(int(state["step"]) == 1
                            for state in optimizer.state.values()))

    def test_resume_rejects_other_hierarchy_and_widths(self):
        model = build_model(make_args())
        checkpoint = {"hierarchy": model.hierarchy.groups,
                      "hidden_widths": model.decoder.hidden_widths}
        validate_resume(checkpoint, model)
        with self.assertRaises(ValueError):
            validate_resume(dict(checkpoint, hierarchy=(tuple(range(7)), (7, 8), (9,))), model)
        with self.assertRaises(ValueError):
            validate_resume(dict(checkpoint, hidden_widths=(4, 8, 16)), model)

    def test_rejects_non_811_configuration(self):
        args = make_args()
        args.level_group_sizes = [7, 2, 1]
        with self.assertRaises(ValueError):
            build_model(args)


if __name__ == "__main__":
    unittest.main()
