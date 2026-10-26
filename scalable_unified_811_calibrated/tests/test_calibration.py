"""Identity, bypass, gradient routing and resume tests; run on CPU."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from scalable_prefix_resnet.ema_training import EMADistortionBalancer, accumulate_ema_prefix_gradients
from scalable_unified_811.train import build_model as build_reference
from scalable_unified_811_calibrated.model import PrefixFeatureCalibration
from scalable_unified_811_calibrated.train import (
    build_model, build_parser, load_checkpoint, save_checkpoint,
)


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


class CalibrationTest(unittest.TestCase):
    def test_affine_identity_and_full_bypass(self):
        module = PrefixFeatureCalibration(4)
        features = torch.randn(1, 3, 5, 4, requires_grad=True)
        for level in range(3):
            self.assertTrue(torch.equal(module(features, level), features))
        with torch.no_grad():
            module.scale[0].fill_(2)
            module.bias[0].fill_(3)
        self.assertTrue(torch.equal(module(features, 0), features * 2 + 3))
        self.assertTrue(torch.equal(module(features, 1), features))
        self.assertIs(module(features, 2), features)
        module(features, 2).sum().backward()
        self.assertIsNone(module.scale.grad)
        self.assertIsNone(module.bias.grad)
        self.assertIsNotNone(features.grad)

    def test_default_counts(self):
        model = build_model(make_args(small=False))
        self.assertEqual(sum(p.numel() for p in model.calibration.parameters()), 576)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2784885)
        self.assertEqual(model.parameter_budget(), (1009613, 1666777, 2784885))
        self.assertEqual(model.active_parameter_budget(), (1009613, 1666489, 2784309))

    def test_reference_initialization_and_outputs_identical_at_every_tier(self):
        args = make_args()
        torch.manual_seed(42)
        reference = build_reference(args)
        torch.manual_seed(42)
        model = build_model(args)
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            for level in range(3):
                self.assertTrue(torch.equal(model(coords, level).reconstruction,
                                            reference(coords, level).reconstruction))
            model.calibration.scale.fill_(2)
            model.calibration.bias.fill_(0.5)
            self.assertTrue(torch.equal(model(coords, 2).reconstruction,
                                        reference(coords, 2).reconstruction))

    def test_calibration_and_grid_gradients_only_follow_selected_tier(self):
        torch.manual_seed(42)
        model = build_model(make_args())
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        for level, active_count in enumerate((8, 9, 10)):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, level).reconstruction, target).backward()
            for index, grid in enumerate(model.grid_encoder.levels):
                self.assertEqual(grid.grid.grad is not None, index < active_count)
            for parameter in model.calibration.parameters():
                if level == 2:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertGreater(parameter.grad[level].abs().sum().item(), 0)
                    self.assertEqual(torch.count_nonzero(parameter.grad[1 - level]).item(), 0)

    def test_one_joint_update_and_checkpoint_round_trip(self):
        args = make_args()
        model = build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        balancer = EMADistortionBalancer()
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        optimizer.zero_grad(set_to_none=True)
        metrics, value = accumulate_ema_prefix_gradients(
            model, coords, target, F.mse_loss, balancer,
            device=torch.device("cpu"), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(value, 0)
        optimizer.step()
        self.assertTrue(all(int(state["step"]) == 1 for state in optimizer.state.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibrated.pth")
            save_checkpoint(path, model, optimizer, balancer, 4, (1, 2, 3), (4, 5, 6), args)
            restored = build_model(args)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
            restored_balancer = EMADistortionBalancer()
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer, restored_balancer,
                                             args, torch.device("cpu")), (5, (1, 2, 3), (4, 5, 6)))
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)
            self.assertEqual(balancer.state_dict(), restored_balancer.state_dict())
            self.assertTrue(all(state["step"].device.type == "cpu"
                                for state in restored_optimizer.state.values()))
            # Resume must reject an uncalibrated checkpoint before touching model state.
            torch.save({"architecture": build_reference(args).architecture}, path)
            with self.assertRaisesRegex(ValueError, "不能混用"):
                load_checkpoint(path, restored, restored_optimizer, restored_balancer,
                                args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
