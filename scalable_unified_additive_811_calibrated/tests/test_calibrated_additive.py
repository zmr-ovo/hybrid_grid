"""The calibrated control must differ only by identity-initialized affine vectors."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from scalable_prefix_resnet.ema_training import EMADistortionBalancer, accumulate_ema_prefix_gradients
from scalable_unified_additive_811.train import build_model as build_plain
from scalable_unified_additive_811_calibrated.train import (
    build_model, build_parser, load_checkpoint, save_checkpoint,
)


def make_args(small=True):
    args = build_parser().parse_args(["-d", "/unused"])
    if small:
        args.base_resolution, args.finest_resolution = 2, 5
        args.aspect_ratio = [1, 1]
        args.time_scale = 1.0
        args.grid_feat_dim, args.pe_freq = 2, 3
        args.hidden_dim, args.gate_hidden_dim, args.time_hidden_dim = 16, 8, 8
    return args


class CalibratedAdditiveTest(unittest.TestCase):
    def test_default_budgets(self):
        model = build_model(make_args(small=False))
        self.assertEqual(model.parameter_budget(), (949733, 1628413, 2784421))
        self.assertEqual(model.active_parameter_budget(), (949733, 1628125, 2783845))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2784421)
        self.assertEqual(sum(p.numel() for p in model.calibration.parameters()), 576)

    def test_all_original_tensors_and_initial_outputs_match_plain(self):
        args = make_args()
        torch.manual_seed(42)
        plain = build_plain(args)
        torch.manual_seed(42)
        model = build_model(args)
        for key, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            for quality in range(3):
                self.assertTrue(torch.equal(plain(coords, quality).reconstruction,
                                            model(coords, quality).reconstruction))
            model.calibration.scale.fill_(2)
            model.calibration.bias.fill_(1)
            self.assertTrue(torch.equal(plain(coords, 2).reconstruction,
                                        model(coords, 2).reconstruction))

    def test_tier_specific_calibration_gradients_and_joint_update(self):
        model = build_model(make_args())
        coords, target = torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12)
        for quality in range(3):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, quality).reconstruction, target).backward()
            for parameter in model.calibration.parameters():
                if quality == 2:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertGreater(parameter.grad[quality].abs().sum().item(), 0)
                    self.assertEqual(torch.count_nonzero(parameter.grad[1 - quality]).item(), 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        metrics, objective = accumulate_ema_prefix_gradients(
            model, coords, target, F.mse_loss, EMADistortionBalancer(),
            device=torch.device("cpu"), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0)
        optimizer.step()
        self.assertTrue(all(int(state["step"]) == 1 for state in optimizer.state.values()))

    def test_checkpoint_and_plain_calibrated_isolation(self):
        args = make_args()
        model = build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        balancer = EMADistortionBalancer()
        balancer.update((0.3, 0.2, 0.1))
        with torch.no_grad():
            model.calibration.scale.fill_(1.5)
            model.calibration.bias.fill_(0.1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibrated.pth")
            save_checkpoint(path, model, optimizer, balancer, 2, (1, 2, 3), (4, 5, 6), args)
            restored = build_model(args)
            self.assertEqual(load_checkpoint(path, restored, torch.optim.Adam(restored.parameters()),
                                             EMADistortionBalancer(), args, torch.device("cpu")),
                             (3, (1, 2, 3), (4, 5, 6)))
            self.assertTrue(torch.equal(model.calibration.scale, restored.calibration.scale))
            plain = build_plain(args)
            with self.assertRaisesRegex(ValueError, "不能混用"):
                load_checkpoint(path, plain, torch.optim.Adam(plain.parameters()),
                                EMADistortionBalancer(), args, torch.device("cpu"))
            save_checkpoint(path, plain, torch.optim.Adam(plain.parameters()),
                            balancer, 2, (1, 2, 3), (4, 5, 6), args)
            with self.assertRaisesRegex(ValueError, "不能混用"):
                load_checkpoint(path, restored, optimizer, EMADistortionBalancer(), args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
