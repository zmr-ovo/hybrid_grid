"""Original Gate/time, no PE, single Additive Skip, calibration and resume."""

import os
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from model import TemporalModulation
from scalable_prefix_resnet.model import PrefixAdditiveSkipDecoder
from scalable_prefix_resnet.ema_training import EMADistortionBalancer, accumulate_ema_prefix_gradients
from scalable_prefix_811_no_pe_calibrated.train import (
    build_model, build_parser, save_checkpoint, load_checkpoint,
)


def make_args(small=True):
    args = build_parser().parse_args(["-d", "/unused"])
    if small:
        args.base_resolution, args.finest_resolution = 2, 5
        args.aspect_ratio = [1, 1]
        args.time_scale = 1.0
        args.grid_feat_dim = 2
        args.hidden_dim = 16
    return args


class OriginalPrefixNoPETest(unittest.TestCase):
    def test_default_topology_and_parameter_counts(self):
        args = make_args(small=False)
        model = build_model(args)
        self.assertEqual(args.pe_freq, 0)
        self.assertFalse(hasattr(model, "pe_encoder"))
        self.assertFalse(hasattr(model, "gate_pe"))
        self.assertFalse(any(key.startswith(("pe_encoder.", "gate_pe.")) for key in model.state_dict()))
        self.assertIsInstance(model.gate_grid, nn.Sequential)
        self.assertEqual(len(model.gate_grid), 2)
        self.assertEqual((model.gate_grid[0].in_features, model.gate_grid[0].out_features), (60, 60))
        self.assertIsInstance(model.time_mod, TemporalModulation)
        self.assertEqual(model.time_mod.mlp[0].out_features, 64)
        self.assertEqual(model.time_mod.mlp[2].out_features, 120)
        self.assertIsInstance(model.decoder, PrefixAdditiveSkipDecoder)
        self.assertEqual(model.decoder.hidden_widths, (128, 192, 256))
        self.assertEqual([(layer.in_features, layer.out_features) for layer in model.decoder.linear],
                         [(60, 256), (256, 256), (256, 256), (256, 128)])
        self.assertFalse(hasattr(model.decoder, "blocks"))
        self.assertEqual(sum(p.numel() for p in model.calibration.parameters()), 240)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2712827)
        self.assertEqual(model.parameter_budget(), (919507, 1577419, 2712827))
        self.assertEqual(model.active_parameter_budget(), (919507, 1577299, 2712587))

    def test_forward_formula_identity_calibration_and_full_bypass(self):
        model = build_model(make_args())
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            for quality in range(3):
                raw, active = model._grid_prefix(coords.permute(0, 2, 3, 1), quality, None)
                gated = model.gate_grid(raw) * raw
                timed = model.time_mod(gated.permute(0, 3, 1, 2), coords).permute(0, 2, 3, 1)
                self.assertTrue(torch.equal(model.calibration(timed, quality), timed))
                expected = model.decoder(timed.reshape(144, -1), quality)
                expected = expected.view(1, 12, 12, 3).permute(0, 3, 1, 2)
                output = model(coords, quality)
                self.assertEqual(output.active_levels, active)
                self.assertTrue(torch.equal(expected, output.reconstruction))
            full = model(coords, 2).reconstruction.clone()
            model.calibration.scale.fill_(2)
            model.calibration.bias.fill_(1)
            self.assertTrue(torch.equal(full, model(coords, 2).reconstruction))

    def test_prefix_gradients_and_only_own_calibration_row(self):
        torch.manual_seed(42)
        model = build_model(make_args())
        coords, pixels = torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12)
        for quality, count in enumerate((8, 9, 10)):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, quality).reconstruction, pixels).backward()
            for index, level in enumerate(model.grid_encoder.levels):
                self.assertEqual(level.grid.grad is not None, index < count)
            self.assertIsNotNone(model.gate_grid[0].weight.grad)
            self.assertIsNotNone(model.time_mod.mlp[2].weight.grad)
            for parameter in model.calibration.parameters():
                if quality == 2:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertGreater(parameter.grad[quality].abs().sum().item(), 0)
                    self.assertEqual(torch.count_nonzero(parameter.grad[1 - quality]).item(), 0)
            width = model.decoder.hidden_widths[quality]
            self.assertEqual(torch.count_nonzero(model.decoder.linear[0].weight.grad[width:]).item(), 0)
            self.assertEqual(torch.count_nonzero(model.decoder.linear[3].weight.grad[:, width:]).item(), 0)

    def test_joint_step_and_checkpoint_round_trip(self):
        args = make_args()
        model = build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        balancer = EMADistortionBalancer()
        optimizer.zero_grad(set_to_none=True)
        metrics, objective = accumulate_ema_prefix_gradients(
            model, torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12),
            F.mse_loss, balancer, device=torch.device("cpu"), metric_fn=lambda *_: 1.0,
        )
        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0)
        optimizer.step()
        self.assertTrue(all(int(state["step"]) == 1 for state in optimizer.state.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "no_pe.pth")
            save_checkpoint(path, model, optimizer, balancer, 2, (1, 2, 3), (4, 5, 6), args)
            restored = build_model(args)
            opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
            ema = EMADistortionBalancer()
            self.assertEqual(load_checkpoint(path, restored, opt, ema, args, torch.device("cpu")),
                             (3, (1, 2, 3), (4, 5, 6)))
            self.assertEqual(ema.state_dict(), balancer.state_dict())
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertTrue(all(state["step"].device.type == "cpu" for state in opt.state.values()))
            torch.save({"architecture": "hybrid_grid_unified_811_no_pe_calibrated_v1"}, path)
            with self.assertRaisesRegex(ValueError, "不能混用"):
                load_checkpoint(path, restored, opt, ema, args, torch.device("cpu"))

    def test_rejects_pe_or_721_config(self):
        args = make_args()
        args.pe_freq = 14
        with self.assertRaisesRegex(ValueError, "无PE"):
            build_model(args)
        args.pe_freq = 0
        args.level_group_sizes = [7, 2, 1]
        with self.assertRaises(ValueError):
            build_model(args)


if __name__ == "__main__":
    unittest.main()
