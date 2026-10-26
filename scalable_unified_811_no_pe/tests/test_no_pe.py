"""Check the true Grid-only feature path, budgets, gradients and resumes."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from scalable_prefix_resnet.ema_training import EMADistortionBalancer, accumulate_ema_prefix_gradients
from scalable_unified_811_calibrated.train import build_model as build_pe_model
from scalable_unified_811_no_pe.train import (
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
        args.hidden_dim = 16
        args.feature_dim = 8
        args.gate_hidden_dim = 8
        args.time_hidden_dim = 8
    return args


class NoPEModelTest(unittest.TestCase):
    def test_default_structure_and_nearly_matched_budget(self):
        args = make_args(small=False)
        model = build_model(args)
        self.assertEqual(model.decoder.hidden_widths, (150, 225, 300))
        self.assertEqual(model.gate_grid.hidden_widths, (24, 36, 48))
        self.assertEqual(model.time_mod.hidden_widths, (32, 48, 64))
        self.assertEqual(model.decoder.input_dim, 60)
        self.assertEqual(model.gate_grid.expand.in_features, 60)
        self.assertEqual(model.gate_grid.project.out_features, 60)
        self.assertEqual(model.time_mod.mlp[2].out_features, 120)
        self.assertFalse(hasattr(model, "pe_encoder"))
        self.assertFalse(hasattr(model, "gate_pe"))
        self.assertFalse(any(key.startswith(("pe_encoder.", "gate_pe."))
                             for key in model.state_dict()))
        self.assertEqual(sum(p.numel() for p in model.calibration.parameters()), 240)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2783263)
        self.assertEqual(model.parameter_budget(), (997917, 1660034, 2783263))
        self.assertEqual(model.active_parameter_budget(), (997917, 1659914, 2783023))
        args.hidden_dim, args.pe_freq = 240, 14
        reference = build_pe_model(args)
        self.assertEqual(sum(p.numel() for p in reference.parameters()) - model.parameter_budget()[2], 1622)

    def test_same_grid_initialization_as_pe_reference(self):
        args = make_args()
        torch.manual_seed(42)
        model = build_model(args)
        args.pe_freq = 3
        torch.manual_seed(42)
        reference = build_pe_model(args)
        for key, value in model.grid_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, reference.grid_encoder.state_dict()[key]), key)

    def test_forward_uses_grid_gate_time_calibration_decoder_in_order(self):
        model = build_model(make_args())
        coords = torch.rand(1, 3, 12, 12)
        with torch.no_grad():
            for quality in range(3):
                raw, active = model._grid_prefix(coords.permute(0, 2, 3, 1), quality, None)
                gated = raw * model.gate_grid(raw, quality)
                timed = model.time_mod(gated.permute(0, 3, 1, 2), coords, quality).permute(0, 2, 3, 1)
                self.assertTrue(torch.equal(model.calibration(timed, quality), timed))
                expected = model.decoder(timed.reshape(144, -1), quality)
                expected = expected.view(1, 12, 12, 3).permute(0, 3, 1, 2)
                output = model(coords, quality)
                self.assertEqual(output.active_levels, active)
                self.assertTrue(torch.equal(output.reconstruction, expected))
                self.assertTrue(torch.isfinite(output.reconstruction).all().item())
            before = model(coords, 2).reconstruction.clone()
            model.calibration.scale.fill_(2)
            model.calibration.bias.fill_(1)
            self.assertTrue(torch.equal(before, model(coords, 2).reconstruction))

    def test_grid_and_hidden_prefix_gradient_routing(self):
        torch.manual_seed(42)
        model = build_model(make_args())
        coords, target = torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12)
        for quality, count in enumerate((8, 9, 10)):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, quality).reconstruction, target).backward()
            for index, level in enumerate(model.grid_encoder.levels):
                self.assertEqual(level.grid.grad is not None, index < count)
            for parameter in model.calibration.parameters():
                if quality == 2:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertGreater(parameter.grad[quality].abs().sum().item(), 0)
                    self.assertEqual(torch.count_nonzero(parameter.grad[1 - quality]).item(), 0)
            modules = [(model.gate_grid.expand, model.gate_grid.project,
                        model.gate_grid.hidden_widths[quality]),
                       (model.time_mod.mlp[0], model.time_mod.mlp[2],
                        model.time_mod.hidden_widths[quality])]
            modules.extend((block.expand, block.project, model.decoder.hidden_widths[quality])
                           for block in model.decoder.blocks)
            for expand, project, width in modules:
                self.assertEqual(torch.count_nonzero(expand.weight.grad[width:]).item(), 0)
                self.assertEqual(torch.count_nonzero(project.weight.grad[:, width:]).item(), 0)

    def test_joint_step_and_resume_with_cpu_optimizer_steps(self):
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
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
            restored_balancer = EMADistortionBalancer()
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer, restored_balancer,
                                             args, torch.device("cpu")), (3, (1, 2, 3), (4, 5, 6)))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertEqual(balancer.state_dict(), restored_balancer.state_dict())
            self.assertTrue(all(state["step"].device.type == "cpu"
                                for state in restored_optimizer.state.values()))
            torch.save({"architecture": "hybrid_grid_unified_811_calibrated_v1"}, path)
            with self.assertRaisesRegex(ValueError, "不能混用"):
                load_checkpoint(path, restored, restored_optimizer, restored_balancer, args, torch.device("cpu"))

    def test_rejects_pe_configuration_and_wrong_grouping(self):
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
