"""Single-Skip Decoder topology, Unified prefixes and isolated resumes."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from scalable_ablation.internal_mlp.model import InternalPrefixMLP
from scalable_prefix_resnet.model import PrefixAdditiveSkipDecoder, PrefixScalableResNet
from scalable_prefix_resnet.ema_training import EMADistortionBalancer, accumulate_ema_prefix_gradients
from scalable_unified_additive_811.train import build_model, build_parser, load_checkpoint, save_checkpoint


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
        args.gate_hidden_dim = 8
        args.time_hidden_dim = 8
    return args


class UnifiedAdditiveTest(unittest.TestCase):
    def test_default_topology_and_counts(self):
        args = make_args(small=False)
        model = build_model(args)
        self.assertIsInstance(model.decoder, PrefixAdditiveSkipDecoder)
        self.assertEqual([(layer.in_features, layer.out_features) for layer in model.decoder.linear],
                         [(144, 256), (256, 256), (256, 256), (256, 128)])
        self.assertEqual((model.decoder.skip_proj.in_features, model.decoder.skip_proj.out_features), (144, 256))
        self.assertEqual(model.decoder.hidden_widths, (128, 192, 256))
        self.assertEqual(model.gate_grid.hidden_widths, (24, 36, 48))
        self.assertEqual(model.time_mod.hidden_widths, (32, 48, 64))
        self.assertFalse(any(isinstance(module, InternalPrefixMLP) for module in model.modules()))
        self.assertFalse(hasattr(model, "calibration"))
        self.assertEqual(model.parameter_budget(), (949445, 1627837, 2783845))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2783845)
        self.assertEqual(args.target_shares, [0.2, 0.3, 0.5])

    def test_decoder_formula_has_only_one_skip_at_third_layer(self):
        model = build_model(make_args())
        decoder = model.decoder
        x = torch.randn(7, decoder.input_dim)
        for quality, width in enumerate(decoder.hidden_widths):
            h = F.gelu(F.linear(x, decoder.linear[0].weight[:width], decoder.linear[0].bias[:width]))
            h = F.gelu(F.linear(h, decoder.linear[1].weight[:width, :width], decoder.linear[1].bias[:width]))
            h = F.gelu(F.linear(h, decoder.linear[2].weight[:width, :width], decoder.linear[2].bias[:width])
                       + F.linear(x, decoder.skip_proj.weight[:width]))
            h = F.gelu(F.linear(h, decoder.linear[3].weight[:, :width], decoder.linear[3].bias))
            expected = decoder.output_layer(h)
            self.assertTrue(torch.equal(decoder(x, quality), expected))

    def test_baseline_decoder_initialization_is_preserved(self):
        args = make_args()
        torch.manual_seed(42)
        model = build_model(args)
        torch.manual_seed(42)
        reference = PrefixScalableResNet(
            grid_levels=10, grid_feat_dim=args.grid_feat_dim,
            base_resolution=args.base_resolution, finest_resolution=args.finest_resolution,
            aspect_ratio=tuple(args.aspect_ratio), time_scale=args.time_scale,
            pe_freq=args.pe_freq, hidden_dim=args.hidden_dim,
        )
        for key, value in reference.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, model.decoder.state_dict()[key]), key)

    def test_grid_and_hidden_gradient_routing(self):
        model = build_model(make_args())
        coords, target = torch.rand(1, 3, 12, 12), torch.rand(1, 3, 12, 12)
        for quality, count in enumerate((8, 9, 10)):
            model.zero_grad(set_to_none=True)
            F.mse_loss(model(coords, quality).reconstruction, target).backward()
            for index, level in enumerate(model.grid_encoder.levels):
                self.assertEqual(level.grid.grad is not None, index < count)
            width = model.decoder.hidden_widths[quality]
            for layer in model.decoder.linear[:3]:
                self.assertEqual(torch.count_nonzero(layer.weight.grad[width:]).item(), 0)
            for layer in model.decoder.linear[1:3]:
                self.assertEqual(torch.count_nonzero(layer.weight.grad[:, width:]).item(), 0)
            self.assertEqual(torch.count_nonzero(model.decoder.skip_proj.weight.grad[width:]).item(), 0)
            self.assertEqual(torch.count_nonzero(model.decoder.linear[3].weight.grad[:, width:]).item(), 0)
            for expand, project, widths in (
                (model.gate_grid.expand, model.gate_grid.project, model.gate_grid.hidden_widths),
                (model.gate_pe.expand, model.gate_pe.project, model.gate_pe.hidden_widths),
                (model.time_mod.mlp[0], model.time_mod.mlp[2], model.time_mod.hidden_widths),
            ):
                self.assertEqual(torch.count_nonzero(expand.weight.grad[widths[quality]:]).item(), 0)
                self.assertEqual(torch.count_nonzero(project.weight.grad[:, widths[quality]:]).item(), 0)

    def test_joint_step_and_checkpoint(self):
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
            path = os.path.join(directory, "plain.pth")
            save_checkpoint(path, model, optimizer, balancer, 1, (1, 2, 3), (4, 5, 6), args)
            restored = build_model(args)
            opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
            self.assertEqual(load_checkpoint(path, restored, opt, EMADistortionBalancer(), args,
                                             torch.device("cpu")), (2, (1, 2, 3), (4, 5, 6)))
            self.assertTrue(all(state["step"].device.type == "cpu" for state in opt.state.values()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            torch.save({"architecture": "hybrid_grid_ablation_unified_internal_width_v1"}, path)
            with self.assertRaisesRegex(ValueError, "不能混用"):
                load_checkpoint(path, restored, opt, EMADistortionBalancer(), args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
