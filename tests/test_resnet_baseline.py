import unittest

import torch
from torch import nn

from model import Decoder, HybridGridNet
from resnet_baseline import (
    DeepResNetDecoder,
    ResidualMLPBlock,
    ResidualSkipDecoder,
    ResNetDecoder,
    ResNetHybridGridNet,
)
from resnet_baseline.train import build_parser


def make_model(model_class=ResNetHybridGridNet):
    return model_class(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
    )


class ResNetBaselineTest(unittest.TestCase):
    def test_additive_skip_is_exactly_equivalent_to_concat(self):
        input_dim = 6
        hidden_dim = 8
        original = Decoder(input_dim, hidden_dim)
        additive = ResidualSkipDecoder(input_dim, hidden_dim)

        with torch.no_grad():
            additive.linear[0].load_state_dict(original.linear[0].state_dict())
            additive.linear[1].load_state_dict(original.linear[1].state_dict())
            additive.skip_proj.weight.copy_(original.linear[2].weight[:, :input_dim])
            additive.linear[2].weight.copy_(original.linear[2].weight[:, input_dim:])
            additive.linear[2].bias.copy_(original.linear[2].bias)
            additive.linear[3].load_state_dict(original.linear[3].state_dict())
            additive.output_layer.load_state_dict(original.output_layer.state_dict())

        inputs = torch.randn(7, input_dim)
        self.assertTrue(torch.allclose(original(inputs), additive(inputs), atol=1e-7))
        self.assertEqual(
            sum(parameter.numel() for parameter in original.parameters()),
            sum(parameter.numel() for parameter in additive.parameters()),
        )

    def test_input_injected_resnet_backpropagates_on_first_step(self):
        decoder = DeepResNetDecoder(input_dim=6, hidden_dim=8, num_blocks=2)
        output = decoder(torch.randn(7, 6))
        output.mean().backward()

        self.assertIsNotNone(decoder.input_skip.weight.grad)
        for block in decoder.blocks:
            self.assertIsNotNone(block.branch[0].weight.grad)
            self.assertGreater(block.branch[0].weight.grad.abs().sum().item(), 0.0)

    def test_residual_block_has_identity_path(self):
        block = ResidualMLPBlock(hidden_dim=8, branch_dim=4)
        for parameter in block.branch.parameters():
            nn.init.zeros_(parameter)

        inputs = torch.randn(3, 8)
        self.assertTrue(torch.equal(block(inputs), inputs))

    def test_forward_returns_rgb_and_backpropagates_to_grid(self):
        model = make_model()
        coords = torch.rand(2, 3, 4, 5)

        output = model(coords)
        output.mean().backward()

        self.assertEqual(output.shape, (2, 3, 4, 5))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))
        self.assertIsNotNone(model.grid_encoder.levels[0].grid.grad)

    def test_only_decoder_differs_from_paper_baseline(self):
        baseline = make_model(HybridGridNet)
        resnet = make_model()

        self.assertIsInstance(resnet.decoder, ResNetDecoder)
        self.assertEqual(
            sum(p.numel() for p in baseline.grid_encoder.parameters()),
            sum(p.numel() for p in resnet.grid_encoder.parameters()),
        )
        self.assertEqual(
            sum(p.numel() for p in baseline.gate_grid.parameters()),
            sum(p.numel() for p in resnet.gate_grid.parameters()),
        )
        self.assertEqual(
            sum(p.numel() for p in baseline.gate_pe.parameters()),
            sum(p.numel() for p in resnet.gate_pe.parameters()),
        )
        self.assertEqual(
            sum(p.numel() for p in baseline.time_mod.parameters()),
            sum(p.numel() for p in resnet.time_mod.parameters()),
        )

    def test_paper_configuration_matches_parameter_budget(self):
        config = dict(
            grid_levels=10,
            grid_feat_dim=6,
            base_resolution=20,
            finest_resolution=100,
            aspect_ratio=(16, 9),
            time_scale=0.1,
            pe_freq=14,
            hidden_dim=256,
        )
        baseline = HybridGridNet(**config)
        resnet = ResNetHybridGridNet(**config)
        baseline_params = sum(p.numel() for p in baseline.parameters())
        resnet_params = sum(p.numel() for p in resnet.parameters())

        self.assertEqual(baseline_params, 2783749)
        self.assertEqual(resnet_params, 2783821)
        self.assertLess(abs(resnet_params - baseline_params) / baseline_params, 0.001)

    def test_architecture_and_parser_are_independent(self):
        model = make_model()
        args = build_parser().parse_args(['-d', './data/example'])

        self.assertEqual(model.architecture, 'hybrid_grid_resnet_baseline_v1')
        self.assertEqual(args.out_dir, './output/resnet_baseline')


if __name__ == '__main__':
    unittest.main()
