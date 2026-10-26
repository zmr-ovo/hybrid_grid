import unittest

import torch
import torch.nn.functional as F

from resnet_baseline.model_1 import ResidualSkipDecoder
from scalable_generation.hierarchy import GridHierarchy
from scalable_module_resnet.training import accumulate_prefix_gradients
from scalable_width_resnet import (
    WidthAdditiveSkipDecoder,
    WidthScalableResNet,
)
from scalable_width_resnet.train import build_parser


def make_model():
    return WidthScalableResNet(
        grid_levels=4,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=8,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
        width_ratios=(0.5, 0.75, 1.0),
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )


class WidthAdditiveSkipDecoderTest(unittest.TestCase):
    def test_full_width_is_exact_model_1_decoder(self):
        original = ResidualSkipDecoder(input_dim=10, hidden_dim=16)
        scalable = WidthAdditiveSkipDecoder(
            input_dim=10,
            hidden_dim=16,
            width_ratios=(0.5, 0.75, 1.0),
        )
        for source, target in zip(original.linear, scalable.linear):
            target.load_state_dict(source.state_dict())
        scalable.skip_proj.load_state_dict(original.skip_proj.state_dict())
        scalable.output_layer.load_state_dict(
            original.output_layer[0].state_dict(),
        )
        inputs = torch.rand(7, 10)

        expected = original(inputs)
        actual, hidden = scalable(inputs, quality_level=2)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(hidden.shape, (7, 8))

    def test_widths_are_strict_nested_prefixes(self):
        decoder = WidthAdditiveSkipDecoder(
            input_dim=10,
            hidden_dim=16,
            width_ratios=(0.5, 0.75, 1.0),
        )

        self.assertEqual(decoder.hidden_widths, (8, 12, 16))
        self.assertEqual(decoder.output_widths, (4, 6, 8))
        self.assertEqual(decoder.input_widths, (5, 8, 10))
        self.assertLess(
            decoder.active_parameter_count(0),
            decoder.active_parameter_count(1),
        )
        self.assertLess(
            decoder.active_parameter_count(1),
            decoder.active_parameter_count(2),
        )


class WidthScalableResNetTest(unittest.TestCase):
    def test_gate_and_time_dimensions_are_causal(self):
        model = make_model()

        self.assertEqual(model.grid_group_dims, (4, 2, 2))
        self.assertEqual(model.feature_widths, (22, 24, 26))
        self.assertEqual(model.base_grid_gate.linear.in_features, 22)
        self.assertEqual(model.enhancement_gates[0].linear.in_features, 24)
        self.assertEqual(model.enhancement_gates[1].linear.in_features, 26)
        self.assertEqual(
            tuple(head.out_features for head in model.time_heads),
            (44, 4, 4),
        )
        self.assertTrue(all(
            torch.count_nonzero(head.weight).item() == 0
            for head in model.time_heads
        ))

    def test_gates_start_as_identity_scaling(self):
        model = make_model()
        inputs = torch.rand(2, 3, 4, 22)

        gate = model.base_grid_gate(inputs)

        self.assertTrue(torch.equal(gate, torch.ones_like(gate)))

    def test_outputs_three_progressive_quality_levels(self):
        model = make_model()
        output = model.forward_all(torch.rand(1, 3, 4, 5))

        self.assertEqual(len(output.prefixes), 3)
        self.assertEqual(
            tuple(item.hidden_width for item in output.prefixes),
            (8, 12, 16),
        )
        self.assertTrue(all(
            item.reconstruction.shape == (1, 3, 4, 5)
            for item in output.prefixes
        ))
        self.assertEqual(output.prefixes[0].active_levels, (0, 1))
        self.assertEqual(output.prefixes[1].active_levels, (0, 1, 2))
        self.assertEqual(output.prefixes[2].active_levels, (0, 1, 2, 3))

    def test_base_does_not_evaluate_enhancement_grids(self):
        model = make_model()

        def fail(*_args, **_kwargs):
            raise AssertionError('enhancement Grid must not be evaluated')

        model.grid_encoder.levels[2].forward = fail
        output = model(torch.rand(1, 3, 3, 4), quality_level=0)

        self.assertEqual(output.active_levels, (0, 1))

    def test_base_gradient_uses_only_base_grid_and_channel_rows(self):
        model = make_model()
        model.zero_grad(set_to_none=True)
        output = model(torch.rand(1, 3, 3, 4), quality_level=0)
        output.reconstruction.mean().backward()

        grid_gradients = [
            level.grid.grad for level in model.grid_encoder.levels
        ]
        first_weight_gradient = model.decoder.linear[0].weight.grad
        base_width = model.decoder.hidden_widths[0]
        base_input_width = model.decoder.input_widths[0]

        self.assertTrue(all(
            gradient is not None for gradient in grid_gradients[:2]
        ))
        self.assertTrue(all(
            gradient is None for gradient in grid_gradients[2:]
        ))
        self.assertGreater(
            first_weight_gradient[:base_width].abs().sum().item(), 0.0,
        )
        self.assertEqual(
            first_weight_gradient[base_width:].abs().sum().item(), 0.0,
        )
        self.assertEqual(
            first_weight_gradient[:base_width, base_input_width:]
            .abs().sum().item(),
            0.0,
        )

    def test_joint_training_reaches_all_grids_and_full_width(self):
        model = make_model()
        coords = torch.rand(1, 3, 4, 5)
        target = torch.rand(1, 3, 4, 5)
        model.zero_grad(set_to_none=True)

        metrics, joint = accumulate_prefix_gradients(
            model,
            coords,
            target,
            F.mse_loss,
            (1.0, 1.0, 1.0),
            torch.device('cpu'),
            metric_fn=lambda *_: 1.0,
        )

        self.assertEqual(len(metrics), 3)
        self.assertGreater(joint, 0.0)
        self.assertTrue(all(
            level.grid.grad is not None for level in model.grid_encoder.levels
        ))
        self.assertGreater(
            model.decoder.linear[0].weight.grad.abs().sum().item(), 0.0,
        )

    def test_paper_configuration_matches_additive_skip_budget(self):
        model = WidthScalableResNet()
        total = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(total, 2781409)
        self.assertEqual(model.parameter_budget()[-1], total)
        self.assertLess(model.parameter_budget()[0], model.parameter_budget()[1])
        self.assertLess(model.parameter_budget()[1], model.parameter_budget()[2])
        self.assertEqual(
            model.architecture,
            'hybrid_grid_scalable_width_causal_additive_skip_v2',
        )

    def test_parser_defaults_to_fair_comparison_configuration(self):
        args = build_parser().parse_args(['-d', './data/example'])

        self.assertEqual(args.level_group_sizes, [7, 2, 1])
        self.assertEqual(args.width_ratios, [0.5, 0.75, 1.0])
        self.assertEqual(args.prefix_weights, [1.0, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()

