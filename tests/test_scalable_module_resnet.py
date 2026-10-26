import unittest

import torch
import torch.nn.functional as F

from scalable_generation.hierarchy import GridHierarchy
from scalable_module_resnet import (
    BaseAdditiveSkipDecoder,
    EnhancementAdditiveSkipAdapter,
    ModuleScalableResNet,
)
from scalable_module_resnet.train import build_parser, hierarchy_from_sizes
from scalable_module_resnet.training import accumulate_prefix_gradients


def make_model(levels=4):
    return ModuleScalableResNet(
        grid_levels=levels,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=8,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
        branch_dim=8,
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )


class ModuleScalableResNetTest(unittest.TestCase):
    def test_base_decoder_uses_exact_additive_skip(self):
        decoder = BaseAdditiveSkipDecoder(input_dim=6, hidden_dim=8)

        self.assertIsNone(decoder.skip_proj.bias)
        self.assertEqual(decoder.output_dim, 4)

    def test_enhancement_adapter_projects_hidden_and_stage_separately(self):
        adapter = EnhancementAdditiveSkipAdapter(
            hidden_dim=8, input_dim=3, bottleneck_dim=4,
        )

        self.assertIsNone(adapter.input_proj.bias)
        self.assertEqual(adapter.hidden_proj.in_features, 8)
        self.assertEqual(adapter.input_proj.in_features, 3)

    def test_gate_and_time_dimensions_follow_hierarchy(self):
        model = make_model()

        # Grid groups are 4/2/2 channels and PE contributes 18 channels.
        self.assertEqual(model.grid_group_dims, (4, 2, 2))
        self.assertEqual(model.base_grid_gate[0].in_features, 22)
        self.assertEqual(model.base_grid_gate[0].out_features, 4)
        self.assertEqual(model.base_pe_gate[0].out_features, 18)
        self.assertEqual(model.enhancement_gates[0][0].in_features, 20)
        self.assertEqual(model.enhancement_gates[1][0].in_features, 20)
        self.assertEqual(
            tuple(head.out_features for head in model.time_heads),
            (44, 4, 4),
        )

    def test_outputs_three_progressive_quality_levels(self):
        model = make_model()
        coords = torch.rand(1, 3, 4, 5)
        output = model.forward_all(coords)

        self.assertEqual(len(output.prefixes), 3)
        self.assertTrue(all(
            item.reconstruction.shape == (1, 3, 4, 5)
            for item in output.prefixes
        ))
        self.assertEqual(output.prefixes[0].active_levels, (0, 1))
        self.assertEqual(output.prefixes[1].active_levels, (0, 1, 2))
        self.assertEqual(output.prefixes[2].active_levels, (0, 1, 2, 3))

    def test_base_prefix_does_not_evaluate_enhancement_grids(self):
        model = make_model()
        coords = torch.rand(1, 3, 3, 4)

        def fail(*_args, **_kwargs):
            raise AssertionError('enhancement Grid must not be evaluated')

        model.grid_encoder.levels[2].forward = fail
        output = model(coords, quality_level=0)

        self.assertEqual(output.active_levels, (0, 1))

    def test_each_prefix_reaches_only_its_grid_and_stage_modules(self):
        model = make_model()
        coords = torch.rand(1, 3, 3, 4)
        expected_levels = (2, 3, 4)

        for quality_level, active_count in enumerate(expected_levels):
            model.zero_grad(set_to_none=True)
            output = model(coords, quality_level)
            output.reconstruction.mean().backward()
            grid_gradients = [
                level.grid.grad for level in model.grid_encoder.levels
            ]
            stage_gradients = [
                next(stage.parameters()).grad for stage in model.stage_blocks
            ]

            self.assertTrue(all(
                gradient is not None
                for gradient in grid_gradients[:active_count]
            ))
            self.assertTrue(all(
                gradient is None
                for gradient in grid_gradients[active_count:]
            ))
            self.assertTrue(all(
                gradient is not None
                for gradient in stage_gradients[:quality_level + 1]
            ))
            self.assertTrue(all(
                gradient is None
                for gradient in stage_gradients[quality_level + 1:]
            ))

    def test_each_prefix_reaches_only_its_gate_and_time_head(self):
        model = make_model()
        coords = torch.rand(1, 3, 3, 4)
        stage_gates = (
            model.base_grid_gate,
            *tuple(model.enhancement_gates),
        )

        for quality_level in range(3):
            model.zero_grad(set_to_none=True)
            model(coords, quality_level).reconstruction.mean().backward()
            gate_gradients = [
                next(gate.parameters()).grad for gate in stage_gates
            ]
            time_gradients = [
                head.weight.grad for head in model.time_heads
            ]

            self.assertTrue(all(
                gradient is not None
                for gradient in gate_gradients[:quality_level + 1]
            ))
            self.assertTrue(all(
                gradient is None
                for gradient in gate_gradients[quality_level + 1:]
            ))
            self.assertTrue(all(
                gradient is not None
                for gradient in time_gradients[:quality_level + 1]
            ))
            self.assertTrue(all(
                gradient is None
                for gradient in time_gradients[quality_level + 1:]
            ))

    def test_sequential_joint_training_reaches_every_stage(self):
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
        self.assertTrue(all(
            next(stage.parameters()).grad is not None
            for stage in model.stage_blocks
        ))

    def test_paper_configuration_stays_near_original_parameter_budget(self):
        model = ModuleScalableResNet()
        total = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(total, 2780133)
        self.assertLess(abs(total - 2783749) / 2783749, 0.002)
        self.assertEqual(model.parameter_budget()[-1], total)
        self.assertLess(model.parameter_budget()[0], model.parameter_budget()[1])
        self.assertLess(model.parameter_budget()[1], model.parameter_budget()[2])
        self.assertEqual(
            model.architecture,
            'hybrid_grid_scalable_hierarchical_additive_skip_v3',
        )

    def test_parser_and_hierarchy_default_to_seven_two_one(self):
        args = build_parser().parse_args(['-d', './data/example'])
        hierarchy = hierarchy_from_sizes(args.level_group_sizes, 10)

        self.assertEqual(args.level_group_sizes, [7, 2, 1])
        self.assertEqual(args.prefix_weights, [1.0, 1.0, 1.0])
        self.assertEqual(
            hierarchy.groups,
            (tuple(range(7)), (7, 8), (9,)),
        )


if __name__ == '__main__':
    unittest.main()
