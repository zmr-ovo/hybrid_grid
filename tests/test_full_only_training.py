import unittest

import torch
import torch.nn.functional as F

from scalable_generation.hierarchy import GridHierarchy
from scalable_module_resnet import ModuleScalableResNet
from test.train_full_only import build_parser
from test.training import backward_full_only


def make_model():
    return ModuleScalableResNet(
        grid_levels=4,
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


class FullOnlyTrainingTest(unittest.TestCase):
    def test_full_only_reaches_all_grid_levels_and_stages(self):
        model = make_model()
        coords = torch.rand(1, 3, 4, 5)
        target = torch.rand(1, 3, 4, 5)

        model.zero_grad(set_to_none=True)
        _, metrics = backward_full_only(
            model, coords, target, F.mse_loss,
        )

        self.assertGreater(metrics.distortion, 0.0)
        self.assertTrue(all(
            level.grid.grad is not None
            for level in model.grid_encoder.levels
        ))
        self.assertTrue(all(
            next(stage.parameters()).grad is not None
            for stage in model.stage_blocks
        ))

    def test_parser_defaults_to_capacity_test_configuration(self):
        args = build_parser().parse_args(['-d', './data/example'])

        self.assertEqual(args.level_group_sizes, [7, 2, 1])
        self.assertEqual(args.out_dir, './output/test/full_only')
        self.assertIsNone(args.branch_dim)
        self.assertEqual(args.target_psnr, 36.8)


if __name__ == '__main__':
    unittest.main()
