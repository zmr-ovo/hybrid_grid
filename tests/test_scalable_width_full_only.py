import unittest

import torch
import torch.nn.functional as F

from scalable_generation.hierarchy import GridHierarchy
from scalable_width_resnet.full_only_training import backward_full_only
from scalable_width_resnet.model import WidthScalableResNet
from scalable_width_resnet.train_full_only import build_parser


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


class WidthFullOnlyTest(unittest.TestCase):
    def test_parser_uses_independent_output(self):
        args = build_parser().parse_args(["-d", "frames"])
        self.assertEqual(args.out_dir, "./output/scalable_width_full_only")
        self.assertEqual(args.level_group_sizes, [7, 2, 1])
        self.assertEqual(args.width_ratios, [0.5, 0.75, 1.0])

    def test_full_backward_reaches_every_grid_and_stage(self):
        model = make_model()
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        model.zero_grad(set_to_none=True)

        reconstruction, metrics = backward_full_only(
            model, coords, target, F.mse_loss,
        )

        self.assertEqual(reconstruction.shape, target.shape)
        self.assertGreater(metrics.distortion, 0.0)
        self.assertTrue(all(
            level.grid.grad is not None
            for level in model.grid_encoder.levels
        ))
        self.assertTrue(all(
            head.weight.grad is not None
            for head in model.time_heads
        ))
        self.assertTrue(all(
            gate.linear.weight.grad is not None
            for gate in model.enhancement_gates
        ))


if __name__ == "__main__":
    unittest.main()
