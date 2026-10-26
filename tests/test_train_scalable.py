import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from compression.model import CompressedHybridGridNet
from model import HybridGridNet
from scalable_generation import GridHierarchy, ScalableHybridGridNet
from scalable_generation.training import accumulate_prefix_gradients
from train_scalable import build_parser, hierarchy_from_sizes


class ScalableTrainingConfigTest(unittest.TestCase):
    def test_builds_default_seven_two_one_hierarchy(self):
        hierarchy = hierarchy_from_sizes([7, 2, 1], 10)

        self.assertEqual(
            hierarchy.groups,
            (tuple(range(7)), (7, 8), (9,)),
        )

    def test_rejects_invalid_group_sizes(self):
        invalid = (
            ([], 10),
            ([7, 2], 10),
            ([7, 2, 0, 1], 10),
            ([7, -2, 5], 10),
        )
        for sizes, levels in invalid:
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    hierarchy_from_sizes(sizes, levels)

    def test_parser_exposes_scheme_a_options(self):
        args = build_parser().parse_args(['-d', './data/example'])

        self.assertEqual(args.level_group_sizes, [7, 2, 1])
        self.assertEqual(args.distortion_weights, [1.0, 1.0, 1.0])
        self.assertEqual(args.rate_weights, [1.0, 1.0, 1.0])
        self.assertEqual(args.out_dir, './output/scalable')


class SequentialPrefixBackwardTest(unittest.TestCase):
    def test_sequential_backward_accumulates_gradients_for_all_levels(self):
        baseline = HybridGridNet(
            grid_levels=4,
            grid_feat_dim=2,
            base_resolution=4,
            finest_resolution=8,
            aspect_ratio=(1, 1),
            time_scale=1.0,
            pe_freq=3,
            hidden_dim=16,
        )
        compressed = CompressedHybridGridNet(
            baseline,
            quant_steps=0.1,
            entropy_model='cross_scale_recursive',
        )
        model = ScalableHybridGridNet(
            compressed, GridHierarchy(((0, 1), (2,), (3,))),
        ).train()
        coords = torch.rand(1, 3, 5, 6)
        target = torch.rand(1, 3, 5, 6)

        model.zero_grad(set_to_none=True)
        with patch(
            'scalable_generation.training.msssim_fn',
            return_value=1.0,
        ):
            metrics, joint_total = accumulate_prefix_gradients(
                model,
                coords,
                target,
                F.mse_loss,
                1e-3,
                (0.5, 0.75, 1.5),
                (1.0, 1.0, 1.0),
                'symbols',
                torch.device('cpu'),
            )

        self.assertEqual(len(metrics), 3)
        self.assertGreater(joint_total, 0.0)
        self.assertTrue(all(
            level.grid.grad is not None
            for level in baseline.grid_encoder.levels
        ))


if __name__ == '__main__':
    unittest.main()
