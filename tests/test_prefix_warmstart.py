import unittest

import torch

from resnet_baseline.model_1 import ResidualSkipHybridGridNet
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.model import PrefixScalableResNet
from scalable_prefix_resnet.train_warmstart import build_parser
from scalable_prefix_resnet.warmstart_training import (
    accumulate_warmstart_gradients,
    validate_distortion_weights,
)


def models():
    kwargs = dict(
        grid_levels=4,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=8,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
    )
    teacher = ResidualSkipHybridGridNet(**kwargs)
    student = PrefixScalableResNet(
        **kwargs,
        width_ratios=(0.5, 0.75, 1.0),
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )
    student.load_state_dict(teacher.state_dict())
    teacher.requires_grad_(False)
    return teacher, student


class PrefixWarmstartTest(unittest.TestCase):
    def test_sequential_backward_updates_student_not_teacher(self):
        teacher, student = models()
        coords = torch.rand(1, 3, 5, 6)
        target = torch.rand(1, 3, 5, 6)
        student.zero_grad(set_to_none=True)

        metrics, total, anchor = accumulate_warmstart_gradients(
            student,
            teacher,
            coords,
            target,
            metric_fn=lambda *_: 1.0,
        )

        self.assertEqual(len(metrics), 3)
        self.assertGreater(total, 0.0)
        self.assertEqual(anchor, 0.0)
        self.assertTrue(all(
            level.grid.grad is not None
            for level in student.grid_encoder.levels
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in teacher.parameters()
        ))

    def test_validates_weights(self):
        self.assertEqual(
            validate_distortion_weights((0.25, 0.5, 1.0)),
            (0.25, 0.5, 1.0),
        )
        with self.assertRaises(ValueError):
            validate_distortion_weights((1.0, 1.0))

    def test_parser_defaults_to_requested_experiment(self):
        args = build_parser().parse_args([
            "--checkpoint", "model.pth", "-d", "frames",
        ])
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.prefix_weights, (0.25, 0.5, 1.0))
        self.assertEqual(args.anchor_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
