import unittest

import torch

from resnet_baseline.model_1 import ResidualSkipHybridGridNet
from scalable_prefix_resnet.evaluate_full_compatibility import (
    SOURCE_ARCHITECTURE,
    build_compatible_models,
    hierarchy_from_sizes,
)


def checkpoint(architecture=SOURCE_ARCHITECTURE):
    config = {
        "grid_levels": 4,
        "grid_feat_dim": 2,
        "base_resolution": 4,
        "finest_resolution": 8,
        "aspect_ratio": [1, 1],
        "time_scale": 1.0,
        "pe_freq": 3,
        "hidden_dim": 16,
    }
    model = ResidualSkipHybridGridNet(**config)
    return {
        "architecture": architecture,
        "state_dict": model.state_dict(),
        "config": config,
    }


class PrefixCheckpointCompatibilityTest(unittest.TestCase):
    def test_checkpoint_full_path_is_exact(self):
        baseline, prefix, _ = build_compatible_models(
            checkpoint(),
            width_ratios=(0.5, 0.75, 1.0),
            level_group_sizes=(2, 1, 1),
        )
        coords = torch.rand(1, 3, 5, 6)

        expected = baseline(coords)
        actual = prefix.forward_prefix(coords, 2).reconstruction

        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in prefix.parameters()),
        )

    def test_rejects_a_different_source_architecture(self):
        with self.assertRaisesRegex(ValueError, "expected checkpoint architecture"):
            build_compatible_models(
                checkpoint("other_model"),
                width_ratios=(0.5, 0.75, 1.0),
                level_group_sizes=(2, 1, 1),
            )

    def test_group_sizes_must_cover_every_grid_level(self):
        with self.assertRaisesRegex(ValueError, "must sum to grid_levels"):
            hierarchy_from_sizes(10, (6, 2, 1))


if __name__ == "__main__":
    unittest.main()
