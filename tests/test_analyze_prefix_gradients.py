import unittest

import torch

from analyze_prefix_gradients import (
    build_parameter_groups,
    vector_pair_statistics,
)
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.model import PrefixScalableResNet


def make_model():
    return PrefixScalableResNet(
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


class GradientConflictStatisticsTest(unittest.TestCase):
    def test_pair_statistics_detect_alignment_and_conflict(self):
        aligned = vector_pair_statistics(
            torch.tensor([1.0, -2.0]), torch.tensor([2.0, -4.0]),
        )
        opposed = vector_pair_statistics(
            torch.tensor([1.0, -2.0]), torch.tensor([-1.0, 2.0]),
        )

        self.assertAlmostEqual(aligned["cosine"], 1.0)
        self.assertAlmostEqual(aligned["opposed_element_ratio"], 0.0)
        self.assertAlmostEqual(opposed["cosine"], -1.0)
        self.assertAlmostEqual(opposed["opposed_element_ratio"], 1.0)

    def test_zero_gradient_returns_na(self):
        values = vector_pair_statistics(
            torch.zeros(3), torch.ones(3),
        )
        self.assertIsNone(values["cosine"])
        self.assertIsNone(values["opposed_element_ratio"])

    def test_decoder_slices_are_disjoint_and_complete(self):
        model = make_model()
        named_parameters, groups = build_parameter_groups(model)
        parameters = [parameter for _, parameter in named_parameters]

        slice_counts = []
        for group_name in (
            "decoder_base_shared",
            "decoder_mid_increment",
            "decoder_full_increment",
        ):
            count = sum(
                parameter.numel() if mask is None else int(mask.sum().item())
                for (index, mask) in groups[group_name]
                for parameter in (parameters[index],)
            )
            slice_counts.append(count)

        decoder_total = sum(
            parameter.numel()
            for name, parameter in named_parameters
            if name.startswith("decoder.")
        )
        self.assertTrue(all(count > 0 for count in slice_counts))
        self.assertEqual(sum(slice_counts), decoder_total)


if __name__ == "__main__":
    unittest.main()
