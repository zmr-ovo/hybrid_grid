import unittest

import torch
import torch.nn.functional as F

from compression.model import CompressedHybridGridNet
from model import HybridGridNet
from scalable_generation import (
    GridHierarchy,
    ScalableHybridGridNet,
    default_hierarchy,
    multi_prefix_rate_distortion_loss,
)


def make_compressed_model(levels=4, entropy_model='factorized'):
    baseline = HybridGridNet(
        grid_levels=levels,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=8,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
    )
    return CompressedHybridGridNet(
        baseline,
        quant_steps=0.1,
        entropy_model=entropy_model,
    )


class GridHierarchyTest(unittest.TestCase):
    def test_default_hierarchy_is_seven_two_one(self):
        hierarchy = default_hierarchy(10)

        self.assertEqual(
            hierarchy.groups,
            (tuple(range(7)), (7, 8), (9,)),
        )
        self.assertEqual(hierarchy.active_levels(0), tuple(range(7)))
        self.assertEqual(hierarchy.active_levels(1), tuple(range(9)))
        self.assertEqual(hierarchy.active_levels(2), tuple(range(10)))

    def test_rejects_non_prefix_partition(self):
        invalid = (
            ((0, 2), (1,)),
            ((0, 1), (1, 2)),
            ((1,), (2,)),
            ((0, 1), ()),
        )
        for groups in invalid:
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    GridHierarchy(groups)

    def test_rejects_invalid_quality_level(self):
        hierarchy = GridHierarchy(((0, 1), (2,), (3,)))

        for quality_level in (-1, 3, True, 1.0):
            with self.subTest(quality_level=quality_level):
                with self.assertRaises((TypeError, ValueError)):
                    hierarchy.active_levels(quality_level)


class ScalableHybridGridNetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.compressed = make_compressed_model().eval()
        self.hierarchy = GridHierarchy(((0, 1), (2,), (3,)))
        self.model = ScalableHybridGridNet(
            self.compressed, self.hierarchy,
        ).eval()
        self.coords = torch.rand(1, 3, 5, 6)

    def test_all_quality_levels_have_the_same_output_shape(self):
        with torch.no_grad():
            outputs = [
                self.model(self.coords, level)
                for level in range(self.hierarchy.num_quality_levels)
            ]

        self.assertTrue(all(
            output.reconstruction.shape == (1, 3, 5, 6)
            for output in outputs
        ))
        self.assertEqual(outputs[0].active_levels, (0, 1))
        self.assertEqual(outputs[1].active_levels, (0, 1, 2))
        self.assertEqual(outputs[2].active_levels, (0, 1, 2, 3))

    def test_missing_higher_levels_are_zero(self):
        with torch.no_grad():
            output = self.model(self.coords, quality_level=0)

        self.assertEqual(output.level_mask.tolist(), [True, True, False, False])
        for index in (0, 1):
            self.assertTrue(torch.equal(
                output.prefix_grids[index], output.quantized_grids[index],
            ))
        for index in (2, 3):
            self.assertTrue(torch.count_nonzero(
                output.prefix_grids[index],
            ).item() == 0)

    def test_full_level_matches_existing_compressed_model(self):
        with torch.no_grad():
            expected = self.compressed(
                self.coords, quant_mode='symbols',
            ).reconstruction
            actual = self.model(
                self.coords, quality_level=2, quant_mode='symbols',
            ).reconstruction

        self.assertTrue(torch.equal(actual, expected))

    def test_forward_all_uses_cumulative_prefix_rates(self):
        with torch.no_grad():
            output = self.model.forward_all(
                self.coords, quant_mode='symbols',
            )
            complete = self.compressed(
                self.coords, quant_mode='symbols',
            )

        self.assertEqual(len(output.prefixes), 3)
        self.assertEqual(
            [len(prefix.rate.level_bits) for prefix in output.prefixes],
            [2, 3, 4],
        )
        prefix_bits = [
            prefix.rate.total_bits.item() for prefix in output.prefixes
        ]
        self.assertLess(prefix_bits[0], prefix_bits[1])
        self.assertLess(prefix_bits[1], prefix_bits[2])
        self.assertEqual(
            output.prefixes[0].rate.normalization_values,
            output.prefixes[2].rate.normalization_values,
        )
        self.assertTrue(torch.equal(
            output.prefixes[-1].rate.total_bits,
            complete.rate.total_bits,
        ))
        self.assertTrue(torch.equal(
            output.prefixes[-1].rate.bits_per_value,
            complete.rate.bits_per_value,
        ))

    def test_symbols_mode_is_deterministic(self):
        with torch.no_grad():
            first = self.model(self.coords, quality_level=1)
            second = self.model(self.coords, quality_level=1)

        self.assertTrue(torch.equal(
            first.reconstruction, second.reconstruction,
        ))

    def test_forward_evaluates_only_requested_prefix(self):
        def fail_forward_all(*args, **kwargs):
            raise AssertionError("forward_all must not be used")

        self.model.forward_all = fail_forward_all
        with torch.no_grad():
            output = self.model(
                self.coords, quality_level=0, quant_mode='symbols',
            )

        self.assertEqual(output.active_levels, (0, 1))
        self.assertEqual(len(output.likelihoods), 2)

    def test_recursive_entropy_prefix_matches_shared_evaluation(self):
        compressed = make_compressed_model(
            entropy_model='cross_scale_recursive',
        ).eval()
        model = ScalableHybridGridNet(
            compressed, self.hierarchy,
        ).eval()

        with torch.no_grad():
            shared = model.forward_all(
                self.coords, quant_mode='symbols',
            ).prefixes[0]
            sequential = model(
                self.coords, quality_level=0, quant_mode='symbols',
            )

        self.assertTrue(torch.equal(
            sequential.reconstruction, shared.reconstruction,
        ))
        self.assertTrue(torch.equal(
            sequential.rate.total_bits, shared.rate.total_bits,
        ))
        self.assertTrue(all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                sequential.likelihoods, shared.likelihoods,
            )
        ))

    def test_scheme_a_loss_uses_all_prefixes(self):
        output = self.model.forward_all(self.coords, quant_mode='noise')
        target = torch.rand_like(output.prefixes[0].reconstruction)

        loss = multi_prefix_rate_distortion_loss(
            output,
            target,
            F.mse_loss,
            lambda_rate=1e-3,
            distortion_weights=(0.5, 0.75, 1.5),
            rate_weights=(1.0, 2.0, 3.0),
        )

        expected = sum(
            distortion_weight * item.distortion
            + 1e-3 * rate_weight * item.rate_per_value
            for item, distortion_weight, rate_weight in zip(
                loss.prefixes,
                (0.5, 0.75, 1.5),
                (1.0, 2.0, 3.0),
            )
        )
        self.assertEqual(len(loss.prefixes), 3)
        self.assertTrue(torch.equal(loss.total, expected))

    def test_prefix_distortion_reaches_only_active_grids(self):
        for quality_level, expected_active in enumerate((2, 3, 4)):
            self.model.zero_grad(set_to_none=True)
            output = self.model(
                self.coords,
                quality_level=quality_level,
                quant_mode='symbols',
            )
            output.reconstruction.mean().backward()
            gradients = [
                level.grid.grad
                for level in self.compressed.reconstruction_model.grid_encoder.levels
            ]

            self.assertTrue(all(
                gradient is not None
                for gradient in gradients[:expected_active]
            ))
            self.assertTrue(all(
                gradient is None
                for gradient in gradients[expected_active:]
            ))

    def test_scheme_a_joint_loss_reaches_every_grid(self):
        output = self.model.forward_all(self.coords, quant_mode='symbols')
        target = torch.rand_like(output.prefixes[0].reconstruction)
        loss = multi_prefix_rate_distortion_loss(
            output,
            target,
            F.mse_loss,
            lambda_rate=1e-3,
            distortion_weights=(0.5, 0.75, 1.5),
            rate_weights=(1.0, 1.0, 1.0),
        )

        loss.total.backward()

        self.assertTrue(all(
            level.grid.grad is not None
            for level in self.compressed.reconstruction_model.grid_encoder.levels
        ))

    def test_rejects_mismatched_hierarchy(self):
        with self.assertRaisesRegex(ValueError, 'same Grid count'):
            ScalableHybridGridNet(
                self.compressed,
                GridHierarchy(((0, 1), (2,))),
            )


if __name__ == '__main__':
    unittest.main()
