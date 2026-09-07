import unittest

import torch
from torch import nn

from compression.rate import (
    estimate_fp32_rate,
    estimate_grid_rate,
    parameter_storage,
)


class GridRateTest(unittest.TestCase):
    def test_known_likelihoods_produce_expected_bits(self):
        result = estimate_grid_rate([
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.25]),
        ])

        self.assertEqual(len(result.level_bits), 2)
        self.assertEqual(result.level_bits[0].item(), 2.0)
        self.assertEqual(result.level_bits[1].item(), 2.0)
        self.assertEqual(result.total_bits.item(), 4.0)
        self.assertEqual(result.total_values, 3)
        self.assertAlmostEqual(result.bits_per_value.item(), 4 / 3)

    def test_rate_uses_fp32_and_preserves_gradient(self):
        logits = torch.tensor([0.0, 1.0], requires_grad=True)
        likelihoods = torch.sigmoid(logits).to(torch.float16)

        result = estimate_grid_rate([likelihoods])
        result.total_bits.backward()

        self.assertEqual(result.total_bits.dtype, torch.float32)
        self.assertEqual(result.bits_per_value.dtype, torch.float32)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.any(logits.grad != 0))

    def test_zero_likelihood_is_bounded(self):
        result = estimate_grid_rate([torch.tensor([0.0, 0.5])])

        self.assertTrue(torch.isfinite(result.total_bits))
        self.assertGreater(result.total_bits.item(), 1.0)

    def test_bits_per_value_is_independent_of_repetition(self):
        likelihood = torch.tensor([0.5, 0.25])

        single = estimate_grid_rate([likelihood])
        repeated = estimate_grid_rate([likelihood, likelihood.clone()])

        self.assertEqual(repeated.total_bits.item(), 2 * single.total_bits.item())
        self.assertEqual(repeated.total_values, 2 * single.total_values)
        self.assertEqual(repeated.bits_per_value.item(), single.bits_per_value.item())

    def test_rejects_invalid_likelihoods(self):
        invalid_cases = (
            (torch.empty(0), ValueError),
            (torch.tensor([1]), TypeError),
            (torch.tensor([-0.1]), ValueError),
            (torch.tensor([1.1]), ValueError),
            (torch.tensor([float('nan')]), ValueError),
        )

        for likelihood, error in invalid_cases:
            with self.subTest(likelihood=likelihood):
                with self.assertRaises(error):
                    estimate_grid_rate([likelihood])

    def test_rejects_invalid_container_and_bound(self):
        with self.assertRaises(TypeError):
            estimate_grid_rate(torch.tensor([0.5]))
        with self.assertRaises(ValueError):
            estimate_grid_rate([])
        for bound in (0, 1, float('inf')):
            with self.subTest(bound=bound):
                with self.assertRaises(ValueError):
                    estimate_grid_rate([torch.tensor([0.5])], bound)
        with self.assertRaises(TypeError):
            estimate_grid_rate([torch.tensor([0.5])], True)


class ParameterStorageTest(unittest.TestCase):
    def test_counts_real_fp32_storage_without_duplicates(self):
        weight = nn.Parameter(torch.zeros(2, 3))
        bias = nn.Parameter(torch.zeros(2))

        result = parameter_storage(
            [weight, bias, weight], required_dtype=torch.float32,
        )

        self.assertEqual(result.tensor_count, 2)
        self.assertEqual(result.parameter_count, 8)
        self.assertEqual(result.total_bits, 8 * 32)
        self.assertEqual(result.bits_by_dtype, (('float32', 8 * 32),))

    def test_required_fp32_rejects_other_dtypes(self):
        parameter = nn.Parameter(torch.zeros(2, dtype=torch.float64))

        with self.assertRaisesRegex(ValueError, 'float32'):
            parameter_storage([parameter], required_dtype=torch.float32)


class Fp32RateBreakdownTest(unittest.TestCase):
    def setUp(self):
        self.non_grid = parameter_storage([
            nn.Parameter(torch.zeros(10)),
        ], required_dtype=torch.float32)
        self.entropy = parameter_storage([
            nn.Parameter(torch.zeros(2)),
        ], required_dtype=torch.float32)

    def test_reports_each_rate_component_and_payload_subtotal(self):
        result = estimate_fp32_rate(
            total_video_pixels=100,
            non_grid_storage=self.non_grid,
            entropy_model_storage=self.entropy,
            grid_bits=400,
            legacy_rate_per_value=2.0,
        )

        self.assertEqual(result.estimated_grid_bits, 400)
        self.assertEqual(result.estimated_grid_bpp, 4.0)
        self.assertEqual(result.non_grid_storage.total_bits, 320)
        self.assertEqual(result.non_grid_bpp, 3.2)
        self.assertEqual(result.entropy_model_storage.total_bits, 64)
        self.assertEqual(result.entropy_model_side_info_bpp, 0.64)
        self.assertEqual(result.estimated_payload_bits, 784)
        self.assertEqual(result.estimated_payload_bpp, 7.84)
        self.assertFalse(result.metadata_included)

    def test_disabled_grid_rate_is_not_reported_as_zero(self):
        result = estimate_fp32_rate(
            total_video_pixels=100,
            non_grid_storage=self.non_grid,
            entropy_model_storage=self.entropy,
        )

        self.assertIsNone(result.legacy_rate_per_value)
        self.assertIsNone(result.estimated_grid_bits)
        self.assertIsNone(result.estimated_grid_bpp)
        self.assertIsNone(result.estimated_payload_bits)
        self.assertIsNone(result.estimated_payload_bpp)
        self.assertEqual(result.non_grid_bpp, 3.2)

    def test_bpp_depends_on_sequence_pixels_not_batch_size(self):
        first = estimate_fp32_rate(
            100, self.non_grid, self.entropy, 400, 2.0,
        )
        second = estimate_fp32_rate(
            100, self.non_grid, self.entropy, 400, 2.0,
        )

        self.assertEqual(first.estimated_payload_bpp,
                         second.estimated_payload_bpp)


if __name__ == '__main__':
    unittest.main()
