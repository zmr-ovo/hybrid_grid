import unittest

import torch

from compression.rate import estimate_grid_rate


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


if __name__ == '__main__':
    unittest.main()
