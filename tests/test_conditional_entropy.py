import unittest

import torch

from compression.conditional_entropy import (
    CrossScaleConditionalEntropy,
    CumulativeCrossScaleEntropy,
)


class CrossScaleConditionalEntropyTest(unittest.TestCase):
    def test_aligns_condition_to_current_grid_shape(self):
        model = CrossScaleConditionalEntropy(3)
        condition = torch.randn(4, 5, 3, 3)

        aligned = model.align_condition(condition, (7, 9, 5, 3))

        self.assertEqual(aligned.shape, (7, 9, 5, 3))
        self.assertTrue(torch.isfinite(aligned).all())

    def test_likelihood_shape_and_range(self):
        model = CrossScaleConditionalEntropy(2)
        condition = torch.randn(4, 4, 3, 2)
        values = torch.randn(7, 6, 5, 2)

        likelihoods = model(values, condition)

        self.assertEqual(likelihoods.shape, values.shape)
        self.assertEqual(likelihoods.dtype, values.dtype)
        self.assertTrue(torch.isfinite(likelihoods).all())
        self.assertTrue(torch.all(likelihoods > 0))
        self.assertTrue(torch.all(likelihoods <= 1))

    def test_gradients_reach_both_grids_and_predictor(self):
        model = CrossScaleConditionalEntropy(2)
        condition = torch.randn(4, 4, 3, 2, requires_grad=True)
        values = torch.randn(6, 7, 5, 2, requires_grad=True)

        loss = -torch.log2(model(values, condition)).mean()
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(condition.grad)
        self.assertTrue(torch.any(values.grad != 0))
        self.assertTrue(torch.any(condition.grad != 0))
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all()
                            for gradient in gradients))

    def test_rejects_invalid_inputs(self):
        model = CrossScaleConditionalEntropy(2)
        valid = torch.zeros(2, 2, 2, 2)

        with self.assertRaisesRegex(ValueError, 'shape'):
            model(torch.zeros(2, 2, 2), valid)
        with self.assertRaisesRegex(ValueError, 'shape'):
            model(valid, torch.zeros(2, 2, 2, 3))
        with self.assertRaisesRegex(TypeError, 'floating point'):
            model(valid.to(torch.int64), valid)


class CumulativeCrossScaleEntropyTest(unittest.TestCase):
    def test_returns_one_likelihood_tensor_per_level(self):
        model = CumulativeCrossScaleEntropy(2, levels=3)
        values = (
            torch.randn(3, 4, 2, 2),
            torch.randn(5, 6, 3, 2),
            torch.randn(7, 8, 4, 2),
        )

        likelihoods = model(values)

        self.assertEqual(len(likelihoods), 3)
        self.assertEqual(
            tuple(item.shape for item in likelihoods),
            tuple(item.shape for item in values),
        )
        self.assertTrue(all(torch.isfinite(item).all() for item in likelihoods))
        self.assertTrue(all(torch.all(item > 0) for item in likelihoods))
        self.assertTrue(all(torch.all(item <= 1) for item in likelihoods))

    def test_rate_gradient_reaches_all_levels_and_context_parameters(self):
        model = CumulativeCrossScaleEntropy(2, levels=3)
        values = tuple(
            torch.randn(size, size + 1, 2, 2, requires_grad=True)
            for size in (2, 3, 4)
        )

        rate = sum(-torch.log2(item).mean() for item in model(values))
        rate.backward()

        self.assertTrue(all(item.grad is not None for item in values))
        self.assertTrue(all(torch.isfinite(item.grad).all() for item in values))
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all()
                            for gradient in gradients))

    def test_rejects_wrong_level_count(self):
        model = CumulativeCrossScaleEntropy(2, levels=3)

        with self.assertRaisesRegex(ValueError, 'configured levels'):
            model((torch.zeros(2, 2, 2, 2),))


if __name__ == '__main__':
    unittest.main()
