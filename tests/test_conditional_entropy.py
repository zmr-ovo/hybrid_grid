import unittest

import torch

from compression.conditional_entropy import CrossScaleConditionalEntropy


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


if __name__ == '__main__':
    unittest.main()
