import unittest

import torch

from compression.entropy_bottleneck import EntropyBottleneck
from compression.quantization import quantize_grid


class EntropyBottleneckTest(unittest.TestCase):
    def test_likelihood_shape_dtype_and_range(self):
        model = EntropyBottleneck(channels=3)
        values = torch.randn(2, 4, 5, 3, dtype=torch.float16)

        likelihoods = model(values)

        self.assertEqual(likelihoods.shape, values.shape)
        self.assertEqual(likelihoods.dtype, torch.float32)
        self.assertTrue(torch.isfinite(likelihoods).all())
        self.assertTrue(torch.all(likelihoods > 0))
        self.assertTrue(torch.all(likelihoods <= 1))

    def test_each_channel_has_independent_parameters(self):
        model = EntropyBottleneck(channels=4)

        for parameter in list(model.matrices) + list(model.biases) + list(model.factors):
            self.assertEqual(parameter.shape[0], 4)

    def test_rate_gradient_reaches_grid_and_entropy_parameters(self):
        for mode in ('noise', 'symbols'):
            with self.subTest(mode=mode):
                grid = (torch.randn(2, 3, 4, 2) * 0.05).requires_grad_()
                model = EntropyBottleneck(channels=2)
                quantized = quantize_grid(grid, quant_step=0.1, mode=mode)

                likelihoods = model(quantized.grid / quantized.quant_step)
                rate = -torch.log2(likelihoods).sum()
                rate.backward()

                self.assertIsNotNone(grid.grad)
                self.assertTrue(torch.isfinite(grid.grad).all())
                self.assertTrue(torch.any(grid.grad != 0))
                entropy_grads = [parameter.grad for parameter in model.parameters()]
                self.assertTrue(all(grad is not None for grad in entropy_grads))
                self.assertTrue(all(torch.isfinite(grad).all() for grad in entropy_grads))
                self.assertTrue(any(torch.any(grad != 0) for grad in entropy_grads))

    def test_zero_and_constant_grids_are_supported(self):
        model = EntropyBottleneck(channels=2)

        for value in (0.0, 1.0, -2.0):
            with self.subTest(value=value):
                likelihoods = model(torch.full((2, 3, 4, 2), value))
                self.assertTrue(torch.isfinite(likelihoods).all())
                self.assertTrue(torch.all(likelihoods > 0))

    def test_rejects_invalid_inputs(self):
        model = EntropyBottleneck(channels=2)
        invalid_cases = (
            ([], TypeError),
            (torch.ones(2, dtype=torch.int64), TypeError),
            (torch.empty(0, 2), ValueError),
            (torch.ones(2, 3), ValueError),
            (torch.tensor([[float('inf'), 0.0]]), ValueError),
        )

        for values, error in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(error):
                    model(values)

    def test_rejects_invalid_configuration(self):
        for kwargs in (
            {'channels': 0},
            {'channels': True},
            {'channels': 2, 'filters': ()},
            {'channels': 2, 'init_scale': 0},
            {'channels': 2, 'likelihood_bound': 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    EntropyBottleneck(**kwargs)


if __name__ == '__main__':
    unittest.main()
