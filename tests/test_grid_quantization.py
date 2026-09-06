import unittest

import torch

from compression.quantization import quantize_grid


class GridQuantizationTest(unittest.TestCase):
    def test_disabled_returns_original_grid(self):
        grid = torch.randn(2, 3, requires_grad=True)

        result = quantize_grid(grid, quant_step=0.1, mode='disabled')

        self.assertIs(result.grid, grid)
        self.assertIsNone(result.symbols)
        self.assertEqual(result.mode, 'disabled')

    def test_zero_step_disables_quantization(self):
        grid = torch.randn(2, 3)

        for mode in ('disabled', 'noise', 'symbols'):
            with self.subTest(mode=mode):
                result = quantize_grid(grid, quant_step=0, mode=mode)
                self.assertIs(result.grid, grid)
                self.assertIsNone(result.symbols)
                self.assertEqual(result.mode, 'disabled')

    def test_noise_is_bounded_and_reproducible(self):
        grid = torch.zeros(128)

        torch.manual_seed(42)
        first = quantize_grid(grid, quant_step=0.2, mode='noise')
        torch.manual_seed(42)
        second = quantize_grid(grid, quant_step=0.2, mode='noise')

        noise = first.grid - grid
        self.assertTrue(torch.all(noise >= -0.1))
        self.assertTrue(torch.all(noise <= 0.1))
        self.assertTrue(torch.equal(first.grid, second.grid))
        self.assertIsNone(first.symbols)

    def test_symbols_are_integer_and_grid_is_dequantized(self):
        grid = torch.tensor([-0.26, -0.09, 0.11, 0.24])

        result = quantize_grid(grid, quant_step=0.1, mode='symbols')

        self.assertEqual(result.symbols.dtype, torch.int32)
        self.assertEqual(result.symbols.tolist(), [-3, -1, 1, 2])
        self.assertTrue(torch.allclose(
            result.grid,
            result.symbols.to(grid.dtype) * result.quant_step,
        ))

    def test_symbols_mode_uses_straight_through_gradient(self):
        grid = torch.randn(2, 3, requires_grad=True)

        result = quantize_grid(grid, quant_step=0.1, mode='symbols')
        result.grid.sum().backward()

        self.assertTrue(torch.equal(grid.grad, torch.ones_like(grid)))

    def test_noise_mode_keeps_grid_gradient(self):
        grid = torch.randn(2, 3, requires_grad=True)

        result = quantize_grid(grid, quant_step=0.1, mode='noise')
        result.grid.sum().backward()

        self.assertTrue(torch.equal(grid.grad, torch.ones_like(grid)))

    def test_shape_device_and_dtype_are_preserved(self):
        grid = torch.randn(2, 3, dtype=torch.float64)

        for mode in ('disabled', 'noise', 'symbols'):
            with self.subTest(mode=mode):
                result = quantize_grid(grid, quant_step=0.1, mode=mode)
                self.assertEqual(result.grid.shape, grid.shape)
                self.assertEqual(result.grid.device, grid.device)
                self.assertEqual(result.grid.dtype, grid.dtype)

    def test_zero_and_constant_grids_are_supported(self):
        for value in (0.0, 1.25, -2.5):
            with self.subTest(value=value):
                grid = torch.full((2, 3), value)
                result = quantize_grid(grid, quant_step=0.25, mode='symbols')
                self.assertTrue(torch.equal(result.grid, grid))

    def test_rejects_invalid_inputs(self):
        invalid_cases = (
            ([], 0.1, 'noise', TypeError),
            (torch.tensor([1]), 0.1, 'noise', TypeError),
            (torch.empty(0), 0.1, 'noise', ValueError),
            (torch.tensor([float('nan')]), 0.1, 'noise', ValueError),
            (torch.tensor([1.0]), -0.1, 'noise', ValueError),
            (torch.tensor([1.0]), float('inf'), 'noise', ValueError),
            (torch.tensor([1.0]), True, 'noise', TypeError),
            (torch.tensor([1.0]), 0.1, 'unknown', ValueError),
        )

        for grid, quant_step, mode, error in invalid_cases:
            with self.subTest(quant_step=quant_step, mode=mode):
                with self.assertRaises(error):
                    quantize_grid(grid, quant_step=quant_step, mode=mode)


if __name__ == '__main__':
    unittest.main()
