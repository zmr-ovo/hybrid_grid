import unittest

import torch

from util import quantize_per_tensor


class QuantizationTest(unittest.TestCase):
    def test_per_tensor_uses_full_integer_range(self):
        values = torch.tensor([0.0, 0.5, 1.0])

        quantized, dequantized, metadata = quantize_per_tensor(values, bit=8)

        self.assertEqual(quantized[[0, -1]].tolist(), [0, 255])
        self.assertIn(quantized[1].item(), (127, 128))
        self.assertTrue(torch.allclose(dequantized, values, atol=metadata.scale.item()))
        self.assertAlmostEqual(metadata.scale.item(), 1 / 255)
        self.assertEqual(metadata.zero_point.item(), 0)

    def test_zero_is_included_in_mixed_tensor_range(self):
        values = torch.tensor([0.0, 1.0, 2.0])

        quantized, dequantized, _ = quantize_per_tensor(values)

        self.assertEqual(quantized[0].item(), 0)
        self.assertEqual(dequantized[0].item(), 0.0)

    def test_per_channel_uses_one_scale_per_channel(self):
        values = torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]])

        quantized, dequantized, metadata = quantize_per_tensor(values, axis=0)

        self.assertEqual(metadata.axis, 0)
        self.assertEqual(metadata.scale.shape, (2, 1))
        self.assertEqual(metadata.zero_point.shape, (2, 1))
        self.assertTrue(torch.allclose(dequantized, values, atol=2 / 255))
        self.assertTrue(torch.all((quantized >= 0) & (quantized <= 255)))

    def test_negative_axis_is_normalized(self):
        values = torch.tensor([[0.0, 1.0], [0.0, 2.0]])

        _, _, metadata = quantize_per_tensor(values, axis=-1)

        self.assertEqual(metadata.axis, 1)
        self.assertEqual(metadata.scale.shape, (1, 2))

    def test_each_value_can_be_a_channel(self):
        values = torch.tensor([1.0, -2.0])

        _, dequantized, metadata = quantize_per_tensor(values, axis=0)

        self.assertEqual(metadata.scale.shape, values.shape)
        self.assertTrue(torch.equal(dequantized, values))

    def test_zero_and_constant_tensors_are_exact(self):
        for value in (0.0, 3.5, -2.25):
            with self.subTest(value=value):
                values = torch.full((2, 3), value)
                _, dequantized, metadata = quantize_per_tensor(values)

                self.assertTrue(torch.equal(dequantized, values))
                self.assertTrue(torch.isfinite(metadata.scale).all())

    def test_metadata_reports_storage_estimate(self):
        values = torch.tensor([[0.0, 1.0], [0.0, 2.0]])

        _, _, metadata = quantize_per_tensor(values, bit=8, axis=0)

        expected = 2 * 32 + 2 * 8 + 2 * 64 + 8 + 64 + 8
        self.assertEqual(metadata.estimated_bits, expected)
        self.assertEqual(metadata.shape, (2, 2))
        self.assertEqual(metadata.dtype, torch.float32)

    def test_rejects_invalid_inputs(self):
        invalid_cases = (
            (torch.empty(0), 8, None, ValueError),
            (torch.tensor([1]), 8, None, TypeError),
            (torch.tensor([float('nan')]), 8, None, ValueError),
            (torch.tensor([1.0]), 0, None, ValueError),
            (torch.tensor([1.0]), 32, None, ValueError),
            (torch.tensor([1.0]), True, None, ValueError),
            (torch.tensor([1.0]), 8, 1, ValueError),
            (torch.tensor([1.0]), 8, 0.5, TypeError),
        )

        for values, bit, axis, error in invalid_cases:
            with self.subTest(values=values, bit=bit, axis=axis):
                with self.assertRaises(error):
                    quantize_per_tensor(values, bit=bit, axis=axis)


if __name__ == '__main__':
    unittest.main()
