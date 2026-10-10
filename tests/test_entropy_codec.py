import unittest
import tempfile
from pathlib import Path

import torch

from compression.codec import (
    decode_recursive_grid_file,
    decode_recursive_grid_symbols,
    encode_recursive_grid_file,
    encode_recursive_grid_symbols,
)
from compression.conditional_entropy import CumulativeCrossScaleEntropy
from compression.quantization import quantize_grid


class RecursiveEntropyCodecTest(unittest.TestCase):
    def test_all_grid_levels_round_trip_exactly(self):
        torch.manual_seed(7)
        model = CumulativeCrossScaleEntropy(2, levels=3)
        symbols = (
            torch.randint(-2, 3, (2, 2, 2, 2), dtype=torch.int32),
            torch.randint(-3, 4, (3, 3, 2, 2), dtype=torch.int32),
            torch.randint(-4, 5, (4, 4, 3, 2), dtype=torch.int32),
        )

        encoded = encode_recursive_grid_symbols(symbols, model)
        restored = decode_recursive_grid_symbols(encoded, model)

        self.assertEqual(len(restored), len(symbols))
        self.assertTrue(all(
            torch.equal(actual, expected)
            for actual, expected in zip(restored, symbols)
        ))
        self.assertEqual(len(encoded.actual_bits_by_level), 3)
        self.assertTrue(all(bits > 0 for bits in encoded.actual_bits_by_level))
        self.assertTrue(all(bits > 0 for bits in encoded.estimated_bits_by_level))

    def test_decoding_does_not_need_original_symbols(self):
        model = CumulativeCrossScaleEntropy(1, levels=2)
        symbols = (
            torch.tensor([[[[-1]], [[1]]]], dtype=torch.int32),
            torch.tensor([[[[2]], [[0]]], [[[-2]], [[1]]]], dtype=torch.int32),
        )

        encoded = encode_recursive_grid_symbols(symbols, model)
        del symbols
        restored = decode_recursive_grid_symbols(encoded, model)

        self.assertEqual(restored[0].tolist(), [[[[-1]], [[1]]]])
        self.assertEqual(
            restored[1].tolist(),
            [[[[2]], [[0]]], [[[-2]], [[1]]]],
        )

    def test_dequantized_grids_match_symbols_mode_exactly(self):
        torch.manual_seed(9)
        model = CumulativeCrossScaleEntropy(2, levels=2)
        steps = (0.1, 0.2)
        quantized = (
            quantize_grid(torch.randn(2, 2, 2, 2), steps[0], 'symbols'),
            quantize_grid(torch.randn(3, 3, 2, 2), steps[1], 'symbols'),
        )
        symbols = tuple(item.symbols for item in quantized)

        encoded = encode_recursive_grid_symbols(symbols, model)
        restored = decode_recursive_grid_symbols(encoded, model)
        restored_grids = tuple(
            item.to(torch.float32) * step
            for item, step in zip(restored, steps)
        )

        self.assertTrue(all(
            torch.equal(actual, expected.grid)
            for actual, expected in zip(restored_grids, quantized)
        ))
        self.assertGreater(encoded.actual_bits, 0)
        self.assertGreater(encoded.estimated_bits, 0)

    def test_grid_file_round_trip_is_exact(self):
        model = CumulativeCrossScaleEntropy(1, levels=2)
        symbols = (
            torch.tensor([[[[-1]], [[1]]]], dtype=torch.int32),
            torch.tensor([[[[2]], [[0]]], [[[-2]], [[1]]]], dtype=torch.int32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grids.hgb'

            encoded, written = encode_recursive_grid_file(
                path, symbols, model, (0.001, 0.002),
            )
            restored, bitstream = decode_recursive_grid_file(path, model)

            self.assertTrue(all(
                torch.equal(actual, expected)
                for actual, expected in zip(restored, symbols)
            ))
            self.assertEqual(bitstream.quant_steps, (0.001, 0.002))
            self.assertEqual(written.total_bits, path.stat().st_size * 8)
            self.assertEqual(
                written.payload_bits,
                encoded.actual_bits,
            )


if __name__ == '__main__':
    unittest.main()
