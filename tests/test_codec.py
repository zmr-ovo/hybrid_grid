import unittest

import torch

from compression.arithmetic_coder import probability_to_cdf
from compression.codec import (
    decode_grid_level,
    decode_grid_symbols,
    encode_grid_level,
    encode_grid_symbols,
)


class GridCodecTest(unittest.TestCase):
    def test_grid_level_round_trip_is_exact(self):
        symbols = torch.tensor(
            [[[-2, 0], [1, 2]], [[0, -1], [2, -2]]],
            dtype=torch.int32,
        )
        channel_cdfs = torch.tensor([
            probability_to_cdf([0.1, 0.2, 0.4, 0.2, 0.1]),
            probability_to_cdf([0.2, 0.2, 0.2, 0.2, 0.2]),
        ])

        encoded = encode_grid_level(symbols, channel_cdfs, symbol_offset=-2)
        restored = decode_grid_level(encoded, channel_cdfs)

        self.assertGreater(len(encoded.payload), 0)
        self.assertTrue(torch.equal(restored, symbols))

    def test_multiple_grid_levels_round_trip_is_exact(self):
        symbols = (
            torch.tensor([[[[-1, 0], [1, 1]]]], dtype=torch.int32),
            torch.tensor(
                [[[[2, -2], [0, 1]]], [[[1, 0], [-1, 2]]]],
                dtype=torch.int32,
            ),
        )
        cdf = probability_to_cdf([0.1, 0.2, 0.4, 0.2, 0.1])
        cdfs = (cdf, cdf)

        encoded = encode_grid_symbols(symbols, cdfs, (-2, -2))
        restored = decode_grid_symbols(encoded, cdfs)

        self.assertEqual(len(encoded), 2)
        self.assertTrue(all(
            torch.equal(actual, expected)
            for actual, expected in zip(restored, symbols)
        ))

    def test_supports_one_cdf_per_symbol(self):
        symbols = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
        cdfs = torch.tensor([
            probability_to_cdf([0.8, 0.1, 0.1]),
            probability_to_cdf([0.1, 0.8, 0.1]),
            probability_to_cdf([0.1, 0.1, 0.8]),
        ])

        encoded = encode_grid_level(symbols, cdfs, symbol_offset=-1)
        restored = decode_grid_level(encoded, cdfs)

        self.assertTrue(torch.equal(restored, symbols))

    def test_rejects_symbols_outside_cdf_support(self):
        symbols = torch.tensor([0, 3], dtype=torch.int32)
        cdf = probability_to_cdf([0.2, 0.6, 0.2])

        with self.assertRaisesRegex(ValueError, 'outside'):
            encode_grid_level(symbols, cdf, symbol_offset=0)


if __name__ == '__main__':
    unittest.main()
