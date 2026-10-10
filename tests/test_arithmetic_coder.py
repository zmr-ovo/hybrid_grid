import unittest

from compression.arithmetic_coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    probability_to_cdf,
)


class ArithmeticCoderTest(unittest.TestCase):
    def test_probability_cdf_is_strict_and_has_fixed_total(self):
        cdf = probability_to_cdf([0.7, 0.2, 0.1], precision=12)

        self.assertEqual(cdf[0], 0)
        self.assertEqual(cdf[-1], 1 << 12)
        self.assertTrue(all(right > left for left, right in zip(cdf, cdf[1:])))

    def test_round_trip_with_changing_distributions(self):
        symbols = [0, 2, 1, 1, 0, 2, 2, 0]
        cdfs = [
            probability_to_cdf(probabilities)
            for probabilities in (
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
                [0.2, 0.6, 0.2],
                [0.3, 0.4, 0.3],
                [0.6, 0.3, 0.1],
                [0.1, 0.1, 0.8],
                [0.2, 0.2, 0.6],
                [0.7, 0.2, 0.1],
            )
        ]
        encoder = ArithmeticEncoder()
        for symbol, cdf in zip(symbols, cdfs):
            encoder.encode(symbol, cdf)

        decoder = ArithmeticDecoder(encoder.finish())
        restored = [decoder.decode(cdf) for cdf in cdfs]

        self.assertEqual(restored, symbols)

    def test_rejects_invalid_cdf_and_symbol(self):
        encoder = ArithmeticEncoder()

        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            encoder.encode(0, [0, 1, 1])
        with self.assertRaisesRegex(ValueError, 'outside'):
            encoder.encode(2, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
