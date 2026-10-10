import tempfile
import unittest
from pathlib import Path

from compression.bitstream import read_grid_bitstream, write_grid_bitstream
from compression.codec import EncodedEntropyGrids, EncodedGridLevel


class GridBitstreamTest(unittest.TestCase):
    @staticmethod
    def _encoded():
        return EncodedEntropyGrids((
            EncodedGridLevel(b'base payload', (2, 3, 1, 2), -4, 9),
            EncodedGridLevel(b'enhancement payload', (4, 5, 2, 2), -7, 15),
        ), (80.0, 144.0))

    def test_file_round_trip_preserves_payload_and_metadata(self):
        encoded = self._encoded()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grids.hgb'

            written = write_grid_bitstream(path, encoded, (0.001, 0.002))
            restored = read_grid_bitstream(path)

            self.assertEqual(restored.encoded.levels, encoded.levels)
            self.assertEqual(restored.quant_steps, (0.001, 0.002))
            self.assertEqual(restored.cdf_precision, 16)
            self.assertEqual(restored.entropy_model, 'cross_scale_recursive')
            self.assertEqual(written, restored.stats)
            self.assertEqual(restored.stats.total_bits, path.stat().st_size * 8)
            self.assertEqual(
                restored.stats.payload_bits,
                sum(len(level.payload) for level in encoded.levels) * 8,
            )

    def test_rejects_truncated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grids.hgb'
            write_grid_bitstream(path, self._encoded(), (0.001, 0.002))
            data = path.read_bytes()
            path.write_bytes(data[:-3])

            with self.assertRaisesRegex(ValueError, 'truncated'):
                read_grid_bitstream(path)

    def test_rejects_corrupted_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grids.hgb'
            write_grid_bitstream(path, self._encoded(), (0.001, 0.002))
            data = bytearray(path.read_bytes())
            data[-1] ^= 0xff
            path.write_bytes(data)

            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_grid_bitstream(path)


if __name__ == '__main__':
    unittest.main()
