"""Small end-to-end tests for the model codec command-line functions."""

import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from codec import main
from compression.network_quantization import configure_network_qat
from train_compression import build_compression_model


def make_config():
    return Namespace(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=[1, 1],
        time_scale=1.0,
        pe_freq=2,
        hidden_dim=16,
        quant_step=0.1,
        quant_steps=[0.1, 0.2],
        entropy_model='cross_scale_recursive',
        network_qat=True,
        network_quant_bits=8,
    )


class CodecCliTest(unittest.TestCase):
    def test_compress_inspect_and_decompress(self):
        config = make_config()
        model = build_compression_model(config, torch.device('cpu'))
        configure_network_qat(model, True, freeze=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pth'
            bitstream = root / 'model.hgm'
            torch.save({
                'architecture': model.architecture,
                'epoch': 4,
                'state_dict': model.state_dict(),
                'config': vars(config),
            }, checkpoint)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                stats = main([
                    'compress',
                    '--checkpoint', str(checkpoint),
                    '--output', str(bitstream),
                    '--device', 'cpu',
                ])
                info = main(['inspect', '--input', str(bitstream)])
                loaded = main([
                    'decompress',
                    '--input', str(bitstream),
                    '--device', 'cpu',
                ])

            text = output.getvalue()
            self.assertTrue(bitstream.is_file())
            self.assertEqual(stats.total_bits, bitstream.stat().st_size * 8)
            self.assertEqual(info.stats.total_bits, stats.total_bits)
            self.assertEqual(loaded.stats.total_bits, stats.total_bits)
            self.assertIn('Command: compress', text)
            self.assertIn('Command: inspect', text)
            self.assertIn('Command: decompress', text)
            self.assertIn('Checkpoint required: NO', text)


if __name__ == '__main__':
    unittest.main()
