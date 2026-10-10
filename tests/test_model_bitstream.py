"""Tests for the checkpoint-independent complete model container.

The tests use a tiny model so they verify exact parameter/Grid restoration and
inference equivalence without depending on a trained video checkpoint.
"""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from compression.model_bitstream import (
    load_model_bitstream,
    write_model_bitstream,
)
from compression.network_quantization import configure_network_qat
from train_compression import build_compression_model


def make_config(network_qat=True):
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
        network_qat=network_qat,
        network_quant_bits=8,
    )


class ModelBitstreamTest(unittest.TestCase):
    def _round_trip(self, network_qat):
        torch.manual_seed(5)
        config = make_config(network_qat)
        model = build_compression_model(config, torch.device('cpu'))
        if network_qat:
            configure_network_qat(model, True, freeze=True)
        model.eval()
        coords = torch.rand(1, 3, 3, 4)
        expected = model(coords, quant_mode='symbols').reconstruction

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.hgm'
            written = write_model_bitstream(path, model, config)
            loaded = load_model_bitstream(path)
            actual = loaded.model(coords, quant_mode='symbols').reconstruction

            self.assertEqual(written.total_bits, path.stat().st_size * 8)
            self.assertGreater(written.grid_bits, 0)
            self.assertGreater(written.non_grid_parameter_bits, 0)
            self.assertGreater(written.entropy_parameter_bits, 0)
            self.assertGreater(written.manifest_bits, 0)
            self.assertGreater(written.container_overhead_bits, 0)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(loaded.config.quant_steps, [0.1, 0.2])
        for index, (expected_level, actual_level) in enumerate(zip(
            model.reconstruction_model.grid_encoder.levels,
            loaded.model.reconstruction_model.grid_encoder.levels,
        )):
            step = config.quant_steps[index]
            expected_grid = torch.round(expected_level.grid / step) * step
            self.assertTrue(torch.equal(actual_level.grid, expected_grid))

    def test_mixed_qat_model_round_trip(self):
        self._round_trip(network_qat=True)

    def test_fp32_non_grid_model_round_trip(self):
        self._round_trip(network_qat=False)

    def test_rejects_unfrozen_qat(self):
        config = make_config(True)
        model = build_compression_model(config, torch.device('cpu'))
        configure_network_qat(model, True, freeze=False)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.hgm'
            with self.assertRaisesRegex(ValueError, 'frozen'):
                write_model_bitstream(path, model, config)


if __name__ == '__main__':
    unittest.main()
