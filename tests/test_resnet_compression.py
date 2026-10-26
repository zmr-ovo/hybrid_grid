import unittest
from argparse import Namespace
from pathlib import Path
import tempfile

import torch

from compression.model_bitstream import load_model_bitstream, write_model_bitstream
from compression.network_quantization import (
    configure_network_qat,
    iter_network_quantizers,
)
from resnet_baseline import ResidualSkipHybridGridNet
from resnet_compression import build_resnet_compression_model
from resnet_compression.train import make_parser
from train_compression import build_compression_model


def make_config(**overrides):
    values = {
        'reconstruction_model': 'additive_skip',
        'grid_levels': 2,
        'grid_feat_dim': 2,
        'base_resolution': 4,
        'finest_resolution': 6,
        'aspect_ratio': [1, 1],
        'time_scale': 1.0,
        'pe_freq': 2,
        'hidden_dim': 16,
        'quant_step': 0.1,
        'quant_steps': None,
        'entropy_model': 'factorized',
        'network_qat': False,
        'network_quant_bits': 8,
    }
    values.update(overrides)
    return Namespace(**values)


class ResNetCompressionTest(unittest.TestCase):
    def test_builds_resnet_reconstruction_model(self):
        model = build_resnet_compression_model(
            make_config(), torch.device('cpu'),
        )

        self.assertIsInstance(
            model.reconstruction_model, ResidualSkipHybridGridNet,
        )
        self.assertEqual(
            model.architecture, 'hybrid_grid_additive_skip_compressed_v1',
        )

    def test_forward_backpropagates_distortion_and_rate(self):
        model = build_resnet_compression_model(
            make_config(), torch.device('cpu'),
        )
        output = model(torch.rand(1, 3, 3, 4), quant_mode='noise')
        loss = output.reconstruction.mean() + 1e-3 * output.rate.bits_per_value
        loss.backward()

        self.assertEqual(output.reconstruction.shape, (1, 3, 3, 4))
        self.assertIsNotNone(
            model.reconstruction_model.grid_encoder.levels[0].grid.grad,
        )
        self.assertIsNotNone(next(model.entropy_models.parameters()).grad)

    def test_recursive_entropy_and_mixed_qat_have_unique_architecture(self):
        model = build_resnet_compression_model(
            make_config(
                entropy_model='cross_scale_recursive',
                network_qat=True,
            ),
            torch.device('cpu'),
        )

        self.assertEqual(
            model.architecture,
            'hybrid_grid_additive_skip_compressed_cross_scale_recursive_mixed_qat_v1',
        )
        self.assertTrue(tuple(iter_network_quantizers(model)))

    def test_default_compression_builder_remains_paper_model(self):
        config = make_config(reconstruction_model='paper')
        model = build_compression_model(config, torch.device('cpu'))

        self.assertNotIsInstance(
            model.reconstruction_model, ResidualSkipHybridGridNet,
        )
        self.assertEqual(model.architecture, 'hybrid_grid_compressed_v1')

    def test_parser_marks_resnet_model_and_separate_output(self):
        args = make_parser().parse_args(['-d', './data/example'])

        self.assertEqual(args.reconstruction_model, 'additive_skip')
        self.assertEqual(args.out_dir, './output/resnet_compression')

    def test_complete_bitstream_restores_resnet_without_checkpoint(self):
        config = make_config(
            quant_steps=[0.1, 0.2],
            entropy_model='cross_scale_recursive',
            network_qat=True,
        )
        model = build_resnet_compression_model(config, torch.device('cpu'))
        configure_network_qat(model, True, freeze=True)
        model.eval()
        coords = torch.rand(1, 3, 3, 4)
        expected = model(coords, quant_mode='symbols').reconstruction

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resnet.hgm'
            write_model_bitstream(path, model, config)
            loaded = load_model_bitstream(path)
            actual = loaded.model(
                coords, quant_mode='symbols',
            ).reconstruction

        self.assertIsInstance(
            loaded.model.reconstruction_model, ResidualSkipHybridGridNet,
        )
        self.assertEqual(loaded.config.reconstruction_model, 'additive_skip')
        self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
