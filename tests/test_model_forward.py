import unittest

import torch
from torch import nn

from model import Decoder, HybridGridNet, TemporalModulation


def make_model():
    return HybridGridNet(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
    )


class PaperBaselineTest(unittest.TestCase):
    def test_forward_returns_rgb_image(self):
        model = make_model()
        coords = torch.rand(2, 3, 4, 5, requires_grad=True)

        output = model(coords)
        output.mean().backward()

        self.assertEqual(output.shape, (2, 3, 4, 5))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))
        self.assertIsNotNone(model.grid_encoder.levels[0].grid.grad)

    def test_decoder_is_mlp_only(self):
        decoder = make_model().decoder

        self.assertIsInstance(decoder, Decoder)
        self.assertFalse(any(isinstance(module, nn.Conv2d) for module in decoder.modules()))

    def test_temporal_modulation_ignores_spatial_coordinates(self):
        modulation = TemporalModulation(input_dim=4, hidden_dim=8)
        features = torch.rand(2, 4, 3, 5)
        first_coords = torch.rand(2, 3, 3, 5)
        second_coords = torch.rand(2, 3, 3, 5)
        second_coords[:, 2] = first_coords[:, 2, :1, :1]
        first_coords[:, 2] = first_coords[:, 2, :1, :1]

        first = modulation(features, first_coords)
        second = modulation(features, second_coords)

        self.assertTrue(torch.equal(first, second))

    def test_position_encoding_keeps_base_two(self):
        model = make_model()
        expected = 2.0 ** torch.arange(3, dtype=model.pe_encoder.freqs.dtype)

        self.assertEqual(model.pe_encoder.base, 2.0)
        self.assertTrue(torch.equal(model.pe_encoder.freqs.detach(), expected))

    def test_architecture_is_fixed_to_paper_baseline(self):
        model = make_model()

        self.assertEqual(model.architecture, 'hybrid_grid_paper_v1')
        self.assertIsInstance(model.time_mod, TemporalModulation)


if __name__ == '__main__':
    unittest.main()
