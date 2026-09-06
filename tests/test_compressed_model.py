import unittest

import torch
from torch import nn

from compression.model import CompressedHybridGridNet
from model import HybridGridNet


def make_baseline():
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


class CompressedModelTest(unittest.TestCase):
    def test_creates_one_independent_entropy_model_per_level(self):
        model = CompressedHybridGridNet(make_baseline(), quant_steps=0.1)

        self.assertEqual(len(model.entropy_models), 2)
        self.assertIsNot(model.entropy_models[0], model.entropy_models[1])
        self.assertTrue(all(
            entropy.channels == 2 for entropy in model.entropy_models
        ))

    def test_disabled_mode_matches_baseline_exactly(self):
        baseline = make_baseline().eval()
        model = CompressedHybridGridNet(baseline, quant_steps=0.1).eval()
        coords = torch.rand(1, 3, 4, 5)

        with torch.no_grad():
            expected = baseline(coords)
            output = model(coords, quant_mode='disabled')

        self.assertTrue(torch.equal(output.reconstruction, expected))
        self.assertEqual(output.quant_mode, 'disabled')
        self.assertIsNone(output.rate)
        self.assertEqual(output.likelihoods, ())

    def test_zero_steps_match_baseline_even_in_symbols_mode(self):
        baseline = make_baseline().eval()
        model = CompressedHybridGridNet(baseline, quant_steps=0).eval()
        coords = torch.rand(1, 3, 4, 5)

        with torch.no_grad():
            expected = baseline(coords)
            output = model(coords, quant_mode='symbols')

        self.assertTrue(torch.equal(output.reconstruction, expected))
        self.assertEqual(output.quant_mode, 'disabled')
        self.assertIsNone(output.rate)

    def test_symbols_mode_reconstructs_from_quantized_grids(self):
        baseline = make_baseline().eval()
        model = CompressedHybridGridNet(baseline, quant_steps=0.1).eval()
        coords = torch.rand(1, 3, 4, 5)
        expected_grids = [
            torch.round(level.grid / 0.1) * 0.1
            for level in baseline.grid_encoder.levels
        ]

        with torch.no_grad():
            expected = baseline(coords, grids=expected_grids)
            output = model(coords, quant_mode='symbols')

        self.assertTrue(torch.equal(output.reconstruction, expected))
        self.assertTrue(all(symbols is not None for symbols in output.symbols))
        self.assertTrue(all(
            torch.equal(actual, expected)
            for actual, expected in zip(output.quantized_grids, expected_grids)
        ))

    def test_compressed_output_contains_rate_and_likelihoods(self):
        baseline = make_baseline()
        model = CompressedHybridGridNet(baseline, quant_steps=[0.1, 0.2])
        coords = torch.rand(1, 3, 4, 5)

        output = model(coords, quant_mode='noise')

        expected_values = sum(
            level.grid.numel() for level in baseline.grid_encoder.levels
        )
        self.assertEqual(len(output.likelihoods), 2)
        self.assertEqual(output.rate.total_values, expected_values)
        self.assertTrue(torch.isfinite(output.rate.total_bits))
        self.assertTrue(torch.isfinite(output.rate.bits_per_value))

    def test_distortion_and_rate_gradients_reach_all_parameters(self):
        baseline = make_baseline()
        model = CompressedHybridGridNet(baseline, quant_steps=0.1)
        coords = torch.rand(1, 3, 4, 5)

        output = model(coords, quant_mode='symbols')
        loss = output.reconstruction.mean() + 1e-3 * output.rate.bits_per_value
        loss.backward()

        grid_grads = [level.grid.grad for level in baseline.grid_encoder.levels]
        self.assertTrue(all(grad is not None for grad in grid_grads))
        entropy_grads = [
            parameter.grad for parameter in model.entropy_models.parameters()
        ]
        self.assertTrue(all(grad is not None for grad in entropy_grads))

    def test_rejects_invalid_reconstruction_model(self):
        with self.assertRaises(TypeError):
            CompressedHybridGridNet(object())
        with self.assertRaisesRegex(ValueError, 'grid_encoder.levels'):
            CompressedHybridGridNet(nn.Linear(2, 1))

    def test_rejects_invalid_quantization_steps(self):
        baseline = make_baseline()
        invalid_steps = ([0.1], [0.0, 0.1], -0.1, float('inf'), True)

        for steps in invalid_steps:
            with self.subTest(steps=steps):
                with self.assertRaises((TypeError, ValueError)):
                    CompressedHybridGridNet(baseline, quant_steps=steps)

    def test_rejects_invalid_quantization_mode(self):
        model = CompressedHybridGridNet(make_baseline(), quant_steps=0.1)

        with self.assertRaisesRegex(ValueError, 'quant_mode'):
            model(torch.rand(1, 3, 2, 2), quant_mode='unknown')


if __name__ == '__main__':
    unittest.main()
