import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from compression.model import CompressedHybridGridNet
from model import HybridGridNet
from train_compression import (
    compression_stage,
    load_compression_checkpoint,
    rate_distortion_loss,
    save_compression_checkpoint,
    set_cosine_learning_rate,
)


def make_model():
    baseline = HybridGridNet(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=2,
        hidden_dim=16,
    )
    return CompressedHybridGridNet(baseline, quant_steps=0.1)


def make_config(**overrides):
    values = {
        'grid_levels': 2,
        'grid_feat_dim': 2,
        'base_resolution': 4,
        'finest_resolution': 6,
        'aspect_ratio': [1, 1],
        'time_scale': 1.0,
        'pe_freq': 2,
        'hidden_dim': 16,
        'quant_step': 0.1,
        'epochs': 300,
        'warmup_epochs': 30,
        'symbol_start_epoch': 270,
        'lambda_max': 5e-4,
    }
    values.update(overrides)
    return Namespace(**values)


class CompressionScheduleTest(unittest.TestCase):
    def test_uses_three_paper_training_stages(self):
        warmup = compression_stage(29)
        first_noise = compression_stage(30)
        last_noise = compression_stage(269)
        symbols = compression_stage(270)

        self.assertEqual(warmup.quant_mode, 'disabled')
        self.assertEqual(warmup.lambda_rate, 0.0)
        self.assertEqual(first_noise.quant_mode, 'noise')
        self.assertGreater(first_noise.lambda_rate, 0.0)
        self.assertLess(first_noise.lambda_rate, 5e-4)
        self.assertEqual(last_noise.quant_mode, 'noise')
        self.assertAlmostEqual(last_noise.lambda_rate, 5e-4)
        self.assertEqual(symbols.quant_mode, 'symbols')
        self.assertEqual(symbols.lambda_rate, 5e-4)

    def test_rejects_invalid_schedule(self):
        invalid = (
            {'epoch': -1},
            {'epoch': 300},
            {'epoch': 0, 'warmup_epochs': 270, 'symbol_start_epoch': 270},
            {'epoch': 0, 'lambda_max': -1.0},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    compression_stage(**arguments)

    def test_cosine_learning_rate_decreases(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=5e-3)

        start = set_cosine_learning_rate(optimizer, 5e-3, 0, 0, 10, 300)
        middle = set_cosine_learning_rate(optimizer, 5e-3, 150, 0, 10, 300)
        end = set_cosine_learning_rate(optimizer, 5e-3, 299, 9, 10, 300)

        self.assertAlmostEqual(start, 5e-3)
        self.assertGreater(start, middle)
        self.assertGreater(middle, end)
        self.assertEqual(optimizer.param_groups[0]['lr'], end)


class RateDistortionLossTest(unittest.TestCase):
    def test_disabled_rate_keeps_distortion_unchanged(self):
        model = make_model()
        output = model(torch.rand(1, 3, 2, 2), quant_mode='disabled')
        distortion = output.reconstruction.mean()

        total, rate = rate_distortion_loss(distortion, output, 0.0)

        self.assertTrue(torch.equal(total, distortion))
        self.assertEqual(rate.item(), 0.0)

    def test_joint_loss_uses_rate_per_value(self):
        model = make_model()
        output = model(torch.rand(1, 3, 2, 2), quant_mode='noise')
        distortion = output.reconstruction.mean()

        total, rate = rate_distortion_loss(distortion, output, 1e-3)

        expected = distortion + 1e-3 * output.rate.bits_per_value
        self.assertTrue(torch.equal(rate, output.rate.bits_per_value))
        self.assertTrue(torch.equal(total, expected))


class CompressionCheckpointTest(unittest.TestCase):
    def test_saves_and_restores_model_entropy_and_optimizer(self):
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        coords = torch.rand(1, 3, 2, 2)
        output = model(coords, quant_mode='noise')
        loss, _ = rate_distortion_loss(
            output.reconstruction.mean(), output, 1e-3,
        )
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'compression.pth'
            save_compression_checkpoint(
                model, optimizer, 4, 30.0, 31.0, make_config(), path,
            )
            restored = make_model()
            restored_optimizer = torch.optim.AdamW(
                restored.parameters(), lr=5e-3,
            )
            start_epoch, best_val, best_train = load_compression_checkpoint(
                path,
                restored,
                restored_optimizer,
                torch.device('cpu'),
                make_config(),
            )

        self.assertEqual(start_epoch, 5)
        self.assertEqual(best_val, 30.0)
        self.assertEqual(best_train, 31.0)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(restored_optimizer.state_dict()['state'])

    def test_rejects_incompatible_compression_config(self):
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'compression.pth'
            save_compression_checkpoint(
                model, optimizer, 0, 0.0, 0.0, make_config(), path,
            )
            restored = make_model()
            restored_optimizer = torch.optim.AdamW(
                restored.parameters(), lr=5e-3,
            )
            with self.assertRaisesRegex(ValueError, 'quant_step'):
                load_compression_checkpoint(
                    path,
                    restored,
                    restored_optimizer,
                    torch.device('cpu'),
                    make_config(quant_step=0.2),
                )


if __name__ == '__main__':
    unittest.main()
