import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from train import evaluate, load_checkpoint, save_checkpoint


def make_config(**overrides):
    values = {
        'grid_levels': 2,
        'grid_feat_dim': 4,
        'base_resolution': 8,
        'finest_resolution': 16,
        'aspect_ratio': [16, 9],
        'time_scale': 0.5,
        'pe_freq': 4,
        'hidden_dim': 32,
    }
    values.update(overrides)
    return Namespace(**values)


class CheckpointTest(unittest.TestCase):
    def test_save_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'checkpoint.pth'
            model = nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters())
            with torch.no_grad():
                model.weight.fill_(2.0)

            save_checkpoint(model, optimizer, 3, 31.5, make_config(), path)

            restored_model = nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(restored_model.parameters())
            start_epoch, best_psnr = load_checkpoint(
                path,
                restored_model,
                torch.device('cpu'),
                make_config(),
                restored_optimizer,
            )

            self.assertEqual(start_epoch, 4)
            self.assertEqual(best_psnr, 31.5)
            self.assertTrue(torch.equal(model.weight, restored_model.weight))

    def test_rejects_incompatible_model_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'checkpoint.pth'
            model = nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters())
            save_checkpoint(model, optimizer, 0, 0.0, make_config(), path)

            with self.assertRaisesRegex(ValueError, 'hidden_dim'):
                load_checkpoint(
                    path,
                    nn.Linear(2, 1),
                    torch.device('cpu'),
                    make_config(hidden_dim=64),
                )

    def test_rejects_incompatible_model_architecture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'checkpoint.pth'
            model = nn.Linear(2, 1)
            model.architecture = 'experimental'
            optimizer = torch.optim.Adam(model.parameters())
            save_checkpoint(model, optimizer, 0, 0.0, make_config(), path)

            restored_model = nn.Linear(2, 1)
            restored_model.architecture = 'hybrid_grid_paper_v1'
            with self.assertRaisesRegex(ValueError, '模型架构与检查点不一致'):
                load_checkpoint(
                    path, restored_model, torch.device('cpu'), make_config(),
                )

    def test_missing_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(
                'missing.pth', nn.Linear(2, 1), torch.device('cpu'), make_config(),
            )


class EvaluateTest(unittest.TestCase):
    @patch('train.msssim_fn', return_value=0.9)
    @patch('train.psnr_fn', return_value=30.0)
    def test_evaluate_restores_model_mode(self, _, __):
        model = nn.Identity()
        model.train()
        batch = {
            'coords': torch.zeros(1, 3, 2, 2),
            'pixels': torch.zeros(1, 3, 2, 2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            psnr, msssim = evaluate(
                model,
                [batch],
                torch.device('cpu'),
                save_dir=temp_dir,
                log_interval=1,
            )

        self.assertEqual(psnr, 30.0)
        self.assertEqual(msssim, 0.9)
        self.assertTrue(model.training)

    def test_evaluate_weights_partial_batches_by_sample_count(self):
        model = nn.Identity()
        batches = [
            {
                'coords': torch.zeros(2, 3, 2, 2),
                'pixels': torch.zeros(2, 3, 2, 2),
            },
            {
                'coords': torch.zeros(1, 3, 2, 2),
                'pixels': torch.zeros(1, 3, 2, 2),
            },
        ]

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch('train.psnr_fn', side_effect=[10.0, 30.0]),
            patch('train.msssim_fn', side_effect=[0.5, 1.0]),
        ):
            psnr, msssim = evaluate(
                model,
                batches,
                torch.device('cpu'),
                save_dir=temp_dir,
                log_interval=1,
            )

        self.assertAlmostEqual(psnr, 50 / 3)
        self.assertAlmostEqual(msssim, 2 / 3)


if __name__ == '__main__':
    unittest.main()
