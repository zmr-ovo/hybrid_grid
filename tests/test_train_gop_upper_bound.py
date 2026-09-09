import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from model import HybridGridNet
from train_gop_lora import GOPLoader
from train_gop_upper_bound import (
    _target_entry,
    _validate_args,
    load_checkpoint,
    save_checkpoint,
)


def make_args(**overrides):
    values = {
        'target_gop': 2,
        'epochs': 3,
        'lr': 5e-3,
        'warmup': 0.2,
        'weight_decay': 0.0,
        'batch_size': 1,
        'workers': 0,
        'gop_size': 2,
        'frame_interval': 1,
        'eval_freq': 1,
        'save_interval': 1,
        'log_interval': 1,
        'dynamic_res': False,
        'grid_levels': 2,
        'grid_feat_dim': 2,
        'base_resolution': 4,
        'finest_resolution': 6,
        'aspect_ratio': [1, 1],
        'time_scale': 1.0,
        'pe_freq': 2,
        'hidden_dim': 16,
    }
    values.update(overrides)
    return Namespace(**values)


def make_model():
    return HybridGridNet(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=2,
        hidden_dim=16,
    )


class DatasetMetadata:
    gop_size = 2
    num_gops = 3
    total_frames = 6


class GOPUpperBoundTest(unittest.TestCase):
    def test_requires_a_later_gop(self):
        with self.assertRaisesRegex(ValueError, 'target_gop'):
            _validate_args(make_args(target_gop=0))

    def test_selects_only_requested_gop(self):
        entries = (
            GOPLoader(0, object(), torch.Generator()),
            GOPLoader(2, object(), torch.Generator()),
        )

        self.assertEqual(_target_entry(entries, 2).gop_index, 2)
        with self.assertRaisesRegex(ValueError, 'outside'):
            _target_entry(entries, 1)

    def test_checkpoint_restores_full_model(self):
        args = make_args()
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        with torch.no_grad():
            next(model.parameters()).fill_(0.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'upper_bound.pth'
            save_checkpoint(
                model, optimizer, 1, 31.0, 32.0, 20.0,
                args, DatasetMetadata(), path,
            )
            restored = make_model()
            restored_optimizer = torch.optim.AdamW(
                restored.parameters(), lr=args.lr,
            )
            progress = load_checkpoint(
                path, restored, restored_optimizer, torch.device('cpu'),
                args, DatasetMetadata(),
            )

        self.assertEqual(progress, (2, 31.0, 32.0, 20.0))
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(all(
            parameter.requires_grad for parameter in restored.parameters()
        ))

    def test_checkpoint_rejects_a_different_target(self):
        args = make_args()
        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'upper_bound.pth'
            save_checkpoint(
                model, optimizer, 0, 0.0, 0.0, 20.0,
                args, DatasetMetadata(), path,
            )
            restored = copy.deepcopy(model)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            with self.assertRaisesRegex(ValueError, '目标 GOP'):
                load_checkpoint(
                    path, restored, restored_optimizer,
                    torch.device('cpu'), make_args(target_gop=1),
                    DatasetMetadata(),
                )


if __name__ == '__main__':
    unittest.main()
