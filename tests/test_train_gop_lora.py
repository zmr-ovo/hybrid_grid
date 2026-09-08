import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from gop_lora import GOPLoRAHybridGridNet, gop_lora_layers, lora_parameters
from model import HybridGridNet
from train_gop_lora import (
    adapter_state_dict,
    load_adapter_checkpoint,
    load_anchor_checkpoint,
    save_adapter_checkpoint,
    save_anchor_checkpoint,
)


def make_args(**overrides):
    values = {
        'grid_levels': 2,
        'grid_feat_dim': 2,
        'base_resolution': 4,
        'finest_resolution': 6,
        'aspect_ratio': [1, 1],
        'time_scale': 1.0,
        'pe_freq': 2,
        'hidden_dim': 16,
        'gop_size': 2,
        'rank': 1,
        'alpha': 1.0,
    }
    values.update(overrides)
    return Namespace(**values)


def make_anchor():
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


class GOPCheckpointTest(unittest.TestCase):
    def test_anchor_checkpoint_restores_model_and_progress(self):
        model = make_anchor()
        optimizer = torch.optim.Adam(model.parameters())
        with torch.no_grad():
            next(model.parameters()).fill_(0.25)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'anchor.pth'
            save_anchor_checkpoint(
                model, optimizer, 4, 31.0, 32.0, make_args(),
                DatasetMetadata(), path,
            )
            restored = make_anchor()
            restored_optimizer = torch.optim.Adam(restored.parameters())
            progress = load_anchor_checkpoint(
                path, restored, torch.device('cpu'), make_args(),
                restored_optimizer, dataset=DatasetMetadata(),
            )

        self.assertEqual(progress, (5, 31.0, 32.0))
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_adapter_checkpoint_contains_only_lora_parameters(self):
        anchor = make_anchor()
        restored_anchor = copy.deepcopy(anchor)
        model = GOPLoRAHybridGridNet(anchor, 3, rank=1)
        optimizer = torch.optim.Adam(lora_parameters(model))
        with torch.no_grad():
            gop_lora_layers(model)[0].lora_b[0].fill_(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'adapter.pth'
            save_adapter_checkpoint(
                model, optimizer, 2, 29.0, 30.0, make_args(),
                DatasetMetadata(), path,
            )
            checkpoint = torch.load(path, map_location='cpu')

            self.assertNotIn('state_dict', checkpoint)
            self.assertEqual(
                set(checkpoint['adapter_state_dict']),
                set(adapter_state_dict(model)),
            )
            self.assertTrue(all(
                'lora_a' in name or 'lora_b' in name
                for name in checkpoint['adapter_state_dict']
            ))

            restored = GOPLoRAHybridGridNet(restored_anchor, 3, rank=1)
            restored_optimizer = torch.optim.Adam(lora_parameters(restored))
            progress = load_adapter_checkpoint(
                path, restored, torch.device('cpu'), make_args(),
                restored_optimizer, dataset=DatasetMetadata(),
            )
            different_anchor = GOPLoRAHybridGridNet(make_anchor(), 3, rank=1)
            with self.assertRaisesRegex(ValueError, '锚点模型不匹配'):
                load_adapter_checkpoint(
                    path, different_anchor, torch.device('cpu'), make_args(),
                    dataset=DatasetMetadata(),
                )

        self.assertEqual(progress, (3, 29.0, 30.0))
        for name, value in adapter_state_dict(model).items():
            self.assertTrue(torch.equal(value, adapter_state_dict(restored)[name]))

    def test_checkpoint_rejects_different_gop_partition(self):
        model = make_anchor()
        optimizer = torch.optim.Adam(model.parameters())

        class OtherDataset:
            gop_size = 2
            num_gops = 4
            total_frames = 7

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'anchor.pth'
            save_anchor_checkpoint(
                model, optimizer, 0, 0.0, 0.0, make_args(),
                DatasetMetadata(), path,
            )
            with self.assertRaisesRegex(ValueError, 'GOP 划分'):
                load_anchor_checkpoint(
                    path, make_anchor(), torch.device('cpu'), make_args(),
                    dataset=OtherDataset(),
                )


if __name__ == '__main__':
    unittest.main()
