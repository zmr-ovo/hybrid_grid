import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from evaluate_global_gop import (
    GOPMetrics,
    build_report,
    evaluate_global_gops,
    format_report,
    load_global_model,
    save_report,
)
from model import HybridGridNet
from train import save_checkpoint


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_coords = None

    def forward(self, coords):
        self.last_coords = coords.detach().clone()
        return coords


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
        base_res=[4, 5],
        fixed_res=[4, 5],
        frame_interval=1,
    )


class GlobalGOPEvaluationTest(unittest.TestCase):
    def test_groups_metrics_and_preserves_global_time(self):
        model = RecordingModel()
        coords = torch.zeros(3, 3, 2, 2)
        coords[0, 2] = 0.0
        coords[1, 2] = 0.5
        coords[2, 2] = 1.0
        batch = {
            'coords': coords,
            'pixels': torch.zeros_like(coords),
            'frame_idx': torch.tensor([0, 1, 2]),
        }

        with patch(
            'evaluate_global_gop.psnr_fn', side_effect=[30.0, 20.0, 10.0],
        ), patch(
            'evaluate_global_gop.msssim_fn',
            side_effect=[0.9, 0.8, 0.7],
        ):
            result = evaluate_global_gops(
                model, [batch], torch.device('cpu'),
                total_frames=3, gop_size=2, log_interval=1,
            )

        self.assertTrue(torch.equal(model.last_coords, coords))
        self.assertAlmostEqual(result.psnr, 20.0)
        self.assertAlmostEqual(result.msssim, 0.8)
        self.assertEqual(result.frames, 3)
        self.assertEqual(result.per_gop[0].frames, 2)
        self.assertAlmostEqual(result.per_gop[0].psnr, 25.0)
        self.assertEqual(result.per_gop[1].frames, 1)
        self.assertAlmostEqual(result.per_gop[1].psnr, 10.0)
        self.assertTrue(model.training)

    def test_rejects_frame_index_outside_video(self):
        batch = {
            'coords': torch.zeros(1, 3, 2, 2),
            'pixels': torch.zeros(1, 3, 2, 2),
            'frame_idx': torch.tensor([3]),
        }

        with self.assertRaisesRegex(IndexError, 'outside the video'):
            evaluate_global_gops(
                RecordingModel(), [batch], torch.device('cpu'),
                total_frames=3, gop_size=2,
            )

    def test_loads_train_checkpoint_configuration(self):
        config = make_config()
        model = HybridGridNet(
            grid_levels=config.grid_levels,
            grid_feat_dim=config.grid_feat_dim,
            base_resolution=config.base_resolution,
            finest_resolution=config.finest_resolution,
            aspect_ratio=tuple(config.aspect_ratio),
            time_scale=config.time_scale,
            pe_freq=config.pe_freq,
            hidden_dim=config.hidden_dim,
        )
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'train_best.pth'
            save_checkpoint(model, optimizer, 4, 31.0, 32.0, config, path)
            restored, restored_config, info = load_global_model(
                path, torch.device('cpu'),
            )

        self.assertEqual(restored_config.hidden_dim, 16)
        self.assertEqual(info['completed_epochs'], 5)
        self.assertEqual(info['best_val_psnr'], 31.0)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_report_is_saved_as_text_and_json(self):
        result = Namespace(
            psnr=30.0,
            msssim=0.9,
            frames=3,
            per_gop=(
                GOPMetrics(
                    gop_index=0, start_frame=0, stop_frame=2,
                    frames=2, psnr=31.0, msssim=0.91,
                ),
                GOPMetrics(
                    gop_index=1, start_frame=2, stop_frame=3,
                    frames=1, psnr=28.0, msssim=0.88,
                ),
            ),
        )
        model = torch.nn.Linear(1, 1)
        config = make_config()
        dataset = Namespace(
            frame_paths=(Path('/tmp/frames/1.png'),),
            total_frames=3,
            frame_interval=1,
            fixed_res=(4, 5),
        )
        report = build_report(
            'model.pth', model, config, {}, dataset, 2, result,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            text_path, json_path = save_report(report, temp_dir)
            text = text_path.read_text(encoding='utf-8')
            saved = json.loads(json_path.read_text(encoding='utf-8'))

        self.assertIn('FULL VIDEO | PSNR: 30.0000 dB', text)
        self.assertIn('GOP 1', format_report(report))
        self.assertEqual(saved['gops'][0]['frames'], 2)


if __name__ == '__main__':
    unittest.main()
