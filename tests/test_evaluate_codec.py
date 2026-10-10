"""Unit tests for exact BPP reporting and CPU-safe codec evaluation helpers."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate_codec import (
    _synchronize,
    build_codec_report,
    format_codec_report,
    save_codec_report,
    run,
)
from fps import FpsBenchmarkResult
import torch


class CodecEvaluationReportTest(unittest.TestCase):
    @staticmethod
    def _loaded():
        level = SimpleNamespace(
            level=0,
            symbol_count=10,
            payload_bits=80,
            distribution_seconds=0.1,
            cdf_seconds=0.2,
            arithmetic_seconds=0.3,
            total_seconds=0.6,
        )
        return SimpleNamespace(
            stats=SimpleNamespace(
                total_bits=1000,
                grid_bits=600,
                non_grid_parameter_bits=200,
                entropy_parameter_bits=80,
                manifest_bits=70,
                container_overhead_bits=50,
            ),
            timing=SimpleNamespace(
                model_build_seconds=0.1,
                parameter_restore_seconds=0.2,
                grid_decode_seconds=0.6,
                total_seconds=1.0,
                grid_levels=(level,),
            ),
        )

    def test_report_uses_exact_file_bits_for_total_bpp(self):
        quality = {
            'psnr_db': 36.5,
            'ms_ssim': 0.98,
            'frames': 5,
            'evaluation_seconds': 2.0,
        }
        fps = FpsBenchmarkResult(12.0, 20.0, 1, 10, 1)

        report = build_codec_report(
            'model.hgm', 'frames', (4, 5), quality, self._loaded(), fps,
        )

        self.assertEqual(report['total_video_pixels'], 100)
        self.assertEqual(report['rate']['actual_total']['bits'], 1000)
        self.assertEqual(report['rate']['actual_total']['bpp'], 10.0)
        self.assertEqual(report['fps']['paper_inference_fps'], 12.0)
        self.assertFalse(report['fps']['includes_entropy_decode'])

    def test_saves_matching_text_and_json_reports(self):
        quality = {
            'psnr_db': 36.5,
            'ms_ssim': 0.98,
            'frames': 5,
            'evaluation_seconds': 2.0,
        }
        report = build_codec_report(
            'model.hgm', 'frames', (4, 5), quality, self._loaded(),
            FpsBenchmarkResult(12.0, 20.0, 1, 10, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            text_path, json_path = save_codec_report(report, directory)
            text = text_path.read_text(encoding='utf-8')

        self.assertIn('Actual total: 1000 bits | 10.000000000 BPP', text)
        self.assertIn('Checkpoint required: NO', text)
        self.assertTrue(json_path.name.endswith('.json'))
        self.assertIn('Grid L0', format_codec_report(report))

    @patch('evaluate_codec.torch.cuda.synchronize')
    def test_cpu_synchronize_does_not_call_cuda(self, synchronize):
        _synchronize(torch.device('cpu'))
        synchronize.assert_not_called()

    @patch('evaluate_codec.load_model_bitstream')
    def test_invalid_data_path_fails_before_entropy_decode(self, load_model):
        args = Namespace(
            seed=42,
            device='cpu',
            fixed_res=[4, 5],
            data_root='/path/that/does/not/exist',
            batch_size=1,
            workers=0,
            frame_interval=1,
            output_dir='unused',
            bitstream='unused.hgm',
            log_interval=1,
            warmup_steps=1,
            repeat_steps=10,
        )

        with self.assertRaisesRegex(ValueError, 'data_root'):
            run(args)

        load_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
