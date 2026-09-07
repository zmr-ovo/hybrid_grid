import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from compression.rate import (
    estimate_fp32_rate,
    estimate_grid_metadata,
    parameter_storage,
)
from evaluate_compression import (
    build_evaluation_report,
    format_evaluation_report,
    load_compression_model,
    save_evaluation_report,
)
from train_compression import (
    CompressionEvaluation,
    build_compression_model,
    save_compression_checkpoint,
)


def make_config():
    return Namespace(
        data_root='unused',
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=[1, 1],
        time_scale=1.0,
        pe_freq=2,
        hidden_dim=16,
        quant_step=0.1,
        epochs=300,
        warmup_epochs=30,
        symbol_start_epoch=270,
        lambda_max=5e-4,
        base_res=[4, 5],
        fixed_res=[4, 5],
        frame_interval=1,
    )


def make_evaluation():
    non_grid = parameter_storage([torch.zeros(10)], torch.float32)
    entropy = parameter_storage([torch.zeros(2)], torch.float32)
    metadata = estimate_grid_metadata(
        [torch.zeros(1, 2), torch.zeros(2, 2)], [0.1, 0.1],
    )
    rate = estimate_fp32_rate(100, non_grid, entropy, metadata, 400, 2.0)
    return CompressionEvaluation(
        psnr=31.25,
        msssim=0.95,
        rate=rate,
        level_bits=(150.0, 250.0),
    )


class CompressionEvaluationReportTest(unittest.TestCase):
    def test_report_contains_quality_and_complete_fp32_subtotal(self):
        report = build_evaluation_report(
            'model.pth', 50, make_config(), 'frames', 5, 4, 5,
            make_evaluation(),
        )

        self.assertEqual(report['quant_mode'], 'symbols')
        self.assertEqual(report['quality']['psnr_db'], 31.25)
        self.assertEqual(report['rate']['grid']['level_bits'], [150.0, 250.0])
        self.assertEqual(report['rate']['non_grid_fp32']['bits'], 320)
        self.assertEqual(report['rate']['entropy_model_fp32']['bits'], 64)
        self.assertEqual(
            report['rate']['estimated_payload_subtotal']['bits'], 784,
        )
        self.assertEqual(report['rate']['quantization_metadata']['bits'], 240)
        self.assertEqual(report['rate']['estimated_total']['bits'], 1024)
        self.assertTrue(report['rate']['metadata_included'])
        self.assertFalse(report['rate']['actual_bitstream_available'])

    def test_saves_matching_text_and_json_reports(self):
        report = build_evaluation_report(
            'model.pth', 50, make_config(), 'frames', 5, 4, 5,
            make_evaluation(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path, json_path = save_evaluation_report(report, temp_dir)
            saved = json.loads(json_path.read_text(encoding='utf-8'))
            text = text_path.read_text(encoding='utf-8')

        self.assertEqual(saved, report)
        self.assertIn('PSNR: 31.2500 dB', text)
        self.assertIn('Grid level 1 bits: 250', text)
        self.assertIn('Quantization metadata', text)
        self.assertIn('Metadata included: YES', text)
        self.assertIn('NOTE: estimated total is not actual total BPP.', text)

    def test_loads_model_configuration_and_full_compressed_state(self):
        config = make_config()
        model = build_compression_model(config, torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'compression.pth'
            save_compression_checkpoint(
                model, optimizer, 4, 30.0, 31.0, config, path,
            )
            restored, restored_config, epoch = load_compression_model(
                path, torch.device('cpu'),
            )

        self.assertEqual(epoch, 5)
        self.assertEqual(restored_config.quant_step, 0.1)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
