import json
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace

import torch

from evaluate_grid_levels import (
    build_grid_level_report,
    format_grid_level_report,
    leave_one_out_grids,
    prefix_grids,
    quantized_grids,
    save_grid_level_report,
)


class GridSelectionTest(unittest.TestCase):
    def test_leave_one_out_zeros_only_requested_level(self):
        grids = (torch.ones(2, 2), torch.full((2, 2), 2.0))

        selected = leave_one_out_grids(grids, 1)

        self.assertTrue(torch.equal(selected[0], grids[0]))
        self.assertTrue(torch.equal(selected[1], torch.zeros_like(grids[1])))
        self.assertTrue(torch.equal(grids[1], torch.full((2, 2), 2.0)))

    def test_prefix_keeps_only_coarse_to_fine_prefix(self):
        grids = tuple(torch.full((1,), float(index + 1)) for index in range(3))

        selected = prefix_grids(grids, 1)
        zero = prefix_grids(grids, -1)

        self.assertEqual([item.item() for item in selected], [1.0, 2.0, 0.0])
        self.assertEqual([item.item() for item in zero], [0.0, 0.0, 0.0])

    def test_quantized_grids_use_each_configured_step(self):
        levels = (
            SimpleNamespace(grid=torch.tensor([0.14, 0.26])),
            SimpleNamespace(grid=torch.tensor([0.24, 0.76])),
        )
        model = SimpleNamespace(
            reconstruction_model=SimpleNamespace(
                grid_encoder=SimpleNamespace(levels=levels),
            ),
            quant_steps=(0.1, 0.5),
        )

        result = quantized_grids(model)

        self.assertTrue(torch.allclose(result[0], torch.tensor([0.1, 0.3])))
        self.assertTrue(torch.allclose(result[1], torch.tensor([0.0, 1.0])))


class GridLevelReportTest(unittest.TestCase):
    def make_report(self):
        config = Namespace(
            entropy_model='cross_scale_recursive',
            quant_step=0.1,
            quant_steps=[0.1, 0.2],
        )
        quality = {
            'samples': 2,
            'full': [60.0, 1.8],
            'zero': [20.0, 1.0],
            'removed': [[58.0, 1.78], [50.0, 1.6]],
            'prefix': [[40.0, 1.4], [60.0, 1.8]],
        }
        return build_grid_level_report(
            'model.pth',
            300,
            config,
            'frames',
            4,
            5,
            (torch.zeros(2, 2), torch.zeros(4, 2)),
            (100.0, 300.0),
            quality,
        )

    def test_report_contains_contributions_and_monotonic_prefix_bits(self):
        report = self.make_report()

        self.assertEqual(report['estimated_grid_bits'], 400.0)
        self.assertEqual(report['full_quality']['psnr_db'], 30.0)
        self.assertEqual(
            report['level_contributions'][1]['psnr_drop_db'], 5.0,
        )
        self.assertEqual(
            [item['cumulative_bits'] for item in report['prefixes']],
            [100.0, 400.0],
        )
        self.assertEqual(report['prefixes'][0]['delta_psnr_db'], 10.0)
        self.assertEqual(report['prefixes'][1]['delta_psnr_db'], 10.0)

    def test_saves_text_and_json_reports(self):
        report = self.make_report()
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path, json_path = save_grid_level_report(report, temp_dir)
            text = text_path.read_text(encoding='utf-8')
            saved = json.loads(json_path.read_text(encoding='utf-8'))

        self.assertIn('Leave-one-out quality contribution', text)
        self.assertIn('Coarse-to-fine prefix quality', text)
        self.assertIn('L0-L1', format_grid_level_report(report))
        self.assertEqual(saved['estimated_grid_bits'], 400.0)


if __name__ == '__main__':
    unittest.main()
