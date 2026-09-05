import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from model import DynamicVideoDataset


class DynamicVideoDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_frame(self, name, size=(12, 8)):
        Image.new('RGB', size, color=(32, 64, 128)).save(self.root / name)

    def test_fixed_resolution_matches_coordinates_and_pixels(self):
        self.add_frame('0.png')
        dataset = DynamicVideoDataset(self.root, fixed_res=(4, 6))

        sample = dataset[0]

        self.assertEqual(sample['coords'].shape, (3, 4, 6))
        self.assertEqual(sample['pixels'].shape, (3, 4, 6))
        self.assertEqual(sample['coords'][0, 0, 0].item(), 0.0)
        self.assertEqual(sample['coords'][0, 0, -1].item(), 1.0)
        self.assertEqual(sample['coords'][1, 0, 0].item(), 0.0)
        self.assertEqual(sample['coords'][1, -1, 0].item(), 1.0)

    def test_time_is_normalized_to_sequence_endpoints(self):
        for index in range(3):
            self.add_frame(f'{index}.png')

        dataset = DynamicVideoDataset(
            self.root, fixed_res=(2, 2), frame_interval=2,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['coords'][2, 0, 0].item(), 0.0)
        self.assertEqual(dataset[1]['frame_idx'], 2)
        self.assertTrue(torch.allclose(dataset[1]['coords'][2], torch.ones(2, 2)))

    def test_filters_non_images_and_uses_natural_order(self):
        for name in ('10.png', '2.png', '1.png'):
            self.add_frame(name)
        (self.root / 'notes.txt').write_text('not a frame', encoding='utf-8')

        dataset = DynamicVideoDataset(self.root, fixed_res=(2, 2))

        self.assertEqual(
            [path.name for path in dataset.frame_paths],
            ['1.png', '2.png', '10.png'],
        )

    def test_dynamic_resolution_uses_scaled_base_resolution(self):
        self.add_frame('0.png')
        dataset = DynamicVideoDataset(
            self.root, base_res=(8, 12), fixed_res=None,
            min_scale=0.5, max_scale=0.5,
        )

        sample = dataset[0]

        self.assertEqual(sample['coords'].shape, (3, 4, 6))
        self.assertEqual(sample['pixels'].shape, (3, 4, 6))

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, 'no supported image'):
            DynamicVideoDataset(self.root)

        self.add_frame('0.png')
        for kwargs in (
            {'frame_interval': 0},
            {'base_res': (0, 10)},
            {'fixed_res': (-1, 10)},
            {'min_scale': 1.0, 'max_scale': 0.5},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DynamicVideoDataset(self.root, **kwargs)


if __name__ == '__main__':
    unittest.main()
