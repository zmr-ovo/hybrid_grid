import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from gop_lora import GOPVideoDataset
from model import DynamicVideoDataset


class GOPVideoDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_frames(self, count):
        for index in range(count):
            Image.new('RGB', (6, 4), color=(index, 32, 64)).save(
                self.root / '{}.png'.format(index)
            )

    def test_adds_gop_metadata_without_changing_image_or_coordinates(self):
        self.add_frames(7)
        base = DynamicVideoDataset(self.root, fixed_res=(4, 6))
        dataset = GOPVideoDataset(base, gop_size=3)

        base_sample = base[4]
        sample = dataset[4]

        self.assertEqual(dataset.num_gops, 3)
        self.assertTrue(torch.equal(sample['coords'], base_sample['coords']))
        self.assertTrue(torch.equal(sample['pixels'], base_sample['pixels']))
        self.assertEqual(sample['frame_idx'], 4)
        self.assertEqual(sample['gop_idx'], 1)
        self.assertEqual(sample['gop_id'], 1)
        self.assertEqual(sample['gop_local_idx'], 1)
        self.assertEqual(sample['gop_local_time'], 0.5)

    def test_local_time_reaches_each_gop_endpoint(self):
        self.add_frames(7)
        dataset = GOPVideoDataset(
            DynamicVideoDataset(self.root, fixed_res=(2, 2)),
            gop_size=3,
        )

        self.assertEqual(dataset[0]['gop_local_time'], 0.0)
        self.assertEqual(dataset[2]['gop_local_time'], 1.0)
        self.assertEqual(dataset[3]['gop_local_time'], 0.0)
        self.assertEqual(dataset[5]['gop_local_time'], 1.0)
        self.assertEqual(dataset[6]['gop_local_time'], 0.0)

    def test_frame_interval_preserves_original_frame_mapping(self):
        self.add_frames(8)
        dataset = GOPVideoDataset(
            DynamicVideoDataset(
                self.root, fixed_res=(2, 2), frame_interval=2,
            ),
            gop_size=4,
        )

        self.assertEqual(dataset.frame_indices, (0, 2, 4, 6))
        self.assertEqual(dataset.gop_sample_indices(0), (0, 1))
        self.assertEqual(dataset.gop_sample_indices(1), (2, 3))
        self.assertEqual(dataset[2]['frame_idx'], 4)
        self.assertEqual(dataset[2]['gop_idx'], 1)


if __name__ == '__main__':
    unittest.main()
