import unittest
from unittest.mock import patch

import torch
from torch import nn

from fps import benchmark_fps


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Identity()

    def forward(self, coords):
        return self.decoder(coords)


class FpsBenchmarkTest(unittest.TestCase):
    @patch('fps.torch.cuda.synchronize')
    def test_cpu_benchmark_reports_full_and_decoder_fps(self, synchronize):
        model = ToyModel()
        model.train()
        coords = torch.zeros(2, 3, 4, 4)

        result = benchmark_fps(
            model, coords, warmup_steps=1, repeat_steps=10,
        )

        self.assertGreater(result.reconstruction_fps, 0.0)
        self.assertGreater(result.decoder_fps, 0.0)
        self.assertEqual(result.paper_inference_fps, result.reconstruction_fps)
        self.assertIsNone(result.codec_encode_ms)
        self.assertIsNone(result.codec_decode_ms)
        self.assertEqual(result.repeat_steps, 10)
        self.assertEqual(result.batch_size, 2)
        self.assertTrue(model.training)
        synchronize.assert_not_called()

    def test_requires_at_least_ten_repeats(self):
        with self.assertRaisesRegex(ValueError, 'at least 10'):
            benchmark_fps(
                ToyModel(), torch.zeros(1, 3, 2, 2), repeat_steps=9,
            )


if __name__ == '__main__':
    unittest.main()
