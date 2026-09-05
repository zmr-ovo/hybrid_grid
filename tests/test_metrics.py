import unittest

import torch

from util import psnr_fn


class MetricTest(unittest.TestCase):
    def test_psnr_is_averaged_per_frame(self):
        pred = torch.zeros(2, 1, 2, 2)
        target = torch.stack([
            torch.full((1, 2, 2), 0.1),
            torch.full((1, 2, 2), 0.01),
        ])

        self.assertAlmostEqual(psnr_fn(pred, target), 30.0, places=4)


if __name__ == '__main__':
    unittest.main()
