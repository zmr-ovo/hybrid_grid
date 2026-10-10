import unittest

import torch

from compression.entropy_bottleneck import EntropyBottleneck
from compression.entropy_cdf import factorized_cdfs, logistic_cdfs


class EntropyCdfTest(unittest.TestCase):
    def test_factorized_model_produces_one_valid_cdf_per_channel(self):
        model = EntropyBottleneck(3)

        cdfs = factorized_cdfs(model, -3, 4)

        self.assertEqual(cdfs.shape, (3, 9))
        self.assertTrue(torch.all(cdfs[:, 0] == 0))
        self.assertTrue(torch.all(cdfs[:, -1] == 1 << 16))
        self.assertTrue(torch.all(cdfs[:, 1:] > cdfs[:, :-1]))

    def test_logistic_model_produces_one_valid_cdf_per_value(self):
        mean = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]]])
        scale = torch.full_like(mean, 2.0)

        cdfs = logistic_cdfs(mean, scale, -2, 2)

        self.assertEqual(cdfs.shape, (4, 6))
        self.assertTrue(torch.all(cdfs[:, 0] == 0))
        self.assertTrue(torch.all(cdfs[:, -1] == 1 << 16))
        self.assertTrue(torch.all(cdfs[:, 1:] > cdfs[:, :-1]))

    def test_rejects_invalid_support_and_scale(self):
        with self.assertRaisesRegex(ValueError, 'smaller'):
            logistic_cdfs(torch.zeros(1), torch.ones(1), 2, 1)
        with self.assertRaisesRegex(ValueError, 'positive'):
            logistic_cdfs(torch.zeros(1), torch.zeros(1), -1, 1)


if __name__ == '__main__':
    unittest.main()
