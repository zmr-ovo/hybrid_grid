import random
import unittest

import numpy as np
import torch

from train import seed_everything, seed_worker


class ReproducibilityTest(unittest.TestCase):
    def test_global_seed_reproduces_random_sequences(self):
        seed_everything(42)
        first = (random.random(), np.random.rand(), torch.rand(1))

        seed_everything(42)
        second = (random.random(), np.random.rand(), torch.rand(1))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_worker_seed_reproduces_python_and_numpy(self):
        torch.manual_seed(7)
        seed_worker(0)
        first = (random.random(), np.random.rand())

        torch.manual_seed(7)
        seed_worker(0)
        second = (random.random(), np.random.rand())

        self.assertEqual(first, second)

    def test_negative_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            seed_everything(-1)


if __name__ == '__main__':
    unittest.main()
