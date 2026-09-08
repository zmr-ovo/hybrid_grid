import unittest

import torch
from torch import nn

from gop_lora.linear import GOPLoRALinear


class GOPLoRALinearTest(unittest.TestCase):
    def test_zero_initialization_matches_shared_linear_exactly(self):
        shared = nn.Linear(5, 4)
        layer = GOPLoRALinear(shared, num_gops=3, rank=2, alpha=2)
        inputs = torch.randn(2, 3, 5)

        expected = shared(inputs)
        actual = layer(inputs, gop_index=1)

        self.assertTrue(torch.equal(actual, expected))

    def test_each_gop_uses_an_independent_adapter(self):
        shared = nn.Linear(3, 2, bias=False)
        layer = GOPLoRALinear(shared, num_gops=2, rank=1)
        inputs = torch.ones(1, 3)
        with torch.no_grad():
            layer.lora_a[0].fill_(1.0)
            layer.lora_b[0].fill_(1.0)

        base = shared(inputs)
        gop_zero = layer(inputs, 0)
        gop_one = layer(inputs, 1)

        self.assertFalse(torch.equal(gop_zero, base))
        self.assertTrue(torch.equal(gop_one, base))

    def test_backward_only_reaches_selected_gop(self):
        layer = GOPLoRALinear(nn.Linear(3, 2), num_gops=2, rank=1)
        with torch.no_grad():
            layer.lora_a[0].fill_(1.0)
            layer.lora_b[0].fill_(1.0)

        layer(torch.ones(1, 3), 0).sum().backward()

        self.assertTrue(torch.any(layer.lora_a[0].grad != 0))
        self.assertTrue(torch.any(layer.lora_b[0].grad != 0))
        self.assertIsNone(layer.lora_a[1].grad)
        self.assertIsNone(layer.lora_b[1].grad)

    def test_accepts_collated_indices_from_one_gop(self):
        layer = GOPLoRALinear(nn.Linear(3, 2), num_gops=2, rank=1)
        inputs = torch.randn(2, 3)

        output = layer(inputs, torch.tensor([1, 1]))

        self.assertEqual(output.shape, (2, 2))

    def test_rejects_mixed_gop_batch(self):
        layer = GOPLoRALinear(nn.Linear(3, 2), num_gops=2, rank=1)

        with self.assertRaisesRegex(ValueError, 'exactly one GOP'):
            layer(torch.randn(2, 3), torch.tensor([0, 1]))

    def test_reuses_shared_parameters_without_copying(self):
        shared = nn.Linear(3, 2)
        layer = GOPLoRALinear(shared, num_gops=2, rank=1)

        self.assertIs(layer.shared, shared)
        self.assertIs(layer.shared.weight, shared.weight)
        self.assertEqual(layer.shared_parameters(), (shared.weight, shared.bias))
        self.assertEqual(len(layer.lora_parameters()), 4)
        self.assertIs(layer.gop_parameters(1)[0], layer.lora_a[1])

    def test_state_dict_restores_identical_output(self):
        layer = GOPLoRALinear(nn.Linear(3, 2), num_gops=2, rank=1)
        with torch.no_grad():
            layer.lora_a[1].fill_(0.25)
            layer.lora_b[1].fill_(0.5)
        inputs = torch.randn(2, 3)
        expected = layer(inputs, 1)

        restored = GOPLoRALinear(nn.Linear(3, 2), num_gops=2, rank=1)
        restored.load_state_dict(layer.state_dict())

        self.assertTrue(torch.equal(restored(inputs, 1), expected))

    def test_rejects_invalid_configuration_and_inputs(self):
        for arguments in (
            {'shared_linear': object(), 'num_gops': 2, 'rank': 1},
            {'shared_linear': nn.Linear(3, 2), 'num_gops': 0, 'rank': 1},
            {'shared_linear': nn.Linear(3, 2), 'num_gops': 2, 'rank': 3},
            {'shared_linear': nn.Linear(3, 2), 'num_gops': 2, 'rank': 1,
             'alpha': 0},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    GOPLoRALinear(**arguments)

        layer = GOPLoRALinear(nn.Linear(3, 2), num_gops=2, rank=1)
        with self.assertRaises(IndexError):
            layer(torch.randn(1, 3), 2)
        with self.assertRaises(ValueError):
            layer(torch.randn(1, 4), 0)
        with self.assertRaises(TypeError):
            layer(torch.randn(1, 3), torch.tensor([0.0]))


if __name__ == '__main__':
    unittest.main()
