import unittest

import torch
from torch.nn.utils import parametrize

from compression.model import CompressedHybridGridNet
from compression.network_quantization import (
    FakeQuantizedParameter,
    configure_network_qat,
    network_qat_state,
    network_qat_storage,
    prepare_network_qat,
)
from model import HybridGridNet


def make_model():
    baseline = HybridGridNet(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=2,
        hidden_dim=16,
    )
    return CompressedHybridGridNet(baseline, quant_steps=0.1)


class FakeQuantizedParameterTest(unittest.TestCase):
    def test_disabled_is_exact_and_enabled_preserves_gradient(self):
        parameter = torch.tensor([-0.83, 0.17, 0.91], requires_grad=True)
        quantizer = FakeQuantizedParameter(parameter, bits=4)

        self.assertTrue(torch.equal(quantizer(parameter), parameter))
        quantizer.configure(True, True)
        output = quantizer(parameter)
        self.assertFalse(torch.equal(output, parameter))
        output.sum().backward()
        self.assertTrue(torch.equal(parameter.grad, torch.ones_like(parameter)))

    def test_freeze_keeps_qparams_fixed(self):
        parameter = torch.tensor([-1.0, 1.0])
        quantizer = FakeQuantizedParameter(parameter, bits=8)
        quantizer.train()
        quantizer.configure(True, True)
        quantizer(parameter)
        scale = quantizer.scale.clone()

        quantizer.configure(True, False)
        quantizer(parameter * 100)
        self.assertTrue(torch.equal(quantizer.scale, scale))

    def test_counts_payload_and_qparam_metadata(self):
        quantizer = FakeQuantizedParameter(torch.zeros(2, 3), bits=8, axis=0)
        storage = quantizer.storage()

        self.assertEqual(storage.parameter_count, 6)
        self.assertEqual(storage.payload_bits, 48)
        self.assertEqual(storage.metadata_bits, 168)


class NetworkQatModelTest(unittest.TestCase):
    def test_prepares_only_linear_weights(self):
        model = make_model()
        grids = tuple(
            level.grid for level in model.reconstruction_model.grid_encoder.levels
        )
        linear_layers = tuple(
            module for module in model.reconstruction_model.modules()
            if isinstance(module, torch.nn.Linear)
        )

        count = prepare_network_qat(model, 8, excluded_parameters=grids)

        self.assertEqual(count, len(linear_layers))
        for layer in linear_layers:
            self.assertTrue(parametrize.is_parametrized(layer, 'weight'))
            self.assertFalse(parametrize.is_parametrized(layer, 'bias'))
        for level in model.reconstruction_model.grid_encoder.levels:
            self.assertFalse(parametrize.is_parametrized(level, 'grid'))
        self.assertFalse(
            parametrize.is_parametrized(
                model.reconstruction_model.pe_encoder, 'freqs',
            )
        )
        for entropy_model in model.entropy_models:
            for module in entropy_model.modules():
                for name, _ in module.named_parameters(recurse=False):
                    self.assertFalse(parametrize.is_parametrized(module, name))

        storage = network_qat_storage(model)
        self.assertEqual(set(storage), {'non_grid'})
        self.assertEqual(
            storage['non_grid'].parameter_count,
            sum(layer.weight.numel() for layer in linear_layers),
        )

    def test_qat_state_is_explicit(self):
        model = make_model()
        grids = tuple(
            level.grid for level in model.reconstruction_model.grid_encoder.levels
        )
        prepare_network_qat(model, 8, excluded_parameters=grids)

        self.assertEqual(network_qat_state(model), 'disabled')
        configure_network_qat(model, True)
        self.assertEqual(network_qat_state(model), 'calibrating')
        configure_network_qat(model, True, freeze=True)
        self.assertEqual(network_qat_state(model), 'frozen')


if __name__ == '__main__':
    unittest.main()
