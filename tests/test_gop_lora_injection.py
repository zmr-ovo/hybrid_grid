import unittest

import torch

from gop_lora.injection import (
    GOPLoRADecoder,
    freeze_shared_parameters,
    gop_lora_layers,
    gop_parameters,
    inject_gop_lora,
    lora_parameters,
    shared_parameters,
)
from gop_lora.linear import GOPLoRALinear
from model import HybridGridNet


def make_model():
    return HybridGridNet(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=2,
        hidden_dim=16,
    )


class GOPLoRAInjectionTest(unittest.TestCase):
    def test_injects_exactly_the_five_decoder_linear_layers(self):
        model = make_model()
        original_weights = (
            [layer.weight for layer in model.decoder.linear]
            + [model.decoder.output_layer[0].weight]
        )

        names = inject_gop_lora(model, num_gops=3, rank=2, alpha=2)

        self.assertEqual(names, (
            'decoder.linear.0',
            'decoder.linear.1',
            'decoder.linear.2',
            'decoder.linear.3',
            'decoder.output_layer.0',
        ))
        self.assertIsInstance(model.decoder, GOPLoRADecoder)
        layers = gop_lora_layers(model)
        self.assertEqual(len(layers), 5)
        for layer, weight in zip(layers, original_weights):
            self.assertIs(layer.shared.weight, weight)
            self.assertEqual(len(layer.lora_a), 2)

    def test_does_not_inject_grid_gate_or_temporal_modulation(self):
        model = make_model()
        inject_gop_lora(model, num_gops=2, rank=1)

        lora_names = tuple(
            name for name, module in model.named_modules()
            if isinstance(module, GOPLoRALinear)
        )

        self.assertTrue(all(name.startswith('decoder.') for name in lora_names))
        self.assertFalse(any('grid_encoder' in name for name in lora_names))
        self.assertFalse(any('gate_' in name for name in lora_names))
        self.assertFalse(any('time_mod' in name for name in lora_names))

    def test_freezes_everything_except_all_lora_parameters(self):
        model = make_model()
        inject_gop_lora(model, num_gops=2, rank=1)

        freeze_shared_parameters(model)

        self.assertTrue(lora_parameters(model))
        self.assertTrue(all(
            parameter.requires_grad for parameter in lora_parameters(model)
        ))
        self.assertTrue(all(
            not parameter.requires_grad for parameter in shared_parameters(model)
        ))
        self.assertEqual(gop_parameters(model, 0), ())
        self.assertEqual(len(gop_parameters(model, 1)), 10)

    def test_backward_only_reaches_the_selected_gop(self):
        model = make_model()
        inject_gop_lora(model, num_gops=3, rank=1)
        freeze_shared_parameters(model)
        for layer in gop_lora_layers(model):
            with torch.no_grad():
                layer.lora_a[0].fill_(1.0)
                layer.lora_b[0].fill_(1.0)

        model.decoder(torch.randn(4, model.decoder.input_dim), 1).sum().backward()

        self.assertTrue(all(
            parameter.grad is not None for parameter in gop_parameters(model, 1)
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in gop_parameters(model, 2)
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in shared_parameters(model)
        ))

    def test_rejects_duplicate_or_unsupported_injection(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, 'only target'):
            inject_gop_lora(model, 2, 1, target='gate')

        inject_gop_lora(model, 2, 1)
        with self.assertRaisesRegex(ValueError, 'already'):
            inject_gop_lora(model, 2, 1)


if __name__ == '__main__':
    unittest.main()
