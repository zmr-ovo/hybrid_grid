import copy
import unittest

import torch

from gop_lora.injection import gop_lora_layers
from gop_lora.model import GOPLoRAHybridGridNet
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


class GOPLoRAModelTest(unittest.TestCase):
    def test_zero_initialization_matches_the_paper_model(self):
        baseline = make_model()
        shared = copy.deepcopy(baseline)
        model = GOPLoRAHybridGridNet(shared, num_gops=2, rank=1)
        coords = torch.rand(1, 3, 3, 4)

        expected = baseline(coords)
        actual = model(coords, gop_index=1)

        self.assertTrue(torch.equal(actual, expected))
        self.assertIs(model.shared_model, shared)

    def test_adapter_change_only_affects_its_own_gop(self):
        baseline = make_model()
        shared = copy.deepcopy(baseline)
        model = GOPLoRAHybridGridNet(shared, num_gops=2, rank=1)
        coords = torch.rand(1, 3, 3, 4)
        expected = baseline(coords)
        output_layer = gop_lora_layers(model)[-1]
        with torch.no_grad():
            output_layer.lora_a[0].fill_(1.0)
            output_layer.lora_b[0].fill_(0.25)

        gop_zero = model(coords, 0)
        gop_one = model(coords, 1)

        self.assertFalse(torch.equal(gop_zero, expected))
        self.assertTrue(torch.equal(gop_one, expected))

    def test_supports_external_grids(self):
        baseline = make_model()
        model = GOPLoRAHybridGridNet(
            copy.deepcopy(baseline), num_gops=2, rank=1,
        )
        coords = torch.rand(1, 3, 3, 4)
        grids = tuple(
            level.grid for level in baseline.grid_encoder.levels
        )

        expected = baseline(coords, grids=grids)
        actual = model(coords, 0, grids=grids)

        self.assertTrue(torch.equal(actual, expected))

    def test_state_dict_restores_identical_reconstruction(self):
        model = GOPLoRAHybridGridNet(make_model(), num_gops=2, rank=1)
        with torch.no_grad():
            layer = gop_lora_layers(model)[0]
            layer.lora_a[1].fill_(0.25)
            layer.lora_b[1].fill_(0.5)
        coords = torch.rand(1, 3, 3, 4)
        expected = model(coords, 1)

        restored = GOPLoRAHybridGridNet(make_model(), num_gops=2, rank=1)
        restored.load_state_dict(model.state_dict())

        self.assertTrue(torch.equal(restored(coords, 1), expected))

    def test_rejects_invalid_shared_model(self):
        with self.assertRaises(TypeError):
            GOPLoRAHybridGridNet(object(), num_gops=2, rank=1)


if __name__ == '__main__':
    unittest.main()
