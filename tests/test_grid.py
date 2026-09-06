import unittest

import torch

from encoding import MultiResGrid, SingleResGrid
from model import HybridGridNet


def make_model():
    return HybridGridNet(
        grid_levels=2,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=6,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
    )


class GridInterfaceTest(unittest.TestCase):
    def test_single_level_external_grid_matches_internal_grid(self):
        encoder = SingleResGrid(n_features=2, base_res=4, aspect_ratio=(1, 1))
        coords = torch.rand(2, 3)

        internal = encoder(coords)
        external = encoder(coords, grid=encoder.grid)

        self.assertTrue(torch.equal(internal, external))

    def test_multi_level_external_grids_match_internal_grids(self):
        encoder = MultiResGrid(
            n_levels=2,
            n_features_per_level=2,
            base_resolution=4,
            finest_resolution=6,
            aspect_ratio=(1, 1),
        )
        coords = torch.rand(2, 3)
        grids = [level.grid for level in encoder.levels]

        internal = encoder(coords)
        external = encoder(coords, grids=grids)

        self.assertTrue(torch.equal(internal, external))

    def test_model_external_grids_match_baseline(self):
        model = make_model().eval()
        coords = torch.rand(1, 3, 4, 5)
        grids = [level.grid for level in model.grid_encoder.levels]

        with torch.no_grad():
            baseline = model(coords)
            external = model(coords, grids=grids)

        self.assertTrue(torch.equal(baseline, external))

    def test_external_grid_is_used_for_interpolation(self):
        encoder = SingleResGrid(n_features=2, base_res=4, aspect_ratio=(1, 1))
        coords = torch.rand(2, 3)
        external_grid = torch.zeros_like(encoder.grid)

        output = encoder(coords, grid=external_grid)

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_reconstruction_gradient_reaches_external_grids(self):
        model = make_model()
        coords = torch.rand(1, 3, 4, 5)
        grids = [
            level.grid.detach().clone().requires_grad_()
            for level in model.grid_encoder.levels
        ]

        model(coords, grids=grids).mean().backward()

        self.assertTrue(all(grid.grad is not None for grid in grids))
        self.assertTrue(all(torch.isfinite(grid.grad).all() for grid in grids))

    def test_rejects_invalid_external_grids(self):
        encoder = MultiResGrid(n_levels=2, n_features_per_level=2)
        coords = torch.rand(2, 3)

        with self.assertRaisesRegex(TypeError, 'list or tuple'):
            encoder(coords, grids=encoder.levels[0].grid)
        with self.assertRaisesRegex(ValueError, 'expected 2 grids'):
            encoder(coords, grids=[encoder.levels[0].grid])
        with self.assertRaisesRegex(ValueError, 'grid shape'):
            encoder(coords, grids=[torch.zeros(1), encoder.levels[1].grid])
        wrong_dtype = encoder.levels[0].grid.detach().double()
        with self.assertRaisesRegex(ValueError, 'grid dtype'):
            encoder(coords, grids=[wrong_dtype, encoder.levels[1].grid])


if __name__ == '__main__':
    unittest.main()
