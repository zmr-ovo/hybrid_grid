import copy
import unittest

import torch

from gop_lora import (
    GOPGridResidualHybridGridNet,
    GOPLowRankGridHybridGridNet,
    GOPStructuredGridHybridGridNet,
    freeze_for_gop,
    gop_adaptation_parameters,
    gop_local_coordinates,
)
from model import HybridGridNet


def make_anchor():
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


class GOPGridResidualTest(unittest.TestCase):
    def test_zero_residual_preserves_anchor_output(self):
        anchor = make_anchor()
        reference = copy.deepcopy(anchor)
        model = GOPGridResidualHybridGridNet(
            anchor, num_gops=3, adapted_gops=(2,), rank=2, alpha=2,
        )
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.5])

        expected = reference(gop_local_coordinates(coords, local_time))
        actual = model(coords, torch.tensor([2]), local_time)

        self.assertTrue(torch.equal(actual, expected))

    def test_only_selected_gop_parameters_are_trainable(self):
        model = GOPGridResidualHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2,
        )
        selected = freeze_for_gop(model, 2)
        selected_ids = {id(parameter) for parameter in selected}

        self.assertEqual(
            selected_ids,
            {
                id(parameter)
                for parameter in gop_adaptation_parameters(model, 2)
            },
        )
        self.assertTrue(all(parameter.requires_grad for parameter in selected))
        self.assertTrue(all(
            parameter.requires_grad == (id(parameter) in selected_ids)
            for parameter in model.parameters()
        ))

    def test_grid_residual_receives_gradient(self):
        model = GOPGridResidualHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2,
        )
        freeze_for_gop(model, 2)
        output = model(
            torch.rand(1, 3, 3, 4), torch.tensor([2]), torch.tensor([0.5]),
        )
        output.mean().backward()

        gradients = [
            parameter.grad for parameter in model.grid_parameters(2)
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(any(
            torch.any(gradient != 0) for gradient in gradients
        ))

    def test_rejects_unconfigured_later_gop(self):
        model = GOPGridResidualHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2,
        )
        with self.assertRaisesRegex(ValueError, 'no Grid residual'):
            model(
                torch.rand(1, 3, 2, 2), torch.tensor([1]),
                torch.tensor([0.5]),
            )


class GOPLowRankGridResidualTest(unittest.TestCase):
    def make_model(self, grid_rank=1):
        return GOPLowRankGridHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2, grid_rank=grid_rank,
            grid_alpha=grid_rank,
        )

    def test_zero_initialization_preserves_anchor_output(self):
        anchor = make_anchor()
        reference = copy.deepcopy(anchor)
        model = GOPLowRankGridHybridGridNet(
            anchor, num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2, grid_rank=1, grid_alpha=1,
        )
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.5])

        expected = reference(gop_local_coordinates(coords, local_time))
        actual = model(coords, torch.tensor([2]), local_time)

        self.assertTrue(torch.equal(actual, expected))

    def test_rank_one_uses_fewer_parameters_than_full_residual(self):
        low_rank = self.make_model(grid_rank=1)
        full = GOPGridResidualHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2,
        )

        low_rank_count = sum(
            parameter.numel() for parameter in low_rank.grid_parameters(2)
        )
        full_count = sum(
            parameter.numel() for parameter in full.grid_parameters(2)
        )
        self.assertLess(low_rank_count, full_count)

    def test_anchor_pca_initializes_an_orthonormal_feature_basis(self):
        model = GOPLowRankGridHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2, grid_rank=2, grid_alpha=2,
            grid_init='anchor_pca',
        )

        for level in model.grid_lora_levels(2):
            identity = torch.eye(level.rank)
            self.assertTrue(torch.allclose(
                level.right.matmul(level.right.transpose(0, 1)),
                identity,
                atol=1e-5,
                rtol=1e-5,
            ))
            self.assertTrue(torch.count_nonzero(level.left) == 0)

    def test_both_factors_receive_gradient_after_an_update(self):
        model = self.make_model(grid_rank=1)
        parameters = freeze_for_gop(model, 2)
        optimizer = torch.optim.Adam(parameters, lr=1e-2)
        coords = torch.rand(1, 3, 3, 4)
        gop_index = torch.tensor([2])
        local_time = torch.tensor([0.5])

        model(coords, gop_index, local_time).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(coords, gop_index, local_time).mean().backward()

        for level in model.grid_lora_levels(2):
            self.assertIsNotNone(level.left.grad)
            self.assertIsNotNone(level.right.grad)
            self.assertTrue(torch.any(level.left.grad != 0))
            self.assertTrue(torch.any(level.right.grad != 0))


class GOPStructuredGridResidualTest(unittest.TestCase):
    def make_model(self, grid_rank=4):
        return GOPStructuredGridHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2, grid_rank=grid_rank,
            grid_alpha=grid_rank,
        )

    def test_zero_initialization_preserves_anchor_output(self):
        anchor = make_anchor()
        reference = copy.deepcopy(anchor)
        model = GOPStructuredGridHybridGridNet(
            anchor, num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2, grid_rank=4, grid_alpha=4,
        )
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.5])

        expected = reference(gop_local_coordinates(coords, local_time))
        actual = model(coords, torch.tensor([2]), local_time)

        self.assertTrue(torch.equal(actual, expected))

    def test_factors_preserve_each_grid_axis(self):
        model = self.make_model(grid_rank=4)

        for grid, level in zip(
            (item.grid for item in model.shared_model.grid_encoder.levels),
            model.grid_lora_levels(2),
        ):
            size_x, size_y, size_t, channels = grid.shape
            self.assertEqual(level.spatial.shape, (4, size_x, size_y))
            self.assertEqual(level.temporal.shape, (4, size_t))
            self.assertEqual(level.channel.shape, (4, channels))
            self.assertEqual(
                sum(parameter.numel() for parameter in level.lora_parameters()),
                4 * (size_x * size_y + size_t + channels),
            )

    def test_structured_rank_four_is_smaller_than_full_residual(self):
        structured = self.make_model(grid_rank=4)
        full = GOPGridResidualHybridGridNet(
            make_anchor(), num_gops=3, adapted_gops=(2,),
            rank=2, alpha=2,
        )

        structured_count = sum(
            parameter.numel() for parameter in structured.grid_parameters(2)
        )
        full_count = sum(
            parameter.numel() for parameter in full.grid_parameters(2)
        )
        self.assertLess(structured_count, full_count)

    def test_all_factors_receive_gradient_after_an_update(self):
        model = self.make_model(grid_rank=4)
        parameters = freeze_for_gop(model, 2)
        optimizer = torch.optim.Adam(parameters, lr=1e-2)
        coords = torch.rand(1, 3, 3, 4)
        gop_index = torch.tensor([2])
        local_time = torch.tensor([0.5])

        model(coords, gop_index, local_time).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(coords, gop_index, local_time).mean().backward()

        for level in model.grid_lora_levels(2):
            for parameter in level.lora_parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.any(parameter.grad != 0))


if __name__ == '__main__':
    unittest.main()
