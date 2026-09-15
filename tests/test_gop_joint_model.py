import copy
import unittest

import torch

from gop_lora import (
    all_gop_parameter_groups,
    assemble_all_gop_model,
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


class AllGOPModelAssemblyTest(unittest.TestCase):
    def make_model(self, num_gops=5):
        return assemble_all_gop_model(
            make_anchor(),
            num_gops=num_gops,
            rank=4,
            alpha=4,
            grid_rank=2,
            grid_alpha=2,
        )

    def test_assembles_every_later_gop_once(self):
        anchor = make_anchor()
        model = assemble_all_gop_model(
            anchor, num_gops=5, rank=4, alpha=4,
            grid_rank=2, grid_alpha=2,
        )
        groups = all_gop_parameter_groups(model)

        self.assertIs(model.shared_model, anchor)
        self.assertEqual(model.num_gops, 5)
        self.assertEqual(model.num_adapters, 4)
        self.assertEqual(model.lora_target, 'all_linear')
        self.assertEqual(model.grid_residuals.adapted_gops, (1, 2, 3, 4))
        self.assertEqual(
            tuple(group.gop_index for group in groups.adapters),
            (1, 2, 3, 4),
        )
        self.assertTrue(all(group.network for group in groups.adapters))
        self.assertTrue(all(group.grid for group in groups.adapters))

    def test_parameter_change_is_routed_to_one_gop_only(self):
        model = self.make_model(num_gops=4)
        groups = all_gop_parameter_groups(model)
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.4])
        before = tuple(
            model(coords, gop_index, local_time)
            for gop_index in range(1, 4)
        )

        selected = groups.for_gop(2)
        with torch.no_grad():
            selected.network[-2].fill_(1.0)
            selected.network[-1].fill_(0.25)

        after = tuple(
            model(coords, gop_index, local_time)
            for gop_index in range(1, 4)
        )
        self.assertTrue(torch.equal(after[0], before[0]))
        self.assertFalse(torch.equal(after[1], before[1]))
        self.assertTrue(torch.equal(after[2], before[2]))

    def test_zero_initialized_adapters_preserve_shared_output(self):
        anchor = make_anchor()
        reference = copy.deepcopy(anchor)
        model = assemble_all_gop_model(
            anchor,
            num_gops=4,
            rank=4,
            alpha=4,
            grid_rank=2,
            grid_alpha=2,
        )
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.4])
        expected = reference(gop_local_coordinates(coords, local_time))

        for gop_index in range(4):
            actual = model(coords, gop_index, local_time)
            self.assertTrue(torch.equal(actual, expected))

    def test_parameter_groups_are_disjoint_and_complete(self):
        model = self.make_model(num_gops=4)
        groups = all_gop_parameter_groups(model)
        grouped_ids = [id(parameter) for parameter in groups.parameters]
        model_ids = [id(parameter) for parameter in model.parameters()]

        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(model_ids))
        self.assertEqual(
            groups.total_count,
            sum(parameter.numel() for parameter in model.parameters()),
        )
        self.assertEqual(
            groups.adaptation_count,
            sum(group.total_count for group in groups.adapters),
        )

    def test_each_gop_owns_independent_adapter_parameters(self):
        groups = all_gop_parameter_groups(self.make_model(num_gops=4))

        for left in groups.adapters:
            left_ids = {id(parameter) for parameter in left.parameters}
            for right in groups.adapters:
                if left.gop_index == right.gop_index:
                    continue
                right_ids = {id(parameter) for parameter in right.parameters}
                self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_gop_zero_has_no_independent_parameter_group(self):
        groups = all_gop_parameter_groups(self.make_model())

        with self.assertRaisesRegex(ValueError, 'no independent adapter'):
            groups.for_gop(0)

    def test_rejects_partial_grid_assembly(self):
        from gop_lora import GOPStructuredGridHybridGridNet

        model = GOPStructuredGridHybridGridNet(
            make_anchor(), num_gops=4, adapted_gops=(2,),
            rank=4, alpha=4, grid_rank=2, grid_alpha=2,
            lora_target='all_linear',
        )

        with self.assertRaisesRegex(ValueError, 'every later GOP'):
            all_gop_parameter_groups(model)

    def test_requires_at_least_two_gops(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            assemble_all_gop_model(
                make_anchor(), num_gops=1, rank=4, alpha=4,
                grid_rank=2, grid_alpha=2,
            )


if __name__ == '__main__':
    unittest.main()
