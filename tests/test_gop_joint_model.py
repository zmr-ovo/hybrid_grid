import copy
import unittest

import torch

from gop_lora import (
    HierarchicalGOPHybridGridNet,
    all_gop_parameter_groups,
    assemble_all_gop_model,
    configure_common_training,
    configure_local_training,
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
            common_rank=6,
            common_alpha=6,
            common_grid_rank=3,
            common_grid_alpha=3,
        )

    def test_assembles_the_complete_hierarchy(self):
        anchor = make_anchor()
        model = assemble_all_gop_model(
            anchor, num_gops=5, rank=4, alpha=4,
            grid_rank=2, grid_alpha=2, common_rank=6,
            common_alpha=6, common_grid_rank=3,
            common_grid_alpha=3,
        )
        groups = all_gop_parameter_groups(model)

        self.assertIsInstance(model, HierarchicalGOPHybridGridNet)
        self.assertIs(model.shared_model, anchor)
        self.assertEqual(model.num_gops, 5)
        self.assertEqual(model.num_adapters, 4)
        self.assertEqual(model.lora_target, 'all_linear')
        self.assertEqual(model.grid_residuals.adapted_gops, (1, 2, 3, 4))
        self.assertEqual(
            tuple(group.gop_index for group in groups.adapters),
            (1, 2, 3, 4),
        )
        self.assertTrue(groups.common_network)
        self.assertTrue(groups.common_grid)
        self.assertTrue(all(group.network for group in groups.adapters))
        self.assertTrue(all(group.grid for group in groups.adapters))

    def test_zero_initialized_adapters_preserve_shared_output(self):
        anchor = make_anchor()
        reference = copy.deepcopy(anchor)
        model = assemble_all_gop_model(
            anchor, num_gops=4, rank=4, alpha=4,
            grid_rank=2, grid_alpha=2, common_rank=6,
            common_alpha=6, common_grid_rank=3,
            common_grid_alpha=3,
        )
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.4])
        expected = reference(gop_local_coordinates(coords, local_time))

        for gop_index in range(4):
            actual = model(coords, gop_index, local_time)
            self.assertTrue(torch.equal(actual, expected))

    def test_common_lora_affects_every_later_gop_but_not_gop_zero(self):
        model = self.make_model(num_gops=4)
        groups = all_gop_parameter_groups(model)
        coords = torch.rand(1, 3, 3, 4)
        local_time = torch.tensor([0.4])
        before = tuple(
            model(coords, gop_index, local_time)
            for gop_index in range(4)
        )

        with torch.no_grad():
            groups.common_network[-2].fill_(1.0)
            groups.common_network[-1].fill_(0.25)

        after = tuple(
            model(coords, gop_index, local_time)
            for gop_index in range(4)
        )
        self.assertTrue(torch.equal(after[0], before[0]))
        for gop_index in range(1, 4):
            self.assertFalse(torch.equal(after[gop_index], before[gop_index]))

    def test_local_lora_affects_only_its_own_gop(self):
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

    def test_common_grid_lora_is_used_only_by_later_gops(self):
        model = self.make_model(num_gops=4)
        grids = tuple(
            level.grid for level in model.shared_model.grid_encoder.levels
        )
        with torch.no_grad():
            level = model.common_grid_levels()[0]
            level.spatial.fill_(0.5)
            level.temporal.fill_(1.0)
            level.channel.fill_(1.0)

        gop_zero = model.grid_residuals(grids, 0)
        gop_one = model.grid_residuals(grids, 1)
        gop_three = model.grid_residuals(grids, 3)

        self.assertTrue(all(
            torch.equal(actual, expected)
            for actual, expected in zip(gop_zero, grids)
        ))
        self.assertFalse(torch.equal(gop_one[0], grids[0]))
        self.assertTrue(torch.equal(gop_one[0], gop_three[0]))

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
            groups.common_count
            + sum(group.total_count for group in groups.adapters),
        )

    def test_each_gop_owns_independent_local_parameters(self):
        groups = all_gop_parameter_groups(self.make_model(num_gops=4))

        for left in groups.adapters:
            left_ids = {id(parameter) for parameter in left.parameters}
            self.assertTrue(left_ids.isdisjoint(
                {id(parameter) for parameter in groups.common}
            ))
            for right in groups.adapters:
                if left.gop_index == right.gop_index:
                    continue
                right_ids = {id(parameter) for parameter in right.parameters}
                self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_gop_zero_has_no_independent_parameter_group(self):
        groups = all_gop_parameter_groups(self.make_model())

        with self.assertRaisesRegex(ValueError, 'no independent adapter'):
            groups.for_gop(0)

    def test_requires_at_least_two_gops(self):
        with self.assertRaisesRegex(ValueError, 'at least two'):
            assemble_all_gop_model(
                make_anchor(), num_gops=1, rank=4, alpha=4,
                grid_rank=2, grid_alpha=2,
            )


class AllGOPTrainingSelectionTest(unittest.TestCase):
    def make_model(self):
        return assemble_all_gop_model(
            make_anchor(), num_gops=4, rank=4, alpha=4,
            grid_rank=2, grid_alpha=2, common_rank=6,
            common_alpha=6, common_grid_rank=3,
            common_grid_alpha=3,
        )

    def parameter_snapshot(self, model):
        return {
            id(parameter): parameter.detach().clone()
            for parameter in model.parameters()
        }

    def changed_parameter_ids(self, model, snapshot):
        return {
            id(parameter)
            for parameter in model.parameters()
            if not torch.equal(parameter.detach(), snapshot[id(parameter)])
        }

    def take_training_step(self, model, selected, gop_index):
        optimizer = torch.optim.SGD(selected.parameters, lr=0.1)
        optimizer.zero_grad()
        output = model(
            torch.rand(1, 3, 3, 4),
            gop_index,
            torch.tensor([0.4]),
        )
        output.mean().backward()
        optimizer.step()

    def test_common_stage_updates_only_common_lora(self):
        model = self.make_model()
        groups = all_gop_parameter_groups(model)
        selected = configure_common_training(model)
        selected_ids = {id(parameter) for parameter in selected.parameters}
        snapshot = self.parameter_snapshot(model)

        self.assertEqual(selected_ids, {
            id(parameter) for parameter in groups.common
        })
        self.assertTrue(all(
            parameter.requires_grad == (id(parameter) in selected_ids)
            for parameter in model.parameters()
        ))

        self.take_training_step(model, selected, gop_index=1)
        changed_ids = self.changed_parameter_ids(model, snapshot)

        self.assertTrue(changed_ids)
        self.assertTrue(changed_ids.issubset(selected_ids))

    def test_local_stage_updates_only_the_selected_gop(self):
        model = self.make_model()
        groups = all_gop_parameter_groups(model)
        selected = configure_local_training(model, gop_index=2)
        selected_ids = {id(parameter) for parameter in selected.parameters}
        snapshot = self.parameter_snapshot(model)

        self.assertEqual(selected_ids, {
            id(parameter) for parameter in groups.for_gop(2).parameters
        })
        self.assertTrue(all(
            parameter.requires_grad == (id(parameter) in selected_ids)
            for parameter in model.parameters()
        ))

        self.take_training_step(model, selected, gop_index=2)
        changed_ids = self.changed_parameter_ids(model, snapshot)

        self.assertTrue(changed_ids)
        self.assertTrue(changed_ids.issubset(selected_ids))

    def test_local_stage_rejects_gop_zero(self):
        with self.assertRaisesRegex(ValueError, 'no independent adapter'):
            configure_local_training(self.make_model(), gop_index=0)


if __name__ == '__main__':
    unittest.main()
