import unittest

import torch
import torch.nn.functional as F

from resnet_baseline.model_1 import (
    ResidualSkipDecoder,
    ResidualSkipHybridGridNet,
)
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet import (
    BalancedPrefixScheduler,
    PrefixAdditiveSkipDecoder,
    PrefixScalableResNet,
)
from scalable_prefix_resnet.training import train_sample


def make_model():
    return PrefixScalableResNet(
        grid_levels=4,
        grid_feat_dim=2,
        base_resolution=4,
        finest_resolution=8,
        aspect_ratio=(1, 1),
        time_scale=1.0,
        pe_freq=3,
        hidden_dim=16,
        width_ratios=(0.5, 0.75, 1.0),
        hierarchy=GridHierarchy(((0, 1), (2,), (3,))),
    )


class PrefixDecoderTest(unittest.TestCase):
    def test_full_width_is_exact_additive_skip_decoder(self):
        original = ResidualSkipDecoder(input_dim=10, hidden_dim=16)
        prefix = PrefixAdditiveSkipDecoder(
            input_dim=10,
            hidden_dim=16,
            width_ratios=(0.5, 0.75, 1.0),
        )
        prefix.load_state_dict(original.state_dict())
        inputs = torch.rand(7, 10)

        self.assertTrue(torch.equal(original(inputs), prefix(inputs, 2)))
        self.assertEqual(prefix.hidden_widths, (8, 12, 16))
        self.assertEqual(
            prefix.active_parameter_count(2),
            sum(parameter.numel() for parameter in prefix.parameters()),
        )


class PrefixModelTest(unittest.TestCase):
    def test_full_model_loads_baseline_weights_and_matches_output(self):
        baseline = ResidualSkipHybridGridNet(
            grid_levels=4,
            grid_feat_dim=2,
            base_resolution=4,
            finest_resolution=8,
            aspect_ratio=(1, 1),
            time_scale=1.0,
            pe_freq=3,
            hidden_dim=16,
        )
        prefix = make_model()
        prefix.load_state_dict(baseline.state_dict())
        coords = torch.rand(1, 3, 4, 5)

        expected = baseline(coords)
        actual = prefix(coords, quality_level=2).reconstruction

        self.assertTrue(torch.equal(expected, actual))

    def test_base_skips_future_grids_but_keeps_fixed_feature_dimension(self):
        model = make_model()

        def fail(*_args, **_kwargs):
            raise AssertionError("future Grid must not be evaluated")

        model.grid_encoder.levels[2].forward = fail
        model.grid_encoder.levels[3].forward = fail
        output = model(torch.rand(1, 3, 3, 4), quality_level=0)

        self.assertEqual(output.active_levels, (0, 1))
        self.assertEqual(output.reconstruction.shape, (1, 3, 3, 4))

    def test_base_gradients_touch_only_grid_and_hidden_prefixes(self):
        model = make_model()
        model.zero_grad(set_to_none=True)
        output = model(torch.rand(1, 3, 3, 4), quality_level=0)
        output.reconstruction.mean().backward()

        grid_gradients = [
            level.grid.grad for level in model.grid_encoder.levels
        ]
        width = model.decoder.hidden_widths[0]
        decoder_gradient = model.decoder.linear[0].weight.grad

        self.assertTrue(all(
            gradient is not None for gradient in grid_gradients[:2]
        ))
        self.assertTrue(all(
            gradient is None for gradient in grid_gradients[2:]
        ))
        self.assertGreater(
            decoder_gradient[:width].abs().sum().item(), 0.0,
        )
        self.assertEqual(
            decoder_gradient[width:].abs().sum().item(), 0.0,
        )

    def test_full_parameter_budget_matches_total_model(self):
        model = make_model()
        budgets = model.parameter_budget()
        total = sum(parameter.numel() for parameter in model.parameters())

        self.assertLess(budgets[0], budgets[1])
        self.assertLess(budgets[1], budgets[2])
        self.assertEqual(budgets[2], total)


class SinglePrefixSamplingTest(unittest.TestCase):
    def test_scheduler_is_balanced_and_resumable(self):
        scheduler = BalancedPrefixScheduler()
        sequence = [scheduler.next() for _ in range(8)]
        restored = BalancedPrefixScheduler()
        restored.load_state_dict(scheduler.state_dict())

        self.assertEqual(sequence, [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(restored.next(), 2)

    def test_one_training_sample_touches_only_selected_prefix(self):
        model = make_model()
        coords = torch.rand(1, 3, 4, 5)
        target = torch.rand(1, 3, 4, 5)
        model.zero_grad(set_to_none=True)

        metrics = train_sample(
            model,
            coords,
            target,
            F.mse_loss,
            quality_level=1,
            device=torch.device("cpu"),
            metric_fn=lambda *_: 1.0,
        )

        self.assertEqual(metrics.quality_level, 1)
        self.assertIsNotNone(model.grid_encoder.levels[2].grid.grad)
        self.assertIsNone(model.grid_encoder.levels[3].grid.grad)


if __name__ == "__main__":
    unittest.main()
