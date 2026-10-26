import unittest

import torch
import torch.nn.functional as F

from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer,
    accumulate_ema_prefix_gradients,
    validate_target_shares,
)
from scalable_prefix_resnet.model import PrefixScalableResNet


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


class EMADistortionBalancerTest(unittest.TestCase):
    def test_weights_make_expected_loss_shares(self):
        balancer = EMADistortionBalancer((0.2, 0.3, 0.5), decay=0.99)
        distortions = (0.0048, 0.0026, 0.00029)
        balancer.update(distortions)
        contributions = [
            weight * distortion
            for weight, distortion in zip(balancer.weights(), distortions)
        ]
        normalized = [value / sum(contributions) for value in contributions]

        for actual, expected in zip(normalized, (0.2, 0.3, 0.5)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_state_round_trip(self):
        source = EMADistortionBalancer((0.2, 0.3, 0.5), decay=0.95)
        source.update((1.0, 2.0, 3.0))
        restored = EMADistortionBalancer((0.2, 0.3, 0.5), decay=0.95)
        restored.load_state_dict(source.state_dict())

        self.assertEqual(source.ema, restored.ema)
        self.assertEqual(source.weights(), restored.weights())

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            validate_target_shares((1.0, 1.0))
        with self.assertRaises(ValueError):
            EMADistortionBalancer(decay=1.0)


class EMASequentialBackwardTest(unittest.TestCase):
    def test_all_prefix_levels_receive_gradients(self):
        model = make_model()
        balancer = EMADistortionBalancer()
        coords = torch.rand(1, 3, 12, 12)
        target = torch.rand(1, 3, 12, 12)
        model.zero_grad(set_to_none=True)

        metrics, objective = accumulate_ema_prefix_gradients(
            model,
            coords,
            target,
            F.mse_loss,
            balancer,
            device=torch.device("cpu"),
            metric_fn=lambda *_: 1.0,
        )

        self.assertEqual(len(metrics), 3)
        self.assertGreater(objective, 0.0)
        self.assertAlmostEqual(sum(item.weight for item in metrics), 1.0)
        self.assertTrue(all(
            level.grid.grad is not None
            for level in model.grid_encoder.levels
        ))


if __name__ == "__main__":
    unittest.main()
