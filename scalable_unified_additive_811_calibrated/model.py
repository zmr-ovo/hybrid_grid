"""Base/Mid affine calibration after time modulation; Full is unchanged."""

from scalable_unified_additive_811.model import UnifiedAdditive811
from scalable_unified_811_calibrated.model import PrefixFeatureCalibration


class CalibratedUnifiedAdditive811(UnifiedAdditive811):
    architecture = "hybrid_grid_unified_additive_811_calibrated_v1"
    training_strategy = "unified_additive_811_calibrated_ema_v1"
    checkpoint_prefix = "additive_calibrated"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration = PrefixFeatureCalibration(self.decoder.input_dim)

    def _calibrate(self, features, quality_level):
        return self.calibration(features, quality_level)

    def parameter_budget(self):
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + per_tier * min(level + 1, 2)
                     for level, count in enumerate(super().parameter_budget()))

    def active_parameter_budget(self):
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + (per_tier if level < 2 else 0)
                     for level, count in enumerate(super().parameter_budget()))
