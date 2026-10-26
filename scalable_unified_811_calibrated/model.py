"""Add identity-initialized channel calibration immediately before Decoder.

Base and Mid each own one scale/bias vector; Full bypasses calibration.
All original Grid, PE, Gate, temporal modulation and Decoder parameters are
constructed unchanged before allocating the new deterministic parameters.
"""

import torch
from torch import nn

from scalable_ablation.unified_internal.model import UnifiedInternalScalableResNet
from scalable_prefix_resnet.model import PrefixOutput


class PrefixFeatureCalibration(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.scale = nn.Parameter(torch.ones(2, dimension))
        self.bias = nn.Parameter(torch.zeros(2, dimension))

    def forward(self, features, quality_level):
        if isinstance(quality_level, bool) or not isinstance(quality_level, int):
            raise TypeError("quality_level must be an integer")
        if not 0 <= quality_level <= 2:
            raise ValueError("quality_level must be Base=0, Mid=1 or Full=2")
        if features.shape[-1] != self.scale.shape[1]:
            raise ValueError("feature channel dimension does not match")
        if quality_level == 2:
            return features
        return features * self.scale[quality_level] + self.bias[quality_level]


class CalibratedUnified811(UnifiedInternalScalableResNet):
    architecture = "hybrid_grid_unified_811_calibrated_v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if tuple(len(group) for group in self.hierarchy.groups) != (8, 1, 1):
            raise ValueError("calibrated experiment requires an explicit 8/1/1 hierarchy")
        self.calibration = PrefixFeatureCalibration(self.decoder.input_dim)

    def forward_prefix(self, coords, quality_level, grids=None):
        if coords.ndim != 4 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (B, 3, H, W)")
        self.hierarchy.validate_quality_level(quality_level)
        grids = self._validate_grids(grids)
        batch, _, height, width = coords.shape
        coords_hw = coords.permute(0, 2, 3, 1)
        grid_feature, active_levels = self._grid_prefix(coords_hw, quality_level, grids)
        pe_feature = self.pe_encoder(coords_hw)
        concatenated = torch.cat((grid_feature, pe_feature), dim=-1)
        fused = torch.cat((
            self.gate_grid(concatenated, quality_level) * grid_feature,
            self.gate_pe(concatenated, quality_level) * pe_feature,
        ), dim=-1)
        modulated = self.time_mod(
            fused.permute(0, 3, 1, 2), coords, quality_level,
        ).permute(0, 2, 3, 1)
        calibrated = self.calibration(modulated, quality_level)
        reconstruction = self.decoder(
            calibrated.reshape(batch * height * width, -1), quality_level,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(
            reconstruction=reconstruction, active_levels=active_levels,
            quality_level=quality_level,
            hidden_width=self.decoder.hidden_widths[quality_level],
        )

    def parameter_budget(self):
        """Cumulative nested storage, including lower-tier calibration vectors."""
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + per_tier * min(level + 1, 2)
                     for level, count in enumerate(super().parameter_budget()))

    def active_parameter_budget(self):
        """Scalars actually used by each forward path (Full has no calibration)."""
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + (per_tier if level < 2 else 0)
                     for level, count in enumerate(super().parameter_budget()))
