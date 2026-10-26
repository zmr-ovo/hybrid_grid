"""No PE branch: Grid -> Gate -> time modulation -> calibration -> Decoder.

Coordinates remain necessary for Grid interpolation, and raw t still drives
temporal modulation. No Frequency encoder, PE Gate, dummy padding or unused
parameters are registered. Wider internal Decoder MLPs nearly match the
parameter budget of the PE-enabled calibrated model.
"""

import torch
from torch import nn

from encoding import MultiResGrid
from scalable_ablation.internal_mlp.model import FixedFeatureDecoder
from scalable_ablation.unified_internal.model import PrefixGate, PrefixTemporalModulation
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.model import PrefixOutput
from scalable_unified_811_calibrated.model import PrefixFeatureCalibration


class GridOnlyCalibrated811(nn.Module):
    architecture = "hybrid_grid_unified_811_no_pe_calibrated_v1"

    def __init__(self, grid_levels=10, grid_feat_dim=6, base_resolution=20,
                 finest_resolution=100, aspect_ratio=(16, 9), time_scale=0.1,
                 hidden_dim=300, feature_dim=128, num_blocks=3,
                 gate_hidden_dim=48, time_hidden_dim=64,
                 width_ratios=(0.5, 0.75, 1.0), hierarchy=None):
        super().__init__()
        if grid_levels != 10:
            raise ValueError("this experiment requires ten Grid levels")
        self.hierarchy = hierarchy or GridHierarchy((tuple(range(8)), (8,), (9,)))
        if not isinstance(self.hierarchy, GridHierarchy):
            raise TypeError("hierarchy must be a GridHierarchy")
        if tuple(len(group) for group in self.hierarchy.groups) != (8, 1, 1):
            raise ValueError("this experiment requires an 8/1/1 hierarchy")
        self.grid_encoder = MultiResGrid(
            n_levels=grid_levels, n_features_per_level=grid_feat_dim,
            base_resolution=base_resolution, finest_resolution=finest_resolution,
            aspect_ratio=aspect_ratio, time_scale=time_scale,
        )
        dimension = self.grid_encoder.output_dim
        self.gate_grid = PrefixGate(dimension, dimension, gate_hidden_dim, width_ratios)
        self.time_mod = PrefixTemporalModulation(dimension, time_hidden_dim, width_ratios)
        self.decoder = FixedFeatureDecoder(
            dimension, hidden_dim, feature_dim, num_blocks, width_ratios,
        )
        self.calibration = PrefixFeatureCalibration(dimension)

    def _grid_prefix(self, coords, quality_level, grids):
        levels = self.grid_encoder.levels
        if grids is None:
            grids = (None,) * len(levels)
        if not isinstance(grids, (tuple, list)) or len(grids) != len(levels):
            raise ValueError("grids must contain one tensor per Grid level")
        active = self.hierarchy.active_levels(quality_level)
        features = [
            level(coords, grid=grids[index]) if index in active else
            coords.new_zeros(*coords.shape[:-1], level.n_features)
            for index, level in enumerate(levels)
        ]
        return torch.cat(features, dim=-1), active

    def forward_prefix(self, coords, quality_level, grids=None):
        if coords.ndim != 4 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (B, 3, H, W)")
        self.hierarchy.validate_quality_level(quality_level)
        batch, _, height, width = coords.shape
        grid_features, active = self._grid_prefix(
            coords.permute(0, 2, 3, 1), quality_level, grids,
        )
        gated = self.gate_grid(grid_features, quality_level) * grid_features
        modulated = self.time_mod(
            gated.permute(0, 3, 1, 2), coords, quality_level,
        ).permute(0, 2, 3, 1)
        calibrated = self.calibration(modulated, quality_level)
        reconstruction = self.decoder(
            calibrated.reshape(batch * height * width, -1), quality_level,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(
            reconstruction=reconstruction, active_levels=active,
            quality_level=quality_level,
            hidden_width=self.decoder.hidden_widths[quality_level],
        )

    def forward(self, coords, quality_level=2, grids=None):
        return self.forward_prefix(coords, quality_level, grids)

    def _uncalibrated_budget(self):
        grid_counts = tuple(sum(p.numel() for p in level.parameters())
                            for level in self.grid_encoder.levels)
        return tuple(
            sum(grid_counts[index] for index in self.hierarchy.active_levels(level))
            + self.gate_grid.active_parameter_count(level)
            + self.time_mod.active_parameter_count(level)
            + self.decoder.active_parameter_count(level)
            for level in range(3)
        )

    def parameter_budget(self):
        """Nested storage includes calibration parameters of lower tiers."""
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + per_tier * min(level + 1, 2)
                     for level, count in enumerate(self._uncalibrated_budget()))

    def active_parameter_budget(self):
        """Full bypasses calibration; Mid does not use Base calibration."""
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + (per_tier if level < 2 else 0)
                     for level, count in enumerate(self._uncalibrated_budget()))
