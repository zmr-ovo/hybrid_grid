"""Keep the original full Gate/time modules and Additive-Skip prefixes.

Only Grid features are used: a single full-width Linear Gate, the original
TemporalModulation, Base/Mid affine calibration, then the original sliced
Decoder. No PE, Unified Gate, internal residual MLP or width compensation.
"""

import torch
from torch import nn

from encoding import MultiResGrid
from model import TemporalModulation
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.model import PrefixAdditiveSkipDecoder, PrefixOutput
from scalable_unified_811_calibrated.model import PrefixFeatureCalibration


class Prefix811NoPECalibrated(nn.Module):
    architecture = "hybrid_grid_prefix_811_no_pe_calibrated_v1"
    training_strategy = "prefix_811_no_pe_calibrated_ema_v1"
    checkpoint_prefix = "prefix_no_pe_calibrated"

    def __init__(self, grid_levels=10, grid_feat_dim=6, base_resolution=20,
                 finest_resolution=100, aspect_ratio=(16, 9), time_scale=0.1,
                 hidden_dim=256, time_hidden_dim=64,
                 width_ratios=(0.5, 0.75, 1.0), hierarchy=None):
        super().__init__()
        if grid_levels != 10:
            raise ValueError("this experiment requires ten Grid levels")
        self.hierarchy = hierarchy or GridHierarchy((tuple(range(8)), (8,), (9,)))
        if not isinstance(self.hierarchy, GridHierarchy):
            raise TypeError("hierarchy must be a GridHierarchy")
        if tuple(len(group) for group in self.hierarchy.groups) != (8, 1, 1):
            raise ValueError("this experiment requires 8/1/1 grouping")
        self.grid_encoder = MultiResGrid(
            n_levels=grid_levels, n_features_per_level=grid_feat_dim,
            base_resolution=base_resolution, finest_resolution=finest_resolution,
            aspect_ratio=aspect_ratio, time_scale=time_scale,
        )
        dimension = self.grid_encoder.output_dim
        self.gate_grid = nn.Sequential(nn.Linear(dimension, dimension), nn.Sigmoid())
        self.time_mod = TemporalModulation(dimension, time_hidden_dim)
        self.decoder = PrefixAdditiveSkipDecoder(dimension, hidden_dim, width_ratios)
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
        features, active = self._grid_prefix(coords.permute(0, 2, 3, 1), quality_level, grids)
        gated = self.gate_grid(features) * features
        modulated = self.time_mod(gated.permute(0, 3, 1, 2), coords).permute(0, 2, 3, 1)
        calibrated = self.calibration(modulated, quality_level)
        reconstruction = self.decoder(
            calibrated.reshape(batch * height * width, -1), quality_level,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(reconstruction=reconstruction, active_levels=active,
                            quality_level=quality_level,
                            hidden_width=self.decoder.hidden_widths[quality_level])

    def forward(self, coords, quality_level=2, grids=None):
        return self.forward_prefix(coords, quality_level, grids)

    def _uncalibrated_budget(self):
        grid_counts = tuple(sum(p.numel() for p in level.parameters())
                            for level in self.grid_encoder.levels)
        common = sum(p.numel() for module in (self.gate_grid, self.time_mod)
                     for p in module.parameters())
        return tuple(common + self.decoder.active_parameter_count(level)
                     + sum(grid_counts[index] for index in self.hierarchy.active_levels(level))
                     for level in range(3))

    def parameter_budget(self):
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + per_tier * min(level + 1, 2)
                     for level, count in enumerate(self._uncalibrated_budget()))

    def active_parameter_budget(self):
        per_tier = 2 * self.decoder.input_dim
        return tuple(count + (per_tier if level < 2 else 0)
                     for level, count in enumerate(self._uncalibrated_budget()))
