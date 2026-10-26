"""Change only Decoder topology relative to the Unified family.

One shared Additive-Skip Decoder runs D -> w -> w -> w -> 128 -> RGB
at the default full width 256. Only the third hidden layer has input Skip.
Gate and temporal modulation retain Unified's internal-channel prefixes.
No InternalPrefixMLP residual blocks are registered.
"""

import torch

from scalable_ablation.unified_internal.model import PrefixGate, PrefixTemporalModulation
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.model import PrefixOutput, PrefixScalableResNet


class UnifiedAdditive811(PrefixScalableResNet):
    architecture = "hybrid_grid_unified_additive_811_v1"
    training_strategy = "unified_additive_811_ema_v1"
    checkpoint_prefix = "additive"

    def __init__(self, gate_hidden_dim=48, time_hidden_dim=64, **kwargs):
        kwargs.setdefault("grid_levels", 10)
        kwargs.setdefault("grid_feat_dim", 6)
        kwargs.setdefault("base_resolution", 20)
        kwargs.setdefault("finest_resolution", 100)
        kwargs.setdefault("time_scale", 0.1)
        kwargs.setdefault("pe_freq", 14)
        kwargs.setdefault("hidden_dim", 256)
        kwargs.setdefault("hierarchy", GridHierarchy((tuple(range(8)), (8,), (9,))))
        super().__init__(**kwargs)
        if tuple(len(group) for group in self.hierarchy.groups) != (8, 1, 1):
            raise ValueError("this experiment requires an 8/1/1 hierarchy")
        dimension = self.decoder.input_dim
        ratios = self.decoder.width_ratios
        self.gate_grid = PrefixGate(dimension, self.grid_encoder.output_dim, gate_hidden_dim, ratios)
        self.gate_pe = PrefixGate(dimension, self.pe_encoder.output_dim, gate_hidden_dim, ratios)
        reference_time = self.time_mod
        self.time_mod = PrefixTemporalModulation(dimension, time_hidden_dim, ratios)
        if time_hidden_dim == 64:
            self.time_mod.load_state_dict(reference_time.state_dict())

    def _calibrate(self, features, quality_level):
        return features

    def forward_prefix(self, coords, quality_level, grids=None):
        if coords.ndim != 4 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (B, 3, H, W)")
        self.hierarchy.validate_quality_level(quality_level)
        grids = self._validate_grids(grids)
        batch, _, height, width = coords.shape
        coords_hw = coords.permute(0, 2, 3, 1)
        grid_feature, active = self._grid_prefix(coords_hw, quality_level, grids)
        pe_feature = self.pe_encoder(coords_hw)
        features = torch.cat((grid_feature, pe_feature), dim=-1)
        fused = torch.cat((
            self.gate_grid(features, quality_level) * grid_feature,
            self.gate_pe(features, quality_level) * pe_feature,
        ), dim=-1)
        modulated = self.time_mod(
            fused.permute(0, 3, 1, 2), coords, quality_level,
        ).permute(0, 2, 3, 1)
        decoder_input = self._calibrate(modulated, quality_level)
        reconstruction = self.decoder(
            decoder_input.reshape(batch * height * width, -1), quality_level,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(reconstruction=reconstruction, active_levels=active,
                            quality_level=quality_level,
                            hidden_width=self.decoder.hidden_widths[quality_level])

    def parameter_budget(self):
        grid_counts = tuple(self._count(level) for level in self.grid_encoder.levels)
        return tuple(
            self._count(self.pe_encoder)
            + sum(grid_counts[index] for index in self.hierarchy.active_levels(level))
            + self.gate_grid.active_parameter_count(level)
            + self.gate_pe.active_parameter_count(level)
            + self.time_mod.active_parameter_count(level)
            + self.decoder.active_parameter_count(level)
            for level in range(3)
        )

    def active_parameter_budget(self):
        return self.parameter_budget()
