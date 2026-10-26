"""Vary only the active Grid prefix; keep all network layers at full width.

Initialization and the Full computation match PrefixScalableResNet exactly.
Gate and temporal modulation keep the shared design of the EMA reference.
"""

import torch

from scalable_prefix_resnet.model import PrefixOutput, PrefixScalableResNet


class GridOnlyScalableResNet(PrefixScalableResNet):
    architecture = "hybrid_grid_ablation_grid_only_additive_skip_v1"

    def forward_prefix(self, coords, quality_level, grids=None):
        if coords.ndim != 4 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (B, 3, H, W)")
        self.hierarchy.validate_quality_level(quality_level)
        grids = self._validate_grids(grids)
        batch, _, height, width = coords.shape
        coords_hw = coords.permute(0, 2, 3, 1)
        grid_feature, active_levels = self._grid_prefix(
            coords_hw, quality_level, grids,
        )
        pe_feature = self.pe_encoder(coords_hw)
        concatenated = torch.cat((grid_feature, pe_feature), dim=-1)
        fused = torch.cat((
            self.gate_grid(concatenated) * grid_feature,
            self.gate_pe(concatenated) * pe_feature,
        ), dim=-1)
        modulated = self.time_mod(
            fused.permute(0, 3, 1, 2), coords,
        ).permute(0, 2, 3, 1)
        reconstruction = self.decoder(
            modulated.reshape(batch * height * width, -1), quality_level=2,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(
            reconstruction=reconstruction,
            active_levels=active_levels,
            quality_level=quality_level,
            hidden_width=self.decoder.hidden_dim,
        )

    def parameter_budget(self):
        network_count = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if not name.startswith("grid_encoder.")
        )
        grid_counts = tuple(self._count(level) for level in self.grid_encoder.levels)
        return tuple(
            network_count + sum(grid_counts[index] for index in
                                self.hierarchy.active_levels(quality_level))
            for quality_level in range(3)
        )
