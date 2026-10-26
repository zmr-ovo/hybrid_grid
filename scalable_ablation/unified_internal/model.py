"""Joint Grid/internal-width prefixes with fixed module interfaces.

Gate: D -> h_gate(k) -> fixed feature-group dimension, followed by Sigmoid.
Time: 1 -> h_time(k) -> 2D, producing gamma/beta as in the reference model.
Decoder: fixed-feature residual MLPs inherited from internal_mlp.
Only hidden neurons are sliced. Lower-prefix neurons remain shared.
"""

import torch
import torch.nn.functional as F
from torch import nn

from scalable_ablation.internal_mlp.model import (
    FixedFeatureDecoder, InternalMLPScalableResNet,
)
from scalable_prefix_resnet.model import PrefixOutput


def prefix_widths(hidden_dim, ratios):
    ratios = tuple(float(value) for value in ratios)
    if hidden_dim < 1 or len(ratios) != 3:
        raise ValueError("positive hidden_dim and three width ratios required")
    if not 0 < ratios[0] < ratios[1] < ratios[2] == 1:
        raise ValueError("width ratios must strictly increase and end at 1")
    widths = tuple(max(1, round(hidden_dim * value)) for value in ratios)
    if len(set(widths)) != 3:
        raise ValueError("hidden dimension must yield three distinct widths")
    return widths


class PrefixGate(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=48,
                 width_ratios=(0.5, 0.75, 1.0)):
        super().__init__()
        self.hidden_widths = prefix_widths(hidden_dim, width_ratios)
        self.expand = nn.Linear(input_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, features, quality_level):
        FixedFeatureDecoder._validate_level(quality_level)
        width = self.hidden_widths[quality_level]
        hidden = F.gelu(F.linear(
            features, self.expand.weight[:width], self.expand.bias[:width],
        ))
        return torch.sigmoid(F.linear(
            hidden, self.project.weight[:, :width], self.project.bias,
        ))

    def active_parameter_count(self, quality_level):
        FixedFeatureDecoder._validate_level(quality_level)
        width = self.hidden_widths[quality_level]
        return width * (self.expand.in_features + 1 + self.project.out_features) + self.project.out_features


class PrefixTemporalModulation(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,
                 width_ratios=(0.5, 0.75, 1.0)):
        super().__init__()
        self.hidden_widths = prefix_widths(hidden_dim, width_ratios)
        # Names and Full computation match TemporalModulation.
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim * 2),
        )
        self.norm = nn.GroupNorm(input_dim, input_dim, affine=False)

    def forward(self, features, coords, quality_level):
        FixedFeatureDecoder._validate_level(quality_level)
        width = self.hidden_widths[quality_level]
        time = coords[:, 2, 0, 0].unsqueeze(1)
        hidden = F.relu(F.linear(
            time, self.mlp[0].weight[:width], self.mlp[0].bias[:width],
        ))
        gamma_beta = F.linear(
            hidden, self.mlp[2].weight[:, :width], self.mlp[2].bias,
        )
        gamma, beta = gamma_beta.chunk(2, dim=1)
        batch, channels = features.shape[:2]
        return (self.norm(features) * gamma.view(batch, channels, 1, 1)
                + beta.view(batch, channels, 1, 1))

    def active_parameter_count(self, quality_level):
        FixedFeatureDecoder._validate_level(quality_level)
        width = self.hidden_widths[quality_level]
        return width * (2 + self.mlp[2].out_features) + self.mlp[2].out_features


class UnifiedInternalScalableResNet(InternalMLPScalableResNet):
    architecture = "hybrid_grid_ablation_unified_internal_width_v1"

    def __init__(self, gate_hidden_dim=48, time_hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        dimension = self.decoder.input_dim
        ratios = self.decoder.width_ratios
        self.gate_grid = PrefixGate(
            dimension, self.grid_encoder.output_dim, gate_hidden_dim, ratios,
        )
        self.gate_pe = PrefixGate(
            dimension, self.pe_encoder.output_dim, gate_hidden_dim, ratios,
        )
        reference_time = self.time_mod
        self.time_mod = PrefixTemporalModulation(dimension, time_hidden_dim, ratios)
        if time_hidden_dim == 64:
            # Keep reference Full temporal initialization for fair comparison.
            self.time_mod.load_state_dict(reference_time.state_dict())

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
        reconstruction = self.decoder(
            modulated.reshape(batch * height * width, -1), quality_level,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(
            reconstruction=reconstruction, active_levels=active_levels,
            quality_level=quality_level,
            hidden_width=self.decoder.hidden_widths[quality_level],
        )

    def parameter_budget(self):
        grid_counts = tuple(self._count(level) for level in self.grid_encoder.levels)
        shared = self._count(self.pe_encoder)
        return tuple(
            shared
            + sum(grid_counts[index] for index in self.hierarchy.active_levels(level))
            + self.gate_grid.active_parameter_count(level)
            + self.gate_pe.active_parameter_count(level)
            + self.time_mod.active_parameter_count(level)
            + self.decoder.active_parameter_count(level)
            for level in range(3)
        )
