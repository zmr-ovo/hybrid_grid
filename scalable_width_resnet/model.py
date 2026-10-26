"""Width-progressive Hybrid Grid based on the model_1 Additive-Skip decoder.

Grid levels use the same default 7/2/1 hierarchy as the module-wise model.
The fused Grid/PE feature keeps the Full model's fixed external dimension;
missing enhancement Grid channels are deterministic zeros.  Inside the
decoder, Base, Mid and Full use strict prefixes of every hidden layer:

    Base widths < Mid widths < Full widths.

The Full path is exactly the model_1 decoder shape.  Lower prefixes therefore
reuse a strict subset of Full's weights instead of owning separate networks.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from encoding import Frequency, MultiResGrid
from scalable_generation.hierarchy import GridHierarchy, default_hierarchy


class IdentityInitializedGate(nn.Module):
    """Channel gate initialized to one through ``2 * sigmoid(logit)``."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feature):
        return 2.0 * torch.sigmoid(self.linear(feature))


class WidthAdditiveSkipDecoder(nn.Module):
    """Additive-Skip MLP whose hidden channels form nested prefixes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        width_ratios=(0.5, 0.75, 1.0),
        input_widths=None,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim < 2:
            raise ValueError('input_dim must be positive and hidden_dim >= 2')
        ratios = tuple(float(value) for value in width_ratios)
        if len(ratios) != 3:
            raise ValueError('width_ratios must contain Base, Mid and Full')
        if not 0 < ratios[0] < ratios[1] < ratios[2]:
            raise ValueError('width_ratios must be strictly increasing')
        if ratios[2] != 1.0:
            raise ValueError('Full width ratio must be 1.0')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim // 2
        self.width_ratios = ratios
        self.hidden_widths = tuple(
            max(1, round(hidden_dim * ratio)) for ratio in ratios
        )
        self.output_widths = tuple(
            max(1, round(self.output_dim * ratio)) for ratio in ratios
        )
        if input_widths is None:
            input_widths = tuple(
                max(1, round(input_dim * ratio)) for ratio in ratios
            )
        self.input_widths = tuple(int(value) for value in input_widths)
        if len(self.input_widths) != 3:
            raise ValueError('input_widths must contain Base, Mid and Full')
        if not (
            0 < self.input_widths[0]
            < self.input_widths[1]
            < self.input_widths[2] == input_dim
        ):
            raise ValueError(
                'input_widths must be strict prefixes ending at input_dim'
            )
        if len(set(self.hidden_widths)) != 3:
            raise ValueError('hidden width prefixes must be distinct')
        if len(set(self.output_widths)) != 3:
            raise ValueError('output width prefixes must be distinct')

        # These are the exact model_1 Full tensors. Prefix decoding uses
        # functional linear calls with leading row/column slices.
        self.linear = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        ])
        self.skip_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(self.output_dim, 3)
        self.act = nn.GELU()

    @staticmethod
    def _linear(x, layer, out_width, in_width=None):
        weight = layer.weight[:out_width]
        if in_width is not None:
            weight = weight[:, :in_width]
        bias = None if layer.bias is None else layer.bias[:out_width]
        return F.linear(x, weight, bias)

    def forward(self, stage_input, quality_level=2):
        if not isinstance(quality_level, int) or isinstance(quality_level, bool):
            raise TypeError('quality_level must be an integer')
        if not 0 <= quality_level < 3:
            raise ValueError('quality_level must be between 0 and 2')
        hidden_width = self.hidden_widths[quality_level]
        output_width = self.output_widths[quality_level]
        input_width = self.input_widths[quality_level]
        active_input = stage_input[:, :input_width]

        hidden = self.act(self._linear(
            active_input, self.linear[0], hidden_width, input_width,
        ))
        hidden = self.act(self._linear(
            hidden, self.linear[1], hidden_width, hidden_width,
        ))
        hidden_path = self._linear(
            hidden, self.linear[2], hidden_width, hidden_width,
        )
        input_path = F.linear(
            active_input,
            self.skip_proj.weight[:hidden_width, :input_width],
            None,
        )
        hidden = self.act(hidden_path + input_path)
        hidden = self.act(self._linear(
            hidden, self.linear[3], output_width, hidden_width,
        ))
        rgb = F.linear(
            hidden,
            self.output_layer.weight[:, :output_width],
            self.output_layer.bias,
        )
        return torch.sigmoid(rgb), hidden

    def active_parameter_count(self, quality_level):
        """Count the unique Decoder scalars required by one prefix."""
        hidden = self.hidden_widths[quality_level]
        output = self.output_widths[quality_level]
        active_input = self.input_widths[quality_level]
        return (
            hidden * active_input + hidden
            + hidden * hidden + hidden
            + hidden * hidden + hidden
            + hidden * active_input
            + output * hidden + output
            + 3 * output + 3
        )


@dataclass(frozen=True)
class WidthPrefixOutput:
    reconstruction: torch.Tensor
    hidden: torch.Tensor
    active_levels: Tuple[int, ...]
    quality_level: int
    hidden_width: int


@dataclass(frozen=True)
class WidthScalableBatchOutput:
    prefixes: Tuple[WidthPrefixOutput, ...]


class WidthScalableResNet(nn.Module):
    """Three Grid prefixes decoded by nested Additive-Skip channel widths."""

    architecture = 'hybrid_grid_scalable_width_causal_additive_skip_v2'

    def __init__(
        self,
        grid_levels: int = 10,
        grid_feat_dim: int = 6,
        base_resolution: int = 20,
        finest_resolution: int = 100,
        aspect_ratio=(16, 9),
        time_scale: float = 0.1,
        pe_freq: int = 14,
        hidden_dim: int = 256,
        width_ratios=(0.5, 0.75, 1.0),
        hierarchy: Optional[GridHierarchy] = None,
    ):
        super().__init__()
        if hierarchy is None:
            hierarchy = default_hierarchy(grid_levels)
        if not isinstance(hierarchy, GridHierarchy):
            raise TypeError('hierarchy must be a GridHierarchy')
        if hierarchy.num_levels != grid_levels:
            raise ValueError('hierarchy and grid_levels must match')
        if hierarchy.num_quality_levels != 3:
            raise ValueError('width model requires exactly three groups')

        self.hierarchy = hierarchy
        self.grid_feat_dim = grid_feat_dim
        self.grid_encoder = MultiResGrid(
            n_levels=grid_levels,
            n_features_per_level=grid_feat_dim,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            aspect_ratio=aspect_ratio,
            time_scale=time_scale,
        )
        self.pe_encoder = Frequency(dim=3, n_levels=pe_freq, base=2.0)
        self.grid_group_dims = tuple(
            len(group) * grid_feat_dim for group in hierarchy.groups
        )
        base_grid_dim, e1_dim, e2_dim = self.grid_group_dims
        pe_dim = self.pe_encoder.output_dim
        self.feature_widths = (
            base_grid_dim + pe_dim,
            base_grid_dim + pe_dim + e1_dim,
            base_grid_dim + pe_dim + e1_dim + e2_dim,
        )
        decoder_input_dim = self.feature_widths[-1]

        # Base is invariant across all prefixes. Enhancement gates are causal:
        # they may inspect already decoded lower groups but never future groups.
        self.base_grid_gate = IdentityInitializedGate(
            base_grid_dim + pe_dim, base_grid_dim,
        )
        self.base_pe_gate = IdentityInitializedGate(
            base_grid_dim + pe_dim, pe_dim,
        )
        self.enhancement_gates = nn.ModuleList([
            IdentityInitializedGate(
                base_grid_dim + e1_dim + pe_dim, e1_dim,
            ),
            IdentityInitializedGate(
                base_grid_dim + e1_dim + e2_dim + pe_dim, e2_dim,
            ),
        ])

        # Time is represented once, while each semantic feature group owns a
        # small head. Zero initialization makes modulation identity-like:
        # Norm(f) * (1 + delta_gamma) + beta.
        time_hidden = 64
        self.time_trunk = nn.Sequential(
            nn.Linear(1, time_hidden), nn.ReLU(inplace=True),
        )
        self.time_heads = nn.ModuleList([
            nn.Linear(time_hidden, 2 * self.feature_widths[0]),
            nn.Linear(time_hidden, 2 * e1_dim),
            nn.Linear(time_hidden, 2 * e2_dim),
        ])
        for head in self.time_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.time_norms = nn.ModuleList([
            nn.GroupNorm(dim, dim, affine=False)
            for dim in (self.feature_widths[0], e1_dim, e2_dim)
        ])
        self.decoder = WidthAdditiveSkipDecoder(
            decoder_input_dim,
            hidden_dim=hidden_dim,
            width_ratios=width_ratios,
            input_widths=self.feature_widths,
        )

    def _validate_grids(
        self,
        grids: Optional[Sequence[torch.Tensor]] = None,
    ):
        levels = self.grid_encoder.levels
        if grids is None:
            return (None,) * len(levels)
        if not isinstance(grids, (tuple, list)):
            raise TypeError('grids must be a tuple or list')
        if len(grids) != len(levels):
            raise ValueError('grids and Grid levels must have the same length')
        return tuple(grids)

    def _grid_group(self, coords_hw, group, grids):
        return torch.cat([
            self.grid_encoder.levels[index](
                coords_hw, grid=grids[index],
            )
            for index in group
        ], dim=-1)

    def _temporal_context(self, coords):
        time = coords[:, 2, 0, 0].unsqueeze(1)
        return self.time_trunk(time)

    def _modulate(self, feature, stage_index, temporal_context):
        batch, _, _, channels = feature.shape
        delta_gamma, beta = self.time_heads[stage_index](
            temporal_context,
        ).chunk(2, dim=1)
        normalized = self.time_norms[stage_index](
            feature.permute(0, 3, 1, 2),
        )
        modulated = (
            normalized * (1.0 + delta_gamma.view(batch, channels, 1, 1))
            + beta.view(batch, channels, 1, 1)
        )
        return modulated.permute(0, 2, 3, 1)

    def forward_prefix(self, coords, quality_level, grids=None):
        if coords.ndim != 4 or coords.shape[1] != 3:
            raise ValueError('coords must have shape (B, 3, H, W)')
        self.hierarchy.validate_quality_level(quality_level)
        grids = self._validate_grids(grids)
        batch, _, height, width = coords.shape
        coords_hw = coords.permute(0, 2, 3, 1)
        pe_feature = self.pe_encoder(coords_hw)
        temporal_context = self._temporal_context(coords)

        base_grid = self._grid_group(
            coords_hw, self.hierarchy.groups[0], grids,
        )
        base_context = torch.cat((base_grid, pe_feature), dim=-1)
        base_feature = torch.cat((
            self.base_grid_gate(base_context) * base_grid,
            self.base_pe_gate(base_context) * pe_feature,
        ), dim=-1)
        features = [self._modulate(
            base_feature, 0, temporal_context,
        )]

        e1_grid = None
        if quality_level >= 1:
            e1_grid = self._grid_group(
                coords_hw, self.hierarchy.groups[1], grids,
            )
            e1_context = torch.cat(
                (base_grid, e1_grid, pe_feature), dim=-1,
            )
            e1_feature = self.enhancement_gates[0](e1_context) * e1_grid
            features.append(self._modulate(
                e1_feature, 1, temporal_context,
            ))

        if quality_level >= 2:
            e2_grid = self._grid_group(
                coords_hw, self.hierarchy.groups[2], grids,
            )
            e2_context = torch.cat(
                (base_grid, e1_grid, e2_grid, pe_feature), dim=-1,
            )
            e2_feature = self.enhancement_gates[1](e2_context) * e2_grid
            features.append(self._modulate(
                e2_feature, 2, temporal_context,
            ))

        active_feature = torch.cat(features, dim=-1)
        missing_channels = self.feature_widths[-1] - active_feature.shape[-1]
        if missing_channels:
            active_feature = torch.cat((
                active_feature,
                active_feature.new_zeros(
                    *active_feature.shape[:-1], missing_channels,
                ),
            ), dim=-1)
        rgb, hidden = self.decoder(
            active_feature.reshape(batch * height * width, -1),
            quality_level,
        )
        return WidthPrefixOutput(
            reconstruction=rgb.view(
                batch, height, width, 3,
            ).permute(0, 3, 1, 2),
            hidden=hidden.view(batch, height, width, -1),
            active_levels=self.hierarchy.active_levels(quality_level),
            quality_level=quality_level,
            hidden_width=self.decoder.hidden_widths[quality_level],
        )

    def forward_all(self, coords, grids=None):
        return WidthScalableBatchOutput(tuple(
            self.forward_prefix(coords, quality_level, grids)
            for quality_level in range(3)
        ))

    def forward(self, coords, quality_level=2, grids=None):
        return self.forward_prefix(coords, quality_level, grids)

    @staticmethod
    def _parameter_count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    def parameter_budget(self):
        """Return cumulative Grid plus network parameters for every prefix."""
        common = (
            self._parameter_count(self.pe_encoder)
            + self._parameter_count(self.time_trunk)
        )
        stage_counts = (
            self._parameter_count(self.base_grid_gate)
            + self._parameter_count(self.base_pe_gate)
            + self._parameter_count(self.time_heads[0]),
            self._parameter_count(self.enhancement_gates[0])
            + self._parameter_count(self.time_heads[1]),
            self._parameter_count(self.enhancement_gates[1])
            + self._parameter_count(self.time_heads[2]),
        )
        grid_counts = tuple(
            self._parameter_count(level)
            for level in self.grid_encoder.levels
        )
        budgets = []
        stage_total = common
        for quality_level in range(3):
            stage_total += stage_counts[quality_level]
            active_levels = self.hierarchy.active_levels(quality_level)
            grid_total = sum(grid_counts[index] for index in active_levels)
            decoder_total = self.decoder.active_parameter_count(quality_level)
            budgets.append(stage_total + grid_total + decoder_total)
        return tuple(budgets)

