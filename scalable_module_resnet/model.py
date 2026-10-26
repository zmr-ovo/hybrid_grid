"""Hierarchical scalable Hybrid Grid with gated stage features.

The 7/2/1 default hierarchy assigns every component to a quality stage:

* Base uses L0--L6, its own Grid/PE gates and the full additive-skip decoder.
* E1 uses L7--L8, a stage gate and a lightweight hidden-state adapter.
* E2 uses L9, a stage gate and a lightweight hidden-state adapter.

All stages share the positional encoding, temporal trunk and RGB head. The
temporal output heads are stage-specific, so decoding a prefix never requires
parameters or Grid values from a later stage. This module is intentionally
uncompressed; it first verifies the representation capacity of the hierarchy.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from encoding import Frequency, MultiResGrid
from scalable_generation.hierarchy import GridHierarchy, default_hierarchy


class BaseAdditiveSkipDecoder(nn.Module):
    """Paper decoder with concat skip written as two additive projections."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        if input_dim <= 0 or hidden_dim < 2:
            raise ValueError('input_dim must be positive and hidden_dim >= 2')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim // 2
        self.linear = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        ])
        self.skip_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, stage_input):
        hidden = stage_input
        for index, layer in enumerate(self.linear):
            if index == 2:
                hidden = layer(hidden) + self.skip_proj(stage_input)
            else:
                hidden = layer(hidden)
            hidden = self.act(hidden)
        return hidden


class EnhancementAdditiveSkipAdapter(nn.Module):
    """Refine an existing hidden state with one newly available Grid group."""

    def __init__(self, hidden_dim: int, input_dim: int, bottleneck_dim: int):
        super().__init__()
        if min(hidden_dim, input_dim, bottleneck_dim) <= 0:
            raise ValueError('adapter dimensions must be positive')
        self.hidden_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.input_proj = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.output_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, hidden, stage_input):
        update = self.hidden_proj(hidden) + self.input_proj(stage_input)
        return hidden + self.output_proj(self.act(update))


@dataclass(frozen=True)
class ModulePrefixOutput:
    reconstruction: torch.Tensor
    hidden: torch.Tensor
    active_levels: Tuple[int, ...]
    quality_level: int


@dataclass(frozen=True)
class ModuleScalableBatchOutput:
    prefixes: Tuple[ModulePrefixOutput, ...]


class ModuleScalableResNet(nn.Module):
    """Three-prefix model with stage-specific Grid, gate and time branches."""

    architecture = 'hybrid_grid_scalable_hierarchical_additive_skip_v3'

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
        branch_dim: Optional[int] = None,
        hierarchy: Optional[GridHierarchy] = None,
    ):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError('hidden_dim must be at least 2')
        if branch_dim is None:
            branch_dim = max(1, hidden_dim // 16)
        if branch_dim <= 0:
            raise ValueError('branch_dim must be positive')
        if hierarchy is None:
            hierarchy = default_hierarchy(grid_levels)
        if not isinstance(hierarchy, GridHierarchy):
            raise TypeError('hierarchy must be a GridHierarchy')
        if hierarchy.num_levels != grid_levels:
            raise ValueError('hierarchy and grid_levels must match')
        if hierarchy.num_quality_levels != 3:
            raise ValueError('hierarchical model requires exactly three groups')

        self.hierarchy = hierarchy
        self.hidden_dim = hidden_dim
        self.branch_dim = branch_dim
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
        base_dim = base_grid_dim + pe_dim

        # Base decides how much Grid and PE information to retain. Enhancement
        # stages gate only their new Grid information; PE remains shared.
        self.base_grid_gate = nn.Sequential(
            nn.Linear(base_dim, base_grid_dim), nn.Sigmoid(),
        )
        self.base_pe_gate = nn.Sequential(
            nn.Linear(base_dim, pe_dim), nn.Sigmoid(),
        )
        self.enhancement_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(e1_dim + pe_dim, e1_dim), nn.Sigmoid()),
            nn.Sequential(nn.Linear(e2_dim + pe_dim, e2_dim), nn.Sigmoid()),
        ])

        # A shared temporal trunk captures common motion. Small heads output
        # gamma/beta only for the feature channels owned by each stage.
        time_hidden = 64
        self.time_trunk = nn.Sequential(
            nn.Linear(1, time_hidden), nn.ReLU(inplace=True),
        )
        self.time_heads = nn.ModuleList([
            nn.Linear(time_hidden, 2 * base_dim),
            nn.Linear(time_hidden, 2 * e1_dim),
            nn.Linear(time_hidden, 2 * e2_dim),
        ])
        self.time_norms = nn.ModuleList([
            nn.GroupNorm(dim, dim, affine=False)
            for dim in (base_dim, e1_dim, e2_dim)
        ])

        hidden_output_dim = hidden_dim // 2
        self.base_decoder = BaseAdditiveSkipDecoder(base_dim, hidden_dim)
        self.enhancement_adapters = nn.ModuleList([
            EnhancementAdditiveSkipAdapter(
                hidden_output_dim, e1_dim, branch_dim,
            ),
            EnhancementAdditiveSkipAdapter(
                hidden_output_dim, e2_dim, branch_dim,
            ),
        ])
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_output_dim, 3), nn.Sigmoid(),
        )

    @property
    def stage_blocks(self):
        """Stage modules in prefix order without duplicate registration."""
        return (self.base_decoder, *tuple(self.enhancement_adapters))

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
        gamma, beta = self.time_heads[stage_index](
            temporal_context,
        ).chunk(2, dim=1)
        normalized = self.time_norms[stage_index](
            feature.permute(0, 3, 1, 2),
        )
        modulated = (
            normalized * gamma.view(batch, channels, 1, 1)
            + beta.view(batch, channels, 1, 1)
        )
        return modulated.permute(0, 2, 3, 1)

    def _hidden_prefixes(self, coords, quality_level, grids=None):
        self.hierarchy.validate_quality_level(quality_level)
        grids = self._validate_grids(grids)
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
        base_feature = self._modulate(
            base_feature, 0, temporal_context,
        )
        hidden = self.base_decoder(base_feature)
        prefixes = [hidden]

        for stage_index in range(1, quality_level + 1):
            stage_grid = self._grid_group(
                coords_hw, self.hierarchy.groups[stage_index], grids,
            )
            gate_context = torch.cat((stage_grid, pe_feature), dim=-1)
            stage_feature = (
                self.enhancement_gates[stage_index - 1](gate_context)
                * stage_grid
            )
            stage_feature = self._modulate(
                stage_feature, stage_index, temporal_context,
            )
            hidden = self.enhancement_adapters[stage_index - 1](
                hidden, stage_feature,
            )
            prefixes.append(hidden)
        return tuple(prefixes)

    def _output(self, hidden, quality_level, height, width):
        batch = hidden.shape[0]
        rgb = self.rgb_head(hidden.reshape(batch * height * width, -1))
        reconstruction = rgb.view(
            batch, height, width, 3,
        ).permute(0, 3, 1, 2)
        return ModulePrefixOutput(
            reconstruction=reconstruction,
            hidden=hidden,
            active_levels=self.hierarchy.active_levels(quality_level),
            quality_level=quality_level,
        )

    @staticmethod
    def _validate_coords(coords):
        if coords.ndim != 4 or coords.shape[1] != 3:
            raise ValueError('coords must have shape (B, 3, H, W)')

    def forward_prefix(self, coords, quality_level, grids=None):
        self._validate_coords(coords)
        _, _, height, width = coords.shape
        hidden = self._hidden_prefixes(
            coords, quality_level, grids,
        )[-1]
        return self._output(hidden, quality_level, height, width)

    def forward_all(self, coords, grids=None):
        self._validate_coords(coords)
        _, _, height, width = coords.shape
        hidden_prefixes = self._hidden_prefixes(
            coords, self.hierarchy.num_quality_levels - 1, grids,
        )
        return ModuleScalableBatchOutput(tuple(
            self._output(hidden, quality_level, height, width)
            for quality_level, hidden in enumerate(hidden_prefixes)
        ))

    def forward(self, coords, quality_level=2, grids=None):
        return self.forward_prefix(coords, quality_level, grids)

    @staticmethod
    def _parameter_count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    def parameter_budget(self):
        """Return cumulative parameters needed by B, Mid and Full."""
        grid_counts = tuple(
            self._parameter_count(level)
            for level in self.grid_encoder.levels
        )
        common = (
            self._parameter_count(self.pe_encoder)
            + self._parameter_count(self.time_trunk)
            + self._parameter_count(self.rgb_head)
        )
        stage_counts = (
            self._parameter_count(self.base_grid_gate)
            + self._parameter_count(self.base_pe_gate)
            + self._parameter_count(self.time_heads[0])
            + self._parameter_count(self.base_decoder),
            self._parameter_count(self.enhancement_gates[0])
            + self._parameter_count(self.time_heads[1])
            + self._parameter_count(self.enhancement_adapters[0]),
            self._parameter_count(self.enhancement_gates[1])
            + self._parameter_count(self.time_heads[2])
            + self._parameter_count(self.enhancement_adapters[1]),
        )
        cumulative = []
        total = common
        for stage_count, group in zip(stage_counts, self.hierarchy.groups):
            total += stage_count
            total += sum(grid_counts[level] for level in group)
            cumulative.append(total)
        return tuple(cumulative)
