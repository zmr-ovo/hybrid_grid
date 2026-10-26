"""Strict Grid/Decoder-prefix model inspired by S-NVRC.

The model owns only one parameter set.  A lower operating point:

1. evaluates only a coarse-to-fine prefix of the multi-resolution Grids;
2. zero-fills missing Grid features before the shared Gate/time modulation;
3. runs a prefix of every hidden channel dimension in the Additive-Skip
   Decoder, while keeping the final feature projection and RGB head shared.

The Full path has the same tensors and computation as ``model_1`` in
``resnet_baseline``.  Consequently a Full baseline state dict can be loaded
without architecture-specific conversion.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model import HybridGridNet
from scalable_generation.hierarchy import GridHierarchy, default_hierarchy


class PrefixAdditiveSkipDecoder(nn.Module):
    """Additive-Skip Decoder with strict prefixes of hidden channels.

    The external input dimension and the last ``hidden_dim // 2`` projection
    stay fixed.  Only the expensive hidden channel dimension is sliced.  At
    quality level Full, this is exactly equivalent to
    ``resnet_baseline.model_1.ResidualSkipDecoder``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        width_ratios=(0.5, 0.75, 1.0),
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim < 2:
            raise ValueError("input_dim must be positive and hidden_dim >= 2")
        ratios = tuple(float(value) for value in width_ratios)
        if len(ratios) != 3:
            raise ValueError("width_ratios must contain Base, Mid and Full")
        if not 0 < ratios[0] < ratios[1] < ratios[2] == 1.0:
            raise ValueError(
                "width_ratios must be strictly increasing and end at 1.0"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim // 2
        self.width_ratios = ratios
        self.hidden_widths = tuple(
            max(1, round(hidden_dim * ratio)) for ratio in ratios
        )
        if len(set(self.hidden_widths)) != 3:
            raise ValueError("width ratios must produce distinct hidden widths")

        # Names and shapes intentionally match ResidualSkipDecoder.
        self.linear = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.output_dim),
        ])
        self.skip_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(self.output_dim, 3),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    @staticmethod
    def _prefix_linear(x, layer, out_width, in_width=None):
        weight = layer.weight[:out_width]
        if in_width is not None:
            weight = weight[:, :in_width]
        bias = None if layer.bias is None else layer.bias[:out_width]
        return F.linear(x, weight, bias)

    def _validate_level(self, quality_level):
        if not isinstance(quality_level, int) or isinstance(quality_level, bool):
            raise TypeError("quality_level must be an integer")
        if not 0 <= quality_level < len(self.hidden_widths):
            raise ValueError("quality_level must be between 0 and 2")

    def forward(self, x, quality_level=2):
        self._validate_level(quality_level)
        hidden_width = self.hidden_widths[quality_level]

        hidden = self.act(self._prefix_linear(
            x, self.linear[0], hidden_width, self.input_dim,
        ))
        hidden = self.act(self._prefix_linear(
            hidden, self.linear[1], hidden_width, hidden_width,
        ))
        hidden_path = self._prefix_linear(
            hidden, self.linear[2], hidden_width, hidden_width,
        )
        input_path = F.linear(
            x,
            self.skip_proj.weight[:hidden_width, :],
            None,
        )
        hidden = self.act(hidden_path + input_path)

        # The output projection and RGB head are shared at every operating
        # point, just like the always-transmitted head in S-NVRC.
        projected = self.act(F.linear(
            hidden,
            self.linear[3].weight[:, :hidden_width],
            self.linear[3].bias,
        ))
        return self.output_layer(projected)

    def active_parameter_count(self, quality_level):
        """Count unique Decoder scalars needed by one nested prefix."""
        self._validate_level(quality_level)
        hidden = self.hidden_widths[quality_level]
        return (
            hidden * self.input_dim + hidden
            + hidden * hidden + hidden
            + hidden * hidden + hidden
            + hidden * self.input_dim
            + self.output_dim * hidden + self.output_dim
            + 3 * self.output_dim + 3
        )


@dataclass(frozen=True)
class PrefixOutput:
    reconstruction: torch.Tensor
    active_levels: Tuple[int, ...]
    quality_level: int
    hidden_width: int


class PrefixScalableResNet(HybridGridNet):
    """One shared model decoded through nested Grid and channel prefixes."""

    architecture = "hybrid_grid_snvrc_prefix_additive_skip_v1"

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
        super().__init__(
            grid_levels=grid_levels,
            grid_feat_dim=grid_feat_dim,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            aspect_ratio=aspect_ratio,
            time_scale=time_scale,
            pe_freq=pe_freq,
            hidden_dim=hidden_dim,
        )
        if hierarchy is None:
            hierarchy = default_hierarchy(grid_levels)
        if not isinstance(hierarchy, GridHierarchy):
            raise TypeError("hierarchy must be a GridHierarchy")
        if hierarchy.num_levels != grid_levels:
            raise ValueError("hierarchy and grid_levels must match")
        if hierarchy.num_quality_levels != 3:
            raise ValueError("prefix model requires exactly three groups")

        self.hierarchy = hierarchy
        decoder_input_dim = (
            self.grid_encoder.output_dim + self.pe_encoder.output_dim
        )
        self.decoder = PrefixAdditiveSkipDecoder(
            decoder_input_dim,
            hidden_dim=hidden_dim,
            width_ratios=width_ratios,
        )

    def _validate_grids(self, grids: Optional[Sequence[torch.Tensor]]):
        levels = self.grid_encoder.levels
        if grids is None:
            return (None,) * len(levels)
        if not isinstance(grids, (tuple, list)):
            raise TypeError("grids must be a tuple or list")
        if len(grids) != len(levels):
            raise ValueError("grids and Grid levels must have the same length")
        return tuple(grids)

    def _grid_prefix(self, coords_hw, quality_level, grids):
        active_levels = self.hierarchy.active_levels(quality_level)
        features = []
        for index, level in enumerate(self.grid_encoder.levels):
            if index in active_levels:
                features.append(level(coords_hw, grid=grids[index]))
            else:
                features.append(coords_hw.new_zeros(
                    *coords_hw.shape[:-1], level.n_features,
                ))
        return torch.cat(features, dim=-1), active_levels

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

        # Gate and temporal modulation are a single shared Full module.  A
        # lower point changes only its zero-filled Grid prefix.
        concatenated = torch.cat((grid_feature, pe_feature), dim=-1)
        fused = torch.cat((
            self.gate_grid(concatenated) * grid_feature,
            self.gate_pe(concatenated) * pe_feature,
        ), dim=-1)
        modulated = self.time_mod(
            fused.permute(0, 3, 1, 2), coords,
        ).permute(0, 2, 3, 1)

        reconstruction = self.decoder(
            modulated.reshape(batch * height * width, -1),
            quality_level,
        ).view(batch, height, width, 3).permute(0, 3, 1, 2)
        return PrefixOutput(
            reconstruction=reconstruction,
            active_levels=active_levels,
            quality_level=quality_level,
            hidden_width=self.decoder.hidden_widths[quality_level],
        )

    def forward(self, coords, quality_level=2, grids=None):
        return self.forward_prefix(coords, quality_level, grids)

    @staticmethod
    def _count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    def parameter_budget(self):
        """Cumulative parameter count visible to Base, Mid and Full."""
        common = (
            self._count(self.pe_encoder)
            + self._count(self.gate_grid)
            + self._count(self.gate_pe)
            + self._count(self.time_mod)
        )
        grid_counts = tuple(
            self._count(level) for level in self.grid_encoder.levels
        )
        return tuple(
            common
            + sum(grid_counts[index] for index in self.hierarchy.active_levels(k))
            + self.decoder.active_parameter_count(k)
            for k in range(3)
        )
