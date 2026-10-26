"""Deterministic quality-level wrapper for a compressed Hybrid Grid model.

The wrapper quantizes Grid parameters with the existing compression settings,
keeps the requested coarse-to-fine prefix, and replaces unavailable higher
levels with zero Grids.  It does not alter entropy models or training code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from compression.quantization import quantize_grid
from compression.rate import estimate_grid_rate
from .hierarchy import GridHierarchy, default_hierarchy


@dataclass(frozen=True)
class PrefixGridRate:
    """Estimated rate of one cumulative Grid prefix.

    ``bits_per_value`` is normalized by the number of values in the complete
    Grid representation.  This keeps the rate scale and ``lambda`` compatible
    across all quality levels.
    """

    level_bits: Tuple[torch.Tensor, ...]
    total_bits: torch.Tensor
    prefix_values: int
    normalization_values: int
    bits_per_value: torch.Tensor


@dataclass(frozen=True)
class ScalableModelOutput:
    reconstruction: torch.Tensor
    quantized_grids: Tuple[torch.Tensor, ...]
    prefix_grids: Tuple[torch.Tensor, ...]
    likelihoods: Tuple[torch.Tensor, ...]
    rate: Optional[PrefixGridRate]
    level_mask: torch.Tensor
    active_levels: Tuple[int, ...]
    quality_level: int
    quant_mode: str


@dataclass(frozen=True)
class ScalableBatchOutput:
    """All quality levels produced from one shared Grid quantization."""

    prefixes: Tuple[ScalableModelOutput, ...]
    quantized_grids: Tuple[torch.Tensor, ...]
    likelihoods: Tuple[torch.Tensor, ...]
    quant_mode: str


class ScalableHybridGridNet(nn.Module):
    """Expose deterministic Grid-prefix quality levels for one compressed model."""

    def __init__(self, compressed_model, hierarchy=None):
        super().__init__()
        if not isinstance(compressed_model, nn.Module):
            raise TypeError("compressed_model must be an nn.Module")
        if not hasattr(compressed_model, 'reconstruction_model'):
            raise ValueError(
                "compressed_model must expose reconstruction_model"
            )
        if not hasattr(compressed_model, 'quant_steps'):
            raise ValueError("compressed_model must expose quant_steps")

        reconstruction_model = compressed_model.reconstruction_model
        if not hasattr(reconstruction_model, 'grid_encoder') or not hasattr(
            reconstruction_model.grid_encoder, 'levels'
        ):
            raise ValueError(
                "reconstruction_model must expose grid_encoder.levels"
            )

        level_count = len(reconstruction_model.grid_encoder.levels)
        if hierarchy is None:
            hierarchy = default_hierarchy(level_count)
        if not isinstance(hierarchy, GridHierarchy):
            raise TypeError("hierarchy must be a GridHierarchy")
        if hierarchy.num_levels != level_count:
            raise ValueError(
                "hierarchy and compressed model must have the same Grid count"
            )
        if len(compressed_model.quant_steps) != level_count:
            raise ValueError(
                "quantization steps and Grid levels must have the same length"
            )

        self.compressed_model = compressed_model
        self.hierarchy = hierarchy

    @property
    def reconstruction_model(self):
        return self.compressed_model.reconstruction_model

    @property
    def quant_steps(self):
        return self.compressed_model.quant_steps

    def _quantized_grids(self, quant_mode):
        levels = self.reconstruction_model.grid_encoder.levels
        results = tuple(
            quantize_grid(level.grid, step, mode=quant_mode)
            for level, step in zip(levels, self.quant_steps)
        )
        return results

    def _likelihoods(self, quantized, active_count=None):
        if active_count is None:
            active_count = len(quantized)
        quantized = quantized[:active_count]
        normalized = tuple(
            result.grid / result.quant_step for result in quantized
        )
        compressed = self.compressed_model
        if compressed.entropy_model_type == 'factorized':
            return tuple(
                entropy_model(values)
                for entropy_model, values in zip(
                    compressed.entropy_models, normalized,
                )
            )
        if compressed.entropy_model_type == 'cross_scale':
            likelihoods = [compressed.entropy_models[0](normalized[0])]
            likelihoods.extend(
                compressed.entropy_models[index](
                    normalized[index], normalized[index - 1],
                )
                for index in range(1, len(normalized))
            )
            return tuple(likelihoods)
        entropy_model = compressed.entropy_models[0]
        if active_count == entropy_model.levels:
            return tuple(entropy_model(normalized))

        likelihoods = [entropy_model.base_entropy(normalized[0])]
        context = entropy_model.initialize_context(normalized[0])
        for level, values in enumerate(normalized[1:], start=1):
            likelihood, context = entropy_model.stages[level - 1](
                values, context,
            )
            likelihoods.append(likelihood)
        return tuple(likelihoods)

    @staticmethod
    def _prefix_rate(likelihoods, active_count, normalization_values):
        if not likelihoods:
            return None
        complete_rate = estimate_grid_rate(likelihoods[:active_count])
        return PrefixGridRate(
            level_bits=complete_rate.level_bits,
            total_bits=complete_rate.total_bits,
            prefix_values=complete_rate.total_values,
            normalization_values=normalization_values,
            bits_per_value=(
                complete_rate.total_bits / normalization_values
            ),
        )

    def prefix_grids(self, grids, quality_level):
        """Zero unavailable higher Grid levels without changing tensor shapes."""
        if not isinstance(grids, (tuple, list)):
            raise TypeError("grids must be a tuple or list")
        if len(grids) != self.hierarchy.num_levels:
            raise ValueError("grids and hierarchy must have the same length")

        mask = self.hierarchy.mask(quality_level)
        return tuple(
            grid if available else torch.zeros_like(grid)
            for grid, available in zip(grids, mask)
        )

    def _prefix_output(self, coords, quality_level, grids, likelihoods,
                       quant_mode, normalization_values):
        active_levels = self.hierarchy.active_levels(quality_level)
        prefix_grids = self.prefix_grids(grids, quality_level)
        reconstruction = self.reconstruction_model(coords, grids=prefix_grids)
        level_mask = torch.tensor(
            self.hierarchy.mask(quality_level),
            dtype=torch.bool,
            device=coords.device,
        )
        return ScalableModelOutput(
            reconstruction=reconstruction,
            quantized_grids=grids,
            prefix_grids=prefix_grids,
            likelihoods=likelihoods[:len(active_levels)],
            rate=self._prefix_rate(
                likelihoods, len(active_levels), normalization_values,
            ),
            level_mask=level_mask,
            active_levels=active_levels,
            quality_level=quality_level,
            quant_mode=quant_mode,
        )

    def forward_all(self, coords, quant_mode='noise'):
        """Evaluate every prefix after quantizing and entropy-modeling once."""
        quantized = self._quantized_grids(quant_mode)
        grids = tuple(result.grid for result in quantized)
        effective_mode = quantized[0].mode
        likelihoods = (
            () if effective_mode == 'disabled'
            else self._likelihoods(quantized)
        )
        normalization_values = sum(grid.numel() for grid in grids)
        prefixes = tuple(
            self._prefix_output(
                coords,
                quality_level,
                grids,
                likelihoods,
                effective_mode,
                normalization_values,
            )
            for quality_level in range(self.hierarchy.num_quality_levels)
        )
        return ScalableBatchOutput(
            prefixes=prefixes,
            quantized_grids=grids,
            likelihoods=likelihoods,
            quant_mode=effective_mode,
        )

    def forward_prefix(self, coords, quality_level, quant_mode='symbols'):
        """Evaluate only one prefix so its graph can be released immediately."""
        self.hierarchy.validate_quality_level(quality_level)
        active_levels = self.hierarchy.active_levels(quality_level)
        active_count = len(active_levels)
        levels = self.reconstruction_model.grid_encoder.levels
        quantized = tuple(
            quantize_grid(level.grid, step, mode=quant_mode)
            for level, step in zip(
                levels[:active_count], self.quant_steps[:active_count],
            )
        )
        active_grids = tuple(result.grid for result in quantized)
        grids = active_grids + tuple(
            torch.zeros_like(level.grid)
            for level in levels[active_count:]
        )
        effective_mode = quantized[0].mode
        likelihoods = (
            () if effective_mode == 'disabled'
            else self._likelihoods(quantized, active_count)
        )
        normalization_values = sum(level.grid.numel() for level in levels)
        level_mask = torch.tensor(
            self.hierarchy.mask(quality_level),
            dtype=torch.bool,
            device=coords.device,
        )
        return ScalableModelOutput(
            reconstruction=self.reconstruction_model(coords, grids=grids),
            quantized_grids=grids,
            prefix_grids=grids,
            likelihoods=likelihoods,
            rate=self._prefix_rate(
                likelihoods, active_count, normalization_values,
            ),
            level_mask=level_mask,
            active_levels=active_levels,
            quality_level=quality_level,
            quant_mode=effective_mode,
        )

    def forward(self, coords, quality_level, quant_mode='symbols'):
        """Reconstruct one deterministic quality level.

        ``quality_level=0`` uses the base Grid prefix, while the final quality
        level uses all Grid levels and therefore matches the existing compressed
        model under the same deterministic quantization mode.
        """
        return self.forward_prefix(coords, quality_level, quant_mode)
