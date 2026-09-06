import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple

import torch
from torch import nn

from .entropy_bottleneck import EntropyBottleneck
from .quantization import QUANTIZATION_MODES, quantize_grid
from .rate import GridRateResult, estimate_grid_rate


@dataclass(frozen=True)
class CompressedModelOutput:
    reconstruction: torch.Tensor
    quantized_grids: Tuple[torch.Tensor, ...]
    symbols: Tuple[Optional[torch.Tensor], ...]
    likelihoods: Tuple[torch.Tensor, ...]
    rate: Optional[GridRateResult]
    quant_mode: str


class CompressedHybridGridNet(nn.Module):
    """Add differentiable Grid compression around a reconstruction model."""

    architecture = 'hybrid_grid_compressed_v1'

    def __init__(
        self,
        reconstruction_model,
        quant_steps=1e-3,
        entropy_filters=(3, 3, 3),
    ):
        super().__init__()
        if not isinstance(reconstruction_model, nn.Module):
            raise TypeError("reconstruction_model must be an nn.Module")
        if not hasattr(reconstruction_model, 'grid_encoder') or not hasattr(
            reconstruction_model.grid_encoder, 'levels'
        ):
            raise ValueError("reconstruction_model must expose grid_encoder.levels")

        levels = tuple(reconstruction_model.grid_encoder.levels)
        if not levels:
            raise ValueError(
                "reconstruction_model must contain at least one Grid level"
            )
        if any(
            not hasattr(level, 'grid') or not hasattr(level, 'n_features')
            for level in levels
        ):
            raise ValueError("each Grid level must expose grid and n_features")

        self.reconstruction_model = reconstruction_model
        self.quant_steps = self._normalize_quant_steps(quant_steps, len(levels))
        self.entropy_models = nn.ModuleList([
            EntropyBottleneck(level.n_features, filters=entropy_filters)
            for level in levels
        ])

    @staticmethod
    def _normalize_quant_steps(quant_steps, num_levels):
        if isinstance(quant_steps, Real) and not isinstance(quant_steps, bool):
            quant_steps = [quant_steps] * num_levels
        elif not isinstance(quant_steps, (list, tuple)):
            raise TypeError("quant_steps must be a real number, list, or tuple")
        if len(quant_steps) != num_levels:
            raise ValueError(f"expected {num_levels} quantization steps")

        normalized = []
        for step in quant_steps:
            if isinstance(step, bool) or not isinstance(step, Real):
                raise TypeError("each quantization step must be a real number")
            step = float(step)
            if not math.isfinite(step) or step < 0:
                raise ValueError("quantization steps must be finite and non-negative")
            normalized.append(step)

        has_zero = any(step == 0 for step in normalized)
        if has_zero and not all(step == 0 for step in normalized):
            raise ValueError("quantization steps must be either all zero or all positive")
        return tuple(normalized)

    def forward(self, coords, quant_mode='noise'):
        if quant_mode not in QUANTIZATION_MODES:
            raise ValueError(
                f"quant_mode must be one of {QUANTIZATION_MODES}, got {quant_mode!r}"
            )

        levels = self.reconstruction_model.grid_encoder.levels
        quantized = tuple(
            quantize_grid(level.grid, step, mode=quant_mode)
            for level, step in zip(levels, self.quant_steps)
        )
        grids = tuple(result.grid for result in quantized)
        symbols = tuple(result.symbols for result in quantized)
        effective_mode = quantized[0].mode

        if effective_mode == 'disabled':
            reconstruction = self.reconstruction_model(coords)
            return CompressedModelOutput(
                reconstruction=reconstruction,
                quantized_grids=grids,
                symbols=symbols,
                likelihoods=(),
                rate=None,
                quant_mode=effective_mode,
            )

        likelihoods = tuple(
            entropy_model(result.grid / result.quant_step)
            for entropy_model, result in zip(self.entropy_models, quantized)
        )
        rate = estimate_grid_rate(likelihoods)
        reconstruction = self.reconstruction_model(coords, grids=grids)
        return CompressedModelOutput(
            reconstruction=reconstruction,
            quantized_grids=grids,
            symbols=symbols,
            likelihoods=likelihoods,
            rate=rate,
            quant_mode=effective_mode,
        )
