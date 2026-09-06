import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional

import torch


QUANTIZATION_MODES = ('disabled', 'noise', 'symbols')


@dataclass(frozen=True)
class GridQuantizationResult:
    grid: torch.Tensor
    symbols: Optional[torch.Tensor]
    quant_step: float
    mode: str


def quantize_grid(grid, quant_step=0.0, mode='disabled'):
    """Quantize one Grid while keeping the reconstruction path differentiable."""
    if not torch.is_tensor(grid):
        raise TypeError("grid must be a torch.Tensor")
    if not torch.is_floating_point(grid):
        raise TypeError("grid must be a floating-point tensor")
    if grid.numel() == 0:
        raise ValueError("grid must not be empty")
    if not torch.isfinite(grid).all():
        raise ValueError("grid must contain only finite values")
    if mode not in QUANTIZATION_MODES:
        raise ValueError(
            f"mode must be one of {QUANTIZATION_MODES}, got {mode!r}"
        )
    if isinstance(quant_step, bool) or not isinstance(quant_step, Real):
        raise TypeError("quant_step must be a real number")

    quant_step = float(quant_step)
    if not math.isfinite(quant_step) or quant_step < 0:
        raise ValueError("quant_step must be finite and non-negative")

    if mode == 'disabled' or quant_step == 0:
        return GridQuantizationResult(
            grid=grid,
            symbols=None,
            quant_step=quant_step,
            mode='disabled',
        )

    if mode == 'noise':
        noise = (torch.rand_like(grid) - 0.5) * quant_step
        return GridQuantizationResult(
            grid=grid + noise,
            symbols=None,
            quant_step=quant_step,
            mode=mode,
        )

    rounded = torch.round(grid / quant_step)
    quantized = rounded * quant_step
    grid_hat = grid + (quantized - grid).detach()
    return GridQuantizationResult(
        grid=grid_hat,
        symbols=rounded.detach().to(torch.int32),
        quant_step=quant_step,
        mode=mode,
    )
