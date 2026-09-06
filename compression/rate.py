import math
from dataclasses import dataclass
from numbers import Real
from typing import Tuple

import torch


@dataclass(frozen=True)
class GridRateResult:
    level_bits: Tuple[torch.Tensor, ...]
    total_bits: torch.Tensor
    total_values: int
    bits_per_value: torch.Tensor


def estimate_grid_rate(likelihoods, likelihood_bound=1e-9):
    """Convert per-level Grid likelihoods to differentiable estimated bits."""
    if not isinstance(likelihoods, (list, tuple)):
        raise TypeError("likelihoods must be a list or tuple of tensors")
    if not likelihoods:
        raise ValueError("likelihoods must contain at least one Grid level")
    if isinstance(likelihood_bound, bool) or not isinstance(likelihood_bound, Real):
        raise TypeError("likelihood_bound must be a real number")

    likelihood_bound = float(likelihood_bound)
    if not math.isfinite(likelihood_bound) or not 0 < likelihood_bound < 1:
        raise ValueError("likelihood_bound must be finite and between 0 and 1")

    level_bits = []
    total_values = 0
    device = None
    for level, likelihood in enumerate(likelihoods):
        if not torch.is_tensor(likelihood):
            raise TypeError(f"likelihoods[{level}] must be a torch.Tensor")
        if not torch.is_floating_point(likelihood):
            raise TypeError(f"likelihoods[{level}] must be floating point")
        if likelihood.numel() == 0:
            raise ValueError(f"likelihoods[{level}] must not be empty")
        if not torch.isfinite(likelihood).all():
            raise ValueError(f"likelihoods[{level}] must contain only finite values")
        if torch.any(likelihood < 0) or torch.any(likelihood > 1):
            raise ValueError(f"likelihoods[{level}] must be in the range [0, 1]")
        if device is None:
            device = likelihood.device
        elif likelihood.device != device:
            raise ValueError("all likelihood tensors must be on the same device")

        probability = likelihood.float().clamp_min(likelihood_bound)
        level_bits.append(-torch.log2(probability).sum())
        total_values += likelihood.numel()

    level_bits = tuple(level_bits)
    total_bits = torch.stack(level_bits).sum()
    bits_per_value = total_bits / total_values
    return GridRateResult(
        level_bits=level_bits,
        total_bits=total_bits,
        total_values=total_values,
        bits_per_value=bits_per_value,
    )
