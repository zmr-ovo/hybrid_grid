import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class GridRateResult:
    level_bits: Tuple[torch.Tensor, ...]
    total_bits: torch.Tensor
    total_values: int
    bits_per_value: torch.Tensor


@dataclass(frozen=True)
class ParameterStorage:
    tensor_count: int
    parameter_count: int
    total_bits: int
    bits_by_dtype: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class GridMetadataStorage:
    """Storage estimate for the small header needed to decode Grid symbols."""

    level_count: int
    shapes: Tuple[Tuple[int, ...], ...]
    quant_steps: Tuple[float, ...]
    header_bits: int
    shape_bits: int
    quant_step_bits: int
    total_bits: int


@dataclass(frozen=True)
class Fp32RateBreakdown:
    """Estimated Grid and network rate before real entropy coding."""

    total_video_pixels: int
    legacy_rate_per_value: Optional[float]
    estimated_grid_bits: Optional[float]
    estimated_grid_bpp: Optional[float]
    non_grid_storage: ParameterStorage
    non_grid_bpp: float
    entropy_model_storage: ParameterStorage
    entropy_model_side_info_bpp: float
    quantization_metadata: GridMetadataStorage
    quantization_metadata_bpp: float
    network_quantization_metadata_bits: int
    network_quantization_metadata_bpp: float
    estimated_payload_bits: Optional[float]
    estimated_payload_bpp: Optional[float]
    estimated_total_bits: Optional[float]
    estimated_total_bpp: Optional[float]
    metadata_included: bool


def parameter_storage(parameters, required_dtype=None):
    """Count parameter storage using each tensor's real dtype."""
    if torch.is_tensor(parameters):
        raise TypeError("parameters must be an iterable of tensors")
    if required_dtype is not None and not isinstance(required_dtype, torch.dtype):
        raise TypeError("required_dtype must be a torch.dtype or None")

    seen = set()
    tensor_count = 0
    parameter_count = 0
    bits_by_dtype = {}
    for index, parameter in enumerate(parameters):
        if not torch.is_tensor(parameter):
            raise TypeError("parameters[{}] must be a tensor".format(index))
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if required_dtype is not None and parameter.dtype != required_dtype:
            raise ValueError(
                "expected {}, got {} at parameters[{}]".format(
                    required_dtype, parameter.dtype, index,
                )
            )

        bits = parameter.numel() * parameter.element_size() * 8
        dtype_name = str(parameter.dtype).replace('torch.', '')
        bits_by_dtype[dtype_name] = bits_by_dtype.get(dtype_name, 0) + bits
        tensor_count += 1
        parameter_count += parameter.numel()

    return ParameterStorage(
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        total_bits=sum(bits_by_dtype.values()),
        bits_by_dtype=tuple(sorted(bits_by_dtype.items())),
    )


def estimate_grid_metadata(grids, quant_steps):
    """Estimate a deterministic Grid header without claiming codec overhead."""
    if not isinstance(grids, (list, tuple)) or not grids:
        raise ValueError("grids must be a non-empty list or tuple")
    if not isinstance(quant_steps, (list, tuple)):
        raise TypeError("quant_steps must be a list or tuple")
    if len(grids) != len(quant_steps):
        raise ValueError("grids and quant_steps must have the same length")

    shapes = []
    steps = []
    for index, (grid, step) in enumerate(zip(grids, quant_steps)):
        if not torch.is_tensor(grid):
            raise TypeError("grids[{}] must be a tensor".format(index))
        if grid.numel() == 0:
            raise ValueError("grids[{}] must not be empty".format(index))
        if grid.ndim > 255:
            raise ValueError("Grid rank must fit in uint8")
        if any(size < 1 or size > 2 ** 32 - 1 for size in grid.shape):
            raise ValueError("Grid dimensions must fit in uint32")
        if isinstance(step, bool) or not isinstance(step, Real):
            raise TypeError("quant_steps[{}] must be real".format(index))
        step = float(step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("quantization steps must be finite and positive")
        shapes.append(tuple(grid.shape))
        steps.append(step)

    # Header: uint16 format version + uint16 number of levels.
    header_bits = 16 + 16
    # Per level: uint8 rank followed by one uint32 for each dimension.
    shape_bits = sum(8 + 32 * len(shape) for shape in shapes)
    # One IEEE-754 float32 quantization step per Grid level.
    quant_step_bits = 32 * len(steps)
    return GridMetadataStorage(
        level_count=len(shapes),
        shapes=tuple(shapes),
        quant_steps=tuple(steps),
        header_bits=header_bits,
        shape_bits=shape_bits,
        quant_step_bits=quant_step_bits,
        total_bits=header_bits + shape_bits + quant_step_bits,
    )


def estimate_fp32_rate(
    total_video_pixels,
    non_grid_storage,
    entropy_model_storage,
    quantization_metadata,
    grid_bits=None,
    legacy_rate_per_value=None,
    network_quantization_metadata_bits=0,
):
    """Combine Grid estimates with network parameter storage costs."""
    if not isinstance(total_video_pixels, int) or isinstance(
        total_video_pixels, bool
    ) or total_video_pixels < 1:
        raise ValueError("total_video_pixels must be a positive integer")
    if not isinstance(non_grid_storage, ParameterStorage):
        raise TypeError("non_grid_storage must be ParameterStorage")
    if not isinstance(entropy_model_storage, ParameterStorage):
        raise TypeError("entropy_model_storage must be ParameterStorage")
    if not isinstance(quantization_metadata, GridMetadataStorage):
        raise TypeError("quantization_metadata must be GridMetadataStorage")
    if (not isinstance(network_quantization_metadata_bits, int) or
            isinstance(network_quantization_metadata_bits, bool) or
            network_quantization_metadata_bits < 0):
        raise ValueError(
            "network_quantization_metadata_bits must be a non-negative integer"
        )

    static_bits = (
        non_grid_storage.total_bits + entropy_model_storage.total_bits
    )
    non_grid_bpp = non_grid_storage.total_bits / total_video_pixels
    entropy_bpp = entropy_model_storage.total_bits / total_video_pixels
    metadata_bpp = quantization_metadata.total_bits / total_video_pixels
    network_metadata_bpp = (
        network_quantization_metadata_bits / total_video_pixels
    )

    if grid_bits is None:
        if legacy_rate_per_value is not None:
            raise ValueError("legacy rate requires an estimated Grid rate")
        estimated_grid_bits = None
        estimated_grid_bpp = None
        estimated_payload_bits = None
        estimated_payload_bpp = None
        estimated_total_bits = None
        estimated_total_bpp = None
        metadata_included = False
    else:
        if isinstance(grid_bits, bool) or not isinstance(grid_bits, Real):
            raise TypeError("grid_bits must be a real number or None")
        if not math.isfinite(grid_bits) or grid_bits < 0:
            raise ValueError("grid_bits must be finite and non-negative")
        if (isinstance(legacy_rate_per_value, bool) or
                not isinstance(legacy_rate_per_value, Real)):
            raise TypeError("legacy_rate_per_value must be a real number")
        if (not math.isfinite(legacy_rate_per_value) or
                legacy_rate_per_value < 0):
            raise ValueError(
                "legacy_rate_per_value must be finite and non-negative"
            )

        estimated_grid_bits = float(grid_bits)
        estimated_grid_bpp = estimated_grid_bits / total_video_pixels
        estimated_payload_bits = estimated_grid_bits + static_bits
        estimated_payload_bpp = estimated_payload_bits / total_video_pixels
        estimated_total_bits = (
            estimated_payload_bits + quantization_metadata.total_bits
            + network_quantization_metadata_bits
        )
        estimated_total_bpp = estimated_total_bits / total_video_pixels
        metadata_included = True

    return Fp32RateBreakdown(
        total_video_pixels=total_video_pixels,
        legacy_rate_per_value=(
            None if legacy_rate_per_value is None
            else float(legacy_rate_per_value)
        ),
        estimated_grid_bits=estimated_grid_bits,
        estimated_grid_bpp=estimated_grid_bpp,
        non_grid_storage=non_grid_storage,
        non_grid_bpp=non_grid_bpp,
        entropy_model_storage=entropy_model_storage,
        entropy_model_side_info_bpp=entropy_bpp,
        quantization_metadata=quantization_metadata,
        quantization_metadata_bpp=metadata_bpp,
        network_quantization_metadata_bits=network_quantization_metadata_bits,
        network_quantization_metadata_bpp=network_metadata_bpp,
        estimated_payload_bits=estimated_payload_bits,
        estimated_payload_bpp=estimated_payload_bpp,
        estimated_total_bits=estimated_total_bits,
        estimated_total_bpp=estimated_total_bpp,
        metadata_included=metadata_included,
    )


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
