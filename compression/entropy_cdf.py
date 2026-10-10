"""Build arithmetic-coding CDFs from trained entropy models."""

import torch


def _validate_support(symbol_min, symbol_max):
    if not isinstance(symbol_min, int) or isinstance(symbol_min, bool):
        raise TypeError("symbol_min must be an integer")
    if not isinstance(symbol_max, int) or isinstance(symbol_max, bool):
        raise TypeError("symbol_max must be an integer")
    if symbol_max < symbol_min:
        raise ValueError("symbol_max must not be smaller than symbol_min")


def _probability_rows_to_cdfs(probabilities, precision):
    """Vectorized equivalent of probability_to_cdf for a batch of rows."""
    if (not isinstance(precision, int) or isinstance(precision, bool) or
            precision < 1 or precision > 24):
        raise ValueError("precision must be an integer between 1 and 24")
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must have shape [N, K]")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = 1 << precision
    alphabet = probabilities.shape[1]
    if alphabet > total:
        raise ValueError("precision is too small for the alphabet")
    row_sums = probabilities.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0):
        raise ValueError("each probability row must contain positive mass")

    scaled = probabilities / row_sums * (total - alphabet)
    floors = torch.floor(scaled)
    counts = floors.to(torch.int64) + 1
    missing = total - counts.sum(dim=1, keepdim=True)
    order = torch.argsort(scaled - floors, dim=1, descending=True)
    ranks = torch.arange(alphabet, device=probabilities.device)[None, :]
    additions = (ranks < missing).to(torch.int64)
    increments = torch.zeros_like(counts).scatter(1, order, additions)
    counts = counts + increments
    zeros = torch.zeros(
        counts.shape[0], 1, dtype=torch.int64, device=counts.device,
    )
    return torch.cat((zeros, counts.cumsum(dim=1)), dim=1).cpu()


@torch.no_grad()
def factorized_cdfs(entropy_model, symbol_min, symbol_max, precision=16):
    """Return one integer CDF per channel of an EntropyBottleneck."""
    _validate_support(symbol_min, symbol_max)
    parameter = next(entropy_model.parameters())
    support = torch.arange(
        symbol_min,
        symbol_max + 1,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    values = support[:, None].expand(-1, entropy_model.channels)
    probabilities = entropy_model(values).transpose(0, 1)
    return _probability_rows_to_cdfs(probabilities, precision)


@torch.no_grad()
def iter_logistic_cdfs(mean, scale, symbol_min, symbol_max, precision=16,
                       chunk_size=4096):
    """Yield bounded-size CDF batches for spatially varying distributions."""
    _validate_support(symbol_min, symbol_max)
    if not torch.is_tensor(mean) or not torch.is_tensor(scale):
        raise TypeError("mean and scale must be tensors")
    if mean.shape != scale.shape or mean.numel() == 0:
        raise ValueError("mean and scale must have the same non-empty shape")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("mean and scale must be finite")
    if torch.any(scale <= 0):
        raise ValueError("scale must be positive")
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or
            chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")

    support = torch.arange(
        symbol_min,
        symbol_max + 1,
        dtype=mean.dtype,
        device=mean.device,
    )
    centers = support.float()[None, :]
    means = mean.float().reshape(-1, 1)
    scales = scale.float().reshape(-1, 1)
    for start in range(0, means.shape[0], chunk_size):
        chunk_mean = means[start:start + chunk_size]
        chunk_scale = scales[start:start + chunk_size]
        upper = torch.sigmoid(
            (centers + 0.5 - chunk_mean) / chunk_scale
        )
        lower = torch.sigmoid(
            (centers - 0.5 - chunk_mean) / chunk_scale
        )
        probabilities = (upper - lower).clamp_min(0)
        yield _probability_rows_to_cdfs(probabilities, precision)


def logistic_cdfs(mean, scale, symbol_min, symbol_max, precision=16):
    """Return all CDF rows; intended for tests and small Grid levels."""
    return torch.cat(tuple(iter_logistic_cdfs(
        mean, scale, symbol_min, symbol_max, precision,
    )), dim=0)
