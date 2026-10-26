"""Rate-distortion objectives for multi-prefix scalable training.

Scheme A evaluates every configured prefix for each batch. Each prefix uses
its own cumulative Grid rate. Evaluation may share one quantization pass;
memory-efficient training may instead recompute prefixes sequentially.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Tuple

import math
import torch


@dataclass(frozen=True)
class PrefixLoss:
    distortion: torch.Tensor
    rate_per_value: torch.Tensor
    total: torch.Tensor
    distortion_weight: float
    rate_weight: float


@dataclass(frozen=True)
class MultiPrefixLoss:
    total: torch.Tensor
    prefixes: Tuple[PrefixLoss, ...]


def _weights(values, count, name):
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise ValueError("{} must contain one value per prefix".format(name))
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("{} must contain real numbers".format(name))
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("{} must be finite and positive".format(name))
        normalized.append(value)
    return tuple(normalized)


def multi_prefix_rate_distortion_loss(
    output,
    target,
    distortion_fn,
    lambda_rate,
    distortion_weights,
    rate_weights,
):
    """Return Scheme A with separate distortion and rate weights."""
    if isinstance(lambda_rate, bool) or not isinstance(lambda_rate, Real):
        raise TypeError("lambda_rate must be a real number")
    lambda_rate = float(lambda_rate)
    if not math.isfinite(lambda_rate) or lambda_rate < 0:
        raise ValueError("lambda_rate must be finite and non-negative")
    if not callable(distortion_fn):
        raise TypeError("distortion_fn must be callable")
    if not hasattr(output, 'prefixes') or not output.prefixes:
        raise ValueError("output must contain at least one prefix")

    count = len(output.prefixes)
    distortion_weights = _weights(
        distortion_weights, count, 'distortion_weights',
    )
    rate_weights = _weights(rate_weights, count, 'rate_weights')
    losses = []
    weighted_total = None
    for prefix, distortion_weight, rate_weight in zip(
        output.prefixes, distortion_weights, rate_weights,
    ):
        distortion = distortion_fn(prefix.reconstruction, target)
        if prefix.rate is None:
            if lambda_rate != 0:
                raise ValueError(
                    "prefix rate is required when lambda_rate is positive"
                )
            rate = distortion.new_zeros(())
        else:
            rate = prefix.rate.bits_per_value
        total = (
            distortion_weight * distortion
            + lambda_rate * rate_weight * rate
        )
        weighted = total
        weighted_total = (
            weighted if weighted_total is None else weighted_total + weighted
        )
        losses.append(PrefixLoss(
            distortion,
            rate,
            total,
            distortion_weight,
            rate_weight,
        ))

    return MultiPrefixLoss(weighted_total, tuple(losses))


def prefix_rate_distortion_loss(
    prefix,
    target,
    distortion_fn,
    lambda_rate,
    distortion_weight=1.0,
    rate_weight=1.0,
):
    """Return one prefix loss for memory-efficient sequential training."""
    if isinstance(lambda_rate, bool) or not isinstance(lambda_rate, Real):
        raise TypeError("lambda_rate must be a real number")
    lambda_rate = float(lambda_rate)
    if not math.isfinite(lambda_rate) or lambda_rate < 0:
        raise ValueError("lambda_rate must be finite and non-negative")
    if not callable(distortion_fn):
        raise TypeError("distortion_fn must be callable")
    distortion_weight = _weights(
        (distortion_weight,), 1, 'distortion_weights',
    )[0]
    rate_weight = _weights((rate_weight,), 1, 'rate_weights')[0]

    distortion = distortion_fn(prefix.reconstruction, target)
    if prefix.rate is None:
        if lambda_rate != 0:
            raise ValueError(
                "prefix rate is required when lambda_rate is positive"
            )
        rate = distortion.new_zeros(())
    else:
        rate = prefix.rate.bits_per_value
    total = (
        distortion_weight * distortion
        + lambda_rate * rate_weight * rate
    )
    return PrefixLoss(
        distortion,
        rate,
        total,
        distortion_weight,
        rate_weight,
    )
