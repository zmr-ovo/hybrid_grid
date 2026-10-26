"""Deterministic scalable decoding for coarse-to-fine Hybrid Grid models.

This package is intentionally separate from ``compression``.  It introduces
quality-level Grid prefixes without changing the existing single-rate model or
codec.
"""

from .hierarchy import GridHierarchy, default_hierarchy
from .losses import (
    MultiPrefixLoss,
    PrefixLoss,
    multi_prefix_rate_distortion_loss,
    prefix_rate_distortion_loss,
)
from .model import (
    PrefixGridRate,
    ScalableBatchOutput,
    ScalableHybridGridNet,
    ScalableModelOutput,
)

__all__ = [
    'GridHierarchy',
    'MultiPrefixLoss',
    'PrefixGridRate',
    'PrefixLoss',
    'ScalableBatchOutput',
    'ScalableHybridGridNet',
    'ScalableModelOutput',
    'default_hierarchy',
    'multi_prefix_rate_distortion_loss',
    'prefix_rate_distortion_loss',
]
