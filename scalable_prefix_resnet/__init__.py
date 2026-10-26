"""S-NVRC-style Grid and hidden-channel prefix experiment.

This package is deliberately independent from the existing scalable models.
It exposes one shared Full model whose lower operating points use strict
prefixes of its Grid levels and Decoder hidden channels.
"""

from .model import PrefixAdditiveSkipDecoder, PrefixScalableResNet
from .training import PREFIX_NAMES, BalancedPrefixScheduler

__all__ = [
    "BalancedPrefixScheduler",
    "PREFIX_NAMES",
    "PrefixAdditiveSkipDecoder",
    "PrefixScalableResNet",
]
