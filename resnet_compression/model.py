"""Build compression around the model_1 additive-skip reconstruction core.

The quantizer, entropy models, rate calculation and network QAT are reused
from the established compression baseline. Only the reconstruction model is
changed to :class:`resnet_baseline.model_1.ResidualSkipHybridGridNet`.
"""

from resnet_baseline.model_1 import ResidualSkipHybridGridNet
from train_compression import build_compression_model


def build_resnet_compression_model(args, device):
    """Construct compression with a checkpoint-safe additive-skip ID."""
    return build_compression_model(
        args,
        device,
        reconstruction_model_class=ResidualSkipHybridGridNet,
        reconstruction_model_name='additive_skip',
    )
