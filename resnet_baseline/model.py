"""ResNet-decoder variant of the uncompressed Hybrid Grid baseline.

Only the decoder is changed. The multi-resolution Grid, positional encoding,
feature gates and temporal modulation are inherited unchanged from the paper
baseline in :mod:`model`.
"""

import torch
from torch import nn

from model import HybridGridNet


class ResidualMLPBlock(nn.Module):
    """Two-layer MLP branch with an additive identity connection."""

    def __init__(self, hidden_dim: int, branch_dim: int):
        super().__init__()
        if hidden_dim <= 0 or branch_dim <= 0:
            raise ValueError("hidden dimensions must be positive")

        self.branch = nn.Sequential(
            nn.Linear(hidden_dim, branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor):
        return inputs + self.branch(inputs)


class ResNetDecoder(nn.Module):
    """Pixel MLP using one two-layer additive residual block.

    The residual branch uses 128% of ``hidden_dim``. This keeps the number of
    hidden Linear layers equal to the original decoder. For the paper setting
    ``hidden_dim=256``, the decoder has 238,923 parameters, only 72 more than
    the original skip-concat decoder (238,851 parameters).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        branch_dim: int = None,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")

        if branch_dim is None:
            branch_dim = max(1, round(hidden_dim * 1.28))
        if branch_dim <= 0:
            raise ValueError("branch_dim must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.branch_dim = branch_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.residual = ResidualMLPBlock(hidden_dim, branch_dim)
        self.tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor):
        features = self.stem(inputs)
        features = self.residual(features)
        return self.output_layer(self.tail(features))


class ResNetHybridGridNet(HybridGridNet):
    """Paper baseline with only its decoder replaced by ``ResNetDecoder``."""

    architecture = "hybrid_grid_resnet_baseline_v1"

    def __init__(
        self,
        grid_levels: int = 6,
        grid_feat_dim: int = 4,
        base_resolution: int = 9,
        finest_resolution: int = 32,
        aspect_ratio=(16, 9),
        time_scale: float = 0.5,
        pe_freq: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__(
            grid_levels=grid_levels,
            grid_feat_dim=grid_feat_dim,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            aspect_ratio=aspect_ratio,
            time_scale=time_scale,
            pe_freq=pe_freq,
            hidden_dim=hidden_dim,
        )
        decoder_input_dim = (
            self.grid_encoder.output_dim + self.pe_encoder.output_dim
        )
        self.decoder = ResNetDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=hidden_dim,
        )
