"""Reference-inspired internal-width scalability for a pixel-wise Decoder.

Each residual block maps d -> h_k -> d. Only h_k changes with quality;
stem, residual stream, input skip, output projection and RGB head stay fixed.
This follows S-NVRC's hidden-channel prefix principle, not its complete
ConvNeXt architecture (no spatial convolution is introduced here).
"""

import torch.nn.functional as F
from torch import nn

from scalable_prefix_resnet.model import PrefixScalableResNet


class InternalPrefixMLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.expand = nn.Linear(feature_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features, hidden_width):
        hidden = F.gelu(F.linear(
            features, self.expand.weight[:hidden_width],
            self.expand.bias[:hidden_width],
        ))
        return features + F.linear(
            hidden, self.project.weight[:, :hidden_width], self.project.bias,
        )

    def active_parameter_count(self, hidden_width):
        dimension = self.expand.in_features
        return 2 * dimension * hidden_width + hidden_width + dimension


class FixedFeatureDecoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=240, feature_dim=128,
        num_blocks=3, width_ratios=(0.5, 0.75, 1.0),
    ):
        super().__init__()
        if min(input_dim, hidden_dim, feature_dim, num_blocks) < 1:
            raise ValueError("Decoder dimensions and num_blocks must be positive")
        ratios = tuple(float(value) for value in width_ratios)
        if len(ratios) != 3 or not 0 < ratios[0] < ratios[1] < ratios[2] == 1:
            raise ValueError("width_ratios must increase strictly and end at 1")
        widths = tuple(max(1, round(hidden_dim * ratio)) for ratio in ratios)
        if len(set(widths)) != 3:
            raise ValueError("width_ratios must produce three distinct widths")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.hidden_widths = widths
        self.width_ratios = ratios
        self.stem = nn.Linear(input_dim, feature_dim)
        self.blocks = nn.ModuleList([
            InternalPrefixMLP(feature_dim, hidden_dim) for _ in range(num_blocks)
        ])
        self.skip_proj = nn.Linear(input_dim, feature_dim, bias=False)
        self.skip_block = min(1, num_blocks - 1)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.output_layer = nn.Sequential(nn.Linear(feature_dim, 3), nn.Sigmoid())

    @staticmethod
    def _validate_level(quality_level):
        if not isinstance(quality_level, int) or isinstance(quality_level, bool):
            raise TypeError("quality_level must be an integer")
        if not 0 <= quality_level < 3:
            raise ValueError("quality_level must be between 0 and 2")

    def forward(self, features, quality_level=2):
        self._validate_level(quality_level)
        hidden_width = self.hidden_widths[quality_level]
        hidden = F.gelu(self.stem(features))
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, hidden_width)
            if index == self.skip_block:
                hidden = F.gelu(hidden + self.skip_proj(features))
        hidden = F.gelu(self.output_proj(hidden))
        return self.output_layer(hidden)

    def active_parameter_count(self, quality_level):
        self._validate_level(quality_level)
        shared = sum(
            parameter.numel()
            for module in (self.stem, self.skip_proj, self.output_proj, self.output_layer)
            for parameter in module.parameters()
        )
        return shared + sum(
            block.active_parameter_count(self.hidden_widths[quality_level])
            for block in self.blocks
        )


class InternalMLPScalableResNet(PrefixScalableResNet):
    architecture = "hybrid_grid_ablation_fixed_feature_internal_mlp_v1"

    def __init__(self, feature_dim=128, num_blocks=3, **kwargs):
        kwargs.setdefault("hidden_dim", 240)
        super().__init__(**kwargs)
        self.decoder = FixedFeatureDecoder(
            input_dim=self.decoder.input_dim,
            hidden_dim=kwargs["hidden_dim"],
            feature_dim=feature_dim,
            num_blocks=num_blocks,
            width_ratios=kwargs.get("width_ratios", (0.5, 0.75, 1.0)),
        )
