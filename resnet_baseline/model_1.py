"""原 Decoder 的等价加法实现，用于验证 concat 与双投影相加的一致性。

原版在第 3 个 Linear 前执行 ``cat([x, h])``。由于 Linear 对各输入分块是
独立线性映射，它可以严格写成 ``hidden_proj(h) + input_proj(x)``。这里保留
原版的层数、激活位置和参数量，只改变代码表达形式，为后续按宽度切片提供干净基线。
"""

import torch
from torch import nn

from model import HybridGridNet


class ResidualSkipDecoder(nn.Module):
    """把 concat Linear 严格拆成两个独立投影的 Decoder。"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, skip_layer: int = 2):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if skip_layer != 2:
            raise ValueError("ResidualSkipDecoder currently requires skip_layer=2")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.skip_layer = skip_layer
        self.linear = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
        ])

        # 不使用 bias：唯一 bias 保留在 hidden projection 中，因而参数量与原
        # Linear(hidden_dim + input_dim, hidden_dim) 完全相同。
        self.skip_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        h = x
        for index, layer in enumerate(self.linear):
            if index == self.skip_layer:
                # 两个分支分别投影后相加，不能先相加再共用一个 Linear。
                h = layer(h) + self.skip_proj(x)
            else:
                h = layer(h)
            h = self.act(h)
        return self.output_layer(h)


class ResidualSkipHybridGridNet(HybridGridNet):
    """论文基线，仅把 concat Decoder 改写为等价双投影形式。"""

    architecture = "hybrid_grid_additive_skip_v2"

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
        decoder_input_dim = self.grid_encoder.output_dim + self.pe_encoder.output_dim
        self.decoder = ResidualSkipDecoder(decoder_input_dim, hidden_dim)


if __name__ == "__main__":
    from model import HybridGridNet as Original

    def count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    original = Original()
    variant = ResidualSkipHybridGridNet()
    print("original decoder params:", count(original.decoder))
    print("additive-skip decoder params:", count(variant.decoder))
