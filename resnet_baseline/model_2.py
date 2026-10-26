"""保留原始输入重注入的瓶颈残差 Decoder。

相比旧版，本实现去除 LayerNorm 和零初始化，并重新加入 Grid + PE 原始特征的
独立深层投影。默认使用两个 ``H -> H/2 -> H`` 残差块；输入投影位于两个残差块
之间，兼顾原论文 Decoder 的信息直达路径与 ResNet 的恒等优化路径。
"""

import torch
from torch import nn

from model import HybridGridNet


class BottleneckResidualBlock(nn.Module):
    """无归一化瓶颈残差块，使用可学习的非零残差系数。"""

    def __init__(self, hidden_dim: int, ff_dim: int, residual_scale: float = 0.1):
        super().__init__()
        if hidden_dim <= 0 or ff_dim <= 0:
            raise ValueError("hidden_dim and ff_dim must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

        self.branch = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor):
        return x + self.scale * self.branch(x)


# 保留旧导入名称，避免已有脚本因类名变化而失败。
PreNormResidualBlock = BottleneckResidualBlock


class DeepResNetDecoder(nn.Module):
    """输入重注入 + 瓶颈残差块组成的逐像素 Decoder。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        ff_dim: int = None,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if num_blocks < 2:
            raise ValueError("num_blocks must be >= 2 so input injection stays between blocks")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        if ff_dim is None:
            ff_dim = self._solve_ff_dim(hidden_dim, num_blocks)
        if ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        self.ff_dim = ff_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            BottleneckResidualBlock(hidden_dim, ff_dim, residual_scale)
            for _ in range(num_blocks)
        ])
        self.input_skip = nn.Linear(input_dim, hidden_dim, bias=False)
        self.injection_index = num_blocks // 2
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    @staticmethod
    def _solve_ff_dim(hidden_dim: int, num_blocks: int) -> int:
        """求瓶颈宽度，使完整 Decoder 与原 Decoder 的参数预算尽量接近。"""
        target = 2 * hidden_dim * hidden_dim + 2 * hidden_dim
        per_block_fixed = hidden_dim + 1
        return max(
            1,
            round(
                (target / num_blocks - per_block_fixed)
                / (2 * hidden_dim + 1)
            ),
        )

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        for index, block in enumerate(self.blocks):
            if index == self.injection_index:
                h = h + self.input_skip(x)
            h = block(h)
        return self.head(h)


class DeepResNetHybridGridNet(HybridGridNet):
    """论文基线，仅将 Decoder 换成带输入重注入的瓶颈 ResNet。"""

    architecture = "hybrid_grid_input_injected_resnet_v2"

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
        num_blocks: int = 2,
        ff_dim: int = None,
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
        self.decoder = DeepResNetDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            ff_dim=ff_dim,
        )


if __name__ == "__main__":
    from model import HybridGridNet as Original

    def count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    original = Original()
    variant = DeepResNetHybridGridNet()
    print("original decoder params:", count(original.decoder))
    print("input-injected ResNet decoder params:", count(variant.decoder))
    print("residual blocks:", variant.decoder.num_blocks)
    print("bottleneck width:", variant.decoder.ff_dim)
