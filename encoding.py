import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence, Tuple


class Frequency(nn.Module):
  def __init__(
    self,
    dim: int,
    n_levels: int = 10,
    learnable_freqs: bool = True,
    base: float = 2.0
  ):
## 频率部分可能被优化为负值或零，可以尝试log频率
    super().__init__()
    if n_levels <= 0:
      raise ValueError("n_levels must be > 0.")
    self.dim = dim
    self.n_levels = n_levels
    self.learnable_freqs = learnable_freqs
    self.base = base

    init_freqs = base ** torch.linspace(0., n_levels-1, n_levels)  # base^0, base^1,..., base^(n_levels-1)
    
    if learnable_freqs:
        self.freqs = nn.Parameter(init_freqs)
    else:  
        self.register_buffer('freqs', init_freqs, persistent=False)

    self.input_dim = dim
    self.output_dim = dim * n_levels * 2
  
  def forward(self, x: torch.Tensor):
    freqs = self.freqs.to(x.device, x.dtype)            # 确保device和dtype一致
    x = x.unsqueeze(dim=-1)                             # (..., dim, 1)
    x = x * self.freqs                                  # (..., dim, L)
    x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1) # (..., dim, L*2)
    return x.flatten(-2, -1)                            # (..., dim * L * 2)


class SingleResGrid(nn.Module):
    # 指定3维
    def __init__(
        self,
        n_features: int,
        base_res: int = 16,
        aspect_ratio = (16, 9),
        time_scale: float = 0.5
    ):
        super().__init__()
        self.n_features = n_features
        
        # 各维度分辨率
        aspect_min = min(aspect_ratio)
        self.res_x = max(4, int(base_res * aspect_ratio[0] / aspect_min))
        self.res_y = max(4, int(base_res * aspect_ratio[1] / aspect_min))
        self.res_t = max(4, int(base_res * time_scale))
        
        # 可学习 3D 网格
        self.grid = nn.Parameter(
            torch.randn(self.res_x, self.res_y, self.res_t, n_features) * 0.01
        )
        
        # 预生成 8 个顶点的上下界模式(纯torch)
        # corner_idx: [0..7]
        corner_idx = torch.arange(8).view(8, 1)           # (8,1)
        dims = torch.arange(3).view(1, 3)                 # (1,3)
        # True -> 用 floor, False -> 用 ceil
        interp_mask = (corner_idx & (1 << dims)) == 0     # (8,3) bool
        self.register_buffer("interp_mask", interp_mask, persistent=False)

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None):
        # x: (..., 3), [x,y,t] ∈ [0,1]
        if grid is None:
            grid = self.grid
        elif not torch.is_tensor(grid):
            raise TypeError("grid must be a torch.Tensor")
        elif grid.shape != self.grid.shape:
            raise ValueError(
                f"grid shape must be {tuple(self.grid.shape)}, got {tuple(grid.shape)}"
            )
        elif grid.device != x.device:
            raise ValueError("grid and coordinates must be on the same device")
        elif grid.dtype != self.grid.dtype:
            raise ValueError(
                f"grid dtype must be {self.grid.dtype}, got {grid.dtype}"
            )

        bdims = x.shape[:-1]

        # 映射到网格坐标
        xg = torch.stack([
            x[..., 0] * (self.res_x - 1),
            x[..., 1] * (self.res_y - 1),
            x[..., 2] * (self.res_t - 1),
        ], dim=-1)                                        # (..., 3)

        xi = xg.floor().to(torch.long)                    # (..., 3)
        xf = xg - xi.float()                              # (..., 3)

        # 扩展出邻居维度
        xi = xi.unsqueeze(-2)                              # (..., 1, 3)
        xf = xf.unsqueeze(-2)                              # (..., 1, 3)

        # 把 mask broadcast 到 batch 维度
        mask = self.interp_mask.view(
            *([1] * len(bdims)), *self.interp_mask.shape
        )                                                  # (..., 8, 3)

        # 生成 8 个邻居坐标：True→floor, False→ceil
        inds = torch.where(mask, xi, xi + 1)               # (..., 8, 3)

        # 边界裁剪
        inds_x = inds[..., 0].clamp_(0, self.res_x - 1)
        inds_y = inds[..., 1].clamp_(0, self.res_y - 1)
        inds_t = inds[..., 2].clamp_(0, self.res_t - 1)
        inds = torch.stack([inds_x, inds_y, inds_t], dim=-1)   # (..., 8, 3)

        # 权重：True→1-xf, False→xf
        ws = torch.where(mask, 1 - xf, xf)                 # (..., 8, 3)
        w = ws.prod(dim=-1, keepdim=True)                  # (..., 8, 1)

        # 从网格中取值（高级索引）
        neig = grid[inds[..., 0], inds[..., 1], inds[..., 2]]  # (..., 8, C)

        # 加权求和
        return torch.sum(neig * w, dim=-2)                 # (..., C)


class MultiResGrid(nn.Module):
    def __init__(
        self,
        n_levels: int = 4,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        finest_resolution: int = 64,
        aspect_ratio: Tuple[float, float] = (16, 9),
        time_scale: float = 0.5
        ):

        super().__init__()

        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if base_resolution <= 0 or finest_resolution <= 0:
            raise ValueError("resolutions must be positive")

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        
        # 计算几何级数比例 b
        if n_levels == 1:
            b = 1.0
        else:
            b = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1))

        self.levels = nn.ModuleList([
            SingleResGrid(
                n_features=n_features_per_level,
                base_res=int(round(base_resolution * (b ** i))),
                aspect_ratio=aspect_ratio,
                time_scale=time_scale
            )
            for i in range(n_levels)
        ])
        
        self.input_dim = 3
        self.output_dim = n_levels * n_features_per_level

    def forward(
        self,
        x: torch.Tensor,
        grids: Optional[Sequence[torch.Tensor]] = None,
    ):
        # x: (..., 3) in [0,1]
        if grids is None:
            return torch.cat([level(x) for level in self.levels], dim=-1)
        if not isinstance(grids, (list, tuple)):
            raise TypeError("grids must be a list or tuple of tensors")
        if len(grids) != self.n_levels:
            raise ValueError(
                f"expected {self.n_levels} grids, got {len(grids)}"
            )

        return torch.cat([
            level(x, grid=grid) for level, grid in zip(self.levels, grids)
        ], dim=-1)
