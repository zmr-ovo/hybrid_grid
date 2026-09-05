import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

from encoding import * 

# --------------------------- 数据集类 ---------------------------
class DynamicVideoDataset(Dataset):
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    def __init__(self, 
                 data_root:str,
                 base_res:Tuple[int, int]=(720, 1280),
                 fixed_res:Optional[Tuple[int, int]]=None,
                 min_scale:float=0.5, 
                 max_scale:float=1.0,
                 frame_interval:int=1,
                 gop_size:int=10):
        """支持动态分辨率的视频帧数据集，返回 (coords, pixels) 配对."""

        root = Path(data_root)
        if not root.is_dir():
            raise ValueError(f"data_root is not a directory: {root}")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
        if gop_size <= 0:
            raise ValueError("gop_size must be positive")
        if min_scale <= 0 or max_scale < min_scale:
            raise ValueError("scales must satisfy 0 < min_scale <= max_scale")
        self.base_res = self._validate_resolution('base_res', base_res)
        self.fixed_res = (
            self._validate_resolution('fixed_res', fixed_res)
            if fixed_res is not None else None
        )
        self.frame_paths = sorted(
            (p for p in root.iterdir()
             if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS),
            key=self._natural_sort_key,
        )
        if not self.frame_paths:
            raise ValueError(f"no supported image frames found in: {root}")

        self.frame_interval = frame_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.total_frames = len(self.frame_paths)
        self.frame_indices = list(range(0, self.total_frames, frame_interval))
        self.gop_size = gop_size
        self.to_tensor = T.ToTensor()

    @staticmethod
    def _natural_sort_key(path: Path):
        return [(1, int(part)) if part.isdigit() else (0, part.lower())
                for part in re.split(r'(\d+)', path.name)]

    @staticmethod
    def _validate_resolution(name: str, resolution: Tuple[int, int]):
        if len(resolution) != 2:
            raise ValueError(f"{name} must contain two positive integers (H, W)")
        values = tuple(int(value) for value in resolution)
        if any(value <= 0 for value in values):
            raise ValueError(f"{name} must contain two positive integers (H, W)")
        return values

    def __len__(self):
        return len(self.frame_indices)
    
    def _generate_coords(self, h: int, w: int, t_norm: float, *, device=None, dtype=None):
        """生成归一化到 [0,1] 的 (H,W,3) 坐标网格."""
        x = torch.linspace(0, 1, w, device=device, dtype=dtype)
        y = torch.linspace(0, 1, h, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        t_grid = torch.full((h, w), t_norm, device=device, dtype=dtype)
        return torch.stack((x_grid, y_grid, t_grid), dim=-1)             # (H,W,3)

    def __getitem__(self, idx):
        """
        返回:
            coords: (3,H,W) 归一化坐标 [x,y,t]
            pixels: (3,H,W) 像素值 [0,1]
            frame_idx: int, 当前样本对应的全局帧号
            gop_id: int, 当前帧所属的 GOP 编号
        """
        frame_idx = self.frame_indices[idx]
        with Image.open(self.frame_paths[frame_idx]) as image:
            img = image.convert("RGB")
        
        if self.fixed_res is not None:                        
            h, w = self.fixed_res
        else:                                                 
            scale = self.min_scale + (self.max_scale - self.min_scale) * np.random.rand()
            h = max(1, int(self.base_res[0] * scale))
            w = max(1, int(self.base_res[1] * scale))

        img = T.functional.resize(img, (h, w))

        img_tensor = self.to_tensor(img)  # (3,H,W) in [0,1]
        
        t_norm = frame_idx / max(self.total_frames - 1, 1)
        coordinates = self._generate_coords(h, w, t_norm, device=img_tensor.device, dtype=img_tensor.dtype)  # (H,W,3) ∈ [0,1]

        if coordinates.shape[:2] != img_tensor.shape[-2:]:
            raise RuntimeError("coordinate and image resolutions do not match")
        
        gop_id = frame_idx // self.gop_size

        return {
            'coords': coordinates.permute(2,0,1),  # (3,H,W)
            'pixels': img_tensor,                  # (3,H,W)
            'frame_idx': frame_idx,
            'gop_id': gop_id
        }

# --------------------------- 解码器模块 0 --------------------------
class DecoderOri(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, skip_layer=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.skip_layer = skip_layer

        # 定义MLP结构
        self.linear = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),        # 第1层 (120→256)
            nn.Linear(hidden_dim, hidden_dim),       # 第2层 (256→256)
            nn.Linear(hidden_dim + input_dim, hidden_dim),  # 第3层 (256+120→256)
            nn.Linear(hidden_dim, hidden_dim//2)     # 第4层 (256→128)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim//2, 3),
            nn.Sigmoid()
        )
        
        self.act = nn.GELU()

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.linear):
            if i == self.skip_layer: 
                h = torch.cat([x, h], dim=-1)
                
            h = layer(h)
            h = self.act(h)
            
        return self.output_layer(h)

# --------------------------- 解码器模块 new --------------------------
class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, skip_layer: int = 2,
                 refine_channels: int = 32, refine_blocks: int = 3):
        """
        MLP + Residual CNN refine 的解码器：
          - 先用 MLP 把 (B*H*W, D) -> (B*H*W, 3) 得到 coarse 结果
          - 再把 coarse reshape成 (B,3,H,W)，经过一个小 CNN 得到 refine
          - 最终输出 out = coarse + refine

        Args:
            input_dim: MLP 输入维度 (与原来一致)
            hidden_dim: MLP 隐层维度
            skip_layer: 使用 skip connection 的层索引（与原来一致）
            refine_channels: 小 CNN 中间通道数
            refine_blocks: refine 里 Conv block 的层数（>=2 比较合适）
        """
        super().__init__()
        self.input_dim = input_dim
        self.skip_layer = skip_layer

        # --------- 原来的 MLP 部分（几乎不动） ---------
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),              # 0
            nn.Linear(hidden_dim, hidden_dim),             # 1
            nn.Linear(hidden_dim + input_dim, hidden_dim), # 2 (skip 连接)
            nn.Linear(hidden_dim, hidden_dim // 2)         # 3
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()   # coarse RGB ∈ [0,1]
        )

        self.act = nn.GELU()

        # --------- 新增：小 CNN refine 模块 ---------
        blocks = []
        in_ch = 3
        ch = refine_channels

        # 第一个 conv，把 3 通道提到 ch
        blocks.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        # 中间 conv blocks
        for _ in range(refine_blocks - 2):
            blocks.append(nn.Conv2d(ch, ch, kernel_size=3, padding=1))
            blocks.append(nn.ReLU(inplace=True))

        # 最后一层把通道数降回 3，作为 residual
        blocks.append(nn.Conv2d(ch, 3, kernel_size=3, padding=1))

        self.refine = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: (B*H*W, D) 解码器输入特征
            b, h, w: 当前 batch 的 B, H, W，用于 reshape

        Return:
            out: (B,3,H,W)
        """
        # ------- 1) MLP 解码，得到 coarse RGB -------
        h_feat = x  # (B*H*W, D)
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                # skip 连接：在指定层把原始输入拼回来
                h_feat = torch.cat([x, h_feat], dim=-1)
            h_feat = layer(h_feat)
            h_feat = self.act(h_feat)

        coarse_flat = self.output_layer(h_feat)           # (B*H*W, 3)
        coarse = coarse_flat.view(b, h, w, 3).permute(0, 3, 1, 2)  # (B,3,H,W)

        # ------- 2) 小 CNN refine -------
        refine = self.refine(coarse)                      # (B,3,H,W)
        out = coarse + refine                             # 残差相加

        # 可选：保证输出在 [0,1] 范围内
        out = out.clamp(0.0, 1.0)

        return out  # (B,3,H,W)

# --------------------------- 调制模块0 ---------------------------
class TemporalModulation(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int=64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim * 2)  # gamma + beta
        )
        
        self.norm = nn.GroupNorm(input_dim, input_dim, affine=False)

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        """
        x: [B, C, H, W]
        coords: [B, 3, H, W]，归一化后的坐标 [x, y, t]
        """
        b, c, _, _ = x.shape
        
        t = coords[:, 2, 0, 0].unsqueeze(1)       # [B,1]

        gamma_beta = self.mlp(t)                  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # [B,C], [B,C]
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        
        x_norm = self.norm(x)

        return x_norm * gamma + beta

# --------------------------- 调制模块new ---------------------------
class SpatioTemporalModulation(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int=64):
        """
        input_dim: 特征通道数 C
        hidden_dim: 用于从 [x,y,t] 生成 gamma/beta 的中间通道数
        """
        super().__init__()

        # 输入: coords [B, 3, H, W]
        # 输出: gamma_beta [B, 2C, H, W]，前 C 个通道是 γ，后 C 个通道是 β
        self.coord_to_gamma_beta = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, input_dim * 2, kernel_size=1, padding=0)
        )
        self.norm = nn.GroupNorm(input_dim, input_dim, affine=False)

        last_conv = self.coord_to_gamma_beta[-1]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        """
        x:      [B, C, H, W]
        coords: [B, 3, H, W]，归一化后的坐标 [x, y, t]
        """
        b, c, h, w = x.shape

        gamma_beta = self.coord_to_gamma_beta(coords)  # [B, 2C, H, W]
        gamma, beta = gamma_beta.chunk(2, dim=1)       # [B,C,H,W], [B,C,H,W]

        x_norm = self.norm(x)

        gamma = 1.0 + gamma

        out = x_norm * gamma + beta
        return out

# --------------------------- 模型架构 ---------------------------
class HybridGridNet(nn.Module):
    def __init__(self,
                 grid_levels: int = 6,
                 grid_feat_dim: int = 4,
                 base_resolution: int = 9,
                 finest_resolution: int = 32,
                 aspect_ratio: Tuple[int, int] = (16, 9),
                 time_scale: float = 0.5,
                 pe_freq: int = 8,
                 hidden_dim: int =256,
                 ):
        super().__init__()
        
        # 编码器
        self.grid_encoder = MultiResGrid(
            n_levels=grid_levels,
            n_features_per_level=grid_feat_dim,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            aspect_ratio=aspect_ratio,
            time_scale=time_scale
        )
        self.pe_encoder = Frequency(dim=3, n_levels=pe_freq)
        
        decoder_input_dim = self.grid_encoder.output_dim + self.pe_encoder.output_dim  

        # 门控
        self.gate_grid = nn.Sequential(
            nn.Linear(decoder_input_dim, self.grid_encoder.output_dim),
            nn.Sigmoid()
        )
        self.gate_pe = nn.Sequential(
            nn.Linear(decoder_input_dim, self.pe_encoder.output_dim),
            nn.Sigmoid()
        )
        
        # 时间调制
        self.time_mod = SpatioTemporalModulation(input_dim=decoder_input_dim, hidden_dim=64)
    
        # 解码器
        self.decoder = Decoder(
            input_dim=decoder_input_dim, 
            hidden_dim=hidden_dim,
            skip_layer=2,
            refine_channels=32
        )
        
    def forward(self, coords):
        """
        Args: coords (B,3,H,W) 归一化坐标[x,y,t] ∈ [0,1]
        Return: (B,3,H,W)
        """
        b, _, h, w = coords.shape
        coords_hw = coords.permute(0, 2, 3, 1)  # (B,H,W,3)

        # 特征编码
        grid_feat = self.grid_encoder(coords_hw)   # (B,H,W,L*C)
        pe_feat   = self.pe_encoder(coords_hw)     # (B,H,W,pe_dim)

        # 门控
        concat_feat = torch.cat([grid_feat, pe_feat], dim=-1)  # (B,H,W,D)
        gated_grid = self.gate_grid(concat_feat) * grid_feat
        gated_pe   = self.gate_pe(concat_feat) * pe_feat
        fused_feat = torch.cat((gated_grid, gated_pe), dim=-1) # (B,H,W,D)

        # 时间调制
        modulated_feat = self.time_mod(
            fused_feat.permute(0, 3, 1, 2),   # [B,D,H,W]
            coords                             # [B,3,H,W]
        ).permute(0, 2, 3, 1)                 # (B,H,W,D)

        # 解码：先 flatten，再交给 Decoder，同时传 b,h,w
        modulated_flat = modulated_feat.reshape(b * h * w, -1)    # (B*H*W,D)
        out = self.decoder(modulated_flat, b, h, w)               # (B,3,H,W)

        return out
