# loss_nerv.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from piqa import SSIM, MS_SSIM

class NervLoss:
    def __init__(self, loss_type: str = 'L2', device: torch.device = 'cuda'):
        self.loss_type = loss_type
        self.device = device

        self.ssim_criterion = SSIM().to(self.device)
        self.ms_ssim_criterion = MS_SSIM(n_channels=3).to(self.device)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach()

        if self.loss_type == 'L2':
            loss = F.mse_loss(pred, target)
        elif self.loss_type == 'L1':
            loss = F.l1_loss(pred, target)
        elif self.loss_type == 'SSIM':
            loss = 1 - self.ssim_criterion(pred, target)
        elif self.loss_type == 'Fusion1':
            loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion2':
            loss = 0.3 * F.l1_loss(pred, target) + 0.7 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion3':
            loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion4':
            loss = 0.5 * F.l1_loss(pred, target) + 0.5 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion5':
            loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion6':
            loss = 0.7 * F.l1_loss(pred, target) + 0.3 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion7':
            loss = 0.7 * F.mse_loss(pred, target) + 0.3 * F.l1_loss(pred, target)
        elif self.loss_type == 'Fusion8':
            loss = 0.5 * F.mse_loss(pred, target) + 0.5 * F.l1_loss(pred, target)
        elif self.loss_type == 'Fusion9':
            loss = 0.9 * F.l1_loss(pred, target) + 0.1 * (1 - self.ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion10':
            loss = 0.7 * F.l1_loss(pred, target) + 0.3 * (1 - self.ms_ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion11':
            loss = 0.9 * F.l1_loss(pred, target) + 0.1 * (1 - self.ms_ssim_criterion(pred, target))
        elif self.loss_type == 'Fusion12':
            loss = 0.8 * F.l1_loss(pred, target) + 0.2 * (1 - self.ms_ssim_criterion(pred, target))
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        return loss

def psnr_fn(pred:torch.Tensor, target: torch.Tensor) -> float:
    """
    计算预测值和真实值之间的PSNR
    
    参数:
        pred (torch.Tensor): 预测图像张量 (B,C,H,W)
        target (torch.Tensor): 真实图像张量 (B,C,H,W)
    
    返回:
        float: PSNR值 (dB)
    """
    pred = pred.detach()
    target = target.detach()
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def msssim_fn(pred: torch.Tensor, target: torch.Tensor, device: torch.device = 'cuda') -> float:
    """
    计算预测值和真实值之间的MS-SSIM
    
    参数:
        pred (torch.Tensor): 预测图像张量 (B,C,H,W)
        target (torch.Tensor): 真实图像张量 (B,C,H,W)
        device (torch.device): 计算设备
    
    返回:
        float: MS-SSIM值 (0-1之间)
    """
    pred = pred.detach()
    target = target.detach()
    ms_ssim_criterion = MS_SSIM(n_channels=3).to(device)
    return ms_ssim_criterion(pred, target).item()


def adjust_lr(optimizer, cur_epoch, cur_iter, num_batches, args):
    cur_epoch = cur_epoch + (float(cur_iter) / num_batches)
    if args.lr_type == 'cosine':
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)
    elif args.lr_type == 'step':
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(args.lr_steps)))
    elif args.lr_type == 'const':
        lr_mult = 1
    elif args.lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < args.warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult

@dataclass(frozen=True)
class QuantizationMetadata:
    bits: int
    axis: Optional[int]
    shape: tuple
    dtype: torch.dtype
    scale: torch.Tensor
    zero_point: torch.Tensor

    @property
    def estimated_bits(self):
        """返回固定宽度存储下的量化元数据 bit 数。"""
        scale_bits = self.scale.numel() * self.scale.element_size() * 8
        zero_point_bits = self.zero_point.numel() * self.bits
        shape_bits = len(self.shape) * 64
        header_bits = 8 + 64 + 8  # bits、axis 和 dtype 标识
        return scale_bits + zero_point_bits + shape_bits + header_bits


def quantize_per_tensor(t, bit=8, axis=None):
    """对浮点张量执行非对称均匀量化。"""
    if not isinstance(bit, int) or isinstance(bit, bool) or not 1 <= bit <= 31:
        raise ValueError("bit must be an integer between 1 and 31")
    if not torch.is_floating_point(t):
        raise TypeError("quantization input must be a floating-point tensor")
    if t.numel() == 0:
        raise ValueError("quantization input must not be empty")
    if not torch.isfinite(t).all():
        raise ValueError("quantization input must contain only finite values")

    if axis is not None:
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise TypeError("axis must be an integer or None")
        if not -t.ndim <= axis < t.ndim:
            raise ValueError(f"axis {axis} is out of range for a {t.ndim}D tensor")
        axis %= t.ndim

    values = t.float() if t.element_size() < 4 else t
    reduce_dims = tuple(dim for dim in range(t.ndim) if dim != axis)
    if axis is None:
        value_min = values.amin()
        value_max = values.amax()
    elif reduce_dims:
        value_min = values.amin(dim=reduce_dims, keepdim=True)
        value_max = values.amax(dim=reduce_dims, keepdim=True)
    else:
        value_min = values
        value_max = values

    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    value_min = torch.minimum(value_min, zero)
    value_max = torch.maximum(value_max, zero)

    quant_max = 2**bit - 1
    value_range = value_max - value_min
    scale = torch.where(value_range == 0, torch.ones_like(value_range), value_range / quant_max)
    zero_point = (-value_min / scale).round().clamp(0, quant_max)

    quantized = (values / scale + zero_point).round().clamp(0, quant_max)
    dequantized = ((quantized - zero_point) * scale).to(t.dtype)
    metadata = QuantizationMetadata(
        bits=bit,
        axis=axis,
        shape=tuple(t.shape),
        dtype=t.dtype,
        scale=scale.detach(),
        zero_point=zero_point.to(torch.int32).detach(),
    )
    return quantized.to(torch.int32), dequantized, metadata
