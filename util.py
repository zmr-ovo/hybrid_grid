# loss_nerv.py
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
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

def quantize_per_tensor(t, bit=8, axis=-1):
    """返回量化后的整数张量"""
    if axis == -1:
        t_valid = t != 0
        t_min, t_max = t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit
    elif axis == 0:  # 按第0维量化
        t_min = t.min(dim=0, keepdim=True)[0]
        t_max = t.max(dim=0, keepdim=True)[0]
        scale = (t_max - t_min) / 2**bit
    elif axis == 1:  # 按第1维量化
        t_min = t.min(dim=1, keepdim=True)[0]
        t_max = t.max(dim=1, keepdim=True)[0]
        scale = (t_max - t_min) / 2**bit
    
    quant_t = ((t - t_min) / (scale + 1e-19)).round().clamp(0, 2**bit-1)
    new_t = t_min + scale * quant_t
    
    return quant_t.to(torch.int32), new_t  # 使用int32避免溢出
