"""Train only the Full prefix of the hierarchical scalable model.

This is a capacity diagnostic, not a new final model.  It uses exactly the
same 7/2/1 architecture as ``scalable_module_resnet`` but excludes Base and
Mid losses.  If Full-only training reaches the non-scalable baseline, the
remaining loss in joint training comes primarily from multi-prefix gradient
conflict.  Otherwise, the hierarchy itself is the limiting factor.
"""

import logging
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from resnet_baseline.train import build_parser as build_baseline_parser
from scalable_module_resnet.model import ModuleScalableResNet
from scalable_module_resnet.train import (
    _learning_rate,
    _load_checkpoint,
    _make_loaders,
    _save_checkpoint,
    hierarchy_from_sizes,
)
from train import seed_everything, setup_logging
from util import NervLoss, msssim_fn, psnr_fn

from .training import backward_full_only


@torch.no_grad()
def evaluate_full(model, loader, device, loss_fn, log_interval, logger):
    """Evaluate only the Full prefix."""
    was_training = model.training
    model.eval()
    sums = {'distortion': 0.0, 'psnr': 0.0, 'msssim': 0.0}
    samples = 0
    for batch_index, batch in enumerate(loader):
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)
        batch_size = pixels.shape[0]
        reconstruction = model.forward_prefix(
            coords, quality_level=2,
        ).reconstruction
        sums['distortion'] += loss_fn(
            reconstruction, pixels,
        ).item() * batch_size
        sums['psnr'] += psnr_fn(reconstruction, pixels) * batch_size
        sums['msssim'] += msssim_fn(
            reconstruction, pixels, device,
        ) * batch_size
        samples += batch_size

        if (batch_index % log_interval == 0 or
                batch_index == len(loader) - 1):
            logger.info(
                'Val Full [%d/%d] | PSNR: %.2f dB | MS-SSIM: %.4f',
                batch_index + 1,
                len(loader),
                sums['psnr'] / samples,
                sums['msssim'] / samples,
            )
    model.train(was_training)
    return {name: value / samples for name, value in sums.items()}


def train_full_only(args):
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError('dynamic resolution requires batch_size=1')
    if args.epochs < 1 or args.workers < 0:
        raise ValueError('epochs must be positive and workers non-negative')
    if args.target_psnr <= 0:
        raise ValueError('target_psnr must be positive')
    hierarchy = hierarchy_from_sizes(
        args.level_group_sizes, args.grid_levels,
    )
    seed_everything(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('使用设备: %s', device)
    logger.info('实验类型: Full-only 层级模型能力上限')
    logger.info('仅 Full 损失参与训练；Base 和 B+E1 不前向、不反向')
    logger.info('层级分组: %s', hierarchy.groups)

    train_loader, val_loader = _make_loaders(args)
    model = ModuleScalableResNet(
        grid_levels=args.grid_levels,
        grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        time_scale=args.time_scale,
        pe_freq=args.pe_freq,
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        hierarchy=hierarchy,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = NervLoss(args.loss_type, device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    budgets = model.parameter_budget()
    logger.info('模型总参数: %d (%.4f M)', total_params, total_params / 1e6)
    logger.info(
        '累计参数 | B: %d | B+E1: %d | Full: %d', *budgets,
    )
    for key, value in vars(args).items():
        logger.info('%s: %s', key, value)

    start_epoch = 0
    best_train = (0.0, 0.0, 0.0)
    best_val = (0.0, 0.0, 0.0)
    if args.resume:
        start_epoch, best_train, best_val = _load_checkpoint(
            args.resume, model, optimizer, device,
        )
        logger.info('从 epoch %d 恢复 Full-only 训练', start_epoch + 1)

    if args.eval_only:
        if not args.resume:
            raise ValueError('eval_only requires --resume')
        values = evaluate_full(
            model, val_loader, device, loss_fn,
            args.log_interval, logger,
        )
        logger.info(
            'Eval Full | distortion: %.6f | PSNR: %.2f dB | '
            'MS-SSIM: %.4f',
            values['distortion'], values['psnr'], values['msssim'],
        )
        logger.info(
            '能力上限判定: %s（目标 %.2f dB）',
            'PASS' if values['psnr'] >= args.target_psnr else 'BELOW TARGET',
            args.target_psnr,
        )
        writer.close()
        return

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        sums = {'distortion': 0.0, 'psnr': 0.0, 'msssim': 0.0}
        samples = 0
        for batch_index, batch in enumerate(train_loader):
            coords = batch['coords'].to(device, non_blocking=True)
            pixels = batch['pixels'].to(device, non_blocking=True)
            batch_size = pixels.shape[0]
            learning_rate = _learning_rate(
                optimizer, args, epoch, batch_index, len(train_loader),
            )
            optimizer.zero_grad(set_to_none=True)
            reconstruction, metrics = backward_full_only(
                model, coords, pixels, loss_fn,
            )
            optimizer.step()

            sums['distortion'] += metrics.distortion * batch_size
            sums['psnr'] += metrics.psnr * batch_size
            sums['msssim'] += msssim_fn(
                reconstruction, pixels, device,
            ) * batch_size
            samples += batch_size

            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                logger.info(
                    'epoch [%d/%d] batch [%d/%d] | Full D: %.6f | '
                    'PSNR: %.2f dB | MS-SSIM: %.4f | lr: %.3e',
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    sums['distortion'] / samples,
                    sums['psnr'] / samples,
                    sums['msssim'] / samples,
                    learning_rate,
                )

        averages = {
            name: value / samples for name, value in sums.items()
        }
        best_train = (
            best_train[0], best_train[1],
            max(best_train[2], averages['psnr']),
        )
        writer.add_scalar('train/full/distortion', averages['distortion'], epoch + 1)
        writer.add_scalar('train/full/psnr', averages['psnr'], epoch + 1)
        writer.add_scalar('train/full/best_psnr', best_train[2], epoch + 1)
        writer.add_scalar('train/full/msssim', averages['msssim'], epoch + 1)
        writer.add_scalar('train/lr', learning_rate, epoch + 1)
        logger.info(
            'Epoch %d | Full-only | distortion: %.6f | PSNR: %.2f dB | '
            'BEST TRAIN: %.2f dB | MS-SSIM: %.4f | time: %.2fs',
            epoch + 1,
            averages['distortion'],
            averages['psnr'],
            best_train[2],
            averages['msssim'],
            time.time() - epoch_start,
        )

        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            values = evaluate_full(
                model, val_loader, device, loss_fn,
                args.log_interval, logger,
            )
            improved = values['psnr'] > best_val[2]
            best_val = (
                best_val[0], best_val[1],
                max(best_val[2], values['psnr']),
            )
            writer.add_scalar('val/full/psnr', values['psnr'], epoch + 1)
            writer.add_scalar('val/full/msssim', values['msssim'], epoch + 1)
            logger.info(
                'Val Full | distortion: %.6f | PSNR: %.2f dB | '
                'BEST: %.2f dB | MS-SSIM: %.4f',
                values['distortion'], values['psnr'],
                best_val[2], values['msssim'],
            )
            if improved:
                _save_checkpoint(
                    os.path.join(log_dir, 'full_only_best.pth'),
                    model, optimizer, epoch, best_train, best_val, args,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            _save_checkpoint(
                os.path.join(log_dir, 'full_only_latest.pth'),
                model, optimizer, epoch, best_train, best_val, args,
            )

    writer.close()
    logger.info(
        'Full-only 能力上限训练完成，总耗时 %.2f 小时',
        (time.time() - training_start) / 3600,
    )
    logger.info(
        '能力上限判定: %s | BEST VAL: %.2f dB | 目标: %.2f dB',
        'PASS' if best_val[2] >= args.target_psnr else 'BELOW TARGET',
        best_val[2],
        args.target_psnr,
    )


def build_parser():
    parser = build_baseline_parser()
    parser.description = '层级模型 Full-only 能力上限实验'
    parser.set_defaults(
        out_dir='./output/test/full_only',
        exp_name='hierarchical_full_only',
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--branch_dim', type=int,
        help='E1/E2 适配器瓶颈宽度，默认 hidden_dim/16',
    )
    parser.add_argument(
        '--level_group_sizes', type=int, nargs=3, default=[7, 2, 1],
    )
    parser.add_argument(
        '--target_psnr', type=float, default=36.8,
        help='用于能力上限判定的 Full PSNR 目标，默认 36.8 dB',
    )
    return parser


def main():
    train_full_only(build_parser().parse_args())


if __name__ == '__main__':
    main()

