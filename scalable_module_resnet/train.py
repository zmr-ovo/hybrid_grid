"""Training entry for gated hierarchical scalable training.

Every batch trains B, Mid and Full on the same frame. Prefixes are recomputed
and backpropagated sequentially, so all three see the complete dataset without
keeping three full-resolution autograd graphs in memory.
"""

import logging
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import DynamicVideoDataset
from resnet_baseline.train import build_parser as build_baseline_parser
from scalable_generation.hierarchy import GridHierarchy
from train import seed_everything, seed_worker, setup_logging
from util import NervLoss

from .model import ModuleScalableResNet
from .training import (
    PREFIX_NAMES,
    accumulate_prefix_gradients,
    average_sums,
    empty_sums,
    evaluate_prefixes,
    normalize_prefix_weights,
    update_sums,
)


def hierarchy_from_sizes(sizes, level_count):
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError('level_group_sizes must contain positive integers')
    if len(sizes) != 3 or sum(sizes) != level_count:
        raise ValueError(
            'module-wise model requires three groups covering every Grid level'
        )
    groups = []
    start = 0
    for size in sizes:
        groups.append(tuple(range(start, start + size)))
        start += size
    return GridHierarchy(tuple(groups))


def _make_loaders(args):
    fixed_res = tuple(args.fixed_res)
    train_dataset = DynamicVideoDataset(
        args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=None if args.dynamic_res else fixed_res,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        frame_interval=args.frame_interval,
    )
    val_dataset = DynamicVideoDataset(
        args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=fixed_res,
        frame_interval=args.frame_interval,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader


def _learning_rate(optimizer, args, epoch, batch_index, num_batches):
    position = epoch + batch_index / num_batches
    warmup_epochs = int(args.warmup * args.epochs)
    if warmup_epochs and position < warmup_epochs:
        multiplier = 0.1 + 0.9 * position / warmup_epochs
    else:
        denominator = max(args.epochs - warmup_epochs, 1)
        progress = (position - warmup_epochs) / denominator
        multiplier = 0.5 * (1.0 + math.cos(progress * math.pi))
    learning_rate = args.lr * multiplier
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    return learning_rate


def _save_checkpoint(path, model, optimizer, epoch, best_train, best_val, args):
    torch.save({
        'checkpoint_version': 1,
        'architecture': model.architecture,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_train_psnr': tuple(best_train),
        'best_val_psnr': tuple(best_val),
        'config': dict(vars(args)),
    }, path)


def _load_checkpoint(path, model, optimizer, device):
    if not os.path.isfile(path):
        raise FileNotFoundError('未找到渐进模型检查点: ' + path)
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('architecture') != model.architecture:
        raise ValueError('检查点与 module-wise ResNet 架构不一致')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value) and name != 'step':
                state[name] = value.to(device)
    return (
        int(checkpoint.get('epoch', -1)) + 1,
        tuple(checkpoint.get('best_train_psnr', (0.0, 0.0, 0.0))),
        tuple(checkpoint.get('best_val_psnr', (0.0, 0.0, 0.0))),
    )


def train_module_scalable(args):
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError('dynamic resolution requires batch_size=1')
    if args.epochs < 1 or args.workers < 0:
        raise ValueError('epochs must be positive and workers non-negative')
    prefix_weights = normalize_prefix_weights(args.prefix_weights)
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
    logger.info(
        '训练结构: 分档 Grid/Gate/时间调制 + Base Additive-Skip '
        '+ E1/E2 轻量适配器（无压缩）'
    )
    logger.info('层级分组: %s', hierarchy.groups)
    logger.info('前缀权重（归一化）: %s', prefix_weights)

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

    budgets = model.parameter_budget()
    logger.info('模型总参数: %d', sum(p.numel() for p in model.parameters()))
    for name, budget in zip(PREFIX_NAMES, budgets):
        logger.info('%s 累计参数: %d (%.4f M)', name, budget, budget / 1e6)
    logger.info('Grid 参数: %d', sum(
        p.numel() for p in model.grid_encoder.parameters()
    ))
    logger.info('非 Grid 参数: %d',
                sum(p.numel() for p in model.parameters()) -
                sum(p.numel() for p in model.grid_encoder.parameters()))
    for key, value in vars(args).items():
        logger.info('%s: %s', key, value)

    start_epoch = 0
    best_train = (0.0, 0.0, 0.0)
    best_val = (0.0, 0.0, 0.0)
    if args.resume:
        start_epoch, best_train, best_val = _load_checkpoint(
            args.resume, model, optimizer, device,
        )
        logger.info('从 epoch %d 恢复训练', start_epoch + 1)

    if args.eval_only:
        if not args.resume:
            raise ValueError('eval_only requires --resume')
        evaluation = evaluate_prefixes(
            model,
            val_loader,
            device,
            loss_fn,
            args.log_interval,
            logger,
        )
        for name, values in zip(PREFIX_NAMES, evaluation):
            logger.info(
                'Eval %s | distortion: %.6f | PSNR: %.2f dB | '
                'MS-SSIM: %.4f',
                name, values['distortion'], values['psnr'], values['msssim'],
            )
        writer.close()
        return

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        sums = empty_sums()
        samples = 0
        for batch_index, batch in enumerate(train_loader):
            coords = batch['coords'].to(device, non_blocking=True)
            pixels = batch['pixels'].to(device, non_blocking=True)
            batch_size = pixels.shape[0]
            learning_rate = _learning_rate(
                optimizer, args, epoch, batch_index, len(train_loader),
            )
            optimizer.zero_grad(set_to_none=True)
            metrics, joint_loss = accumulate_prefix_gradients(
                model,
                coords,
                pixels,
                loss_fn,
                prefix_weights,
                device,
            )
            optimizer.step()
            update_sums(sums, metrics, batch_size)
            samples += batch_size

            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                averages = average_sums(sums, samples)
                logger.info(
                    'epoch [%d/%d] batch [%d/%d] | joint: %.6f | %s | lr: %.3e',
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    joint_loss,
                    ' | '.join(
                        '{} D: {:.6f} PSNR: {:.2f}'.format(
                            name, values['distortion'], values['psnr'],
                        )
                        for name, values in zip(PREFIX_NAMES, averages)
                    ),
                    learning_rate,
                )

        averages = average_sums(sums, samples)
        best_train = tuple(
            max(best, values['psnr'])
            for best, values in zip(best_train, averages)
        )
        for name, values, best in zip(PREFIX_NAMES, averages, best_train):
            tag = name.lower().replace('+', '_plus_')
            writer.add_scalar('train/{}/distortion'.format(tag),
                              values['distortion'], epoch + 1)
            writer.add_scalar('train/{}/psnr'.format(tag),
                              values['psnr'], epoch + 1)
            writer.add_scalar('train/{}/msssim'.format(tag),
                              values['msssim'], epoch + 1)
            writer.add_scalar('train/{}/best_psnr'.format(tag),
                              best, epoch + 1)
            logger.info(
                'Train %s | distortion: %.6f | PSNR: %.2f dB | '
                'BEST: %.2f dB | MS-SSIM: %.4f',
                name, values['distortion'], values['psnr'], best,
                values['msssim'],
            )
        writer.add_scalar('train/lr', learning_rate, epoch + 1)
        writer.add_scalar('time/epoch_sec', time.time() - epoch_start, epoch + 1)

        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            evaluation = evaluate_prefixes(
                model,
                val_loader,
                device,
                loss_fn,
                args.log_interval,
                logger,
            )
            full_improved = evaluation[-1]['psnr'] > best_val[-1]
            best_val = tuple(
                max(best, values['psnr'])
                for best, values in zip(best_val, evaluation)
            )
            for name, values, best in zip(PREFIX_NAMES, evaluation, best_val):
                tag = name.lower().replace('+', '_plus_')
                writer.add_scalar('val/{}/psnr'.format(tag),
                                  values['psnr'], epoch + 1)
                writer.add_scalar('val/{}/msssim'.format(tag),
                                  values['msssim'], epoch + 1)
                logger.info(
                    'Val %s | distortion: %.6f | PSNR: %.2f dB | '
                    'BEST: %.2f dB | MS-SSIM: %.4f',
                    name, values['distortion'], values['psnr'], best,
                    values['msssim'],
                )
            if full_improved:
                _save_checkpoint(
                    os.path.join(log_dir, 'module_resnet_best_full.pth'),
                    model, optimizer, epoch, best_train, best_val, args,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            _save_checkpoint(
                os.path.join(log_dir, 'module_resnet_latest.pth'),
                model, optimizer, epoch, best_train, best_val, args,
            )

    total_seconds = time.time() - training_start
    writer.close()
    logger.info('按模块渐进训练完成，总耗时 %.2f 小时', total_seconds / 3600)


def build_parser():
    parser = build_baseline_parser()
    parser.description = 'Gate/时间调制/Decoder 分档的三前缀训练（无压缩）'
    parser.set_defaults(
        out_dir='./output/scalable_module_resnet',
        exp_name='scalable_hierarchical_additive_skip',
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--branch_dim', type=int,
        help='E1/E2 轻量适配器瓶颈宽度，默认 hidden_dim/16',
    )
    parser.add_argument(
        '--level_group_sizes', type=int, nargs=3, default=[7, 2, 1],
    )
    parser.add_argument(
        '--prefix_weights', type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help='B、Mid、Full 的失真权重；训练时自动归一化',
    )
    return parser


def main():
    train_module_scalable(build_parser().parse_args())


if __name__ == '__main__':
    main()
