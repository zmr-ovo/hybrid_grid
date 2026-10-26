"""Training entry for width-progressive Additive-Skip Hybrid Grid.

Every batch trains Base, Mid and Full on the same frame.  Prefixes are
recomputed and backpropagated sequentially, so the three widths see all frames
without retaining three full-resolution autograd graphs at once.
"""

import logging
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from resnet_baseline.train import build_parser as build_baseline_parser
from scalable_module_resnet.train import (
    _learning_rate,
    _load_checkpoint,
    _make_loaders,
    _save_checkpoint,
    hierarchy_from_sizes,
)
from scalable_module_resnet.training import (
    PREFIX_NAMES,
    accumulate_prefix_gradients,
    average_sums,
    empty_sums,
    evaluate_prefixes,
    normalize_prefix_weights,
    update_sums,
)
from train import seed_everything, setup_logging
from util import NervLoss

from .model import WidthScalableResNet


def train_width_scalable(args):
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
        '训练结构: 分层因果 Gate/时间调制 + 嵌套 Additive-Skip 通道前缀'
    )
    logger.info('Grid 层级分组: %s', hierarchy.groups)
    logger.info('Decoder 宽度比例: %s', tuple(args.width_ratios))
    logger.info('前缀权重（归一化）: %s', prefix_weights)

    train_loader, val_loader = _make_loaders(args)
    model = WidthScalableResNet(
        grid_levels=args.grid_levels,
        grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        time_scale=args.time_scale,
        pe_freq=args.pe_freq,
        hidden_dim=args.hidden_dim,
        width_ratios=tuple(args.width_ratios),
        hierarchy=hierarchy,
    ).to(device)
    logger.info('Decoder 输入前缀: %s', model.decoder.input_widths)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = NervLoss(args.loss_type, device)

    budgets = model.parameter_budget()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    grid_params = sum(
        parameter.numel() for parameter in model.grid_encoder.parameters()
    )
    logger.info('模型总参数: %d (%.4f M)', total_params, total_params / 1e6)
    logger.info('Grid 参数: %d', grid_params)
    logger.info('非 Grid 参数: %d', total_params - grid_params)
    for name, width, output_width, budget in zip(
        PREFIX_NAMES,
        model.decoder.hidden_widths,
        model.decoder.output_widths,
        budgets,
    ):
        logger.info(
            '%s | hidden width: %d | output width: %d | '
            '累计参数: %d (%.4f M)',
            name, width, output_width, budget, budget / 1e6,
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
        logger.info('从 epoch %d 恢复宽度渐进训练', start_epoch + 1)

    if args.eval_only:
        if not args.resume:
            raise ValueError('eval_only requires --resume')
        evaluation = evaluate_prefixes(
            model, val_loader, device, loss_fn,
            args.log_interval, logger,
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
                    'epoch [%d/%d] batch [%d/%d] | joint: %.6f | %s | '
                    'lr: %.3e',
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
        logger.info(
            'Epoch %d | widths: %s | time: %.2fs',
            epoch + 1,
            model.decoder.hidden_widths,
            time.time() - epoch_start,
        )
        for name, values, best in zip(PREFIX_NAMES, averages, best_train):
            tag = name.lower().replace('+', '_plus_')
            writer.add_scalar(
                'train/{}/distortion'.format(tag),
                values['distortion'], epoch + 1,
            )
            writer.add_scalar(
                'train/{}/psnr'.format(tag), values['psnr'], epoch + 1,
            )
            writer.add_scalar(
                'train/{}/msssim'.format(tag), values['msssim'], epoch + 1,
            )
            writer.add_scalar(
                'train/{}/best_psnr'.format(tag), best, epoch + 1,
            )
            logger.info(
                'Train %s | distortion: %.6f | PSNR: %.2f dB | '
                'BEST: %.2f dB | MS-SSIM: %.4f',
                name, values['distortion'], values['psnr'], best,
                values['msssim'],
            )
        writer.add_scalar('train/lr', learning_rate, epoch + 1)

        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            evaluation = evaluate_prefixes(
                model, val_loader, device, loss_fn,
                args.log_interval, logger,
            )
            full_improved = evaluation[-1]['psnr'] > best_val[-1]
            best_val = tuple(
                max(best, values['psnr'])
                for best, values in zip(best_val, evaluation)
            )
            for name, values, best in zip(PREFIX_NAMES, evaluation, best_val):
                tag = name.lower().replace('+', '_plus_')
                writer.add_scalar(
                    'val/{}/psnr'.format(tag), values['psnr'], epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/msssim'.format(tag), values['msssim'], epoch + 1,
                )
                logger.info(
                    'Val %s | distortion: %.6f | PSNR: %.2f dB | '
                    'BEST: %.2f dB | MS-SSIM: %.4f',
                    name, values['distortion'], values['psnr'], best,
                    values['msssim'],
                )
            if full_improved:
                _save_checkpoint(
                    os.path.join(log_dir, 'width_resnet_best_full.pth'),
                    model, optimizer, epoch, best_train, best_val, args,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            _save_checkpoint(
                os.path.join(log_dir, 'width_resnet_latest.pth'),
                model, optimizer, epoch, best_train, best_val, args,
            )

    writer.close()
    logger.info(
        '按宽度渐进训练完成，总耗时 %.2f 小时',
        (time.time() - training_start) / 3600,
    )


def build_parser():
    parser = build_baseline_parser()
    parser.description = '按嵌套通道宽度分层的三前缀 Additive-Skip 训练'
    parser.set_defaults(
        out_dir='./output/scalable_width_resnet',
        exp_name='scalable_width_additive_skip',
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--level_group_sizes', type=int, nargs=3, default=[7, 2, 1],
    )
    parser.add_argument(
        '--width_ratios', type=float, nargs=3, default=[0.5, 0.75, 1.0],
        help='Base、Mid、Full 的嵌套 Decoder 宽度比例',
    )
    parser.add_argument(
        '--prefix_weights', type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help='B、Mid、Full 的失真权重；训练时自动归一化',
    )
    return parser


def main():
    train_width_scalable(build_parser().parse_args())


if __name__ == '__main__':
    main()

