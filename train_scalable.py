"""Train three deterministic Grid prefixes with a joint RD objective.

Scheme A evaluates B, B+E1 and Full for every input batch. During training,
the prefixes run and backpropagate sequentially to avoid retaining three
full-resolution graphs at once. Their gradients accumulate before one shared
optimizer step. Existing single-rate training and codec entry points are not
modified.
"""

import logging
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from compression.network_quantization import (
    configure_network_qat,
    network_qat_state,
)
from scalable_generation import (
    GridHierarchy,
    ScalableHybridGridNet,
)
from scalable_generation.training import (
    accumulate_prefix_gradients,
    average_metric_sums,
    empty_metric_sums,
    evaluate_prefixes,
    prefix_names,
    update_metric_sums_from_batch,
)
from train import seed_everything, setup_logging
from train_compression import (
    _make_loaders,
    _validate_args,
    build_compression_model,
    build_parser as build_compression_parser,
    compression_grid_metadata,
    compression_model_storage,
    compression_network_metadata_bits,
    compression_stage,
    set_cosine_learning_rate,
)
from util import NervLoss


def hierarchy_from_sizes(group_sizes, grid_levels):
    if not isinstance(group_sizes, (tuple, list)) or not group_sizes:
        raise ValueError("level_group_sizes must not be empty")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1
        for size in group_sizes
    ):
        raise ValueError("level_group_sizes must contain positive integers")
    if sum(group_sizes) != grid_levels:
        raise ValueError(
            "level_group_sizes must sum to grid_levels"
        )

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(tuple(range(start, start + size)))
        start += size
    return GridHierarchy(tuple(groups))


def _validate_scalable_args(args):
    _validate_args(args)
    hierarchy = hierarchy_from_sizes(
        args.level_group_sizes, args.grid_levels,
    )
    for name in ('distortion_weights', 'rate_weights'):
        weights = getattr(args, name)
        if len(weights) != hierarchy.num_quality_levels:
            raise ValueError(
                "{} must contain one value per quality level".format(name)
            )
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in weights
        ):
            raise ValueError("{} must be finite and positive".format(name))
    return hierarchy


def _load_initial_checkpoint(path, compressed_model, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("未找到初始化检查点: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('architecture') != compressed_model.architecture:
        raise ValueError(
            "初始化模型架构不一致: checkpoint={}, current={}".format(
                checkpoint.get('architecture'), compressed_model.architecture,
            )
        )
    compressed_model.load_state_dict(checkpoint['state_dict'])
    return int(checkpoint.get('epoch', -1)) + 1


def _checkpoint(model, optimizer, epoch, best_train, best_val, args):
    return {
        'checkpoint_version': 1,
        'architecture': 'hybrid_grid_scalable_scheme_a_v1',
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_train_psnr': list(best_train),
        'best_val_psnr': list(best_val),
        'hierarchy': [list(group) for group in model.hierarchy.groups],
        'config': dict(vars(args)),
    }


def _save_checkpoint(path, model, optimizer, epoch, best_train, best_val,
                     args):
    torch.save(
        _checkpoint(
            model, optimizer, epoch, best_train, best_val, args,
        ),
        path,
    )


def _load_resume(path, model, optimizer, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("未找到渐进训练检查点: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('architecture') != 'hybrid_grid_scalable_scheme_a_v1':
        raise ValueError("检查点不是 Scheme-A 渐进训练模型")
    if tuple(tuple(group) for group in checkpoint.get('hierarchy', ())) != (
        model.hierarchy.groups
    ):
        raise ValueError("检查点层级分组与当前配置不一致")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        step = state.get('step')
        if torch.is_tensor(step) and step.is_cuda:
            state['step'] = step.cpu()
    count = model.hierarchy.num_quality_levels
    best_train = tuple(checkpoint.get('best_train_psnr', [0.0] * count))
    best_val = tuple(checkpoint.get('best_val_psnr', [0.0] * count))
    return int(checkpoint.get('epoch', -1)) + 1, best_train, best_val


def _write_prefix_metrics(writer, split, names, metrics, epoch):
    for name, values in zip(names, metrics):
        tag = name.lower().replace('+', '_plus_')
        for key in ('distortion', 'rate', 'total', 'psnr', 'msssim',
                    'grid_bits'):
            writer.add_scalar(
                '{}/{}/{}'.format(split, tag, key), values[key], epoch,
            )


def train_scalable(args):
    hierarchy = _validate_scalable_args(args)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, train_generator = _make_loaders(args)

    compressed_model = build_compression_model(args, device)
    if args.init_checkpoint and args.resume:
        raise ValueError("init_checkpoint and resume cannot be used together")
    if args.init_checkpoint:
        source_epoch = _load_initial_checkpoint(
            args.init_checkpoint, compressed_model, device,
        )
        logger.info(
            "从单档压缩检查点 epoch %d 初始化", source_epoch,
        )

    model = ScalableHybridGridNet(compressed_model, hierarchy).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = NervLoss(loss_type=args.loss_type, device=device)
    names = prefix_names(hierarchy.num_quality_levels)
    distortion_weights = tuple(args.distortion_weights)
    rate_weights = tuple(args.rate_weights)
    total_video_pixels = (
        len(val_loader.dataset) * args.fixed_res[0] * args.fixed_res[1]
    )
    non_grid, entropy = compression_model_storage(compressed_model)
    metadata = compression_grid_metadata(compressed_model)
    network_metadata_bits = compression_network_metadata_bits(compressed_model)
    static_bits = (
        non_grid.total_bits + entropy.total_bits + metadata.total_bits
        + network_metadata_bits
    )

    logger.info("使用设备: %s", device)
    logger.info(
        "训练方案: Scheme A（三前缀顺序反向传播，每 batch 一次参数更新）"
    )
    logger.info("层级分组: %s", [list(group) for group in hierarchy.groups])
    logger.info("前缀失真权重: %s", list(distortion_weights))
    logger.info("前缀码率权重: %s", list(rate_weights))
    logger.info("联合损失: sum(w_D * distortion + lambda * w_R * rate)")
    logger.info("熵模型类型: %s", compressed_model.entropy_model_type)
    logger.info("Grid 量化步长: %s", list(compressed_model.quant_steps))
    logger.info("固定公共开销估计: %d bits", static_bits)
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    start_epoch = 0
    best_train = tuple(0.0 for _ in names)
    best_val = tuple(0.0 for _ in names)
    if args.resume:
        start_epoch, best_train, best_val = _load_resume(
            args.resume, model, optimizer, device,
        )
        logger.info("从 epoch %d 恢复渐进训练", start_epoch + 1)

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.network_qat:
            configure_network_qat(
                compressed_model,
                epoch >= args.network_quant_start_epoch,
                epoch >= args.network_quant_freeze_epoch,
            )
        train_generator.manual_seed(args.seed + epoch)
        stage = compression_stage(
            epoch,
            args.epochs,
            args.warmup_epochs,
            args.symbol_start_epoch,
            args.lambda_max,
        )
        sums = empty_metric_sums(hierarchy.num_quality_levels)
        samples = 0
        epoch_start = time.time()

        for batch_index, batch in enumerate(train_loader):
            coords = batch['coords'].to(device, non_blocking=True)
            pixels = batch['pixels'].to(device, non_blocking=True)
            batch_size = pixels.size(0)
            learning_rate = set_cosine_learning_rate(
                optimizer,
                args.lr,
                epoch,
                batch_index,
                len(train_loader),
                args.epochs,
            )
            optimizer.zero_grad(set_to_none=True)
            batch_metrics, joint_total = accumulate_prefix_gradients(
                model,
                coords,
                pixels,
                loss_fn,
                stage.lambda_rate,
                distortion_weights,
                rate_weights,
                stage.quant_mode,
                device,
            )
            optimizer.step()

            update_metric_sums_from_batch(sums, batch_metrics, batch_size)
            samples += batch_size
            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                averages = average_metric_sums(sums, samples)
                logger.info(
                    "epoch [%d/%d] batch [%d/%d] | mode: %s | "
                    "lambda: %.3e | joint: %.6f | %s | lr: %.3e",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    stage.quant_mode,
                    stage.lambda_rate,
                    joint_total,
                    " | ".join(
                        "{} D: {:.6f} R: {:.4f} PSNR: {:.2f}".format(
                            name,
                            values['distortion'],
                            values['rate'],
                            values['psnr'],
                        )
                        for name, values in zip(names, averages)
                    ),
                    learning_rate,
                )

        averages = average_metric_sums(sums, samples)
        best_train = tuple(
            max(previous, values['psnr'])
            for previous, values in zip(best_train, averages)
        )
        epoch_seconds = time.time() - epoch_start
        _write_prefix_metrics(writer, 'train', names, averages, epoch + 1)
        for name, values in zip(names, averages):
            tag = name.lower().replace('+', '_plus_')
            grid_bpp = values['grid_bits'] / total_video_pixels
            writer.add_scalar(
                'train/{}/grid_bpp'.format(tag), grid_bpp, epoch + 1,
            )
            writer.add_scalar(
                'train/{}/estimated_total_bits'.format(tag),
                values['grid_bits'] + static_bits,
                epoch + 1,
            )
            writer.add_scalar(
                'train/{}/estimated_total_bpp'.format(tag),
                (values['grid_bits'] + static_bits) / total_video_pixels,
                epoch + 1,
            )
        writer.add_scalar('train/joint_lambda', stage.lambda_rate, epoch + 1)
        writer.add_scalar('train/lr', learning_rate, epoch + 1)
        writer.add_scalar('time/epoch_sec', epoch_seconds, epoch + 1)
        logger.info(
            "Epoch %d | mode: %s | QAT: %s | joint lambda: %.3e | "
            "time: %.2fs",
            epoch + 1,
            stage.quant_mode,
            network_qat_state(compressed_model),
            stage.lambda_rate,
            epoch_seconds,
        )
        for name, values, best_value in zip(names, averages, best_train):
            logger.info(
                "Train %s | distortion: %.6f | rate/value: %.4f | "
                "loss: %.6f | Grid: %.0f bits (%.6f BPP) | "
                "estimated total: %.0f bits (%.6f BPP) | PSNR: %.2f dB | "
                "BEST: %.2f dB | MS-SSIM: %.4f",
                name,
                values['distortion'],
                values['rate'],
                values['total'],
                values['grid_bits'],
                values['grid_bits'] / total_video_pixels,
                values['grid_bits'] + static_bits,
                (values['grid_bits'] + static_bits) / total_video_pixels,
                values['psnr'],
                best_value,
                values['msssim'],
            )

        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            evaluation_mode = (
                'disabled' if stage.quant_mode == 'disabled' else 'symbols'
            )
            evaluation = evaluate_prefixes(
                model,
                val_loader,
                device,
                evaluation_mode,
                loss_fn,
                0.0 if evaluation_mode == 'disabled' else stage.lambda_rate,
                distortion_weights,
                rate_weights,
                total_video_pixels,
                args.log_interval,
                logger,
            )
            full_improved = evaluation[-1].psnr > best_val[-1]
            best_val = tuple(
                max(previous, item.psnr)
                for previous, item in zip(best_val, evaluation)
            )
            for name, item, best_value in zip(names, evaluation, best_val):
                tag = name.lower().replace('+', '_plus_')
                writer.add_scalar(
                    'val/{}/psnr'.format(tag), item.psnr, epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/distortion'.format(tag),
                    item.distortion,
                    epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/rate_per_value'.format(tag),
                    item.rate_per_value,
                    epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/grid_bits'.format(tag),
                    item.grid_bits,
                    epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/msssim'.format(tag), item.msssim, epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/grid_bpp'.format(tag), item.grid_bpp, epoch + 1,
                )
                writer.add_scalar(
                    'val/{}/estimated_total_bpp'.format(tag),
                    (item.grid_bits + static_bits) / total_video_pixels,
                    epoch + 1,
                )
            logger.info("Val epoch %d | mode: %s", epoch + 1, evaluation_mode)
            for name, item, best_value in zip(names, evaluation, best_val):
                logger.info(
                    "Val %s | distortion: %.6f | rate/value: %.4f | "
                    "loss: %.6f | Grid: %.0f bits (%.6f BPP) | "
                    "estimated total: %.0f bits (%.6f BPP) | "
                    "PSNR: %.2f dB | BEST: %.2f dB | MS-SSIM: %.4f",
                    name,
                    item.distortion,
                    item.rate_per_value,
                    item.total_loss,
                    item.grid_bits,
                    item.grid_bpp,
                    item.grid_bits + static_bits,
                    (item.grid_bits + static_bits) / total_video_pixels,
                    item.psnr,
                    best_value,
                    item.msssim,
                )
            if full_improved:
                _save_checkpoint(
                    os.path.join(log_dir, 'scalable_best_full.pth'),
                    model,
                    optimizer,
                    epoch,
                    best_train,
                    best_val,
                    args,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            _save_checkpoint(
                os.path.join(log_dir, 'scalable_latest.pth'),
                model,
                optimizer,
                epoch,
                best_train,
                best_val,
                args,
            )
    total_seconds = time.time() - training_start
    writer.add_scalar('time/total_sec', total_seconds, 0)
    writer.close()
    logger.info("渐进式联合训练完成，总耗时 %.2f 小时", total_seconds / 3600)


def build_parser():
    parser = build_compression_parser()
    parser.description = 'Hybrid Grid 三前缀联合率失真训练（Scheme A）'
    parser.set_defaults(
        out_dir='./output/scalable',
        exp_name='scalable_scheme_a',
    )
    parser.add_argument(
        '--level_group_sizes',
        type=int,
        nargs='+',
        default=[7, 2, 1],
        help='基础层和增强层各自包含的 Grid 层数，默认 7 2 1',
    )
    parser.add_argument(
        '--distortion_weights',
        type=float,
        nargs='+',
        default=[1.0, 1.0, 1.0],
        help='各前缀的失真权重，默认等权',
    )
    parser.add_argument(
        '--rate_weights',
        type=float,
        nargs='+',
        default=[1.0, 1.0, 1.0],
        help='各前缀的累计码率权重，默认等权',
    )
    parser.add_argument(
        '--init_checkpoint',
        help='可选的单档 compression checkpoint，仅加载模型参数作为初始化',
    )
    return parser


if __name__ == '__main__':
    train_scalable(build_parser().parse_args())
