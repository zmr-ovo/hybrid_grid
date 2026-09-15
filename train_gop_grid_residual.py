import argparse
import logging
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from gop_lora import (
    GOPGridResidualHybridGridNet,
    GOPLowRankGridHybridGridNet,
    GOPStructuredGridHybridGridNet,
    freeze_for_gop,
    gop_lora_layers,
    gop_parameters,
)
from train import seed_everything, setup_logging
from train_gop_lora import (
    MODEL_CONFIG_KEYS,
    build_model,
    evaluate,
    load_anchor_checkpoint,
    make_datasets,
    make_gop_loaders,
    train_epoch,
)
from train_gop_upper_bound import (
    _gop_metadata,
    _log_evaluation,
    _random_state,
    _restore_random_state,
    _target_entry,
    make_parser as make_base_parser,
)
from util import NervLoss


CHECKPOINT_CONFIG_KEYS = MODEL_CONFIG_KEYS + (
    'target_gop', 'rank', 'alpha', 'epochs', 'lr', 'warmup',
    'weight_decay', 'grid_rank', 'grid_alpha', 'grid_init',
    'lora_target', 'grid_lr', 'network_lora_lr', 'grid_adapter',
)


def _grid_adapter(args):
    if args.grid_adapter != 'auto':
        return args.grid_adapter
    return 'full' if args.grid_rank == 0 else 'matrix'


def _validate_saved_config(saved, current):
    if saved is None:
        raise ValueError("Grid 残差检查点缺少配置")
    mismatches = []
    for key in CHECKPOINT_CONFIG_KEYS:
        if key not in saved:
            continue
        saved_value = saved[key]
        current_value = getattr(current, key)
        if isinstance(saved_value, list):
            saved_value = tuple(saved_value)
        if isinstance(current_value, list):
            current_value = tuple(current_value)
        if saved_value != current_value:
            mismatches.append(
                '{}: checkpoint={}, current={}'.format(
                    key, saved_value, current_value,
                )
            )
    if mismatches:
        raise ValueError(
            "Grid 残差检查点配置不一致: " + '; '.join(mismatches)
        )


def save_checkpoint(model, optimizer, epoch, best_val_psnr,
                    best_train_psnr, initial_psnr, args, dataset, path):
    torch.save({
        'checkpoint_version': 1,
        'checkpoint_type': 'gop_grid_adapter_network_lora',
        'architecture': model.architecture,
        'target_gop': args.target_gop,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_psnr': best_val_psnr,
        'best_train_psnr': best_train_psnr,
        'initial_psnr': initial_psnr,
        'config': dict(vars(args)),
        'gop': _gop_metadata(dataset),
        'random_state': _random_state(),
    }, path)


def load_checkpoint(path, model, optimizer, device, args, dataset):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "未找到 Grid 残差检查点: {}".format(path)
        )
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('checkpoint_type') not in (
        'gop_grid_residual_decoder_lora',
        'gop_grid_adapter_network_lora',
    ):
        raise ValueError("检查点不是 GOP Grid 适配 + 网络 LoRA 模型")
    if checkpoint.get('architecture') != model.architecture:
        raise ValueError("Grid 残差模型架构与检查点不一致")
    if checkpoint.get('target_gop') != args.target_gop:
        raise ValueError("检查点的目标 GOP 不一致")
    if checkpoint.get('gop') != _gop_metadata(dataset):
        raise ValueError("GOP 划分与检查点不一致")
    _validate_saved_config(checkpoint.get('config'), args)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    _restore_random_state(checkpoint.get('random_state'))
    return (
        int(checkpoint.get('epoch', -1)) + 1,
        float(checkpoint.get('best_val_psnr', 0.0)),
        float(checkpoint.get('best_train_psnr', 0.0)),
        float(checkpoint.get('initial_psnr', 0.0)),
    )


def _validate_args(args):
    if args.target_gop < 1:
        raise ValueError("target_gop must identify a later GOP (>= 1)")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not 0 <= args.warmup < 1:
        raise ValueError("warmup must be in [0, 1)")
    if args.rank < 1:
        raise ValueError("rank must be positive")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if args.grid_rank < 0:
        raise ValueError("grid_rank must be non-negative")
    adapter = _grid_adapter(args)
    if adapter == 'full' and args.grid_rank != 0:
        raise ValueError("full Grid residual requires grid_rank=0")
    if adapter in ('matrix', 'structured') and args.grid_rank < 1:
        raise ValueError(
            "matrix and structured Grid LoRA require grid_rank >= 1"
        )
    if adapter == 'structured' and args.grid_init != 'random':
        raise ValueError(
            "structured Grid LoRA currently uses random initialization"
        )
    if not math.isfinite(args.grid_alpha) or args.grid_alpha <= 0:
        raise ValueError("grid_alpha must be finite and positive")
    for name in ('grid_lr', 'network_lora_lr'):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value) or value <= 0
        ):
            raise ValueError("{} must be finite and positive".format(name))
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if args.gop_size < 1 or args.frame_interval < 1:
        raise ValueError("gop_size and frame_interval must be positive")
    if args.frame_interval > args.gop_size:
        raise ValueError("frame_interval must not exceed gop_size")
    if min(args.eval_freq, args.save_interval, args.log_interval) < 1:
        raise ValueError("log, save and eval intervals must be positive")
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")


def run(args):
    _validate_args(args)
    seed_everything(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(output_dir)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("使用设备: %s", device)
    logger.info("训练结构: GOP Grid 适配 + Decoder LoRA")
    logger.info("随机种子: %d", args.seed)
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    train_dataset, val_dataset = make_datasets(args)
    train_entry = _target_entry(
        make_gop_loaders(train_dataset, args, shuffle=True),
        args.target_gop,
    )
    val_entry = _target_entry(
        make_gop_loaders(val_dataset, args, shuffle=False),
        args.target_gop,
    )
    logger.info(
        "视频帧: %d | GOP size: %d | GOP 数量: %d | 目标 GOP: %d | "
        "目标帧数: %d",
        train_dataset.total_frames,
        train_dataset.gop_size,
        train_dataset.num_gops,
        args.target_gop,
        len(train_entry.loader.dataset),
    )

    anchor = build_model(args, device)
    load_anchor_checkpoint(
        args.anchor_checkpoint, anchor, device, args, dataset=train_dataset,
    )
    adapter = _grid_adapter(args)
    model_class = {
        'full': GOPGridResidualHybridGridNet,
        'matrix': GOPLowRankGridHybridGridNet,
        'structured': GOPStructuredGridHybridGridNet,
    }[adapter]
    model_arguments = {
        'shared_model': anchor,
        'num_gops': train_dataset.num_gops,
        'adapted_gops': (args.target_gop,),
        'rank': args.rank,
        'alpha': args.alpha,
        'lora_target': args.lora_target,
    }
    if adapter in ('matrix', 'structured'):
        model_arguments.update({
            'grid_rank': args.grid_rank,
            'grid_alpha': args.grid_alpha,
        })
    if adapter == 'matrix':
        model_arguments['grid_init'] = args.grid_init
    model = model_class(**model_arguments).to(device)
    phase_name = {
        'full': 'GRID RESIDUAL',
        'matrix': 'MATRIX GRID LORA',
        'structured': 'STRUCTURED GRID LORA',
    }[adapter]
    trainable = freeze_for_gop(model, args.target_gop)
    network_parameters = gop_parameters(model, args.target_gop)
    grid_parameters = model.grid_parameters(args.target_gop)
    grid_lr = args.lr if args.grid_lr is None else args.grid_lr
    network_lora_lr = (
        args.lr if args.network_lora_lr is None
        else args.network_lora_lr
    )
    logger.info(
        "模型总参数: %d | 目标 GOP 可训练参数: %d | "
        "Grid 适配: %d | 网络 LoRA: %d",
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in trainable),
        sum(parameter.numel() for parameter in grid_parameters),
        sum(parameter.numel() for parameter in network_parameters),
    )
    logger.info(
        "LoRA 注入范围: %s | Grid lr: %.3e | 网络 LoRA lr: %.3e",
        args.lora_target, grid_lr, network_lora_lr,
    )
    for name, layer in zip(model.injected_layers, gop_lora_layers(model)):
        logger.info(
            "Network LoRA layer: %s | effective rank: %d | "
            "alpha: %.4g | scaling: %.4g",
            name, layer.rank, layer.alpha, layer.scaling,
        )
    if adapter == 'full':
        logger.info("Grid 适配类型: 完整残差")
    else:
        logger.info(
            "Grid 适配类型: %s LoRA | requested rank: %d | alpha: %.4g | "
            "initialization: %s",
            adapter, args.grid_rank, args.grid_alpha, args.grid_init,
        )
        for index, level in enumerate(
            model.grid_lora_levels(args.target_gop)
        ):
            logger.info(
                "Grid LoRA level %d | shape: %s | effective rank: %d | "
                "alpha: %.4g | scaling: %.4g | params: %d",
                index,
                level.grid_shape,
                level.rank,
                level.alpha,
                level.scaling,
                sum(
                    parameter.numel()
                    for parameter in level.lora_parameters()
                ),
            )
            if adapter == 'structured':
                logger.info(
                    "Grid LoRA level %d factors | spatial: %d | "
                    "temporal: %d | channel: %d",
                    index,
                    level.spatial.numel(),
                    level.temporal.numel(),
                    level.channel.numel(),
                )

    if args.grid_lr is None and args.network_lora_lr is None:
        optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW([
            {
                'params': grid_parameters,
                'lr': grid_lr,
                'base_lr': grid_lr,
            },
            {
                'params': network_parameters,
                'lr': network_lora_lr,
                'base_lr': network_lora_lr,
            },
        ], weight_decay=args.weight_decay)
    loss_fn = NervLoss(args.loss_type, device)
    start_epoch = 0
    best_val_psnr = 0.0
    best_train_psnr = 0.0
    initial_psnr = 0.0

    if args.resume:
        (start_epoch, best_val_psnr, best_train_psnr,
         initial_psnr) = load_checkpoint(
            args.resume, model, optimizer, device, args, train_dataset,
        )
        if start_epoch >= args.epochs:
            raise ValueError("Grid 残差训练已经达到 epochs")
        logger.info("从 epoch %d 恢复", start_epoch + 1)
    else:
        initial = evaluate(
            model, (val_entry,), device, args.log_interval,
        )
        initial_psnr = initial.psnr
        best_val_psnr = initial.psnr
        _log_evaluation('INITIAL TRANSFER', initial, args.target_gop)
        save_checkpoint(
            model, optimizer, -1, best_val_psnr, best_train_psnr,
            initial_psnr, args, train_dataset,
            os.path.join(output_dir, 'gop_{}_best.pth'.format(
                args.target_gop,
            )),
        )

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        metrics = train_epoch(
            model, (train_entry,), optimizer, loss_fn, device, epoch,
            args.epochs, args.lr, args.warmup, args.log_interval,
            phase_name, args.seed + 300000,
        )
        best_train_psnr = max(best_train_psnr, metrics.psnr)
        writer.add_scalar('grid_residual/train_loss', metrics.loss, epoch + 1)
        writer.add_scalar('grid_residual/train_psnr', metrics.psnr, epoch + 1)
        writer.add_scalar(
            'grid_residual/best_train_psnr', best_train_psnr, epoch + 1,
        )
        writer.add_scalar(
            'grid_residual/train_msssim', metrics.msssim, epoch + 1,
        )
        logger.info(
            "%s Epoch %d | GOP: %d | loss: %.6f | "
            "PSNR: %.2f dB | BEST TRAIN: %.2f dB | MS-SSIM: %.4f | "
            "time: %.2fs",
            phase_name, epoch + 1, args.target_gop,
            metrics.loss, metrics.psnr,
            best_train_psnr, metrics.msssim, time.time() - start,
        )

        should_evaluate = (
            (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1
        )
        if should_evaluate:
            result = evaluate(
                model, (val_entry,), device, args.log_interval,
            )
            writer.add_scalar(
                'grid_residual/val_psnr', result.psnr, epoch + 1,
            )
            writer.add_scalar(
                'grid_residual/val_msssim', result.msssim, epoch + 1,
            )
            _log_evaluation(
                phase_name + ' VAL', result, args.target_gop,
            )
            if result.psnr > best_val_psnr:
                best_val_psnr = result.psnr
                save_checkpoint(
                    model, optimizer, epoch, best_val_psnr,
                    best_train_psnr, initial_psnr, args, train_dataset,
                    os.path.join(output_dir, 'gop_{}_best.pth'.format(
                        args.target_gop,
                    )),
                )

        if ((epoch + 1) % args.save_interval == 0 or
                epoch == args.epochs - 1):
            save_checkpoint(
                model, optimizer, epoch, best_val_psnr, best_train_psnr,
                initial_psnr, args, train_dataset,
                os.path.join(output_dir, 'gop_{}_latest.pth'.format(
                    args.target_gop,
                )),
            )

    logger.info(
        "%s COMPLETE | GOP %d | initial: %.2f dB | "
        "best train: %.2f dB | best val: %.2f dB",
        phase_name, args.target_gop, initial_psnr,
        best_train_psnr, best_val_psnr,
    )
    writer.close()


def make_parser():
    parser = make_base_parser()
    parser.description = (
        "共享 GOP 0 Grid，训练指定 GOP 的 Grid 适配和 Decoder LoRA"
    )
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument(
        '--grid_rank', type=int, default=0,
        help='Grid LoRA rank；auto 模式下 0 表示完整 Grid 残差',
    )
    parser.add_argument(
        '--grid_adapter',
        choices=('auto', 'full', 'matrix', 'structured'),
        default='auto',
        help='Grid 适配结构；auto 保持已有 grid_rank 行为',
    )
    parser.add_argument('--grid_alpha', type=float, default=1.0)
    parser.add_argument(
        '--grid_init', choices=('random', 'anchor_pca'), default='random',
    )
    parser.add_argument(
        '--lora_target', choices=('decoder', 'all_linear'),
        default='decoder',
    )
    parser.add_argument('--grid_lr', type=float)
    parser.add_argument('--network_lora_lr', type=float)
    parser.set_defaults(
        out_dir='./output/gop_grid_residual',
        exp_name='gop_grid_residual',
    )
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
