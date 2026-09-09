import argparse
import logging
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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
from util import NervLoss


CHECKPOINT_CONFIG_KEYS = MODEL_CONFIG_KEYS + (
    'target_gop', 'epochs', 'lr', 'warmup', 'weight_decay',
)


def _random_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state):
    if not state:
        logging.warning(
            "检查点未保存随机状态，恢复结果可能不完全一致"
        )
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def _gop_metadata(dataset):
    return {
        'gop_size': dataset.gop_size,
        'num_gops': dataset.num_gops,
        'total_frames': dataset.total_frames,
    }


def _validate_saved_config(saved, current):
    if saved is None:
        raise ValueError("GOP 上限检查点缺少配置")
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
        raise ValueError("GOP 上限检查点配置不一致: " + '; '.join(mismatches))


def save_checkpoint(model, optimizer, epoch, best_val_psnr,
                    best_train_psnr, initial_psnr, args, dataset, path):
    torch.save({
        'checkpoint_version': 1,
        'checkpoint_type': 'gop_full_finetune_upper_bound',
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
        raise FileNotFoundError("未找到 GOP 上限检查点: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('checkpoint_type') != 'gop_full_finetune_upper_bound':
        raise ValueError("检查点不是 GOP 全参数微调上限模型")
    if checkpoint.get('architecture') != model.architecture:
        raise ValueError("GOP 上限模型架构与检查点不一致")
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


def _target_entry(entries, target_gop):
    for entry in entries:
        if entry.gop_index == target_gop:
            return entry
    raise ValueError("target_gop {} is outside the video".format(target_gop))


def _log_evaluation(prefix, result, target_gop):
    psnr, msssim, frames = result.per_gop[target_gop]
    logging.info(
        "%s | GOP %d | frames: %d | PSNR: %.2f dB | MS-SSIM: %.4f",
        prefix, target_gop, frames, psnr, msssim,
    )


def run(args):
    _validate_args(args)
    seed_everything(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(output_dir)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("使用设备: %s", device)
    logger.info("诊断模式: 后续 GOP 全参数微调上限")
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

    model = build_model(args, device)
    load_anchor_checkpoint(
        args.anchor_checkpoint, model, device, args, dataset=train_dataset,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        "已加载 GOP 0 锚点: %s | 全模型可训练参数: %d",
        args.anchor_checkpoint, trainable_parameters,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
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
            raise ValueError("GOP 上限训练已经达到 epochs")
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
            'UPPER BOUND', args.seed + 200000,
        )
        best_train_psnr = max(best_train_psnr, metrics.psnr)
        writer.add_scalar('upper_bound/train_loss', metrics.loss, epoch + 1)
        writer.add_scalar('upper_bound/train_psnr', metrics.psnr, epoch + 1)
        writer.add_scalar(
            'upper_bound/best_train_psnr', best_train_psnr, epoch + 1,
        )
        writer.add_scalar(
            'upper_bound/train_msssim', metrics.msssim, epoch + 1,
        )
        logger.info(
            "UPPER BOUND Epoch %d | GOP: %d | loss: %.6f | "
            "PSNR: %.2f dB | BEST TRAIN: %.2f dB | MS-SSIM: %.4f | "
            "time: %.2fs",
            epoch + 1, args.target_gop, metrics.loss, metrics.psnr,
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
                'upper_bound/val_psnr', result.psnr, epoch + 1,
            )
            writer.add_scalar(
                'upper_bound/val_msssim', result.msssim, epoch + 1,
            )
            _log_evaluation('UPPER BOUND VAL', result, args.target_gop)
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
        "UPPER BOUND COMPLETE | GOP %d | initial: %.2f dB | "
        "best train: %.2f dB | best val: %.2f dB",
        args.target_gop, initial_psnr, best_train_psnr, best_val_psnr,
    )
    writer.close()


def make_parser():
    parser = argparse.ArgumentParser(
        description="从 GOP 0 锚点出发，对指定后续 GOP 做全参数微调",
    )
    parser.add_argument('-d', '--data_root', required=True)
    parser.add_argument('--target_gop', type=int, default=2)
    parser.add_argument('--anchor_checkpoint', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--gop_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--warmup', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--grid_levels', type=int, default=10)
    parser.add_argument('--grid_feat_dim', type=int, default=6)
    parser.add_argument('--base_resolution', type=int, default=20)
    parser.add_argument('--finest_resolution', type=int, default=100)
    parser.add_argument('--aspect_ratio', type=int, nargs=2, default=[16, 9])
    parser.add_argument('--time_scale', type=float, default=0.1)
    parser.add_argument('--pe_freq', type=int, default=14)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--base_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--fixed_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--dynamic_res', action='store_true')
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--max_scale', type=float, default=1.2)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_dir', default='./output/gop_upper_bound')
    parser.add_argument('--exp_name', default='gop_upper_bound')
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--resume')
    parser.add_argument(
        '--loss_type', default='L2',
        choices=(
            'L2', 'L1', 'SSIM', 'Fusion1', 'Fusion2', 'Fusion3', 'Fusion4',
            'Fusion5', 'Fusion6', 'Fusion7', 'Fusion8', 'Fusion9', 'Fusion10',
            'Fusion11', 'Fusion12',
        ),
    )
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
