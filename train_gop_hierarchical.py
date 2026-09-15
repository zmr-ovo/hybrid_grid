import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime

import torch

from gop_lora import (
    all_gop_parameter_groups,
    assemble_all_gop_model,
    configure_common_training,
    configure_local_training,
)
from train import seed_everything, setup_logging
from train_gop_lora import (
    Evaluation,
    _log_evaluation,
    _random_state,
    _restore_random_state,
    _validate_gop_metadata,
    _validate_saved_config,
    build_model,
    evaluate,
    load_anchor_checkpoint,
    make_datasets,
    make_gop_loaders,
    train_epoch,
)
from util import NervLoss


HIERARCHICAL_CONFIG_KEYS = (
    'grid_levels', 'grid_feat_dim', 'base_resolution', 'finest_resolution',
    'aspect_ratio', 'time_scale', 'pe_freq', 'hidden_dim', 'gop_size',
    'common_rank', 'common_alpha', 'common_grid_rank',
    'common_grid_alpha', 'local_rank', 'local_alpha', 'local_grid_rank',
    'local_grid_alpha',
)


@dataclass(frozen=True)
class StageResult:
    best_train_psnr: float
    best_val_psnr: float
    best_epoch: int


def _positive_number(name, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("{} must be finite and positive".format(name))


def _validate_args(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if args.gop_size < 1 or args.frame_interval < 1:
        raise ValueError("gop_size and frame_interval must be positive")
    if args.frame_interval > args.gop_size:
        raise ValueError("frame_interval must not exceed gop_size")
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if min(args.common_epochs, args.local_epochs, args.short_gop_epochs,
           args.log_interval, args.eval_freq, args.save_interval) < 1:
        raise ValueError("epochs and intervals must be positive")
    for name in (
        'common_rank', 'common_grid_rank', 'local_rank', 'local_grid_rank',
    ):
        if getattr(args, name) < 1:
            raise ValueError("{} must be positive".format(name))
    for name in (
        'common_alpha', 'common_grid_alpha',
        'local_alpha', 'local_grid_alpha',
        'common_network_lr', 'common_grid_lr',
        'local_network_lr', 'local_grid_lr',
    ):
        _positive_number(name, getattr(args, name))
    for name in ('common_warmup', 'local_warmup'):
        value = getattr(args, name)
        if not 0 <= value < 1:
            raise ValueError("{} must be in [0, 1)".format(name))
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if args.target_params is not None and args.target_params < 1:
        raise ValueError("target_params must be positive")
    if args.resume and args.stage not in ('common', 'local'):
        raise ValueError("--resume requires --stage common or local")
    if args.stage == 'local' and not (
        args.hierarchical_checkpoint or args.resume
    ):
        raise ValueError(
            "local stage requires --hierarchical_checkpoint or --resume"
        )
    if args.stage == 'eval' and not args.hierarchical_checkpoint:
        raise ValueError("eval stage requires --hierarchical_checkpoint")
    if args.resume and args.stage == 'local' and (
        not args.target_gops or len(args.target_gops) != 1
    ):
        raise ValueError("resuming local training requires one --target_gops")


def make_optimizer(parameters, network_lr, grid_lr, weight_decay):
    """Create separate learning-rate groups for network and Grid LoRA."""
    return torch.optim.AdamW([
        {
            'params': parameters.network,
            'lr': network_lr,
            'base_lr': network_lr,
            'name': 'network_lora',
        },
        {
            'params': parameters.grid,
            'lr': grid_lr,
            'base_lr': grid_lr,
            'name': 'grid_lora',
        },
    ], weight_decay=weight_decay)


def _selected_evaluation(result, gop_indices):
    """Aggregate an Evaluation over an explicit set of GOPs."""
    selected = {}
    for gop_index in gop_indices:
        if gop_index not in result.per_gop:
            raise ValueError("validation is missing GOP {}".format(gop_index))
        selected[gop_index] = result.per_gop[gop_index]
    samples = sum(values[2] for values in selected.values())
    return Evaluation(
        psnr=sum(values[0] * values[2] for values in selected.values()) / samples,
        msssim=sum(
            values[1] * values[2] for values in selected.values()
        ) / samples,
        per_gop=selected,
    )


def _loader_states(entries):
    return {
        entry.gop_index: entry.generator.get_state()
        for entry in entries
    }


def _restore_loader_states(entries, states):
    if not states:
        logging.warning("检查点未保存 DataLoader 状态")
        return
    for entry in entries:
        state = states.get(entry.gop_index)
        if state is not None:
            entry.generator.set_state(state)


def save_hierarchical_checkpoint(model, optimizer, epoch, best_val_psnr,
                                 best_train_psnr, best_epoch, args, dataset,
                                 entries, stage, path, gop_index=None):
    torch.save({
        'checkpoint_version': 1,
        'checkpoint_type': 'gop_hierarchical',
        'architecture': model.architecture,
        'stage': stage,
        'gop_index': gop_index,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'best_val_psnr': best_val_psnr,
        'best_train_psnr': best_train_psnr,
        'best_epoch': best_epoch,
        'config': dict(vars(args)),
        'gop': {
            'gop_size': dataset.gop_size,
            'num_gops': dataset.num_gops,
            'total_frames': dataset.total_frames,
        },
        'random_state': _random_state(),
        'loader_states': _loader_states(entries),
    }, path)


def load_hierarchical_checkpoint(path, model, device, args, dataset,
                                 optimizer=None, entries=(),
                                 restore_training=False,
                                 expected_stage=None,
                                 expected_gop=None):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("未找到分层 GOP 检查点: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('checkpoint_type') != 'gop_hierarchical':
        raise ValueError("检查点不是分层 GOP-LoRA 模型")
    if checkpoint.get('architecture') != model.architecture:
        raise ValueError("分层模型架构与检查点不一致")
    _validate_saved_config(
        checkpoint.get('config'), args, HIERARCHICAL_CONFIG_KEYS,
    )
    _validate_gop_metadata(checkpoint, dataset)
    if expected_stage and checkpoint.get('stage') != expected_stage:
        raise ValueError(
            "检查点阶段应为 {}，实际为 {}".format(
                expected_stage, checkpoint.get('stage'),
            )
        )
    if expected_gop is not None and checkpoint.get('gop_index') != expected_gop:
        raise ValueError(
            "检查点 GOP 应为 {}，实际为 {}".format(
                expected_gop, checkpoint.get('gop_index'),
            )
        )
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        if checkpoint.get('optimizer') is None:
            raise ValueError("检查点不包含优化器状态")
        optimizer.load_state_dict(checkpoint['optimizer'])
    if restore_training:
        _restore_random_state(checkpoint.get('random_state'))
        _restore_loader_states(entries, checkpoint.get('loader_states'))
    return StageResult(
        best_train_psnr=float(checkpoint.get('best_train_psnr', 0.0)),
        best_val_psnr=float(checkpoint.get('best_val_psnr', 0.0)),
        best_epoch=int(checkpoint.get('best_epoch', -1)),
    ), int(checkpoint.get('epoch', -1)) + 1


def _should_run(epoch, epochs, interval):
    return (epoch + 1) % interval == 0 or epoch == epochs - 1


def _train_stage(model, entries, parameters, loss_fn, device, epochs,
                 network_lr, grid_lr, warmup, log_interval, phase, seed,
                 weight_decay, val_entries=(), selection_gops=(),
                 eval_freq=1, save_interval=1, output_dir=None, args=None,
                 dataset=None, checkpoint_stage=None, gop_index=None,
                 resume=None):
    optimizer = make_optimizer(
        parameters, network_lr, grid_lr, weight_decay,
    )
    best_train_psnr = 0.0
    best_val_psnr = 0.0
    best_epoch = -1
    start_epoch = 0
    if resume:
        restored, start_epoch = load_hierarchical_checkpoint(
            resume, model, device, args, dataset,
            optimizer=optimizer,
            entries=entries,
            restore_training=True,
            expected_stage=checkpoint_stage,
            expected_gop=gop_index,
        )
        best_train_psnr = restored.best_train_psnr
        best_val_psnr = restored.best_val_psnr
        best_epoch = restored.best_epoch
        logging.info("%s 从 epoch %d 恢复", phase, start_epoch + 1)
        if start_epoch >= epochs:
            raise ValueError("恢复的训练阶段已经达到指定 epochs")

    for epoch in range(start_epoch, epochs):
        start = time.time()
        metrics = train_epoch(
            model=model,
            entries=entries,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            epochs=epochs,
            base_lr=network_lr,
            warmup_ratio=warmup,
            log_interval=log_interval,
            phase=phase,
            seed=seed,
        )
        best_train_psnr = max(best_train_psnr, metrics.psnr)
        logging.info(
            "%s Epoch %d | loss: %.6f | PSNR: %.2f dB | "
            "BEST TRAIN: %.2f dB | MS-SSIM: %.4f | "
            "network lr: %.3e | Grid lr: %.3e | time: %.2fs",
            phase,
            epoch + 1,
            metrics.loss,
            metrics.psnr,
            best_train_psnr,
            metrics.msssim,
            optimizer.param_groups[0]['lr'],
            optimizer.param_groups[1]['lr'],
            time.time() - start,
        )

        if val_entries and _should_run(epoch, epochs, eval_freq):
            full_result = evaluate(
                model, val_entries, device, log_interval,
            )
            selected_result = _selected_evaluation(
                full_result, selection_gops,
            )
            _log_evaluation('{} VAL'.format(phase), full_result)
            logging.info(
                "%s SELECTED | GOPs: %s | PSNR: %.2f dB | "
                "MS-SSIM: %.4f",
                phase,
                ','.join(str(index) for index in selection_gops),
                selected_result.psnr,
                selected_result.msssim,
            )
            if selected_result.psnr > best_val_psnr:
                best_val_psnr = selected_result.psnr
                best_epoch = epoch
                if output_dir:
                    save_hierarchical_checkpoint(
                        model, optimizer, epoch, best_val_psnr,
                        best_train_psnr, best_epoch, args, dataset, entries,
                        checkpoint_stage,
                        os.path.join(output_dir, '{}_best.pth'.format(
                            checkpoint_stage if gop_index is None
                            else 'gop_{}'.format(gop_index)
                        )),
                        gop_index,
                    )

        if output_dir and _should_run(epoch, epochs, save_interval):
            save_hierarchical_checkpoint(
                model, optimizer, epoch, best_val_psnr,
                best_train_psnr, best_epoch, args, dataset, entries,
                checkpoint_stage,
                os.path.join(output_dir, '{}_latest.pth'.format(
                    checkpoint_stage if gop_index is None
                    else 'gop_{}'.format(gop_index)
                )),
                gop_index,
            )
    return StageResult(best_train_psnr, best_val_psnr, best_epoch)


def _later_entries(entries):
    return tuple(entry for entry in entries if entry.gop_index > 0)


def _target_entries(entries, target_gops):
    available = {entry.gop_index: entry for entry in entries}
    targets = (
        tuple(sorted(available))
        if target_gops is None
        else tuple(target_gops)
    )
    if len(targets) != len(set(targets)):
        raise ValueError("--target_gops must not contain duplicates")
    invalid = [index for index in targets if index == 0 or index not in available]
    if invalid:
        raise ValueError(
            "invalid later GOP indices: {}".format(invalid)
        )
    return tuple(available[index] for index in targets)


def _local_epochs(entry, args, total_frames):
    start = entry.gop_index * args.gop_size
    frame_count = min(start + args.gop_size, total_frames) - start
    if frame_count < args.gop_size:
        return args.short_gop_epochs
    return args.local_epochs


def _log_parameter_counts(model, target_count=None):
    groups = all_gop_parameter_groups(model)
    logging.info(
        "参数统计 | 共享模型: %d | 公共网络 LoRA: %d | "
        "公共 Grid LoRA: %d | 全部独立 LoRA: %d | 模型总参数: %d",
        groups.shared_count,
        groups.common_network_count,
        groups.common_grid_count,
        groups.local_count,
        groups.total_count,
    )
    for adapter in groups.adapters:
        logging.info(
            "参数统计 | GOP %d | 独立网络 LoRA: %d | "
            "独立 Grid LoRA: %d | 合计: %d",
            adapter.gop_index,
            adapter.network_count,
            adapter.grid_count,
            adapter.total_count,
        )
    if target_count is not None:
        difference = groups.total_count - target_count
        logging.info(
            "参数预算 | 目标: %d | 实际: %d | 差值: %+d (%.3f%%)",
            target_count,
            groups.total_count,
            difference,
            100.0 * difference / target_count,
        )


def run(args):
    _validate_args(args)
    seed_everything(args.seed)
    if args.resume:
        output_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("使用设备: %s", device)
    logger.info("训练结构: 冻结共享模型 + 公共 LoRA + GOP 独立 LoRA")
    logger.info("运行阶段: %s", args.stage)
    logger.info("随机种子: %d", args.seed)
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    train_dataset, val_dataset = make_datasets(args)
    train_entries = make_gop_loaders(train_dataset, args, shuffle=True)
    val_entries = make_gop_loaders(val_dataset, args, shuffle=False)
    if len(train_entries) != train_dataset.num_gops:
        raise ValueError("帧采样导致部分 GOP 没有训练样本")
    if train_dataset.num_gops < 2:
        raise ValueError("分层 GOP-LoRA训练至少需要两个 GOP")
    later_train_entries = _later_entries(train_entries)
    target_train_entries = _target_entries(
        later_train_entries, args.target_gops,
    )
    logger.info(
        "视频帧: %d | GOP size: %d | GOP 数量: %d | "
        "公共 LoRA训练帧: %d",
        train_dataset.total_frames,
        train_dataset.gop_size,
        train_dataset.num_gops,
        sum(len(entry.loader.dataset) for entry in later_train_entries),
    )

    shared_model = build_model(args, device)
    load_anchor_checkpoint(
        args.anchor_checkpoint,
        shared_model,
        device,
        args,
        dataset=train_dataset,
    )
    model = assemble_all_gop_model(
        shared_model=shared_model,
        num_gops=train_dataset.num_gops,
        rank=args.local_rank,
        alpha=args.local_alpha,
        grid_rank=args.local_grid_rank,
        grid_alpha=args.local_grid_alpha,
        common_rank=args.common_rank,
        common_alpha=args.common_alpha,
        common_grid_rank=args.common_grid_rank,
        common_grid_alpha=args.common_grid_alpha,
    ).to(device)
    _log_parameter_counts(model, args.target_params)
    loss_fn = NervLoss(args.loss_type, device)

    if args.stage in ('local', 'eval'):
        if args.hierarchical_checkpoint:
            load_hierarchical_checkpoint(
                args.hierarchical_checkpoint, model, device, args,
                train_dataset,
            )
            logger.info(
                "已加载分层模型: %s", args.hierarchical_checkpoint,
            )

    if args.stage in ('all', 'common'):
        common_parameters = configure_common_training(model)
        logger.info(
            "开始公共 LoRA训练 | epochs: %d | network lr: %.3e | "
            "Grid lr: %.3e",
            args.common_epochs,
            args.common_network_lr,
            args.common_grid_lr,
        )
        _train_stage(
            model=model,
            entries=later_train_entries,
            parameters=common_parameters,
            loss_fn=loss_fn,
            device=device,
            epochs=args.common_epochs,
            network_lr=args.common_network_lr,
            grid_lr=args.common_grid_lr,
            warmup=args.common_warmup,
            log_interval=args.log_interval,
            phase='COMMON LORA',
            seed=args.seed + 400000,
            weight_decay=args.weight_decay,
            val_entries=val_entries,
            selection_gops=tuple(
                entry.gop_index for entry in later_train_entries
            ),
            eval_freq=args.eval_freq,
            save_interval=args.save_interval,
            output_dir=output_dir,
            args=args,
            dataset=train_dataset,
            checkpoint_stage='common',
            resume=args.resume if args.stage == 'common' else None,
        )
        common_best = os.path.join(output_dir, 'common_best.pth')
        load_hierarchical_checkpoint(
            common_best, model, device, args, train_dataset,
            expected_stage='common',
        )
        logger.info("独立 LoRA阶段使用最佳公共模型: %s", common_best)
        if args.stage == 'common':
            return

    if args.stage == 'eval':
        result = evaluate(
            model, val_entries, device, args.log_interval,
            os.path.join(output_dir, 'eval') if args.dump_images else None,
        )
        _log_evaluation('FINAL EVAL', result)
        return

    for entry in target_train_entries:
        epochs = _local_epochs(entry, args, train_dataset.total_frames)
        local_parameters = configure_local_training(model, entry.gop_index)
        logger.info(
            "开始独立 LoRA训练 | GOP: %d | frames: %d | epochs: %d | "
            "network lr: %.3e | Grid lr: %.3e",
            entry.gop_index, len(entry.loader.dataset), epochs,
            args.local_network_lr, args.local_grid_lr,
        )
        resume = args.resume if args.stage == 'local' else None
        _train_stage(
            model=model,
            entries=(entry,),
            parameters=local_parameters,
            loss_fn=loss_fn,
            device=device,
            epochs=epochs,
            network_lr=args.local_network_lr,
            grid_lr=args.local_grid_lr,
            warmup=args.local_warmup,
            log_interval=args.log_interval,
            phase='LOCAL LORA GOP {}'.format(entry.gop_index),
            seed=args.seed + 500000 + entry.gop_index * 10000,
            weight_decay=args.weight_decay,
            val_entries=val_entries,
            selection_gops=(entry.gop_index,),
            eval_freq=args.eval_freq,
            save_interval=args.save_interval,
            output_dir=output_dir,
            args=args,
            dataset=train_dataset,
            checkpoint_stage='local',
            gop_index=entry.gop_index,
            resume=resume,
        )
        best_path = os.path.join(
            output_dir, 'gop_{}_best.pth'.format(entry.gop_index),
        )
        load_hierarchical_checkpoint(
            best_path, model, device, args, train_dataset,
            expected_stage='local', expected_gop=entry.gop_index,
        )
        logger.info("后续阶段使用 GOP %d 最佳模型", entry.gop_index)
        args.resume = None

    final_result = evaluate(
        model, val_entries, device, args.log_interval,
        os.path.join(output_dir, 'final') if args.dump_images else None,
    )
    _log_evaluation('FINAL', final_result)
    save_hierarchical_checkpoint(
        model=model,
        optimizer=None,
        epoch=-1,
        best_val_psnr=final_result.psnr,
        best_train_psnr=0.0,
        best_epoch=-1,
        args=args,
        dataset=train_dataset,
        entries=train_entries,
        stage='final',
        path=os.path.join(output_dir, 'final.pth'),
    )

    logger.info("分层 GOP-LoRA训练完成: %s", output_dir)


def make_parser():
    parser = argparse.ArgumentParser(
        description="冻结 GOP0共享模型，训练公共与独立分层 GOP-LoRA",
    )
    parser.add_argument('-d', '--data_root', required=True)
    parser.add_argument('--anchor_checkpoint', required=True)
    parser.add_argument(
        '--stage', choices=('all', 'common', 'local', 'eval'), default='all',
    )
    parser.add_argument('--hierarchical_checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--target_gops', type=int, nargs='+')
    parser.add_argument('--target_params', type=int)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--gop_size', type=int, default=30)
    parser.add_argument('--common_epochs', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--short_gop_epochs', type=int, default=5)
    parser.add_argument('--common_network_lr', type=float, default=1e-3)
    parser.add_argument('--common_grid_lr', type=float, default=5e-3)
    parser.add_argument('--local_network_lr', type=float, default=1e-3)
    parser.add_argument('--local_grid_lr', type=float, default=5e-3)
    parser.add_argument('--common_warmup', type=float, default=0.05)
    parser.add_argument('--local_warmup', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--common_rank', type=int, default=8)
    parser.add_argument('--common_alpha', type=float, default=8.0)
    parser.add_argument('--common_grid_rank', type=int, default=2)
    parser.add_argument('--common_grid_alpha', type=float, default=2.0)
    parser.add_argument('--local_rank', type=int, default=8)
    parser.add_argument('--local_alpha', type=float, default=8.0)
    parser.add_argument('--local_grid_rank', type=int, default=2)
    parser.add_argument('--local_grid_alpha', type=float, default=2.0)

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

    parser.add_argument('--out_dir', default='./output/gop_hierarchical')
    parser.add_argument('--exp_name', default='gop_hierarchical')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=25)
    parser.add_argument('--save_interval', type=int, default=25)
    parser.add_argument('--dump_images', action='store_true')
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
