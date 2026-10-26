"""EMA training shared only by the two new single-Skip Additive experiments.

Reuses the existing loaders, LR, EMA gradient accumulation and metrics. Logs
and checkpoints are isolated; uncalibrated/721 checkpoints are not accepted
as resumes. No QAT, entropy model, teacher, or warm-start is introduced.
"""

import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from scalable_ablation.unified_internal.train import _empty_sums, _averages, _log_evaluation
from scalable_module_resnet.train import _learning_rate, _make_loaders, hierarchy_from_sizes
from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer, PREFIX_NAMES, accumulate_ema_prefix_gradients,
)
from scalable_prefix_resnet.training import evaluate_prefixes
from resnet_baseline.train import build_parser as build_baseline_parser
from train import seed_everything, setup_logging
from util import NervLoss

from .model import UnifiedAdditive811


MODEL_KEYS = (
    "grid_levels", "grid_feat_dim", "base_resolution", "finest_resolution",
    "aspect_ratio", "time_scale", "pe_freq", "hidden_dim", "width_ratios",
    "level_group_sizes", "gate_hidden_dim", "time_hidden_dim",
)


def build_parser():
    parser = build_baseline_parser()
    parser.description = "Unified Gate/时间调制 + 原Additive-Skip Decoder通道前缀"
    parser.set_defaults(
        epochs=300, lr=5e-3, weight_decay=0.0, warmup=0.2,
        grid_levels=10, grid_feat_dim=6, base_resolution=20, finest_resolution=100,
        time_scale=0.1, pe_freq=14, hidden_dim=256,
        eval_freq=25, save_interval=25,
        out_dir="./output/scalable_unified_additive_811",
        exp_name="bunny_unified_additive_811_ema_20_30_50_300epoch",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--level_group_sizes", type=int, nargs=3, default=[8, 1, 1])
    parser.add_argument("--width_ratios", type=float, nargs=3, default=[0.5, 0.75, 1.0])
    parser.add_argument("--gate_hidden_dim", type=int, default=48)
    parser.add_argument("--time_hidden_dim", type=int, default=64)
    parser.add_argument("--target_shares", type=float, nargs=3, default=[0.2, 0.3, 0.5])
    parser.add_argument("--ema_decay", type=float, default=0.99)
    return parser


def build_model(args, model_class=UnifiedAdditive811):
    if args.grid_levels != 10 or tuple(args.level_group_sizes) != (8, 1, 1):
        raise ValueError("this experiment requires 10 Grid levels and 8/1/1 groups")
    return model_class(
        grid_levels=args.grid_levels, grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution, finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio), time_scale=args.time_scale,
        pe_freq=args.pe_freq, hidden_dim=args.hidden_dim,
        width_ratios=tuple(args.width_ratios),
        hierarchy=hierarchy_from_sizes(args.level_group_sizes, args.grid_levels),
        gate_hidden_dim=args.gate_hidden_dim, time_hidden_dim=args.time_hidden_dim,
    )


def save_checkpoint(path, model, optimizer, balancer, epoch, best_train, best_val, args):
    torch.save({
        "checkpoint_version": 1, "architecture": model.architecture,
        "training_strategy": model.training_strategy, "epoch": epoch,
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "ema_balancer": balancer.state_dict(),
        "best_train_psnr": tuple(best_train), "best_val_psnr": tuple(best_val),
        "hierarchy": model.hierarchy.groups,
        "hidden_widths": model.decoder.hidden_widths,
        "gate_hidden_widths": model.gate_grid.hidden_widths,
        "time_hidden_widths": model.time_mod.hidden_widths,
        "calibration_dimension": model.decoder.input_dim if hasattr(model, "calibration") else 0,
        "config": dict(vars(args)),
    }, path)


def load_checkpoint(path, model, optimizer, balancer, args, device):
    checkpoint = torch.load(path, map_location="cpu")
    if (checkpoint.get("architecture") != model.architecture or
            checkpoint.get("training_strategy") != model.training_strategy):
        raise ValueError("断点架构/策略不一致，不能混用三残差MLP、校准或其他模型")
    if tuple(tuple(group) for group in checkpoint.get("hierarchy", ())) != model.hierarchy.groups:
        raise ValueError("checkpoint hierarchy does not match")
    for key in MODEL_KEYS:
        if checkpoint["config"].get(key) != getattr(args, key):
            raise ValueError("checkpoint model configuration differs: " + key)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.cpu() if name == "step" else value.to(device)
    balancer.load_state_dict(checkpoint["ema_balancer"])
    return (int(checkpoint["epoch"]) + 1,
            tuple(checkpoint["best_train_psnr"]), tuple(checkpoint["best_val_psnr"]))


def train(args, model_class=UnifiedAdditive811):
    if min(args.epochs, args.eval_freq, args.save_interval, args.log_interval) < 1:
        raise ValueError("epochs and evaluation/save/log intervals must be positive")
    if not 0 <= args.warmup < 1 or args.workers < 0:
        raise ValueError("invalid warmup ratio or worker count")
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if args.eval_only and not args.resume:
        raise ValueError("eval_only requires --resume")
    balancer = EMADistortionBalancer(args.target_shares, args.ema_decay)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = _make_loaders(args)
    model = build_model(args, model_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = NervLoss(args.loss_type, device)
    log_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d-%H%M%S"), args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    try:
        logger.info("Device: %s | 架构: %s", device, model.architecture)
        logger.info("训练: 从零三档顺序反向传播，每batch一次更新；无teacher、无压缩")
        logger.info("Grid分组: %s | Decoder隐藏通道/Gate内部/Time内部宽度: %s / %s / %s",
                    model.hierarchy.groups, model.decoder.hidden_widths,
                    model.gate_grid.hidden_widths, model.time_mod.hidden_widths)
        logger.info("EMA目标损失占比: %s | decay: %.5f", balancer.target_shares, balancer.decay)
        calibration = getattr(model, "calibration", None)
        logger.info("Decoder: D -> w -> w -> w -> %d -> RGB；只有第三隐藏层有输入Additive Skip",
                    model.decoder.output_dim)
        logger.info("校准: %s | 位置: 时间调制后、Decoder前 | 新增参数: %d",
                    "Base/Mid启用、Full直通" if calibration is not None else "禁用",
                    sum(p.numel() for p in calibration.parameters()) if calibration is not None else 0)
        logger.info("累计存储参数 B/M/F: %s | 各档实际前向参数: %s",
                    model.parameter_budget(), model.active_parameter_budget())
        logger.info("模型总参数: %d | 累计存储与实际前向参数分开统计",
                    sum(p.numel() for p in model.parameters()))
        for key, value in vars(args).items():
            logger.info("%s: %s", key, value)
        start, best_train, best_val = 0, (0.0,) * 3, (0.0,) * 3
        if args.resume:
            start, best_train, best_val = load_checkpoint(
                args.resume, model, optimizer, balancer, args, device)
            logger.info("从epoch %d恢复Additive前缀实验", start + 1)
        if args.eval_only:
            result = evaluate_prefixes(model, val_loader, device, loss_fn, args.log_interval, logger)
            _log_evaluation(logger, start, result, best_val)
            return
        total_started = time.time()
        for epoch in range(start, args.epochs):
            model.train()
            started = time.time()
            sums, samples, objective_sum = _empty_sums(), 0, 0.0
            for index, batch in enumerate(train_loader):
                coords = batch["coords"].to(device, non_blocking=True)
                pixels = batch["pixels"].to(device, non_blocking=True)
                lr = _learning_rate(optimizer, args, epoch, index, len(train_loader))
                optimizer.zero_grad(set_to_none=True)
                metrics, objective = accumulate_ema_prefix_gradients(
                    model, coords, pixels, loss_fn, balancer, device)
                optimizer.step()
                count = pixels.shape[0]
                samples += count
                objective_sum += objective * count
                for tier, item in enumerate(metrics):
                    for key in sums[tier]:
                        sums[tier][key] += getattr(item, key) * count
                if index % args.log_interval == 0 or index == len(train_loader) - 1:
                    logger.info("epoch [%d/%d] batch [%d/%d] | objective: %.7f | weights: %s | PSNR: %s | lr: %.3e",
                                epoch + 1, args.epochs, index + 1, len(train_loader), objective,
                                " / ".join("{:.4f}".format(item.weight) for item in metrics),
                                " / ".join("{:.2f}".format(item.psnr) for item in metrics), lr)
            averages = _averages(sums, samples)
            best_train = tuple(max(best, item["psnr"]) for best, item in zip(best_train, averages))
            logger.info("Epoch %d | objective: %.7f | time: %.2fs", epoch + 1,
                        objective_sum / samples, time.time() - started)
            for name, values, best in zip(PREFIX_NAMES, averages, best_train):
                logger.info("Train %s | distortion: %.7f | weight: %.5f | contribution: %.7f | PSNR: %.2f dB | BEST: %.2f dB | MS-SSIM: %.4f",
                            name, values["distortion"], values["weight"], values["contribution"],
                            values["psnr"], best, values["msssim"])
                for key, value in dict(values, best_psnr=best).items():
                    writer.add_scalar("train/{}/{}".format(name.lower(), key), value, epoch + 1)
            writer.add_scalar("train/lr", lr, epoch + 1)
            if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
                result = evaluate_prefixes(model, val_loader, device, loss_fn, args.log_interval, logger)
                improved = result[2]["psnr"] > best_val[2]
                best_val = tuple(max(best, item["psnr"]) for best, item in zip(best_val, result))
                _log_evaluation(logger, epoch + 1, result, best_val)
                for name, values, best in zip(PREFIX_NAMES, result, best_val):
                    for key, value in dict(values, best_psnr=best).items():
                        writer.add_scalar("val/{}/{}".format(name.lower(), key), value, epoch + 1)
                if improved:
                    save_checkpoint(os.path.join(log_dir, model.checkpoint_prefix + "_best_full.pth"), model,
                                    optimizer, balancer, epoch, best_train, best_val, args)
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                save_checkpoint(os.path.join(log_dir, model.checkpoint_prefix + "_latest.pth"), model,
                                optimizer, balancer, epoch, best_train, best_val, args)
        logger.info("Additive前缀实验完成 | %.2f小时 | BEST VAL B/M/F: %s",
                    (time.time() - total_started) / 3600, best_val)
    finally:
        writer.close()


if __name__ == "__main__":
    train(build_parser().parse_args())
