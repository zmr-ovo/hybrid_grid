"""Train Grid-only prefixes from scratch with EMA-normalized distortion loss.

Base, Mid and Full are recomputed and backpropagated sequentially for every
mini-batch, so memory usage stays close to one prefix graph.  Their gradients
are accumulated before a single optimizer update.  Detached EMA statistics
adapt the effective weights so the three loss shares follow the configured
target ratio instead of being determined by raw distortion magnitudes.

Every prefix runs the full-width shared Decoder. The only experimental
change relative to train_ema is disabling Decoder width truncation.
This is a pure-distortion capacity experiment.  It deliberately excludes
quantization, entropy rate and teacher/checkpoint anchoring.
"""

import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from resnet_baseline.train import build_parser as build_baseline_parser
from scalable_module_resnet.train import (
    _learning_rate,
    _make_loaders,
    hierarchy_from_sizes,
)
from train import seed_everything, setup_logging
from util import NervLoss

from scalable_prefix_resnet.ema_training import (
    EMADistortionBalancer,
    PREFIX_NAMES,
    accumulate_ema_prefix_gradients,
    validate_target_shares,
)
from .model import GridOnlyScalableResNet
from scalable_prefix_resnet.training import evaluate_prefixes


def _empty_sums():
    return [
        {
            "distortion": 0.0,
            "psnr": 0.0,
            "msssim": 0.0,
            "weight": 0.0,
            "contribution": 0.0,
        }
        for _ in PREFIX_NAMES
    ]


def _averages(sums, samples):
    return tuple(
        {key: value / samples for key, value in item.items()}
        for item in sums
    )


def _save_checkpoint(
    path,
    model,
    optimizer,
    balancer,
    epoch,
    best_train,
    best_val,
    args,
):
    torch.save({
        "checkpoint_version": 1,
        "architecture": model.architecture,
        "training_strategy": "grid_only_ema_distortion_v1",
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema_balancer": balancer.state_dict(),
        "best_train_psnr": tuple(best_train),
        "best_val_psnr": tuple(best_val),
        "hierarchy": tuple(model.hierarchy.groups),
        "hidden_widths": (model.decoder.hidden_dim,) * 3,
        "config": dict(vars(args)),
    }, path)


def _load_checkpoint(path, model, optimizer, balancer, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("未找到 EMA 前缀训练检查点: " + path)
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("检查点与严格前缀模型架构不一致")
    if checkpoint.get("training_strategy") != "grid_only_ema_distortion_v1":
        raise ValueError("检查点不是 EMA 归一化训练产生的")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.cpu() if name == "step" else value.to(device)
    balancer.load_state_dict(checkpoint["ema_balancer"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        tuple(checkpoint.get("best_train_psnr", (0.0, 0.0, 0.0))),
        tuple(checkpoint.get("best_val_psnr", (0.0, 0.0, 0.0))),
    )


def _log_evaluation(logger, epoch, evaluation, best_val):
    logger.info("========== Val epoch %d ==========", epoch)
    for name, values, best in zip(PREFIX_NAMES, evaluation, best_val):
        logger.info(
            "Val %s | distortion: %.7f | PSNR: %.2f dB | "
            "BEST: %.2f dB | MS-SSIM: %.4f",
            name,
            values["distortion"],
            values["psnr"],
            best,
            values["msssim"],
        )


def train(args):
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if args.epochs < 1 or args.workers < 0:
        raise ValueError("epochs must be positive and workers non-negative")
    target_shares = validate_target_shares(args.target_shares)
    hierarchy = hierarchy_from_sizes(args.level_group_sizes, args.grid_levels)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = _make_loaders(args)
    model = GridOnlyScalableResNet(
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    balancer = EMADistortionBalancer(target_shares, args.ema_decay)
    loss_fn = NervLoss(args.loss_type, device)

    logger.info("使用设备: %s", device)
    logger.info("训练结构: Grid-only前缀 + 全宽共享Decoder")
    logger.info("架构: %s", model.architecture)
    logger.info("训练策略: 每batch三前缀顺序反向传播 + EMA归一化")
    logger.info("初始化: 随机初始化（无Full预训练、无teacher）")
    logger.info("目标损失占比 Base/Mid/Full: %s", target_shares)
    logger.info("EMA decay: %.5f", args.ema_decay)
    logger.info("压缩模块: 未启用（纯失真能力实验）")
    logger.info("Grid层级: %s", hierarchy.groups)
    logger.info("实际Decoder隐藏宽度: %s", (model.decoder.hidden_dim,) * 3)
    logger.info("累计有效参数: %s", model.parameter_budget())
    logger.info(
        "模型参数: %d",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    start_epoch = 0
    best_train = (0.0, 0.0, 0.0)
    best_val = (0.0, 0.0, 0.0)
    if args.resume:
        start_epoch, best_train, best_val = _load_checkpoint(
            args.resume, model, optimizer, balancer, device,
        )
        logger.info("从 epoch %d 恢复 EMA 归一化训练", start_epoch + 1)

    if args.eval_only:
        if not args.resume:
            raise ValueError("eval_only requires --resume")
        evaluation = evaluate_prefixes(
            model, val_loader, device, loss_fn, args.log_interval, logger,
        )
        _log_evaluation(logger, start_epoch, evaluation, best_val)
        writer.close()
        return

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        sums = _empty_sums()
        samples = 0
        objective_sum = 0.0
        for batch_index, batch in enumerate(train_loader):
            coords = batch["coords"].to(device, non_blocking=True)
            pixels = batch["pixels"].to(device, non_blocking=True)
            lr = _learning_rate(
                optimizer, args, epoch, batch_index, len(train_loader),
            )

            optimizer.zero_grad(set_to_none=True)
            metrics, objective = accumulate_ema_prefix_gradients(
                model,
                coords,
                pixels,
                loss_fn,
                balancer,
                device,
            )
            optimizer.step()

            batch_size = pixels.shape[0]
            samples += batch_size
            objective_sum += objective * batch_size
            for index, values in enumerate(metrics):
                sums[index]["distortion"] += values.distortion * batch_size
                sums[index]["psnr"] += values.psnr * batch_size
                sums[index]["msssim"] += values.msssim * batch_size
                sums[index]["weight"] += values.weight * batch_size
                sums[index]["contribution"] += values.contribution * batch_size

            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                logger.info(
                    "epoch [%d/%d] batch [%d/%d] | objective: %.7f | "
                    "EMA: %s | weights: %s | PSNR: %s | lr: %.3e",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    objective,
                    " / ".join("{:.3e}".format(value) for value in balancer.ema),
                    " / ".join("{:.4f}".format(item.weight) for item in metrics),
                    " / ".join("{:.2f}".format(item.psnr) for item in metrics),
                    lr,
                )

        averages = _averages(sums, samples)
        best_train = tuple(
            max(best, values["psnr"])
            for best, values in zip(best_train, averages)
        )
        logger.info(
            "Epoch %d | objective: %.7f | EMA: %s | time: %.2fs",
            epoch + 1,
            objective_sum / samples,
            " / ".join("{:.3e}".format(value) for value in balancer.ema),
            time.time() - epoch_start,
        )
        for name, values, best in zip(PREFIX_NAMES, averages, best_train):
            tag = name.lower()
            logger.info(
                "Train %s | distortion: %.7f | effective weight: %.5f | "
                "contribution: %.7f | PSNR: %.2f dB | BEST: %.2f dB | "
                "MS-SSIM: %.4f",
                name,
                values["distortion"],
                values["weight"],
                values["contribution"],
                values["psnr"],
                best,
                values["msssim"],
            )
            for key in ("distortion", "psnr", "msssim", "weight", "contribution"):
                writer.add_scalar(
                    "train/{}/{}".format(tag, key), values[key], epoch + 1,
                )
        writer.add_scalar("train/objective", objective_sum / samples, epoch + 1)
        writer.add_scalar("train/lr", lr, epoch + 1)

        should_evaluate = (
            (epoch + 1) % args.eval_freq == 0
            or epoch >= args.epochs - 10
        )
        if should_evaluate:
            evaluation = evaluate_prefixes(
                model, val_loader, device, loss_fn, args.log_interval, logger,
            )
            full_improved = evaluation[2]["psnr"] > best_val[2]
            best_val = tuple(
                max(best, values["psnr"])
                for best, values in zip(best_val, evaluation)
            )
            _log_evaluation(logger, epoch + 1, evaluation, best_val)
            for name, values in zip(PREFIX_NAMES, evaluation):
                tag = name.lower()
                writer.add_scalar("val/{}/psnr".format(tag), values["psnr"], epoch + 1)
                writer.add_scalar("val/{}/msssim".format(tag), values["msssim"], epoch + 1)
            if full_improved:
                _save_checkpoint(
                    os.path.join(log_dir, "grid_only_best_full.pth"),
                    model, optimizer, balancer, epoch,
                    best_train, best_val, args,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            _save_checkpoint(
                os.path.join(log_dir, "grid_only_latest.pth"),
                model, optimizer, balancer, epoch,
                best_train, best_val, args,
            )

    writer.close()
    logger.info(
        "Grid-only EMA训练完成，总耗时 %.2f 小时",
        (time.time() - training_start) / 3600,
    )


def build_parser():
    parser = build_baseline_parser()
    parser.description = "Grid-only三前缀EMA纯失真对照实验"
    parser.set_defaults(
        out_dir="./output/scalable_ablation/grid_only",
        exp_name="bunny_grid_only_ema_20_30_50",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--level_group_sizes", type=int, nargs=3, default=[7, 2, 1],
    )
    # Retain the reference initialization; every forward uses Decoder level 2.
    parser.set_defaults(width_ratios=[0.5, 0.75, 1.0])
    parser.add_argument(
        "--target_shares", type=float, nargs=3, default=[0.2, 0.3, 0.5],
        help="EMA归一化后的Base、Mid、Full目标损失占比",
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.99,
        help="三个前缀distortion滑动平均的衰减系数",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
