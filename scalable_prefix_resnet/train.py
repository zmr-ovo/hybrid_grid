"""Train the strict Grid/hidden-channel prefix model.

This first implementation is an uncompressed capacity experiment.  It trains
one sampled operating point per batch and evaluates all three prefixes on the
full validation video.  Quantization, entropy rate and real bitstreams remain
outside this entry so architecture/training conflicts can be measured first.
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

from .model import PrefixScalableResNet
from .training import (
    PREFIX_NAMES,
    BalancedPrefixScheduler,
    average_sums,
    empty_sums,
    evaluate_prefixes,
    train_sample,
    update_sums,
)


def _save_checkpoint(
    path, model, optimizer, scheduler, epoch, best_train, best_val, args,
):
    torch.save({
        "checkpoint_version": 1,
        "architecture": model.architecture,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "prefix_scheduler": scheduler.state_dict(),
        "best_train_psnr": tuple(best_train),
        "best_val_psnr": tuple(best_val),
        "hierarchy": tuple(model.hierarchy.groups),
        "hidden_widths": tuple(model.decoder.hidden_widths),
        "config": dict(vars(args)),
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("未找到前缀模型检查点: " + path)
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("检查点与严格前缀模型架构不一致")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # PyTorch 1.x requires Adam's scalar step tensor to remain on CPU.
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value) and name != "step":
                state[name] = value.to(device)
    scheduler.load_state_dict(checkpoint["prefix_scheduler"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        tuple(checkpoint.get("best_train_psnr", (0.0, 0.0, 0.0))),
        tuple(checkpoint.get("best_val_psnr", (0.0, 0.0, 0.0))),
    )


def train_prefix_scalable(args):
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if args.epochs < 1 or args.workers < 0:
        raise ValueError("epochs must be positive and workers non-negative")
    hierarchy = hierarchy_from_sizes(
        args.level_group_sizes, args.grid_levels,
    )
    seed_everything(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)
    logger.info("训练结构: Grid前缀 + Decoder隐藏通道前缀")
    logger.info("训练采样: 每个batch仅训练一个档位，按Base/Mid/Full轮转")
    logger.info("压缩模块: 未启用（先验证联合表示能力上限）")
    logger.info("Grid层级: %s", hierarchy.groups)

    train_loader, val_loader = _make_loaders(args)
    model = PrefixScalableResNet(
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
    prefix_scheduler = BalancedPrefixScheduler()
    loss_fn = NervLoss(args.loss_type, device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    grid_params = sum(
        parameter.numel() for parameter in model.grid_encoder.parameters()
    )
    logger.info("模型总参数: %d (%.4f M)", total_params, total_params / 1e6)
    logger.info("Grid参数: %d", grid_params)
    logger.info("非Grid参数: %d", total_params - grid_params)
    logger.info("Decoder隐藏宽度: %s", model.decoder.hidden_widths)
    for name, budget in zip(PREFIX_NAMES, model.parameter_budget()):
        logger.info(
            "%s累计参数前缀: %d (%.4f M)",
            name, budget, budget / 1e6,
        )
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    start_epoch = 0
    best_train = (0.0, 0.0, 0.0)
    best_val = (0.0, 0.0, 0.0)
    if args.resume:
        start_epoch, best_train, best_val = _load_checkpoint(
            args.resume,
            model,
            optimizer,
            prefix_scheduler,
            device,
        )
        logger.info("从epoch %d恢复严格前缀训练", start_epoch + 1)

    if args.eval_only:
        if not args.resume:
            raise ValueError("eval_only requires --resume")
        evaluation = evaluate_prefixes(
            model, val_loader, device, loss_fn, args.log_interval, logger,
        )
        for name, values in zip(PREFIX_NAMES, evaluation):
            logger.info(
                "Eval %s | distortion: %.6f | PSNR: %.2f dB | "
                "MS-SSIM: %.4f",
                name,
                values["distortion"],
                values["psnr"],
                values["msssim"],
            )
        writer.close()
        return

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        sums = empty_sums()
        for batch_index, batch in enumerate(train_loader):
            coords = batch["coords"].to(device, non_blocking=True)
            pixels = batch["pixels"].to(device, non_blocking=True)
            quality_level = prefix_scheduler.next()
            learning_rate = _learning_rate(
                optimizer, args, epoch, batch_index, len(train_loader),
            )

            optimizer.zero_grad(set_to_none=True)
            metrics = train_sample(
                model,
                coords,
                pixels,
                loss_fn,
                quality_level,
                device,
            )
            optimizer.step()
            update_sums(sums, metrics, pixels.shape[0])

            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                averages = average_sums(sums)
                logger.info(
                    "epoch [%d/%d] batch [%d/%d] | sampled: %s | "
                    "distortion: %.6f | PSNR: %.2f dB | lr: %.3e",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    PREFIX_NAMES[quality_level],
                    metrics.distortion,
                    metrics.psnr,
                    learning_rate,
                )
                logger.info(
                    "Train sampled averages | %s",
                    " | ".join(
                        "{}: {:.2f} dB ({} samples)".format(
                            name, values["psnr"], values["count"],
                        )
                        for name, values in zip(PREFIX_NAMES, averages)
                    ),
                )

        averages = average_sums(sums)
        best_train = tuple(
            max(best, values["psnr"])
            for best, values in zip(best_train, averages)
        )
        logger.info(
            "Epoch %d | time: %.2fs | sampled counts: %s",
            epoch + 1,
            time.time() - epoch_start,
            tuple(values["count"] for values in averages),
        )
        for name, values, best in zip(PREFIX_NAMES, averages, best_train):
            tag = name.lower()
            writer.add_scalar(
                "train/{}/distortion".format(tag),
                values["distortion"], epoch + 1,
            )
            writer.add_scalar(
                "train/{}/psnr".format(tag), values["psnr"], epoch + 1,
            )
            writer.add_scalar(
                "train/{}/msssim".format(tag), values["msssim"], epoch + 1,
            )
            writer.add_scalar(
                "train/{}/best_psnr".format(tag), best, epoch + 1,
            )

        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            evaluation = evaluate_prefixes(
                model,
                val_loader,
                device,
                loss_fn,
                args.log_interval,
                logger,
            )
            full_improved = evaluation[-1]["psnr"] > best_val[-1]
            best_val = tuple(
                max(best, values["psnr"])
                for best, values in zip(best_val, evaluation)
            )
            for name, values, best in zip(PREFIX_NAMES, evaluation, best_val):
                tag = name.lower()
                writer.add_scalar(
                    "val/{}/psnr".format(tag), values["psnr"], epoch + 1,
                )
                writer.add_scalar(
                    "val/{}/msssim".format(tag),
                    values["msssim"], epoch + 1,
                )
                logger.info(
                    "Val %s | distortion: %.6f | PSNR: %.2f dB | "
                    "BEST: %.2f dB | MS-SSIM: %.4f",
                    name,
                    values["distortion"],
                    values["psnr"],
                    best,
                    values["msssim"],
                )
            if full_improved:
                _save_checkpoint(
                    os.path.join(log_dir, "prefix_best_full.pth"),
                    model,
                    optimizer,
                    prefix_scheduler,
                    epoch,
                    best_train,
                    best_val,
                    args,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            _save_checkpoint(
                os.path.join(log_dir, "prefix_latest.pth"),
                model,
                optimizer,
                prefix_scheduler,
                epoch,
                best_train,
                best_val,
                args,
            )

    writer.close()
    logger.info(
        "严格前缀训练完成，总耗时 %.2f 小时",
        (time.time() - training_start) / 3600,
    )


def build_parser():
    parser = build_baseline_parser()
    parser.description = "论文式Grid/Decoder严格前缀单档位采样训练"
    parser.set_defaults(
        out_dir="./output/scalable_prefix_resnet",
        exp_name="scalable_prefix_additive_skip",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--level_group_sizes", type=int, nargs=3, default=[7, 2, 1],
        help="Base、E1、E2新增的Grid层数",
    )
    parser.add_argument(
        "--width_ratios", type=float, nargs=3, default=[0.5, 0.75, 1.0],
        help="Base、Mid、Full的Decoder隐藏通道前缀比例",
    )
    return parser


def main():
    train_prefix_scalable(build_parser().parse_args())


if __name__ == "__main__":
    main()
