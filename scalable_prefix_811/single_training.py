"""Single-prefix capacity controls for the 8/1/1 experiment.

Train Base or Mid from scratch on every frame, without EMA, other-prefix
losses, quantization or a teacher. Model initialization, data loaders and LR
schedule match the joint experiment. Unused Grids are frozen, while Decoder
prefixes are evaluated by the original slicing implementation. Checkpoints
retain the full model tensors for compatibility; active parameters are logged
separately and must not be confused with checkpoint size or compressed bits.
"""

import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from scalable_module_resnet.train import _learning_rate, _make_loaders
from train import seed_everything, setup_logging
from util import NervLoss, msssim_fn, psnr_fn

from .train import build_model, build_parser as build_joint_parser, validate_resume


QUALITY_LEVELS = {"base": 0, "mid": 1}
STRATEGY = "prefix_811_single_distortion_v1"
MODEL_KEYS = (
    "grid_levels", "grid_feat_dim", "base_resolution", "finest_resolution",
    "aspect_ratio", "time_scale", "pe_freq", "hidden_dim", "width_ratios",
    "level_group_sizes",
)


def build_parser(quality):
    if quality not in QUALITY_LEVELS:
        raise ValueError("quality must be base or mid")
    parser = build_joint_parser()
    parser.description = "8/1/1 {}-only 单档纯失真能力对照".format(quality)
    parser.set_defaults(
        quality=quality,
        out_dir="./output/scalable_prefix_811_single/{}_only".format(quality),
        exp_name="bunny_811_{}_only_300epoch".format(quality),
    )
    return parser


def freeze_unused_grids(model, quality_level):
    active = model.hierarchy.active_levels(quality_level)
    for index, level in enumerate(model.grid_encoder.levels):
        level.requires_grad_(index in active)


def run_epoch(model, loader, device, loss_fn, args, epoch=0, optimizer=None,
              logger=None, metric_fn=msssim_fn):
    """Use only the selected prefix for every training or validation batch."""
    training = optimizer is not None
    was_training = model.training
    model.train(training)
    sums = dict(distortion=0.0, psnr=0.0, msssim=0.0)
    samples = 0
    quality_level = QUALITY_LEVELS[args.quality]
    try:
        with torch.set_grad_enabled(training):
            for index, batch in enumerate(loader):
                coords = batch["coords"].to(device, non_blocking=True)
                pixels = batch["pixels"].to(device, non_blocking=True)
                if training:
                    lr = _learning_rate(optimizer, args, epoch, index, len(loader))
                    optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_prefix(coords, quality_level).reconstruction
                loss = loss_fn(prediction, pixels)
                if training:
                    loss.backward()
                    optimizer.step()
                values = dict(
                    distortion=loss.detach().item(),
                    psnr=psnr_fn(prediction, pixels),
                    msssim=metric_fn(prediction, pixels, device),
                )
                count = pixels.shape[0]
                samples += count
                for key, value in values.items():
                    sums[key] += value * count
                if logger is not None and (
                    index % args.log_interval == 0 or index == len(loader) - 1
                ):
                    logger.info(
                        "%s %s [%d/%d] | distortion: %.7f | PSNR: %.2f dB | MS-SSIM: %.4f",
                        "Train" if training else "Val", args.quality.title(),
                        index + 1, len(loader), sums["distortion"] / samples,
                        sums["psnr"] / samples, sums["msssim"] / samples,
                    )
        if not samples:
            raise ValueError("dataset contains no samples")
        result = {key: value / samples for key, value in sums.items()}
        if training:
            result["lr"] = lr
        return result
    finally:
        model.train(was_training)


def save_checkpoint(path, model, optimizer, epoch, best_train, best_val, args):
    torch.save({
        "checkpoint_version": 1,
        "architecture": model.architecture,
        "training_strategy": STRATEGY,
        "quality": args.quality,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_train_psnr": best_train,
        "best_val_psnr": best_val,
        "hierarchy": model.hierarchy.groups,
        "hidden_widths": model.decoder.hidden_widths,
        "config": dict(vars(args)),
    }, path)


def load_checkpoint(path, model, optimizer, args, device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("training_strategy") != STRATEGY:
        raise ValueError("只能恢复相同的单档训练，不能加载联合训练断点")
    if checkpoint.get("quality") != args.quality:
        raise ValueError("Base-only 与 Mid-only 断点不能混用")
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("checkpoint architecture does not match")
    validate_resume(checkpoint, model)
    config = checkpoint["config"]
    for key in MODEL_KEYS:
        if config.get(key) != getattr(args, key):
            raise ValueError("checkpoint configuration differs: " + key)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.cpu() if key == "step" else value.to(device)
    return checkpoint["epoch"] + 1, checkpoint["best_train_psnr"], checkpoint["best_val_psnr"]


def train(args):
    if args.epochs < 1 or args.workers < 0:
        raise ValueError("epochs must be positive and workers non-negative")
    if min(args.eval_freq, args.save_interval, args.log_interval) < 1:
        raise ValueError("evaluation/save/log intervals must be positive")
    if not 0 <= args.warmup < 1:
        raise ValueError("warmup must be in [0, 1)")
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if args.eval_only and not args.resume:
        raise ValueError("eval_only requires --resume")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = _make_loaders(args)
    model = build_model(args).to(device)
    quality_level = QUALITY_LEVELS[args.quality]
    freeze_unused_grids(model, quality_level)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = NervLoss(args.loss_type, device)
    log_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d-%H%M%S"), args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    name = args.quality.title()
    logger.info("Device: %s | 单档实验: %s-only | 无EMA、无预训练、无压缩", device, name)
    logger.info("Grid: %s | Decoder width: %d", model.hierarchy.active_levels(quality_level),
                model.decoder.hidden_widths[quality_level])
    logger.info("Active parameters: %d | stored model parameters: %d",
                model.parameter_budget()[quality_level], sum(p.numel() for p in model.parameters()))
    logger.info("注意：存储完整模型是为了兼容，不代表该档实际需要的参数或bits")
    logger.info("target_shares 和 ema_decay 为继承参数，单档实验不使用")
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)
    start, best_train, best_val = 0, 0.0, 0.0
    if args.resume:
        start, best_train, best_val = load_checkpoint(args.resume, model, optimizer, args, device)
        logger.info("从 epoch %d 恢复", start + 1)
    try:
        if args.eval_only:
            values = run_epoch(model, loaders[1], device, loss_fn, args, logger=logger)
            logger.info("Eval %s | PSNR: %.4f dB | MS-SSIM: %.6f", name, values["psnr"], values["msssim"])
            return
        for epoch in range(start, args.epochs):
            started = time.time()
            values = run_epoch(model, loaders[0], device, loss_fn, args, epoch, optimizer, logger)
            best_train = max(best_train, values["psnr"])
            logger.info("Epoch %d | %s-only | distortion: %.7f | PSNR: %.2f dB | BEST TRAIN: %.2f dB | time: %.2fs",
                        epoch + 1, name, values["distortion"], values["psnr"], best_train, time.time() - started)
            for key, value in dict(values, best_psnr=best_train).items():
                writer.add_scalar("train/" + key, value, epoch + 1)
            if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
                values = run_epoch(model, loaders[1], device, loss_fn, args, logger=logger)
                improved = values["psnr"] > best_val
                best_val = max(best_val, values["psnr"])
                logger.info("Val %s | distortion: %.7f | PSNR: %.2f dB | BEST: %.2f dB | MS-SSIM: %.4f",
                            name, values["distortion"], values["psnr"], best_val, values["msssim"])
                for key, value in dict(values, best_psnr=best_val).items():
                    writer.add_scalar("val/" + key, value, epoch + 1)
                if improved:
                    save_checkpoint(os.path.join(log_dir, "single_best.pth"), model, optimizer,
                                    epoch, best_train, best_val, args)
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                save_checkpoint(os.path.join(log_dir, "single_latest.pth"), model, optimizer,
                                epoch, best_train, best_val, args)
        logger.info("%s-only 完成 | BEST TRAIN: %.4f dB | BEST VAL: %.4f dB", name, best_train, best_val)
    finally:
        writer.close()
