"""Warm-start three-prefix pure-distortion fine-tuning.

The student is initialized from a validated Additive-Skip Full checkpoint.
For every frame, Base, Mid and Full are differentiated sequentially and their
gradients are accumulated before one optimizer update.  A frozen copy of the
source Additive-Skip model anchors the Full output.  This entry deliberately
contains no quantization, entropy model or rate loss.
"""

import argparse
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import DynamicVideoDataset
from scalable_prefix_resnet.evaluate_full_compatibility import (
    build_compatible_models,
)
from scalable_prefix_resnet.training import (
    PREFIX_NAMES,
    evaluate_prefixes,
)
from scalable_prefix_resnet.warmstart_training import (
    accumulate_warmstart_gradients,
    validate_distortion_weights,
)
from train import seed_everything, seed_worker, setup_logging
from util import NervLoss


def _resolution(config, key, fallback):
    value = config.get(key, fallback)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("{} must contain H and W".format(key))
    return tuple(int(item) for item in value)


def make_loaders(config, args):
    base_res = _resolution(config, "base_res", (720, 1280))
    fixed_res = _resolution(config, "fixed_res", base_res)
    frame_interval = int(config.get("frame_interval", 1))
    train_dataset = DynamicVideoDataset(
        args.data_root,
        base_res=base_res,
        fixed_res=fixed_res,
        frame_interval=frame_interval,
    )
    val_dataset = DynamicVideoDataset(
        args.data_root,
        base_res=base_res,
        fixed_res=fixed_res,
        frame_interval=frame_interval,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader


def learning_rate(optimizer, args, epoch, batch_index, num_batches):
    position = epoch + batch_index / num_batches
    warmup_epochs = args.warmup * args.epochs
    if warmup_epochs and position < warmup_epochs:
        multiplier = 0.1 + 0.9 * position / warmup_epochs
    else:
        duration = max(args.epochs - warmup_epochs, 1.0)
        progress = min(max((position - warmup_epochs) / duration, 0.0), 1.0)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    value = args.lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def empty_sums():
    return [
        {"distortion": 0.0, "psnr": 0.0, "msssim": 0.0}
        for _ in PREFIX_NAMES
    ]


def average_sums(sums, count):
    return tuple(
        {key: value / count for key, value in values.items()}
        for values in sums
    )


def save_checkpoint(
    path, student, optimizer, epoch, best_val, args, source_checkpoint,
    source_config,
):
    torch.save({
        "checkpoint_version": 1,
        "architecture": student.architecture,
        "epoch": epoch,
        "state_dict": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_psnr": tuple(best_val),
        "source_checkpoint": str(source_checkpoint),
        "hierarchy": tuple(student.hierarchy.groups),
        "hidden_widths": tuple(student.decoder.hidden_widths),
        "model_config": dict(source_config),
        "training_config": dict(vars(args)),
    }, path)


def load_resume(path, student, optimizer, device):
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("architecture") != student.architecture:
        raise ValueError("resume checkpoint architecture does not match")
    student.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[name] = value.cpu() if name == "step" else value.to(device)
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        tuple(checkpoint.get("best_val_psnr", (0.0, 0.0, 0.0))),
    )


def log_evaluation(logger, label, values, best):
    logger.info("========== %s ==========", label)
    for name, result, best_psnr in zip(PREFIX_NAMES, values, best):
        logger.info(
            "%s | distortion: %.9f | PSNR: %.4f dB | "
            "BEST: %.4f dB | MS-SSIM: %.6f",
            name,
            result["distortion"],
            result["psnr"],
            best_psnr,
            result["msssim"],
        )


def train(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.workers < 0 or not 0 <= args.warmup < 1:
        raise ValueError("workers must be non-negative and warmup in [0, 1)")
    prefix_weights = validate_distortion_weights(args.prefix_weights)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = torch.load(args.checkpoint, map_location="cpu")
    teacher, student, source_config = build_compatible_models(
        source,
        args.width_ratios,
        args.level_group_sizes,
    )
    teacher.to(device).eval().requires_grad_(False)
    student.to(device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))
    logger.info("Device: %s", device)
    logger.info("Source checkpoint: %s", args.checkpoint)
    logger.info("Source epoch: %s", source.get("epoch", "N/A"))
    logger.info("Source best val PSNR: %s", source.get("best_val_psnr", "N/A"))
    logger.info("Objective: pure L2 distortion; rate/quantization/entropy disabled")
    logger.info("Prefix weights: %s", prefix_weights)
    logger.info("Full teacher anchor weight: %.4f", args.anchor_weight)
    logger.info("Hierarchy: %s", student.hierarchy.groups)
    logger.info("Hidden widths: %s", student.decoder.hidden_widths)

    train_loader, val_loader = make_loaders(source_config, args)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = NervLoss("L2", device)
    start_epoch = 0

    initial = evaluate_prefixes(
        student, val_loader, device, loss_fn, args.log_interval, logger,
    )
    best_val = tuple(result["psnr"] for result in initial)
    log_evaluation(logger, "INITIAL DIRECT PREFIXES", initial, best_val)
    save_checkpoint(
        os.path.join(log_dir, "warmstart_initial.pth"),
        student, optimizer, -1, best_val, args, args.checkpoint, source_config,
    )

    if args.resume:
        start_epoch, best_val = load_resume(
            args.resume, student, optimizer, device,
        )
        logger.info("Resume warm-start training from epoch %d", start_epoch + 1)

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        student.train()
        sums = empty_sums()
        samples = 0
        total_objective = 0.0
        total_anchor = 0.0
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_loader):
            coords = batch["coords"].to(device, non_blocking=True)
            pixels = batch["pixels"].to(device, non_blocking=True)
            lr = learning_rate(
                optimizer, args, epoch, batch_index, len(train_loader),
            )
            optimizer.zero_grad(set_to_none=True)
            metrics, objective, anchor = accumulate_warmstart_gradients(
                student,
                teacher,
                coords,
                pixels,
                prefix_weights,
                args.anchor_weight,
                device,
            )
            optimizer.step()

            batch_size = pixels.shape[0]
            samples += batch_size
            total_objective += objective * batch_size
            total_anchor += anchor * batch_size
            for index, values in enumerate(metrics):
                sums[index]["distortion"] += values.distortion * batch_size
                sums[index]["psnr"] += values.psnr * batch_size
                sums[index]["msssim"] += values.msssim * batch_size

            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                averages = average_sums(sums, samples)
                logger.info(
                    "epoch [%d/%d] batch [%d/%d] | objective: %.7f | "
                    "anchor: %.7f | %s | lr: %.3e",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    total_objective / samples,
                    total_anchor / samples,
                    " | ".join(
                        "{} PSNR: {:.2f}".format(name, value["psnr"])
                        for name, value in zip(PREFIX_NAMES, averages)
                    ),
                    lr,
                )

        averages = average_sums(sums, samples)
        logger.info(
            "Epoch %d | objective: %.7f | anchor: %.7f | time: %.2fs",
            epoch + 1,
            total_objective / samples,
            total_anchor / samples,
            time.time() - epoch_start,
        )
        for name, value in zip(PREFIX_NAMES, averages):
            tag = name.lower()
            writer.add_scalar("train/{}/distortion".format(tag), value["distortion"], epoch + 1)
            writer.add_scalar("train/{}/psnr".format(tag), value["psnr"], epoch + 1)
            writer.add_scalar("train/{}/msssim".format(tag), value["msssim"], epoch + 1)
        writer.add_scalar("train/objective", total_objective / samples, epoch + 1)
        writer.add_scalar("train/full_anchor", total_anchor / samples, epoch + 1)
        writer.add_scalar("train/lr", lr, epoch + 1)

        should_evaluate = (
            (epoch + 1) % args.eval_freq == 0
            or epoch == args.epochs - 1
        )
        if should_evaluate:
            evaluation = evaluate_prefixes(
                student,
                val_loader,
                device,
                loss_fn,
                args.log_interval,
                logger,
            )
            full_improved = evaluation[2]["psnr"] > best_val[2]
            best_val = tuple(
                max(best, result["psnr"])
                for best, result in zip(best_val, evaluation)
            )
            log_evaluation(logger, "VAL EPOCH {}".format(epoch + 1), evaluation, best_val)
            if full_improved:
                save_checkpoint(
                    os.path.join(log_dir, "warmstart_best_full.pth"),
                    student, optimizer, epoch, best_val, args,
                    args.checkpoint, source_config,
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                os.path.join(log_dir, "warmstart_latest.pth"),
                student, optimizer, epoch, best_val, args,
                args.checkpoint, source_config,
            )

    writer.close()
    logger.info(
        "Warm-start pure-distortion training finished in %.2f hours",
        (time.time() - training_start) / 3600,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-checkpoint warm-start pure-distortion prefix training",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-d", "--data_root", required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--prefix_weights", nargs=3, type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--level_group_sizes", nargs=3, type=int, default=(7, 2, 1))
    parser.add_argument("--width_ratios", nargs=3, type=float, default=(0.5, 0.75, 1.0))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out_dir", default="./output/scalable_prefix_warmstart")
    parser.add_argument("--exp_name", default="bunny_prefix_warmstart_distortion")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
