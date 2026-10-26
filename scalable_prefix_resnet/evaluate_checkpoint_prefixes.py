"""Evaluate direct Base/Mid/Full truncation of an Additive-Skip checkpoint.

No training or parameter conversion is performed.  The checkpoint is loaded
strictly into ``PrefixScalableResNet`` and every validation frame is decoded
through all three deterministic prefixes.
"""

import argparse
import logging
from pathlib import Path

import torch

from scalable_prefix_resnet.evaluate_full_compatibility import (
    build_compatible_models,
    make_loader,
)
from scalable_prefix_resnet.training import PREFIX_NAMES, evaluate_prefixes
from util import NervLoss


def format_report(args, checkpoint, prefix, resolution, results):
    lines = [
        "========== Direct Prefix Evaluation ==========",
        "Checkpoint: {}".format(Path(args.checkpoint).resolve()),
        "Checkpoint epoch: {}".format(checkpoint.get("epoch", "N/A")),
        "Checkpoint best val PSNR: {}".format(
            checkpoint.get("best_val_psnr", "N/A")
        ),
        "Hierarchy: {}".format(prefix.hierarchy.groups),
        "Decoder hidden widths: {}".format(prefix.decoder.hidden_widths),
        "Resolution: {} x {}".format(*resolution),
    ]
    for name, budget, values in zip(
        PREFIX_NAMES, prefix.parameter_budget(), results,
    ):
        lines.append(
            "{} | parameters: {} | distortion: {:.9f} | PSNR: {:.6f} dB | "
            "MS-SSIM: {:.9f}".format(
                name,
                budget,
                values["distortion"],
                values["psnr"],
                values["msssim"],
            )
        )
    monotonic = (
        results[0]["psnr"] < results[1]["psnr"] < results[2]["psnr"]
    )
    lines.append(
        "MONOTONIC PSNR: {}".format("PASS" if monotonic else "FAIL")
    )
    return "\n".join(lines)


def run(args):
    if args.batch_size <= 0 or args.workers < 0 or args.log_interval <= 0:
        raise ValueError(
            "batch_size/log_interval must be positive and workers non-negative"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device), flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    _, prefix, config = build_compatible_models(
        checkpoint,
        args.width_ratios,
        args.level_group_sizes,
    )
    prefix.to(device)
    loader, resolution = make_loader(
        config, args.data_root, args.batch_size, args.workers,
    )
    loss_fn = NervLoss("L2", device)
    results = evaluate_prefixes(
        prefix,
        loader,
        device,
        loss_fn,
        args.log_interval,
        logger,
    )
    report = format_report(
        args, checkpoint, prefix, resolution, results,
    )
    print(report, flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
        print("Saved report: {}".format(output.resolve()), flush=True)
    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate direct truncation of a Full Additive-Skip checkpoint",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-d", "--data_root", required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--level_group_sizes", nargs=3, type=int, default=(7, 2, 1),
    )
    parser.add_argument(
        "--width_ratios", nargs=3, type=float, default=(0.5, 0.75, 1.0),
    )
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
