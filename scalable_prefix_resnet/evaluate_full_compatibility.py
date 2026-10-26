"""Verify that strict-prefix Full exactly reproduces an Additive-Skip model.

This is a diagnostic entry point, not a training script.  It loads one
``hybrid_grid_additive_skip_v2`` checkpoint into both the original
Additive-Skip model and ``PrefixScalableResNet``.  The two Full paths are then
evaluated on the same video frames and compared pixel by pixel.

Two independent conclusions are reported:

1. PATH EQUIVALENCE: the strict-prefix Full computation matches the original
   Additive-Skip computation within a numerical tolerance;
2. QUALITY TARGET: the selected checkpoint reproduces the expected full-video
   PSNR (37.03 dB by default).

Separating these checks makes architecture errors distinguishable from using
the wrong checkpoint, dataset or resolution.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import DynamicVideoDataset
from resnet_baseline.model_1 import ResidualSkipHybridGridNet
from scalable_generation.hierarchy import GridHierarchy
from scalable_prefix_resnet.model import PrefixScalableResNet
from util import msssim_fn, psnr_fn


SOURCE_ARCHITECTURE = "hybrid_grid_additive_skip_v2"
MODEL_CONFIG_KEYS = (
    "grid_levels",
    "grid_feat_dim",
    "base_resolution",
    "finest_resolution",
    "aspect_ratio",
    "time_scale",
    "pe_freq",
    "hidden_dim",
)


def _tuple(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("{} must contain two values".format(name))
    return tuple(value)


def model_kwargs(config):
    """Extract the architecture arguments required by both model classes."""
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(
            "checkpoint config is missing: {}".format(", ".join(missing))
        )
    kwargs = {key: config[key] for key in MODEL_CONFIG_KEYS}
    kwargs["aspect_ratio"] = _tuple(kwargs["aspect_ratio"], "aspect_ratio")
    return kwargs


def hierarchy_from_sizes(num_levels, group_sizes):
    sizes = tuple(int(size) for size in group_sizes)
    if len(sizes) != 3 or any(size <= 0 for size in sizes):
        raise ValueError("level_group_sizes must contain three positive values")
    if sum(sizes) != num_levels:
        raise ValueError(
            "level_group_sizes must sum to grid_levels ({} != {})".format(
                sum(sizes), num_levels,
            )
        )
    groups = []
    start = 0
    for size in sizes:
        groups.append(tuple(range(start, start + size)))
        start += size
    return GridHierarchy(tuple(groups))


def build_compatible_models(checkpoint, width_ratios, level_group_sizes):
    """Load one baseline state dict strictly into both equivalent models."""
    architecture = checkpoint.get("architecture")
    if architecture != SOURCE_ARCHITECTURE:
        raise ValueError(
            "expected checkpoint architecture {}, got {}".format(
                SOURCE_ARCHITECTURE, architecture,
            )
        )
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint is missing state_dict")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise KeyError("checkpoint is missing its training config")

    kwargs = model_kwargs(config)
    hierarchy = hierarchy_from_sizes(
        int(kwargs["grid_levels"]), level_group_sizes,
    )
    baseline = ResidualSkipHybridGridNet(**kwargs)
    prefix = PrefixScalableResNet(
        **kwargs,
        width_ratios=tuple(width_ratios),
        hierarchy=hierarchy,
    )
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    prefix.load_state_dict(checkpoint["state_dict"], strict=True)
    return baseline, prefix, config


def make_loader(config, data_root, batch_size, workers):
    base_res = _tuple(config.get("base_res", (720, 1280)), "base_res")
    fixed_res = _tuple(config.get("fixed_res", base_res), "fixed_res")
    dataset = DynamicVideoDataset(
        data_root=data_root,
        base_res=base_res,
        fixed_res=fixed_res,
        frame_interval=int(config.get("frame_interval", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    ), fixed_res


@torch.inference_mode()
def evaluate_equivalence(baseline, prefix, loader, device, log_interval):
    baseline.eval()
    prefix.eval()
    total_baseline_psnr = 0.0
    total_prefix_psnr = 0.0
    total_baseline_msssim = 0.0
    total_prefix_msssim = 0.0
    absolute_error_sum = 0.0
    compared_values = 0
    samples_seen = 0
    max_absolute_error = 0.0

    for batch_index, batch in enumerate(loader):
        coords = batch["coords"].to(device, non_blocking=True)
        target = batch["pixels"].to(device, non_blocking=True)
        reference = baseline(coords)
        candidate = prefix.forward_prefix(coords, 2).reconstruction

        difference = (reference - candidate).abs()
        max_absolute_error = max(
            max_absolute_error, float(difference.max().item()),
        )
        absolute_error_sum += float(difference.sum().item())
        compared_values += difference.numel()

        batch_size = target.size(0)
        baseline_psnr = psnr_fn(reference, target)
        prefix_psnr = psnr_fn(candidate, target)
        baseline_msssim = msssim_fn(reference, target, device)
        prefix_msssim = msssim_fn(candidate, target, device)
        total_baseline_psnr += baseline_psnr * batch_size
        total_prefix_psnr += prefix_psnr * batch_size
        total_baseline_msssim += baseline_msssim * batch_size
        total_prefix_msssim += prefix_msssim * batch_size
        samples_seen += batch_size

        if batch_index % log_interval == 0 or batch_index == len(loader) - 1:
            print(
                "Compare [{}/{}] | Additive PSNR: {:.4f} dB | "
                "Prefix Full PSNR: {:.4f} dB | max abs diff: {:.3e}".format(
                    batch_index + 1,
                    len(loader),
                    total_baseline_psnr / samples_seen,
                    total_prefix_psnr / samples_seen,
                    max_absolute_error,
                ),
                flush=True,
            )

    if samples_seen == 0:
        raise RuntimeError("the evaluation dataset is empty")
    return {
        "frames": samples_seen,
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": absolute_error_sum / compared_values,
        "baseline_psnr": total_baseline_psnr / samples_seen,
        "prefix_psnr": total_prefix_psnr / samples_seen,
        "baseline_msssim": total_baseline_msssim / samples_seen,
        "prefix_msssim": total_prefix_msssim / samples_seen,
    }


def format_report(
    args, checkpoint, baseline, prefix, resolution, metrics,
):
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    prefix_parameters = sum(p.numel() for p in prefix.parameters())
    psnr_delta = metrics["prefix_psnr"] - metrics["baseline_psnr"]
    msssim_delta = metrics["prefix_msssim"] - metrics["baseline_msssim"]
    path_pass = (
        baseline_parameters == prefix_parameters
        and metrics["max_absolute_error"] <= args.output_tolerance
        and abs(psnr_delta) <= args.metric_tolerance
    )
    target_pass = (
        metrics["prefix_psnr"]
        >= args.expected_psnr - args.expected_psnr_tolerance
    )
    overall_pass = path_pass and target_pass
    lines = [
        "========== Additive-Skip / Prefix Full Compatibility ==========",
        "Checkpoint: {}".format(Path(args.checkpoint).resolve()),
        "Checkpoint epoch: {}".format(checkpoint.get("epoch", "N/A")),
        "Checkpoint architecture: {}".format(checkpoint.get("architecture")),
        "Checkpoint best val PSNR: {}".format(
            checkpoint.get("best_val_psnr", "N/A")
        ),
        "Frames: {}".format(metrics["frames"]),
        "Resolution: {} x {}".format(*resolution),
        "Additive-Skip parameters: {}".format(baseline_parameters),
        "Prefix Full parameters: {}".format(prefix_parameters),
        "Max absolute output difference: {:.9e}".format(
            metrics["max_absolute_error"]
        ),
        "Mean absolute output difference: {:.9e}".format(
            metrics["mean_absolute_error"]
        ),
        "Additive-Skip PSNR: {:.6f} dB".format(metrics["baseline_psnr"]),
        "Prefix Full PSNR: {:.6f} dB".format(metrics["prefix_psnr"]),
        "PSNR delta (Prefix - Additive): {:+.9f} dB".format(psnr_delta),
        "Additive-Skip MS-SSIM: {:.9f}".format(
            metrics["baseline_msssim"]
        ),
        "Prefix Full MS-SSIM: {:.9f}".format(metrics["prefix_msssim"]),
        "MS-SSIM delta (Prefix - Additive): {:+.9e}".format(msssim_delta),
        "PATH EQUIVALENCE: {}".format("PASS" if path_pass else "FAIL"),
        "QUALITY TARGET ({:.2f} +/- {:.2f} dB lower bound): {}".format(
            args.expected_psnr,
            args.expected_psnr_tolerance,
            "PASS" if target_pass else "FAIL",
        ),
        "OVERALL: {}".format("PASS" if overall_pass else "FAIL"),
    ]
    return "\n".join(lines), overall_pass


def run(args):
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device), flush=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    baseline, prefix, config = build_compatible_models(
        checkpoint,
        width_ratios=args.width_ratios,
        level_group_sizes=args.level_group_sizes,
    )
    baseline.to(device)
    prefix.to(device)
    loader, resolution = make_loader(
        config,
        args.data_root,
        args.batch_size,
        args.workers,
    )
    metrics = evaluate_equivalence(
        baseline, prefix, loader, device, args.log_interval,
    )
    report, passed = format_report(
        args, checkpoint, baseline, prefix, resolution, metrics,
    )
    print(report, flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")
        print("Saved report: {}".format(output.resolve()), flush=True)
    return passed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify Additive-Skip checkpoint equivalence on Prefix Full",
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
    parser.add_argument("--output_tolerance", type=float, default=1e-6)
    parser.add_argument("--metric_tolerance", type=float, default=1e-5)
    parser.add_argument("--expected_psnr", type=float, default=37.03)
    parser.add_argument("--expected_psnr_tolerance", type=float, default=0.02)
    parser.add_argument("--output", type=str, default=None)
    return parser


if __name__ == "__main__":
    success = run(build_parser().parse_args())
    raise SystemExit(0 if success else 1)
