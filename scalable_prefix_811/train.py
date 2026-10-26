"""Compare 8/1/1 against 7/2/1 without changing the EMA training algorithm.

Run from the project root with ``python -m scalable_prefix_811.train``.
Each batch trains all three prefixes sequentially, followed by one optimizer
step. This experiment uses distortion only: no QAT, entropy model or teacher.
The original trainer owns logging, evaluation and checkpoint saving; only the
hierarchy, output defaults and resume safety checks are specific to this file.
"""

import torch

from scalable_module_resnet.train import hierarchy_from_sizes
from scalable_prefix_resnet.model import PrefixScalableResNet
from scalable_prefix_resnet.train_ema import (
    build_parser as build_reference_parser,
    train as train_reference,
)


def build_parser():
    parser = build_reference_parser()
    parser.description = "独立 8/1/1 双前缀 EMA 纯失真对照实验"
    parser.set_defaults(
        epochs=300,
        lr=5e-3,
        weight_decay=0.0,
        grid_levels=10,
        grid_feat_dim=6,
        base_resolution=20,
        finest_resolution=100,
        time_scale=0.1,
        pe_freq=14,
        hidden_dim=256,
        level_group_sizes=[8, 1, 1],
        width_ratios=[0.5, 0.75, 1.0],
        target_shares=[0.2, 0.3, 0.5],
        out_dir="./output/scalable_prefix_811",
        exp_name="bunny_prefix_811_ema_20_30_50_300epoch",
    )
    return parser


def build_model(args):
    if args.grid_levels != 10 or tuple(args.level_group_sizes) != (8, 1, 1):
        raise ValueError("本独立实验固定使用 10 层 Grid 和 8/1/1 分组")
    return PrefixScalableResNet(
        grid_levels=args.grid_levels,
        grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        time_scale=args.time_scale,
        pe_freq=args.pe_freq,
        hidden_dim=args.hidden_dim,
        width_ratios=tuple(args.width_ratios),
        hierarchy=hierarchy_from_sizes(args.level_group_sizes, args.grid_levels),
    )


def validate_resume(checkpoint, model):
    hierarchy = tuple(tuple(group) for group in checkpoint.get("hierarchy", ()))
    if hierarchy != model.hierarchy.groups:
        raise ValueError("checkpoint Grid 分组不一致；不能把 7/2/1 当作 8/1/1 续训")
    if tuple(checkpoint.get("hidden_widths", ())) != model.decoder.hidden_widths:
        raise ValueError("checkpoint Decoder 隐藏宽度不一致")


def train(args):
    # This CPU-only preflight does not affect the experiment's initialization:
    # the reference trainer resets all seeds before constructing its model.
    model = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume(checkpoint, model)
        del checkpoint
    print("独立实验: 8/1/1；对照: 原 7/2/1 EMA", flush=True)
    print("累计有效参数 Base/Mid/Full: {}".format(model.parameter_budget()), flush=True)
    del model
    train_reference(args)


if __name__ == "__main__":
    train(build_parser().parse_args())
