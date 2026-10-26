"""Train Unified Internal with 8/1/1 Grid groups instead of 7/2/1.

Only grouping and experiment paths differ from the Unified 721 control.
The existing model and EMA training loop are reused without modification.
This entry trains from scratch unless explicitly resuming a matching 811
checkpoint. No compression, additional calibration or teacher is introduced.
"""

import torch

from scalable_ablation.unified_internal.model import UnifiedInternalScalableResNet
from scalable_ablation.unified_internal.train import (
    build_parser as build_reference_parser,
    train as train_reference,
)
from scalable_module_resnet.train import hierarchy_from_sizes


def build_parser():
    parser = build_reference_parser()
    parser.description = "独立 Unified 811 三前缀 EMA 纯失真实验"
    parser.set_defaults(
        epochs=300, lr=5e-3, weight_decay=0.0, warmup=0.2,
        grid_levels=10, grid_feat_dim=6,
        base_resolution=20, finest_resolution=100,
        time_scale=0.1, pe_freq=14, hidden_dim=240,
        feature_dim=128, num_blocks=3,
        gate_hidden_dim=48, time_hidden_dim=64,
        level_group_sizes=[8, 1, 1],
        width_ratios=[0.5, 0.75, 1.0],
        target_shares=[0.2, 0.3, 0.5], ema_decay=0.99,
        eval_freq=25, save_interval=25,
        out_dir="./output/scalable_unified_811",
        exp_name="bunny_unified_811_ema_20_30_50_300epoch",
    )
    return parser


def build_model(args):
    if args.grid_levels != 10 or tuple(args.level_group_sizes) != (8, 1, 1):
        raise ValueError("本实验固定使用10层Grid和8/1/1分组")
    return UnifiedInternalScalableResNet(
        grid_levels=args.grid_levels, grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution, finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio), time_scale=args.time_scale,
        pe_freq=args.pe_freq, hidden_dim=args.hidden_dim,
        width_ratios=tuple(args.width_ratios),
        hierarchy=hierarchy_from_sizes(args.level_group_sizes, args.grid_levels),
        feature_dim=args.feature_dim, num_blocks=args.num_blocks,
        gate_hidden_dim=args.gate_hidden_dim, time_hidden_dim=args.time_hidden_dim,
    )


def validate_resume(checkpoint, model):
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("checkpoint不是Unified模型，不能混用原双前缀断点")
    if checkpoint.get("training_strategy") != "unified_internal_ema_distortion_v1":
        raise ValueError("checkpoint不是Unified EMA训练产生的")
    groups = tuple(tuple(group) for group in checkpoint.get("hierarchy", ()))
    if groups != model.hierarchy.groups:
        raise ValueError("checkpoint分组不一致，不能把Unified 721当作811续训")
    for key, widths in (
        ("hidden_widths", model.decoder.hidden_widths),
        ("gate_hidden_widths", model.gate_grid.hidden_widths),
        ("time_hidden_widths", model.time_mod.hidden_widths),
    ):
        if tuple(checkpoint.get(key, ())) != widths:
            raise ValueError("checkpoint宽度不一致: " + key)


def train(args):
    if min(args.epochs, args.eval_freq, args.save_interval, args.log_interval) < 1:
        raise ValueError("epochs及日志/评测/保存间隔必须为正数")
    if not 0 <= args.warmup < 1:
        raise ValueError("warmup必须为[0, 1)内的比例")
    if args.eval_only and not args.resume:
        raise ValueError("eval_only requires --resume")
    # CPU preflight only. The reference trainer resets the seed before creating
    # its model, preserving the original initialization and data order policy.
    model = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume(checkpoint, model)
        del checkpoint
    print("独立实验: Unified 811；原Unified 721与原双前缀811均保持不变", flush=True)
    print("累计有效参数 Base/Mid/Full: {}".format(model.parameter_budget()), flush=True)
    del model
    train_reference(args)


if __name__ == "__main__":
    train(build_parser().parse_args())
