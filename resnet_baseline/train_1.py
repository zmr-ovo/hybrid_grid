"""训练入口：未压缩 Hybrid Grid，残差 skip 干净对照（model_1）。

复用稳定的基线数据、检查点、指标与 FPS 流程，不开启量化、熵模型或可缩放前缀。
模型为 :class:`resnet_baseline.model_1.ResidualSkipHybridGridNet`，只把原 decoder
的 concat skip 换成"投影后相加"的加法 skip。
"""

import argparse

from train import train

from resnet_baseline.model_1 import ResidualSkipHybridGridNet


def build_parser():
    parser = argparse.ArgumentParser(
        description="训练未压缩 Hybrid Grid 残差 Skip 对照基线"
    )
    parser.add_argument('-d', '--data_root', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr_type', type=str, default='cosine')
    parser.add_argument('--warmup', type=float, default=0.2)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--fps_warmup', type=int, default=5)
    parser.add_argument('--fps_repeats', type=int, default=20)
    parser.add_argument('--fixed_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--dynamic_res', action='store_true')

    parser.add_argument('--grid_levels', type=int, default=12)
    parser.add_argument('--grid_feat_dim', type=int, default=4)
    parser.add_argument('--base_resolution', type=int, default=9)
    parser.add_argument('--finest_resolution', type=int, default=32)
    parser.add_argument('--aspect_ratio', type=int, nargs=2, default=[16, 9])
    parser.add_argument('--time_scale', type=float, default=0.5)
    parser.add_argument('--pe_freq', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)

    parser.add_argument('--base_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--max_scale', type=float, default=1.2)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_dir', type=str, default='./output/resnet_baseline_1')
    parser.add_argument('--exp_name', type=str, default='residual_skip')
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--dump_images', action='store_true', default=False)
    parser.add_argument(
        '--loss_type',
        type=str,
        default='L2',
        choices=[
            'L2', 'L1', 'SSIM', 'Fusion1', 'Fusion2', 'Fusion3', 'Fusion4',
            'Fusion5', 'Fusion6', 'Fusion7', 'Fusion8', 'Fusion9', 'Fusion10',
            'Fusion11', 'Fusion12',
        ],
    )
    return parser


def main():
    args = build_parser().parse_args()
    train(args, model_class=ResidualSkipHybridGridNet)


if __name__ == '__main__':
    main()
