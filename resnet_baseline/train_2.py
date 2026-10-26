"""训练入口：未压缩 Hybrid Grid，输入重注入残差 decoder（model_2）。

复用稳定的基线数据、检查点、指标与 FPS 流程，不开启量化、熵模型或可缩放前缀。
模型为 :class:`resnet_baseline.model_2.DeepResNetHybridGridNet`。默认使用两个无归一化
瓶颈残差块，并在两个块之间重新注入 Grid + PE 原始特征。
"""

import argparse
from functools import partial

from train import train

from resnet_baseline.model_2 import DeepResNetHybridGridNet


def build_parser():
    parser = argparse.ArgumentParser(
        description="训练未压缩 Hybrid Grid 深度残差 Decoder 基线"
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

    # 深度残差 decoder 专属超参
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='残差块个数（至少 2，默认 2）')
    parser.add_argument('--ff_dim', type=int, default=None,
                        help='每个块内 FFN 宽度；缺省时自动求解以对齐原 decoder 参数量')

    parser.add_argument('--base_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--max_scale', type=float, default=1.2)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_dir', type=str, default='./output/resnet_baseline_2')
    parser.add_argument('--exp_name', type=str, default='deep_resnet')
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

    # train() 固定以 8 个配置参数构造 model_class(...)，这里用 partial 把
    # decoder 专属的 num_blocks / ff_dim 预先绑定进去。
    model_class = partial(
        DeepResNetHybridGridNet,
        num_blocks=args.num_blocks,
        ff_dim=args.ff_dim,
    )
    train(args, model_class=model_class)


if __name__ == '__main__':
    main()
