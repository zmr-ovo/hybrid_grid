"""Independent calibrated entry; shares only the new Additive training loop."""

from scalable_unified_additive_811.train import (
    build_parser as build_plain_parser,
    build_model as build_plain_model,
    train as train_plain,
    save_checkpoint,
    load_checkpoint,
)
from .model import CalibratedUnifiedAdditive811


def build_parser():
    parser = build_plain_parser()
    parser.description = "Unified 811 + 原Additive Decoder + Base/Mid特征校准"
    parser.set_defaults(
        out_dir="./output/scalable_unified_additive_811_calibrated",
        exp_name="bunny_unified_additive_811_calibrated_ema_20_30_50_300epoch",
    )
    return parser


def build_model(args):
    return build_plain_model(args, CalibratedUnifiedAdditive811)


def train(args):
    return train_plain(args, CalibratedUnifiedAdditive811)


if __name__ == "__main__":
    train(build_parser().parse_args())
