"""Training entry for model_1 Hybrid Grid rate-distortion optimization.

This entry supports the same Grid quantization, entropy models, mixed network
QAT, checkpointing and rate reports as ``train_compression.py``.
"""

from train_compression import build_parser, train_compression

from resnet_compression.model import build_resnet_compression_model


def make_parser():
    parser = build_parser()
    parser.description = 'model_1 Additive-Skip Hybrid Grid 率失真训练'
    parser.set_defaults(
        reconstruction_model='additive_skip',
        out_dir='./output/resnet_compression',
        exp_name='additive_skip_compression',
    )
    return parser


def main():
    args = make_parser().parse_args()
    train_compression(args, model_builder=build_resnet_compression_model)


if __name__ == '__main__':
    main()
