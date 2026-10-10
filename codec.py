"""Command-line interface for the complete Hybrid Grid model bitstream.

Commands
--------
``compress`` converts a trained compression checkpoint into a standalone
``.hgm`` file. ``inspect`` reports its configuration and exact storage
breakdown without constructing the neural network. ``decompress`` restores
the inference model and verifies that every Grid level can be decoded without
access to the original checkpoint.

This tool handles model coding only.  Dataset PSNR, MS-SSIM, BPP and timing
experiments belong to the later ``evaluate_codec.py`` evaluation entry point.
"""

import argparse
from pathlib import Path
import time

import torch

from compression.model_bitstream import (
    inspect_model_bitstream,
    load_model_bitstream,
    write_model_bitstream,
)
from compression.network_quantization import network_qat_state
from evaluate_compression import load_compression_model


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    return torch.device(name)


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _mib(bits):
    return bits / 8 / 1024 ** 2


def _print_stats(stats):
    items = (
        ('Grid file', stats.grid_bits),
        ('non-Grid parameters', stats.non_grid_parameter_bits),
        ('entropy parameters', stats.entropy_parameter_bits),
        ('manifest', stats.manifest_bits),
        ('container overhead', stats.container_overhead_bits),
        ('TOTAL', stats.total_bits),
    )
    for label, bits in items:
        print('{}: {} bits ({:.6f} MiB)'.format(label, bits, _mib(bits)))


def compress_checkpoint(checkpoint, output, device_name='auto',
                        cdf_precision=16):
    device = _device(device_name)
    checkpoint = Path(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, config, epoch = load_compression_model(checkpoint, device)
    model.eval()

    print('Command: compress')
    print('Device: {}'.format(device))
    print('Checkpoint: {}'.format(checkpoint))
    print('Checkpoint epoch: {}'.format(epoch))
    print('Architecture: {}'.format(model.architecture))
    print('Entropy model: {}'.format(model.entropy_model_type))
    print('Network QAT: {}'.format(network_qat_state(model)))

    def progress(level, symbols, encoded_level, estimated_bits):
        print(
            'Encoded Grid L{} | symbols: {} | estimated: {:.0f} bits | '
            'actual payload: {} bits'.format(
                level,
                symbols.numel(),
                estimated_bits,
                len(encoded_level.payload) * 8,
            ),
            flush=True,
        )

    _synchronize(device)
    start = time.perf_counter()
    stats = write_model_bitstream(
        output,
        model,
        config,
        cdf_precision=cdf_precision,
        progress=progress,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - start

    print('Output: {}'.format(output))
    _print_stats(stats)
    print('Encoding time: {:.3f} s'.format(elapsed))
    return stats


def inspect_bitstream(path):
    path = Path(path)
    info = inspect_model_bitstream(path)
    config = info.config
    print('Command: inspect')
    print('Input: {}'.format(path))
    print('Architecture: {}'.format(info.architecture))
    print('Entropy model: {}'.format(config['entropy_model']))
    print('Network QAT: {}'.format(config['network_qat']))
    print('Network quantization bits: {}'.format(
        config['network_quant_bits'] if config['network_qat'] else 'N/A'
    ))
    print('Grid levels: {}'.format(len(info.grid_shapes)))
    print('Grid shapes: {}'.format(info.grid_shapes))
    print('Quantization steps: {}'.format(info.quant_steps))
    print('Estimated Grid bits: {:.0f}'.format(info.estimated_grid_bits))
    print('Actual arithmetic payload bits: {}'.format(
        info.actual_grid_payload_bits
    ))
    _print_stats(info.stats)
    return info


def decompress_bitstream(path, device_name='auto'):
    device = _device(device_name)
    path = Path(path)
    print('Command: decompress')
    print('Device: {}'.format(device))
    print('Input: {}'.format(path))

    def progress(item):
        print(
            'Decoded Grid L{} | symbols: {} | payload: {} bits | '
            'distribution: {:.3f} s | CDF: {:.3f} s | arithmetic: {:.3f} s '
            '| total: {:.3f} s'.format(
                item.level,
                item.symbol_count,
                item.payload_bits,
                item.distribution_seconds,
                item.cdf_seconds,
                item.arithmetic_seconds,
                item.total_seconds,
            ),
            flush=True,
        )

    _synchronize(device)
    start = time.perf_counter()
    loaded = load_model_bitstream(path, device, progress=progress)
    _synchronize(device)
    elapsed = time.perf_counter() - start

    parameter_count = sum(
        parameter.numel() for parameter in loaded.model.parameters()
    )
    shapes = tuple(
        tuple(level.grid.shape)
        for level in loaded.model.reconstruction_model.grid_encoder.levels
    )
    print('Architecture: {}'.format(loaded.model.architecture))
    print('Model parameters: {}'.format(parameter_count))
    print('Decoded Grid levels: {}'.format(len(shapes)))
    print('Decoded Grid shapes: {}'.format(shapes))
    print('Network QAT: {}'.format(network_qat_state(loaded.model)))
    _print_stats(loaded.stats)
    print('Model build time: {:.3f} s'.format(
        loaded.timing.model_build_seconds
    ))
    print('Parameter restore time: {:.3f} s'.format(
        loaded.timing.parameter_restore_seconds
    ))
    print('Grid entropy decode time: {:.3f} s'.format(
        loaded.timing.grid_decode_seconds
    ))
    print('Decoding time: {:.3f} s'.format(elapsed))
    print('Checkpoint required: NO')
    return loaded


def make_parser():
    parser = argparse.ArgumentParser(
        description='Hybrid Grid真实模型编解码工具',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    compress = subparsers.add_parser(
        'compress', help='将训练 checkpoint 转换为完整 .hgm 码流',
    )
    compress.add_argument('--checkpoint', required=True)
    compress.add_argument('--output', required=True)
    compress.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                          default='auto')
    compress.add_argument('--cdf_precision', type=int, default=16)

    inspect = subparsers.add_parser(
        'inspect', help='不构造模型，查看 .hgm 配置与真实大小',
    )
    inspect.add_argument('--input', required=True)

    decompress = subparsers.add_parser(
        'decompress', help='不依赖 checkpoint，恢复并验证完整模型',
    )
    decompress.add_argument('--input', required=True)
    decompress.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                            default='auto')
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == 'compress':
        return compress_checkpoint(
            args.checkpoint,
            args.output,
            args.device,
            args.cdf_precision,
        )
    if args.command == 'inspect':
        return inspect_bitstream(args.input)
    return decompress_bitstream(args.input, args.device)


if __name__ == '__main__':
    main()
