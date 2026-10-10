"""Evaluate a standalone ``.hgm`` model bitstream on a video sequence.

The evaluator never reads the training checkpoint.  It first performs real
entropy decoding, restores the complete model, reconstructs every source
frame and reports PSNR, MS-SSIM, exact file BPP and timing.  One-time entropy
decoding is reported separately from neural reconstruction FPS; the latter is
the inference speed appropriate for comparison in the thesis.

Both a human-readable text report and a machine-readable JSON report are
written to the requested output directory.
"""

import argparse
import json
from pathlib import Path
import re
import time

from PIL import Image
from piqa import MS_SSIM
import torch
from torch.utils.data import DataLoader

from compression.model_bitstream import load_model_bitstream
from fps import benchmark_fps
from model import DynamicVideoDataset
from train import seed_everything, seed_worker
from util import psnr_fn


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    return torch.device(name)


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _natural_key(path):
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r'(\d+)', path.name)
    ]


def _source_resolution(data_root):
    root = Path(data_root)
    paths = sorted(
        (
            path for path in root.iterdir()
            if path.is_file()
            and path.suffix.lower() in DynamicVideoDataset.IMAGE_EXTENSIONS
        ),
        key=_natural_key,
    )
    if not paths:
        raise ValueError('no supported image frames found in: {}'.format(root))
    with Image.open(paths[0]) as image:
        width, height = image.size
    return height, width


def _loader(data_root, resolution, batch_size, workers, seed,
            frame_interval):
    dataset = DynamicVideoDataset(
        data_root=data_root,
        base_res=resolution,
        fixed_res=resolution,
        frame_interval=frame_interval,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, log_interval=10):
    """Measure average per-frame quality using already decoded parameters."""
    model.eval()
    criterion = MS_SSIM(n_channels=3).to(device)
    psnr_sum = 0.0
    msssim_sum = 0.0
    frames = 0
    first_coords = None
    _synchronize(device)
    start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)
        if first_coords is None:
            first_coords = coords[:1].clone()
        prediction = model(coords)
        batch_size = pixels.shape[0]
        psnr_sum += psnr_fn(prediction, pixels) * batch_size
        msssim_sum += criterion(prediction, pixels).item() * batch_size
        frames += batch_size
        if batch_index % log_interval == 0 or batch_index == len(loader) - 1:
            print(
                'Evaluate [{}/{}] | PSNR: {:.4f} dB | MS-SSIM: {:.6f}'.format(
                    batch_index + 1,
                    len(loader),
                    psnr_sum / frames,
                    msssim_sum / frames,
                ),
                flush=True,
            )
    _synchronize(device)
    seconds = time.perf_counter() - start
    return {
        'psnr_db': psnr_sum / frames,
        'ms_ssim': msssim_sum / frames,
        'frames': frames,
        'evaluation_seconds': seconds,
    }, first_coords


def _part(bits, pixels):
    return {'bits': int(bits), 'bpp': bits / pixels}


def build_codec_report(bitstream_path, data_root, resolution, quality,
                       loaded, fps_result):
    """Build the complete report from exact file sizes and measured quality."""
    height, width = resolution
    total_pixels = quality['frames'] * height * width
    stats = loaded.stats
    timing = loaded.timing
    level_timing = [
        {
            'level': item.level,
            'symbols': item.symbol_count,
            'payload_bits': item.payload_bits,
            'distribution_seconds': item.distribution_seconds,
            'cdf_seconds': item.cdf_seconds,
            'arithmetic_seconds': item.arithmetic_seconds,
            'total_seconds': item.total_seconds,
        }
        for item in timing.grid_levels
    ]
    return {
        'bitstream': str(Path(bitstream_path).resolve()),
        'data_root': str(Path(data_root).resolve()),
        'checkpoint_required': False,
        'frames': quality['frames'],
        'resolution': {'height': height, 'width': width},
        'total_video_pixels': total_pixels,
        'quality': {
            'psnr_db': quality['psnr_db'],
            'ms_ssim': quality['ms_ssim'],
        },
        'rate': {
            'grid_file': _part(stats.grid_bits, total_pixels),
            'non_grid_parameters': _part(
                stats.non_grid_parameter_bits, total_pixels,
            ),
            'entropy_parameters': _part(
                stats.entropy_parameter_bits, total_pixels,
            ),
            'manifest': _part(stats.manifest_bits, total_pixels),
            'container_overhead': _part(
                stats.container_overhead_bits, total_pixels,
            ),
            'actual_total': _part(stats.total_bits, total_pixels),
        },
        'timing': {
            'model_build_seconds': timing.model_build_seconds,
            'parameter_restore_seconds': timing.parameter_restore_seconds,
            'grid_entropy_decode_seconds': timing.grid_decode_seconds,
            'complete_model_restore_seconds': timing.total_seconds,
            'quality_evaluation_seconds': quality['evaluation_seconds'],
            'grid_levels': level_timing,
        },
        'fps': {
            'paper_inference_fps': fps_result.paper_inference_fps,
            'full_reconstruction_fps': fps_result.reconstruction_fps,
            'decoder_only_fps': fps_result.decoder_fps,
            'warmup_steps': fps_result.warmup_steps,
            'repeat_steps': fps_result.repeat_steps,
            'includes_entropy_decode': False,
        },
    }


def format_codec_report(report):
    rate = report['rate']
    timing = report['timing']
    fps = report['fps']
    lines = [
        '========== Complete Codec Evaluation ==========',
        'Bitstream: ' + report['bitstream'],
        'Data: ' + report['data_root'],
        'Checkpoint required: NO',
        'Frames: {}'.format(report['frames']),
        'Resolution: {height} x {width}'.format(**report['resolution']),
        'PSNR: {:.4f} dB'.format(report['quality']['psnr_db']),
        'MS-SSIM: {:.6f}'.format(report['quality']['ms_ssim']),
        'Actual total: {} bits | {:.9f} BPP'.format(
            rate['actual_total']['bits'], rate['actual_total']['bpp'],
        ),
        'Grid file: {} bits | {:.9f} BPP'.format(
            rate['grid_file']['bits'], rate['grid_file']['bpp'],
        ),
        'Non-Grid parameters: {} bits | {:.9f} BPP'.format(
            rate['non_grid_parameters']['bits'],
            rate['non_grid_parameters']['bpp'],
        ),
        'Entropy parameters: {} bits | {:.9f} BPP'.format(
            rate['entropy_parameters']['bits'],
            rate['entropy_parameters']['bpp'],
        ),
        'Manifest: {} bits | {:.9f} BPP'.format(
            rate['manifest']['bits'], rate['manifest']['bpp'],
        ),
        'Container overhead: {} bits | {:.9f} BPP'.format(
            rate['container_overhead']['bits'],
            rate['container_overhead']['bpp'],
        ),
        'Grid entropy decode: {:.3f} s'.format(
            timing['grid_entropy_decode_seconds'],
        ),
        'Complete model restore: {:.3f} s'.format(
            timing['complete_model_restore_seconds'],
        ),
        'Paper inference FPS: {:.4f}'.format(fps['paper_inference_fps']),
        'Decoder-only FPS: {:.4f}'.format(fps['decoder_only_fps']),
        'NOTE: inference FPS excludes one-time entropy decoding.',
    ]
    for item in timing['grid_levels']:
        lines.append(
            'Grid L{level}: {symbols} symbols | distribution '
            '{distribution_seconds:.3f} s | CDF {cdf_seconds:.3f} s | '
            'arithmetic {arithmetic_seconds:.3f} s | total '
            '{total_seconds:.3f} s'.format(**item)
        )
    return '\n'.join(lines)


def save_codec_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / 'codec_evaluation.txt'
    json_path = output_dir / 'codec_evaluation.json'
    text_path.write_text(format_codec_report(report) + '\n', encoding='utf-8')
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return text_path, json_path


def run(args):
    seed_everything(args.seed)
    device = _device(args.device)
    resolution = (
        tuple(args.fixed_res)
        if args.fixed_res is not None
        else _source_resolution(args.data_root)
    )
    # Validate all dataset inputs before starting the expensive entropy decode.
    loader = _loader(
        args.data_root,
        resolution,
        args.batch_size,
        args.workers,
        args.seed,
        args.frame_interval,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def decode_progress(item):
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

    print('Device: {}'.format(device))
    print('Bitstream: {}'.format(args.bitstream))
    loaded = load_model_bitstream(
        args.bitstream, device, progress=decode_progress,
    )
    reconstruction_model = loaded.model.reconstruction_model
    quality, benchmark_coords = evaluate_reconstruction(
        reconstruction_model,
        loader,
        device,
        args.log_interval,
    )
    fps_result = benchmark_fps(
        reconstruction_model,
        benchmark_coords,
        warmup_steps=args.warmup_steps,
        repeat_steps=args.repeat_steps,
    )
    report = build_codec_report(
        args.bitstream,
        args.data_root,
        resolution,
        quality,
        loaded,
        fps_result,
    )
    text_path, json_path = save_codec_report(report, args.output_dir)
    print(format_codec_report(report))
    print('Saved text report: {}'.format(text_path))
    print('Saved JSON report: {}'.format(json_path))
    return report


def make_parser():
    parser = argparse.ArgumentParser(
        description='评测真实 Hybrid Grid完整模型码流',
    )
    parser.add_argument('--bitstream', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                        default='auto')
    parser.add_argument('--fixed_res', type=int, nargs=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--repeat_steps', type=int, default=20)
    parser.add_argument('--output_dir', default='./output/codec_evaluation')
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
