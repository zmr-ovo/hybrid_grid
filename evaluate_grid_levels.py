import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_compression import _device, load_compression_model
from model import DynamicVideoDataset
from train import seed_everything, seed_worker
from util import msssim_fn, psnr_fn


def quantized_grids(model):
    """Return the checkpoint Grids quantized with their configured steps."""
    levels = tuple(model.reconstruction_model.grid_encoder.levels)
    if len(levels) != len(model.quant_steps):
        raise ValueError("Grid levels and quantization steps do not match")
    return tuple(
        torch.round(level.grid / step) * step
        for level, step in zip(levels, model.quant_steps)
    )


def leave_one_out_grids(grids, removed_level):
    if not isinstance(grids, (tuple, list)) or not grids:
        raise ValueError("grids must be a non-empty tuple or list")
    if (not isinstance(removed_level, int) or
            isinstance(removed_level, bool) or
            not 0 <= removed_level < len(grids)):
        raise ValueError("removed_level is out of range")
    return tuple(
        torch.zeros_like(grid) if index == removed_level else grid
        for index, grid in enumerate(grids)
    )


def prefix_grids(grids, last_level):
    if not isinstance(grids, (tuple, list)) or not grids:
        raise ValueError("grids must be a non-empty tuple or list")
    if (not isinstance(last_level, int) or isinstance(last_level, bool) or
            not -1 <= last_level < len(grids)):
        raise ValueError("last_level is out of range")
    return tuple(
        grid if index <= last_level else torch.zeros_like(grid)
        for index, grid in enumerate(grids)
    )


def _quality_accumulator(level_count):
    return {
        'full': [0.0, 0.0],
        'zero': [0.0, 0.0],
        'removed': [[0.0, 0.0] for _ in range(level_count)],
        'prefix': [[0.0, 0.0] for _ in range(level_count)],
        'samples': 0,
    }


def _accumulate_quality(accumulator, key, prediction, target, device,
                        level=None):
    values = accumulator[key] if level is None else accumulator[key][level]
    for item_prediction, item_target in zip(prediction, target):
        item_prediction = item_prediction.unsqueeze(0)
        item_target = item_target.unsqueeze(0)
        values[0] += psnr_fn(item_prediction, item_target)
        values[1] += msssim_fn(item_prediction, item_target, device)


@torch.no_grad()
def evaluate_grid_levels(model, loader, device, log_interval=10):
    """Evaluate full, zero, leave-one-out, and coarse-to-fine prefixes."""
    if log_interval < 1:
        raise ValueError("log_interval must be positive")
    was_training = model.training
    model.eval()
    grids = tuple(grid.detach() for grid in quantized_grids(model))
    level_count = len(grids)
    zero = tuple(torch.zeros_like(grid) for grid in grids)
    removed = tuple(
        tuple(
            zero[index] if index == level else grid
            for index, grid in enumerate(grids)
        )
        for level in range(level_count)
    )
    prefixes = tuple(
        tuple(
            grid if index <= level else zero[index]
            for index, grid in enumerate(grids)
        )
        for level in range(level_count - 1)
    )
    quality = _quality_accumulator(level_count)
    level_bits = None

    for batch_index, batch in enumerate(loader):
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)

        if level_bits is None:
            full_output = model(coords, quant_mode='symbols')
            if full_output.rate is None:
                raise RuntimeError("symbols evaluation did not produce a rate")
            full_prediction = full_output.reconstruction
            level_bits = tuple(
                float(bits.item()) for bits in full_output.rate.level_bits
            )
        else:
            full_prediction = model.reconstruction_model(coords, grids=grids)

        _accumulate_quality(
            quality, 'full', full_prediction, pixels, device,
        )
        _accumulate_quality(
            quality,
            'zero',
            model.reconstruction_model(coords, grids=zero),
            pixels,
            device,
        )
        for level, candidate in enumerate(removed):
            prediction = model.reconstruction_model(coords, grids=candidate)
            _accumulate_quality(
                quality, 'removed', prediction, pixels, device, level,
            )
        for level, candidate in enumerate(prefixes):
            prediction = model.reconstruction_model(coords, grids=candidate)
            _accumulate_quality(
                quality, 'prefix', prediction, pixels, device, level,
            )
        quality['samples'] += pixels.shape[0]

        if batch_index % log_interval == 0 or batch_index == len(loader) - 1:
            logging.info(
                "Grid analysis [%d/%d] | full PSNR: %.2f dB",
                batch_index + 1,
                len(loader),
                quality['full'][0] / quality['samples'],
            )

    if level_bits is None or quality['samples'] == 0:
        raise ValueError("evaluation loader must not be empty")
    # The complete prefix is exactly the full quantized model; do not run it twice.
    quality['prefix'][-1] = list(quality['full'])
    model.train(was_training)
    return grids, level_bits, quality


def _average_quality(values, samples):
    return {
        'psnr_db': values[0] / samples,
        'ms_ssim': values[1] / samples,
    }


def build_grid_level_report(checkpoint, checkpoint_epoch, config, data_root,
                            height, width, grids, level_bits, quality):
    level_count = len(grids)
    if len(level_bits) != level_count:
        raise ValueError("level_bits must contain one value per Grid level")
    if len(quality['removed']) != level_count:
        raise ValueError("removed quality count does not match Grid levels")
    if len(quality['prefix']) != level_count:
        raise ValueError("prefix quality count does not match Grid levels")
    samples = quality['samples']
    if samples < 1:
        raise ValueError("quality samples must be positive")
    total_video_pixels = samples * height * width
    full = _average_quality(quality['full'], samples)
    zero = _average_quality(quality['zero'], samples)

    contributions = []
    for level, (grid, step, bits, removed_values) in enumerate(zip(
        grids, config.quant_steps or [config.quant_step] * level_count,
        level_bits, quality['removed'],
    )):
        removed_quality = _average_quality(removed_values, samples)
        contributions.append({
            'level': level,
            'quant_step': float(step),
            'parameters': grid.numel(),
            'estimated_bits': float(bits),
            'bits_per_value': float(bits) / grid.numel(),
            'grid_bpp': float(bits) / total_video_pixels,
            'removed_quality': removed_quality,
            'psnr_drop_db': full['psnr_db'] - removed_quality['psnr_db'],
            'ms_ssim_drop': full['ms_ssim'] - removed_quality['ms_ssim'],
        })

    prefixes = []
    cumulative_bits = 0.0
    previous_bits = 0.0
    previous_quality = zero
    for level, (bits, values) in enumerate(zip(level_bits, quality['prefix'])):
        cumulative_bits += float(bits)
        current_quality = _average_quality(values, samples)
        delta_bits = cumulative_bits - previous_bits
        delta_psnr = current_quality['psnr_db'] - previous_quality['psnr_db']
        prefixes.append({
            'last_level': level,
            'name': 'L0' if level == 0 else 'L0-L{}'.format(level),
            'cumulative_bits': cumulative_bits,
            'grid_bpp': cumulative_bits / total_video_pixels,
            'quality': current_quality,
            'delta_bits': delta_bits,
            'delta_grid_bpp': delta_bits / total_video_pixels,
            'delta_psnr_db': delta_psnr,
            'psnr_gain_per_mbit': (
                delta_psnr / (delta_bits / 1_000_000)
                if delta_bits > 0 else None
            ),
        })
        previous_bits = cumulative_bits
        previous_quality = current_quality

    return {
        'schema_version': 1,
        'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_epoch': checkpoint_epoch,
        'entropy_model': config.entropy_model,
        'quant_mode': 'symbols',
        'sequence': {
            'data_root': str(Path(data_root).resolve()),
            'frames': samples,
            'height': height,
            'width': width,
            'total_video_pixels': total_video_pixels,
        },
        'full_quality': full,
        'zero_grid_quality': zero,
        'estimated_grid_bits': sum(level_bits),
        'estimated_grid_bpp': sum(level_bits) / total_video_pixels,
        'level_contributions': contributions,
        'prefixes': prefixes,
        'notes': {
            'leave_one_out_rate': (
                'Quality-only ablation. Removing an intermediate level breaks '
                'the trained entropy-condition chain, so no ablated rate is claimed.'
            ),
            'prefix_rate': (
                'Cumulative prefix bits preserve the coarse-to-fine entropy '
                'decoding order and are valid estimated Grid rates.'
            ),
            'actual_bitstream_available': False,
        },
    }


def format_grid_level_report(report):
    sequence = report['sequence']
    full = report['full_quality']
    zero = report['zero_grid_quality']
    lines = [
        '========== Grid Level Analysis ==========',
        'Checkpoint: {}'.format(report['checkpoint']),
        'Checkpoint epoch: {}'.format(report['checkpoint_epoch']),
        'Entropy model: {}'.format(report['entropy_model']),
        'Frames: {}'.format(sequence['frames']),
        'Resolution: {} x {}'.format(sequence['height'], sequence['width']),
        'Full PSNR: {:.4f} dB'.format(full['psnr_db']),
        'Full MS-SSIM: {:.6f}'.format(full['ms_ssim']),
        'Zero-Grid PSNR: {:.4f} dB'.format(zero['psnr_db']),
        'Estimated Grid bits: {:.0f}'.format(report['estimated_grid_bits']),
        'Estimated Grid BPP: {:.8f}'.format(report['estimated_grid_bpp']),
        '',
        'Leave-one-out quality contribution',
        'level step params bits bits/value grid_bpp removed_psnr psnr_drop '
        'removed_ms_ssim ms_ssim_drop',
    ]
    for item in report['level_contributions']:
        removed = item['removed_quality']
        lines.append(
            'L{level} {quant_step:.6g} {parameters} {estimated_bits:.0f} '
            '{bits_per_value:.6f} {grid_bpp:.8f} {psnr:.4f} '
            '{psnr_drop_db:.4f} {msssim:.6f} {ms_ssim_drop:.6f}'.format(
                psnr=removed['psnr_db'],
                msssim=removed['ms_ssim'],
                **item
            )
        )
    lines.extend([
        '',
        'Coarse-to-fine prefix quality',
        'prefix cumulative_bits grid_bpp psnr ms_ssim delta_bits '
        'delta_psnr psnr_gain_per_mbit',
    ])
    for item in report['prefixes']:
        gain = item['psnr_gain_per_mbit']
        lines.append(
            '{name} {cumulative_bits:.0f} {grid_bpp:.8f} {psnr:.4f} '
            '{msssim:.6f} {delta_bits:.0f} {delta_psnr_db:.4f} {gain}'.format(
                psnr=item['quality']['psnr_db'],
                msssim=item['quality']['ms_ssim'],
                gain='N/A' if gain is None else '{:.6f}'.format(gain),
                **item
            )
        )
    lines.extend([
        '',
        'NOTE: leave-one-out reports quality only; its rate is not a valid '
        'conditional bitstream rate.',
        'NOTE: all rates are entropy estimates; actual bitstream is unavailable.',
    ])
    return '\n'.join(lines) + '\n'


def save_grid_level_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'grid_level_analysis.json'
    text_path = output_dir / 'grid_level_analysis.txt'
    with json_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    text_path.write_text(format_grid_level_report(report), encoding='utf-8')
    return text_path, json_path


def analyze_checkpoint(args):
    if args.batch_size < 1 or args.workers < 0 or args.log_interval < 1:
        raise ValueError("batch_size/log_interval must be positive; workers >= 0")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    device = _device(args.device)
    seed_everything(args.seed)
    model, config, checkpoint_epoch = load_compression_model(
        args.checkpoint, device,
    )
    data_root = args.data_root or config.data_root
    fixed_res = tuple(config.fixed_res)
    dataset = DynamicVideoDataset(
        data_root=data_root,
        base_res=tuple(config.base_res),
        fixed_res=fixed_res,
        frame_interval=config.frame_interval,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        worker_init_fn=seed_worker,
        generator=generator,
    )
    logging.info("Device: %s", device)
    logging.info("Checkpoint: %s", args.checkpoint)
    logging.info("Frames: %d | Grid levels: %d", len(dataset), config.grid_levels)
    grids, level_bits, quality = evaluate_grid_levels(
        model, loader, device, args.log_interval,
    )
    report = build_grid_level_report(
        args.checkpoint,
        checkpoint_epoch,
        config,
        data_root,
        fixed_res[0],
        fixed_res[1],
        grids,
        level_bits,
        quality,
    )
    output_dir = args.output_dir or str(
        Path(args.checkpoint).resolve().parent / 'grid_level_analysis'
    )
    text_path, json_path = save_grid_level_report(report, output_dir)
    print(format_grid_level_report(report), end='')
    logging.info("Saved text report: %s", text_path)
    logging.info("Saved JSON report: %s", json_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Analyze leave-one-out and coarse-to-fine Grid quality',
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('-d', '--data_root')
    parser.add_argument('--output_dir')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto',
    )
    return parser


if __name__ == '__main__':
    analyze_checkpoint(build_parser().parse_args())
