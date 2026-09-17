import argparse
import json
import logging
import os
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gop_lora import FixedGOPPartition
from model import DynamicVideoDataset, HybridGridNet
from train import load_checkpoint, seed_everything, seed_worker
from util import msssim_fn, psnr_fn


MODEL_KEYS = (
    'grid_levels', 'grid_feat_dim', 'base_resolution', 'finest_resolution',
    'aspect_ratio', 'time_scale', 'pe_freq', 'hidden_dim',
)


@dataclass(frozen=True)
class GOPMetrics:
    gop_index: int
    start_frame: int
    stop_frame: int
    frames: int
    psnr: float
    msssim: float


@dataclass(frozen=True)
class GlobalGOPEvaluation:
    psnr: float
    msssim: float
    frames: int
    per_gop: tuple


def load_global_model(checkpoint_path, device):
    """Build a baseline HybridGridNet from a train.py checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "checkpoint not found: {}".format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get('config')
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint does not contain a valid config")
    missing = [key for key in MODEL_KEYS if key not in saved_config]
    if missing:
        raise ValueError(
            "checkpoint config is missing: {}".format(', '.join(missing))
        )

    config = Namespace(**saved_config)
    model = HybridGridNet(
        grid_levels=config.grid_levels,
        grid_feat_dim=config.grid_feat_dim,
        base_resolution=config.base_resolution,
        finest_resolution=config.finest_resolution,
        aspect_ratio=tuple(config.aspect_ratio),
        time_scale=config.time_scale,
        pe_freq=config.pe_freq,
        hidden_dim=config.hidden_dim,
    ).to(device)
    start_epoch, best_val_psnr, best_train_psnr = load_checkpoint(
        checkpoint_path, model, device, config,
    )
    return model, config, {
        'completed_epochs': start_epoch,
        'best_val_psnr': best_val_psnr,
        'best_train_psnr': best_train_psnr,
    }


@torch.no_grad()
def evaluate_global_gops(model, data_loader, device, total_frames, gop_size,
                         log_interval=50):
    """Evaluate global-time coordinates and aggregate metrics per GOP."""
    partition = FixedGOPPartition(total_frames, gop_size)
    accumulators = {
        index: {'psnr': 0.0, 'msssim': 0.0, 'frames': 0}
        for index in range(partition.num_gops)
    }
    total_psnr = 0.0
    total_msssim = 0.0
    samples_seen = 0
    was_training = model.training
    model.eval()

    for batch_index, batch in enumerate(data_loader):
        if 'frame_idx' not in batch:
            raise KeyError("evaluation batches must contain frame_idx")
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)
        prediction = model(coords)
        frame_indices = batch['frame_idx'].reshape(-1)
        if frame_indices.numel() != pixels.size(0):
            raise ValueError("frame_idx count must match the batch size")

        for item, frame_index in enumerate(frame_indices):
            frame_index = int(frame_index)
            position = partition.locate(frame_index)
            item_prediction = prediction[item:item + 1]
            item_pixels = pixels[item:item + 1]
            psnr = psnr_fn(item_prediction, item_pixels)
            msssim = msssim_fn(item_prediction, item_pixels, device)
            values = accumulators[position.gop_index]
            values['psnr'] += psnr
            values['msssim'] += msssim
            values['frames'] += 1
            total_psnr += psnr
            total_msssim += msssim
            samples_seen += 1

        if (batch_index % log_interval == 0 or
                batch_index == len(data_loader) - 1):
            logging.info(
                "Global eval [%d/%d] | frames: %d | PSNR: %.2f dB | "
                "MS-SSIM: %.4f",
                batch_index + 1,
                len(data_loader),
                samples_seen,
                total_psnr / samples_seen,
                total_msssim / samples_seen,
            )

    model.train(was_training)
    if samples_seen == 0:
        raise ValueError("evaluation data loader is empty")

    per_gop = []
    for gop_index, values in accumulators.items():
        if values['frames'] == 0:
            continue
        start, stop = partition.bounds(gop_index)
        per_gop.append(GOPMetrics(
            gop_index=gop_index,
            start_frame=start,
            stop_frame=stop,
            frames=values['frames'],
            psnr=values['psnr'] / values['frames'],
            msssim=values['msssim'] / values['frames'],
        ))
    return GlobalGOPEvaluation(
        psnr=total_psnr / samples_seen,
        msssim=total_msssim / samples_seen,
        frames=samples_seen,
        per_gop=tuple(per_gop),
    )


def build_report(checkpoint_path, model, config, checkpoint_info, dataset,
                 gop_size, result):
    return {
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'checkpoint_info': checkpoint_info,
        'model_parameters': sum(
            parameter.numel() for parameter in model.parameters()
        ),
        'model_config': {
            key: getattr(config, key) for key in MODEL_KEYS
        },
        'data': {
            'root': str(Path(dataset.frame_paths[0]).parent.resolve()),
            'total_frames': dataset.total_frames,
            'evaluated_frames': result.frames,
            'frame_interval': dataset.frame_interval,
            'resolution': list(dataset.fixed_res),
            'gop_size': gop_size,
        },
        'full_video': {
            'frames': result.frames,
            'psnr_db': result.psnr,
            'ms_ssim': result.msssim,
        },
        'gops': [asdict(metrics) for metrics in result.per_gop],
    }


def format_report(report):
    lines = [
        'Global-time GOP evaluation',
        'Checkpoint: {}'.format(report['checkpoint']),
        'Model parameters: {:,}'.format(report['model_parameters']),
        'Evaluated frames: {}'.format(report['data']['evaluated_frames']),
        'GOP size: {}'.format(report['data']['gop_size']),
        'FULL VIDEO | PSNR: {:.4f} dB | MS-SSIM: {:.6f}'.format(
            report['full_video']['psnr_db'],
            report['full_video']['ms_ssim'],
        ),
    ]
    for metrics in report['gops']:
        lines.append(
            'GOP {gop_index} | source frames: {start_frame}-{last_frame} | '
            'evaluated: {frames} | PSNR: {psnr:.4f} dB | '
            'MS-SSIM: {msssim:.6f}'.format(
                last_frame=metrics['stop_frame'] - 1,
                **metrics,
            )
        )
    return '\n'.join(lines) + '\n'


def save_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / 'global_gop_metrics.txt'
    json_path = output_dir / 'global_gop_metrics.json'
    text_path.write_text(format_report(report), encoding='utf-8')
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return text_path, json_path


def _dataset_config(config, args):
    fixed_res = (
        tuple(args.fixed_res)
        if args.fixed_res is not None
        else tuple(getattr(config, 'fixed_res', [720, 1280]))
    )
    base_res = tuple(getattr(config, 'base_res', fixed_res))
    frame_interval = (
        args.frame_interval
        if args.frame_interval is not None
        else int(getattr(config, 'frame_interval', 1))
    )
    return base_res, fixed_res, frame_interval


def run(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if args.gop_size < 1 or args.log_interval < 1:
        raise ValueError("gop_size and log_interval must be positive")
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config, checkpoint_info = load_global_model(
        args.checkpoint, device,
    )
    base_res, fixed_res, frame_interval = _dataset_config(config, args)
    dataset = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=base_res,
        fixed_res=fixed_res,
        frame_interval=frame_interval,
    )
    generator = torch.Generator().manual_seed(args.seed)
    data_loader = DataLoader(
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
    logging.info("Model parameters: %d", sum(
        parameter.numel() for parameter in model.parameters()
    ))
    logging.info(
        "Frames: %d | evaluated: %d | GOP size: %d | resolution: %s",
        dataset.total_frames, len(dataset), args.gop_size, fixed_res,
    )
    result = evaluate_global_gops(
        model, data_loader, device, dataset.total_frames, args.gop_size,
        args.log_interval,
    )
    report = build_report(
        args.checkpoint, model, config, checkpoint_info, dataset,
        args.gop_size, result,
    )
    text_path, json_path = save_report(report, args.output_dir)
    logging.info('\n%s', format_report(report).rstrip())
    logging.info("Saved text report: %s", text_path)
    logging.info("Saved JSON report: %s", json_path)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a global-time baseline checkpoint per GOP",
    )
    parser.add_argument('-d', '--data_root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gop_size', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fixed_res', type=int, nargs=2)
    parser.add_argument('--frame_interval', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--output_dir', default='./output/global_gop_eval')
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run(make_parser().parse_args())
