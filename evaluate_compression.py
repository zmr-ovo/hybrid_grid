import argparse
import json
import logging
import math
import os
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import DynamicVideoDataset
from train import seed_everything, seed_worker
from train_compression import (
    build_compression_model,
    compression_grid_metadata,
    compression_model_storage,
    compression_network_metadata_bits,
    evaluate_compression,
)


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(name)


def load_compression_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "未找到压缩检查点: {}".format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_config = checkpoint.get('config')
    if not isinstance(saved_config, dict):
        raise ValueError("压缩检查点缺少完整 config")

    config = Namespace(**saved_config)
    for name, value in (
        ('network_qat', False),
        ('network_quant_bits', 8),
        ('network_quant_start_epoch', 30),
        ('network_quant_freeze_epoch', 270),
    ):
        if not hasattr(config, name):
            setattr(config, name, value)
    try:
        model = build_compression_model(config, device)
    except AttributeError as error:
        raise ValueError("压缩检查点的模型配置不完整") from error

    if checkpoint.get('architecture') != model.architecture:
        raise ValueError(
            "压缩模型架构与检查点不一致: checkpoint={}, current={}".format(
                checkpoint.get('architecture'), model.architecture,
            )
        )
    model.load_state_dict(checkpoint['state_dict'])
    return model, config, int(checkpoint.get('epoch', -1)) + 1


def _evaluation_loader(config, data_root, batch_size, workers, seed):
    fixed_res = tuple(config.fixed_res)
    dataset = DynamicVideoDataset(
        data_root=data_root,
        base_res=tuple(config.base_res),
        fixed_res=fixed_res,
        frame_interval=config.frame_interval,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, fixed_res


def _storage_report(storage, bpp):
    return {
        'tensor_count': storage.tensor_count,
        'parameter_count': storage.parameter_count,
        'bits': storage.total_bits,
        'bytes': storage.total_bits // 8,
        'mib': storage.total_bits / 8 / 1024 ** 2,
        'bpp': bpp,
        'bits_by_dtype': dict(storage.bits_by_dtype),
    }


def build_evaluation_report(checkpoint_path, checkpoint_epoch, config,
                            data_root, frame_count, height, width, evaluation):
    rate = evaluation.rate
    if rate.estimated_grid_bits is None:
        raise ValueError("symbols evaluation must provide an estimated Grid rate")
    if rate.total_video_pixels != frame_count * height * width:
        raise ValueError("sequence dimensions do not match total_video_pixels")
    if not math.isclose(
        sum(evaluation.level_bits),
        rate.estimated_grid_bits,
        rel_tol=1e-6,
        abs_tol=1.0,
    ):
        raise ValueError("Grid level bits do not sum to estimated_grid_bits")

    metadata = rate.quantization_metadata
    qat_enabled = getattr(config, 'network_qat', False)
    qat_bits = getattr(config, 'network_quant_bits', 8)
    non_grid_key = (
        'non_grid_uint{}'.format(qat_bits)
        if qat_enabled else 'non_grid_fp32'
    )
    entropy_key = (
        'entropy_model_uint{}'.format(qat_bits)
        if qat_enabled else 'entropy_model_fp32'
    )

    report = {
        'schema_version': 3,
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'checkpoint_epoch': checkpoint_epoch,
        'quant_mode': 'symbols',
        'quant_step': config.quant_step,
        'network_qat': {
            'enabled': qat_enabled,
            'bits': qat_bits if qat_enabled else None,
            'start_epoch': (
                getattr(config, 'network_quant_start_epoch', 30)
                if qat_enabled else None
            ),
            'freeze_epoch': (
                getattr(config, 'network_quant_freeze_epoch', 270)
                if qat_enabled else None
            ),
        },
        'model_config': {
            'grid_levels': config.grid_levels,
            'grid_feat_dim': config.grid_feat_dim,
            'base_resolution': config.base_resolution,
            'finest_resolution': config.finest_resolution,
            'aspect_ratio': list(config.aspect_ratio),
            'time_scale': config.time_scale,
            'pe_freq': config.pe_freq,
            'hidden_dim': config.hidden_dim,
        },
        'sequence': {
            'data_root': str(Path(data_root).resolve()),
            'frames': frame_count,
            'height': height,
            'width': width,
            'total_video_pixels': rate.total_video_pixels,
        },
        'quality': {
            'psnr_db': evaluation.psnr,
            'ms_ssim': evaluation.msssim,
        },
        'rate': {
            'legacy_rate_per_value': rate.legacy_rate_per_value,
            'grid': {
                'level_bits': list(evaluation.level_bits),
                'bits': rate.estimated_grid_bits,
                'mib': rate.estimated_grid_bits / 8 / 1024 ** 2,
                'bpp': rate.estimated_grid_bpp,
            },
            'estimated_payload_subtotal': {
                'bits': rate.estimated_payload_bits,
                'mib': rate.estimated_payload_bits / 8 / 1024 ** 2,
                'bpp': rate.estimated_payload_bpp,
            },
            'quantization_metadata': {
                'level_count': metadata.level_count,
                'shapes': [list(shape) for shape in metadata.shapes],
                'quant_steps': list(metadata.quant_steps),
                'header_bits': metadata.header_bits,
                'shape_bits': metadata.shape_bits,
                'quant_step_bits': metadata.quant_step_bits,
                'bits': metadata.total_bits,
                'mib': metadata.total_bits / 8 / 1024 ** 2,
                'bpp': rate.quantization_metadata_bpp,
            },
            'network_quantization_metadata': {
                'bits': rate.network_quantization_metadata_bits,
                'mib': (
                    rate.network_quantization_metadata_bits / 8 / 1024 ** 2
                ),
                'bpp': rate.network_quantization_metadata_bpp,
            },
            'estimated_total': {
                'bits': rate.estimated_total_bits,
                'mib': rate.estimated_total_bits / 8 / 1024 ** 2,
                'bpp': rate.estimated_total_bpp,
            },
            'metadata_included': rate.metadata_included,
            'actual_bitstream_available': False,
        },
    }
    report['rate'][non_grid_key] = _storage_report(
        rate.non_grid_storage, rate.non_grid_bpp,
    )
    report['rate'][entropy_key] = _storage_report(
        rate.entropy_model_storage, rate.entropy_model_side_info_bpp,
    )
    return report


def format_evaluation_report(report):
    sequence = report['sequence']
    quality = report['quality']
    rate = report['rate']
    grid = rate['grid']
    network_qat = report['network_qat']
    storage_suffix = (
        'uint{}'.format(network_qat['bits']) if network_qat['enabled'] else 'fp32'
    )
    non_grid = rate['non_grid_' + storage_suffix]
    entropy = rate['entropy_model_' + storage_suffix]
    payload = rate['estimated_payload_subtotal']
    metadata = rate['quantization_metadata']
    network_metadata = rate['network_quantization_metadata']
    estimated_total = rate['estimated_total']
    lines = [
        '========== Compression Evaluation ==========',
        'Checkpoint: {}'.format(report['checkpoint']),
        'Checkpoint epoch: {}'.format(report['checkpoint_epoch']),
        'Quant mode: symbols',
        'Quant step: {}'.format(report['quant_step']),
        'Network QAT: {}'.format(
            '{}-bit'.format(network_qat['bits'])
            if network_qat['enabled'] else 'disabled'
        ),
        'Model config: {}'.format(json.dumps(report['model_config'])),
        'Frames: {}'.format(sequence['frames']),
        'Resolution: {} x {}'.format(sequence['height'], sequence['width']),
        'Total video pixels: {}'.format(sequence['total_video_pixels']),
        '',
        'Quality',
        'PSNR: {:.4f} dB'.format(quality['psnr_db']),
        'MS-SSIM: {:.6f}'.format(quality['ms_ssim']),
        '',
        'Estimated Grid rate',
        'Legacy rate/value: {:.6f}'.format(rate['legacy_rate_per_value']),
        'Grid bits: {:.0f}'.format(grid['bits']),
        'Grid size: {:.6f} MiB'.format(grid['mib']),
        'Grid BPP: {:.8f}'.format(grid['bpp']),
    ]
    lines.extend(
        'Grid level {} bits: {:.0f}'.format(index, bits)
        for index, bits in enumerate(grid['level_bits'])
    )
    lines.extend([
        '',
        'Non-Grid {} storage'.format(storage_suffix.upper()),
        'Tensors: {}'.format(non_grid['tensor_count']),
        'Parameters: {}'.format(non_grid['parameter_count']),
        'Bits: {}'.format(non_grid['bits']),
        'Size: {:.6f} MiB'.format(non_grid['mib']),
        'BPP: {:.8f}'.format(non_grid['bpp']),
        '',
        'Entropy model {} side information'.format(storage_suffix.upper()),
        'Tensors: {}'.format(entropy['tensor_count']),
        'Parameters: {}'.format(entropy['parameter_count']),
        'Bits: {}'.format(entropy['bits']),
        'Size: {:.6f} MiB'.format(entropy['mib']),
        'BPP: {:.8f}'.format(entropy['bpp']),
        '',
        'Estimated payload subtotal',
        'Bits: {:.0f}'.format(payload['bits']),
        'Size: {:.6f} MiB'.format(payload['mib']),
        'BPP: {:.8f}'.format(payload['bpp']),
        '',
        'Quantization metadata',
        'Levels: {}'.format(metadata['level_count']),
        'Header bits: {}'.format(metadata['header_bits']),
        'Shape bits: {}'.format(metadata['shape_bits']),
        'Quant-step bits: {}'.format(metadata['quant_step_bits']),
        'Bits: {}'.format(metadata['bits']),
        'Size: {:.6f} MiB'.format(metadata['mib']),
        'BPP: {:.8f}'.format(metadata['bpp']),
        '',
        'Network QAT metadata',
        'Bits: {}'.format(network_metadata['bits']),
        'Size: {:.6f} MiB'.format(network_metadata['mib']),
        'BPP: {:.8f}'.format(network_metadata['bpp']),
        '',
        'Estimated total',
        'Bits: {:.0f}'.format(estimated_total['bits']),
        'Size: {:.6f} MiB'.format(estimated_total['mib']),
        'BPP: {:.8f}'.format(estimated_total['bpp']),
        'Metadata included: YES',
        'Actual bitstream available: NO',
        'NOTE: estimated total is not actual total BPP.',
    ])
    return '\n'.join(lines) + '\n'


def save_evaluation_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'evaluation.json'
    text_path = output_dir / 'evaluation.txt'
    with json_path.open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    text_path.write_text(format_evaluation_report(report), encoding='utf-8')
    return text_path, json_path


def evaluate_checkpoint(args):
    if args.batch_size < 1 or args.workers < 0 or args.log_interval < 1:
        raise ValueError("batch_size/log_interval must be positive; workers >= 0")
    device = _device(args.device)
    seed_everything(args.seed)
    model, config, checkpoint_epoch = load_compression_model(
        args.checkpoint, device,
    )
    data_root = args.data_root or config.data_root
    loader, (height, width) = _evaluation_loader(
        config, data_root, args.batch_size, args.workers, args.seed,
    )
    total_video_pixels = len(loader.dataset) * height * width
    non_grid_storage, entropy_storage = compression_model_storage(model)
    quantization_metadata = compression_grid_metadata(model)
    network_metadata_bits = compression_network_metadata_bits(model)

    output_dir = args.output_dir or str(
        Path(args.checkpoint).resolve().parent / 'evaluation'
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                str(Path(output_dir) / 'evaluation.log'),
                mode='w',
                encoding='utf-8',
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("使用设备: %s", device)
    logging.info("使用 symbols 硬量化进行确定性评估")

    evaluation = evaluate_compression(
        model,
        loader,
        device,
        'symbols',
        total_video_pixels,
        non_grid_storage,
        entropy_storage,
        quantization_metadata,
        network_metadata_bits,
        save_dir=str(Path(output_dir) / 'reconstructions'),
        dump_images=args.dump_images,
        log_interval=args.log_interval,
    )
    report = build_evaluation_report(
        args.checkpoint,
        checkpoint_epoch,
        config,
        data_root,
        len(loader.dataset),
        height,
        width,
        evaluation,
    )
    text_path, json_path = save_evaluation_report(report, output_dir)
    for line in format_evaluation_report(report).splitlines():
        logging.info(line)
    logging.info("文本结果: %s", text_path)
    logging.info("JSON 结果: %s", json_path)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description='评估压缩 Hybrid Grid checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('-d', '--data_root')
    parser.add_argument('--output_dir')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--dump_images', action='store_true')
    return parser


if __name__ == '__main__':
    evaluate_checkpoint(build_parser().parse_args())
