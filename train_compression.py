import argparse
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from compression.model import CompressedHybridGridNet
from compression.network_quantization import (
    configure_network_qat,
    network_qat_state,
    network_qat_storage,
    prepare_network_qat,
)
from compression.rate import (
    Fp32RateBreakdown,
    ParameterStorage,
    estimate_fp32_rate,
    estimate_grid_metadata,
    parameter_storage,
)
from model import DynamicVideoDataset, HybridGridNet
from train import seed_everything, seed_worker, setup_logging
from util import NervLoss, msssim_fn, psnr_fn


CHECKPOINT_CONFIG_KEYS = (
    'grid_levels', 'grid_feat_dim', 'base_resolution', 'finest_resolution',
    'aspect_ratio', 'time_scale', 'pe_freq', 'hidden_dim', 'quant_step',
    'epochs', 'warmup_epochs', 'symbol_start_epoch', 'lambda_max',
    'network_qat', 'network_quant_bits', 'network_quant_start_epoch',
    'network_quant_freeze_epoch',
)


@dataclass(frozen=True)
class CompressionStage:
    quant_mode: str
    lambda_rate: float


@dataclass(frozen=True)
class CompressionEvaluation:
    psnr: float
    msssim: float
    rate: Fp32RateBreakdown
    level_bits: Tuple[float, ...]


def compression_stage(
    epoch,
    epochs=300,
    warmup_epochs=30,
    symbol_start_epoch=270,
    lambda_max=5e-4,
):
    """Return the quantization mode and rate weight for one epoch."""
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError("epoch must be an integer")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (epochs, warmup_epochs, symbol_start_epoch)
    ):
        raise TypeError("epoch boundaries must be integers")
    if epochs < 1 or not 0 <= warmup_epochs < symbol_start_epoch < epochs:
        raise ValueError(
            "expected 0 <= warmup_epochs < symbol_start_epoch < epochs"
        )
    if not math.isfinite(lambda_max) or lambda_max < 0:
        raise ValueError("lambda_max must be finite and non-negative")
    if not 0 <= epoch < epochs:
        raise ValueError("epoch must be in [0, epochs)")

    if epoch < warmup_epochs:
        return CompressionStage('disabled', 0.0)
    if epoch >= symbol_start_epoch:
        return CompressionStage('symbols', float(lambda_max))

    joint_epochs = symbol_start_epoch - warmup_epochs
    progress = (epoch - warmup_epochs + 1) / joint_epochs
    return CompressionStage('noise', float(lambda_max) * progress)


def rate_distortion_loss(distortion, output, lambda_rate):
    """Combine reconstruction distortion with rate per Grid value."""
    if lambda_rate < 0 or not math.isfinite(lambda_rate):
        raise ValueError("lambda_rate must be finite and non-negative")
    if output.rate is None:
        if lambda_rate != 0:
            raise ValueError("rate is required when lambda_rate is positive")
        rate = distortion.new_zeros(())
    else:
        rate = output.rate.bits_per_value
    return distortion + lambda_rate * rate, rate


def compression_model_storage(model):
    qat_storage = network_qat_storage(model)
    if qat_storage:
        non_grid = qat_storage['non_grid']
        entropy = qat_storage['entropy']
        return (
            ParameterStorage(
                non_grid.tensor_count,
                non_grid.parameter_count,
                non_grid.payload_bits,
                (('uint{}'.format(model.network_quant_bits),
                  non_grid.payload_bits),),
            ),
            ParameterStorage(
                entropy.tensor_count,
                entropy.parameter_count,
                entropy.payload_bits,
                (('uint{}'.format(model.network_quant_bits),
                  entropy.payload_bits),),
            ),
        )

    grid_parameter_ids = {
        id(level.grid)
        for level in model.reconstruction_model.grid_encoder.levels
    }
    non_grid_parameters = (
        parameter
        for parameter in model.reconstruction_model.parameters()
        if id(parameter) not in grid_parameter_ids
    )
    return (
        parameter_storage(non_grid_parameters, required_dtype=torch.float32),
        parameter_storage(
            model.entropy_models.parameters(), required_dtype=torch.float32,
        ),
    )


def compression_network_metadata_bits(model):
    return sum(
        storage.metadata_bits
        for storage in network_qat_storage(model).values()
    )


def compression_grid_metadata(model):
    grids = tuple(
        level.grid for level in model.reconstruction_model.grid_encoder.levels
    )
    return estimate_grid_metadata(grids, model.quant_steps)


def _averaged_rate_summary(grid_bits_sum, rate_sum, samples_seen,
                           total_video_pixels, non_grid_storage,
                           entropy_model_storage, quantization_metadata,
                           network_metadata_bits=0):
    if grid_bits_sum is None:
        return estimate_fp32_rate(
            total_video_pixels,
            non_grid_storage,
            entropy_model_storage,
            quantization_metadata,
            network_quantization_metadata_bits=network_metadata_bits,
        )
    return estimate_fp32_rate(
        total_video_pixels,
        non_grid_storage,
        entropy_model_storage,
        quantization_metadata,
        grid_bits=grid_bits_sum / samples_seen,
        legacy_rate_per_value=rate_sum / samples_seen,
        network_quantization_metadata_bits=network_metadata_bits,
    )


def _mib(bits):
    return bits / 8 / 1024 ** 2


def _storage_label(storage):
    if len(storage.bits_by_dtype) == 1:
        return storage.bits_by_dtype[0][0].upper()
    return 'MIXED'


def _log_rate_summary(logger, prefix, summary, level_bits=()):
    non_grid = summary.non_grid_storage
    entropy = summary.entropy_model_storage
    logger.info(
        "%s BIT | video pixels: %d | non-Grid %s: %d tensors, %d params, "
        "%d bits (%.4f MiB, %.6f BPP) | entropy side %s: %d tensors, "
        "%d params, %d bits (%.4f MiB, %.6f BPP)",
        prefix,
        summary.total_video_pixels,
        _storage_label(non_grid),
        non_grid.tensor_count,
        non_grid.parameter_count,
        non_grid.total_bits,
        _mib(non_grid.total_bits),
        summary.non_grid_bpp,
        _storage_label(entropy),
        entropy.tensor_count,
        entropy.parameter_count,
        entropy.total_bits,
        _mib(entropy.total_bits),
        summary.entropy_model_side_info_bpp,
    )
    if summary.network_quantization_metadata_bits:
        logger.info(
            "%s NETWORK QAT METADATA BIT | scales, zero-points and shapes: "
            "%d bits (%.6f BPP)",
            prefix,
            summary.network_quantization_metadata_bits,
            summary.network_quantization_metadata_bpp,
        )
    if summary.estimated_grid_bits is None:
        logger.info(
            "%s BIT | Grid: N/A (quantization disabled) | estimated payload: "
            "N/A | quantization metadata: NOT APPLICABLE",
            prefix,
        )
        return

    logger.info(
        "%s BIT | legacy rate/value: %.4f | Grid: %.0f bits "
        "(%.4f MiB, %.6f BPP) | estimated payload subtotal: %.0f bits "
        "(%.4f MiB, %.6f BPP)",
        prefix,
        summary.legacy_rate_per_value,
        summary.estimated_grid_bits,
        _mib(summary.estimated_grid_bits),
        summary.estimated_grid_bpp,
        summary.estimated_payload_bits,
        _mib(summary.estimated_payload_bits),
        summary.estimated_payload_bpp,
    )
    metadata = summary.quantization_metadata
    logger.info(
        "%s METADATA BIT | %d levels | header: %d bits | shapes: %d bits | "
        "quant steps: %d bits | total: %d bits (%.6f BPP)",
        prefix,
        metadata.level_count,
        metadata.header_bits,
        metadata.shape_bits,
        metadata.quant_step_bits,
        metadata.total_bits,
        summary.quantization_metadata_bpp,
    )
    logger.info(
        "%s TOTAL BIT | estimated total: %.0f bits (%.4f MiB, %.6f BPP) | "
        "actual bitstream: NOT AVAILABLE",
        prefix,
        summary.estimated_total_bits,
        _mib(summary.estimated_total_bits),
        summary.estimated_total_bpp,
    )
    if level_bits:
        logger.info(
            "%s GRID LEVEL BITS | %s",
            prefix,
            ' | '.join(
                'L{}: {:.0f}'.format(index, bits)
                for index, bits in enumerate(level_bits)
            ),
        )


def _write_rate_summary(writer, prefix, summary, level_bits, step):
    scalars = {
        'non_grid_parameter_bits': summary.non_grid_storage.total_bits,
        'non_grid_parameter_bpp': summary.non_grid_bpp,
        'entropy_model_parameter_bits': summary.entropy_model_storage.total_bits,
        'entropy_model_parameter_bpp': summary.entropy_model_side_info_bpp,
        'network_quantization_metadata_bits': (
            summary.network_quantization_metadata_bits
        ),
        'network_quantization_metadata_bpp': (
            summary.network_quantization_metadata_bpp
        ),
    }
    if summary.estimated_grid_bits is not None:
        scalars.update({
            'legacy_rate_per_value': summary.legacy_rate_per_value,
            'estimated_grid_bits': summary.estimated_grid_bits,
            'estimated_grid_bpp': summary.estimated_grid_bpp,
            'estimated_payload_bits': summary.estimated_payload_bits,
            'estimated_payload_bpp': summary.estimated_payload_bpp,
            'quantization_metadata_bits': (
                summary.quantization_metadata.total_bits
            ),
            'quantization_metadata_bpp': summary.quantization_metadata_bpp,
            'estimated_total_bits': summary.estimated_total_bits,
            'estimated_total_bpp': summary.estimated_total_bpp,
        })
    for name, value in scalars.items():
        writer.add_scalar(prefix + '/' + name, value, step)
    for index, bits in enumerate(level_bits):
        writer.add_scalar(
            prefix + '/grid_level_{}_bits'.format(index), bits, step,
        )


def set_cosine_learning_rate(optimizer, base_lr, epoch, batch_index,
                             num_batches, epochs):
    progress = (epoch + batch_index / num_batches) / epochs
    learning_rate = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    return learning_rate


def _validate_saved_config(saved_config, current_config):
    if saved_config is None:
        logging.warning("压缩检查点未保存配置，跳过配置一致性检查")
        return

    mismatches = []
    for key in CHECKPOINT_CONFIG_KEYS:
        if key not in saved_config:
            continue
        saved_value = saved_config[key]
        current_value = getattr(current_config, key)
        if isinstance(saved_value, list):
            saved_value = tuple(saved_value)
        if isinstance(current_value, list):
            current_value = tuple(current_value)
        if saved_value != current_value:
            mismatches.append(
                "{}: checkpoint={}, current={}".format(
                    key, saved_value, current_value,
                )
            )
    if mismatches:
        raise ValueError("压缩配置与检查点不一致: " + '; '.join(mismatches))


def _random_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state):
    if not state:
        logging.warning("压缩检查点未保存随机状态，恢复结果可能不完全一致")
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_compression_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_psnr,
    best_train_psnr,
    config,
    path,
):
    checkpoint = {
        'checkpoint_version': 1,
        'architecture': model.architecture,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_psnr': best_val_psnr,
        'best_train_psnr': best_train_psnr,
        'config': dict(vars(config)),
        'random_state': _random_state(),
    }
    torch.save(checkpoint, path)


def load_compression_checkpoint(path, model, optimizer, device, config):
    if not os.path.isfile(path):
        raise FileNotFoundError("未找到压缩检查点: {}".format(path))

    checkpoint = torch.load(path, map_location=device)
    architecture = checkpoint.get('architecture')
    if architecture != model.architecture:
        raise ValueError(
            "压缩模型架构与检查点不一致: checkpoint={}, current={}".format(
                architecture, model.architecture,
            )
        )
    _validate_saved_config(checkpoint.get('config'), config)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    _restore_random_state(checkpoint.get('random_state'))

    return (
        int(checkpoint.get('epoch', -1)) + 1,
        float(checkpoint.get('best_val_psnr', 0.0)),
        float(checkpoint.get('best_train_psnr', 0.0)),
    )


@torch.no_grad()
def evaluate_compression(model, loader, device, quant_mode, total_video_pixels,
                         non_grid_storage, entropy_model_storage,
                         quantization_metadata, network_metadata_bits=0,
                         save_dir=None,
                         dump_images=False, log_interval=50):
    was_training = model.training
    model.eval()
    total_psnr = 0.0
    total_msssim = 0.0
    total_rate = 0.0
    grid_bits_sum = None
    level_bits_sum = None
    samples_seen = 0

    if dump_images:
        os.makedirs(save_dir, exist_ok=True)

    for batch_index, batch in enumerate(loader):
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)
        batch_size = pixels.size(0)
        output = model(coords, quant_mode=quant_mode)

        psnr = psnr_fn(output.reconstruction, pixels)
        msssim = msssim_fn(output.reconstruction, pixels, device)
        rate = 0.0 if output.rate is None else output.rate.bits_per_value.item()
        total_psnr += psnr * batch_size
        total_msssim += msssim * batch_size
        total_rate += rate * batch_size
        if output.rate is not None:
            if grid_bits_sum is None:
                grid_bits_sum = 0.0
                level_bits_sum = [0.0] * len(output.rate.level_bits)
            grid_bits_sum += output.rate.total_bits.item() * batch_size
            for index, bits in enumerate(output.rate.level_bits):
                level_bits_sum[index] += bits.item() * batch_size

        if dump_images:
            frame_indices = batch.get('frame_idx')
            for item in range(batch_size):
                frame_index = (
                    int(frame_indices[item])
                    if frame_indices is not None else samples_seen + item
                )
                save_image(
                    output.reconstruction[item].detach().cpu(),
                    os.path.join(save_dir, 'pred_{:05d}.png'.format(frame_index)),
                )

        samples_seen += batch_size
        if batch_index % log_interval == 0 or batch_index == len(loader) - 1:
            logging.info(
                "Val [%d/%d] | PSNR: %.2f dB | MS-SSIM: %.4f | "
                "rate/value: %.4f",
                batch_index + 1,
                len(loader),
                total_psnr / samples_seen,
                total_msssim / samples_seen,
                total_rate / samples_seen,
            )

    model.train(was_training)
    summary = _averaged_rate_summary(
        grid_bits_sum,
        total_rate,
        samples_seen,
        total_video_pixels,
        non_grid_storage,
        entropy_model_storage,
        quantization_metadata,
        network_metadata_bits,
    )
    level_bits = (
        () if level_bits_sum is None
        else tuple(bits / samples_seen for bits in level_bits_sum)
    )
    return CompressionEvaluation(
        psnr=total_psnr / samples_seen,
        msssim=total_msssim / samples_seen,
        rate=summary,
        level_bits=level_bits,
    )


def _validate_args(args):
    compression_stage(
        0,
        args.epochs,
        args.warmup_epochs,
        args.symbol_start_epoch,
        args.lambda_max,
    )
    if not math.isfinite(args.quant_step) or args.quant_step <= 0:
        raise ValueError("quant_step must be finite and positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if any(value < 1 for value in (
        args.save_interval, args.log_interval, args.eval_freq,
    )):
        raise ValueError("save, log and evaluation intervals must be positive")
    if args.network_qat:
        if not 2 <= args.network_quant_bits <= 16:
            raise ValueError("network_quant_bits must be between 2 and 16")
        if not (0 <= args.network_quant_start_epoch <=
                args.network_quant_freeze_epoch < args.epochs):
            raise ValueError(
                "expected 0 <= network quant start <= freeze < epochs"
            )


def _make_loaders(args):
    fixed_res = tuple(args.fixed_res)
    train_dataset = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=None if args.dynamic_res else fixed_res,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        frame_interval=args.frame_interval,
    )
    val_dataset = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=fixed_res,
        frame_interval=args.frame_interval,
    )

    train_generator = torch.Generator()
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader, train_generator


def build_compression_model(args, device):
    reconstruction_model = HybridGridNet(
        grid_levels=args.grid_levels,
        grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        time_scale=args.time_scale,
        pe_freq=args.pe_freq,
        hidden_dim=args.hidden_dim,
    )
    model = CompressedHybridGridNet(
        reconstruction_model,
        quant_steps=args.quant_step,
    )
    if getattr(args, 'network_qat', False):
        grid_parameters = tuple(
            level.grid for level in reconstruction_model.grid_encoder.levels
        )
        bits = args.network_quant_bits
        prepare_network_qat(model, bits, excluded_parameters=grid_parameters)
        model.network_quant_bits = bits
        model.architecture = 'hybrid_grid_compressed_network_qat_v1'
    return model.to(device)


def train_compression(args):
    _validate_args(args)
    seed_everything(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("使用设备: %s", device)
    logger.info("随机种子: %d", args.seed)

    train_loader, val_loader, train_generator = _make_loaders(args)
    model = build_compression_model(args, device)
    total_video_pixels = (
        len(val_loader.dataset) * args.fixed_res[0] * args.fixed_res[1]
    )
    non_grid_storage, entropy_model_storage = compression_model_storage(model)
    quantization_metadata = compression_grid_metadata(model)
    network_metadata_bits = compression_network_metadata_bits(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = NervLoss(loss_type=args.loss_type, device=device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    reconstruction_params = sum(
        parameter.numel()
        for parameter in model.reconstruction_model.parameters()
    )
    entropy_params = total_params - reconstruction_params
    logger.info("压缩模型参数: %d", total_params)
    logger.info("重建模型参数: %d", reconstruction_params)
    logger.info("熵模型参数: %d", entropy_params)
    initial_rate = estimate_fp32_rate(
        total_video_pixels,
        non_grid_storage,
        entropy_model_storage,
        quantization_metadata,
        network_quantization_metadata_bits=network_metadata_bits,
    )
    _log_rate_summary(logger, 'MODEL', initial_rate)
    logger.info("非 Grid 网络 QAT 状态: %s", network_qat_state(model))
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    start_epoch = 0
    best_val_psnr = 0.0
    best_train_psnr = 0.0
    if args.resume:
        start_epoch, best_val_psnr, best_train_psnr = (
            load_compression_checkpoint(
                args.resume, model, optimizer, device, args,
            )
        )
        logger.info("从 epoch %d 恢复压缩训练", start_epoch + 1)

    training_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.network_qat:
            qat_enabled = epoch >= args.network_quant_start_epoch
            qat_frozen = epoch >= args.network_quant_freeze_epoch
            configure_network_qat(model, qat_enabled, qat_frozen)
        train_generator.manual_seed(args.seed + epoch)
        stage = compression_stage(
            epoch,
            args.epochs,
            args.warmup_epochs,
            args.symbol_start_epoch,
            args.lambda_max,
        )
        epoch_start = time.time()
        distortion_sum = 0.0
        rate_sum = 0.0
        total_sum = 0.0
        psnr_sum = 0.0
        msssim_sum = 0.0
        grid_bits_sum = None
        level_bits_sum = None
        samples_seen = 0

        for batch_index, batch in enumerate(train_loader):
            coords = batch['coords'].to(device, non_blocking=True)
            pixels = batch['pixels'].to(device, non_blocking=True)
            batch_size = pixels.size(0)
            learning_rate = set_cosine_learning_rate(
                optimizer,
                args.lr,
                epoch,
                batch_index,
                len(train_loader),
                args.epochs,
            )

            output = model(coords, quant_mode=stage.quant_mode)
            distortion = loss_fn(output.reconstruction, pixels)
            total_loss, rate = rate_distortion_loss(
                distortion, output, stage.lambda_rate,
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            psnr = psnr_fn(output.reconstruction, pixels)
            msssim = msssim_fn(output.reconstruction, pixels, device)
            distortion_sum += distortion.item() * batch_size
            rate_sum += rate.item() * batch_size
            if output.rate is not None:
                if grid_bits_sum is None:
                    grid_bits_sum = 0.0
                    level_bits_sum = [0.0] * len(output.rate.level_bits)
                grid_bits_sum += output.rate.total_bits.item() * batch_size
                for index, bits in enumerate(output.rate.level_bits):
                    level_bits_sum[index] += bits.item() * batch_size
            total_sum += total_loss.item() * batch_size
            psnr_sum += psnr * batch_size
            msssim_sum += msssim * batch_size
            samples_seen += batch_size

            if (batch_index % args.log_interval == 0 or
                    batch_index == len(train_loader) - 1):
                logger.info(
                    "epoch [%d/%d] batch [%d/%d] | mode: %s | "
                    "network QAT: %s | "
                    "lambda: %.3e | distortion: %.6f | rate/value: %.4f | "
                    "total: %.6f | PSNR: %.2f dB | lr: %.3e",
                    epoch + 1,
                    args.epochs,
                    batch_index + 1,
                    len(train_loader),
                    stage.quant_mode,
                    network_qat_state(model),
                    stage.lambda_rate,
                    distortion.item(),
                    rate.item(),
                    total_loss.item(),
                    psnr_sum / samples_seen,
                    learning_rate,
                )

        avg_distortion = distortion_sum / samples_seen
        avg_rate = rate_sum / samples_seen
        avg_total = total_sum / samples_seen
        avg_psnr = psnr_sum / samples_seen
        avg_msssim = msssim_sum / samples_seen
        epoch_rate = _averaged_rate_summary(
            grid_bits_sum,
            rate_sum,
            samples_seen,
            total_video_pixels,
            non_grid_storage,
            entropy_model_storage,
            quantization_metadata,
            network_metadata_bits,
        )
        level_bits = (
            () if level_bits_sum is None
            else tuple(bits / samples_seen for bits in level_bits_sum)
        )
        best_train_psnr = max(best_train_psnr, avg_psnr)
        epoch_seconds = time.time() - epoch_start

        writer.add_scalar('train/distortion_loss', avg_distortion, epoch + 1)
        writer.add_scalar('train/rate_per_value', avg_rate, epoch + 1)
        writer.add_scalar('train/total_loss', avg_total, epoch + 1)
        writer.add_scalar('train/psnr', avg_psnr, epoch + 1)
        writer.add_scalar('train/best_psnr', best_train_psnr, epoch + 1)
        writer.add_scalar('train/msssim', avg_msssim, epoch + 1)
        writer.add_scalar('train/lambda_rate', stage.lambda_rate, epoch + 1)
        writer.add_scalar('train/quant_step', args.quant_step, epoch + 1)
        writer.add_scalar('train/lr', learning_rate, epoch + 1)
        writer.add_scalar(
            'train/quant_mode',
            {'disabled': 0, 'noise': 1, 'symbols': 2}[stage.quant_mode],
            epoch + 1,
        )
        writer.add_scalar(
            'train/network_qat_state',
            {'not prepared': 0, 'disabled': 0, 'calibrating': 1, 'frozen': 2}[
                network_qat_state(model)
            ],
            epoch + 1,
        )
        writer.add_scalar('time/epoch_sec', epoch_seconds, epoch + 1)
        _write_rate_summary(writer, 'train_rate', epoch_rate, level_bits, epoch + 1)
        logger.info(
            "Epoch %d | mode: %s | network QAT: %s | lambda: %.3e | "
            "distortion: %.6f | "
            "rate/value: %.4f | total: %.6f | PSNR: %.2f dB | "
            "BEST TRAIN: %.2f dB | MS-SSIM: %.4f | time: %.2fs",
            epoch + 1,
            stage.quant_mode,
            network_qat_state(model),
            stage.lambda_rate,
            avg_distortion,
            avg_rate,
            avg_total,
            avg_psnr,
            best_train_psnr,
            avg_msssim,
            epoch_seconds,
        )
        _log_rate_summary(logger, 'Train epoch', epoch_rate, level_bits)

        if (epoch + 1) % args.eval_freq == 0 or epoch >= args.epochs - 10:
            evaluation_mode = (
                'disabled' if stage.quant_mode == 'disabled' else 'symbols'
            )
            evaluation = evaluate_compression(
                model,
                val_loader,
                device,
                evaluation_mode,
                total_video_pixels,
                non_grid_storage,
                entropy_model_storage,
                quantization_metadata,
                network_metadata_bits,
                save_dir=os.path.join(log_dir, 'visualize'),
                dump_images=args.dump_images,
                log_interval=args.log_interval,
            )
            writer.add_scalar('val/psnr', evaluation.psnr, epoch + 1)
            writer.add_scalar('val/msssim', evaluation.msssim, epoch + 1)
            _write_rate_summary(
                writer,
                'val_rate',
                evaluation.rate,
                evaluation.level_bits,
                epoch + 1,
            )
            logger.info(
                "Val epoch %d | mode: %s | PSNR: %.2f dB | "
                "MS-SSIM: %.4f | rate/value: %.4f",
                epoch + 1,
                evaluation_mode,
                evaluation.psnr,
                evaluation.msssim,
                0.0 if evaluation.rate.legacy_rate_per_value is None
                else evaluation.rate.legacy_rate_per_value,
            )
            _log_rate_summary(
                logger, 'Val epoch', evaluation.rate, evaluation.level_bits,
            )
            if (evaluation_mode == 'symbols' and
                    evaluation.psnr > best_val_psnr):
                best_val_psnr = evaluation.psnr
                save_compression_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_val_psnr,
                    best_train_psnr,
                    args,
                    os.path.join(log_dir, 'compression_best.pth'),
                )

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_compression_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_psnr,
                best_train_psnr,
                args,
                os.path.join(log_dir, 'compression_latest.pth'),
            )

    total_seconds = time.time() - training_start
    writer.add_scalar('time/total_sec', total_seconds, 0)
    writer.close()
    logger.info("压缩训练完成，总耗时 %.2f 小时", total_seconds / 3600)


def build_parser():
    parser = argparse.ArgumentParser(description='Hybrid Grid 率失真训练')
    parser.add_argument('-d', '--data_root', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--symbol_start_epoch', type=int, default=270)
    parser.add_argument('--lambda_max', type=float, default=5e-4)
    parser.add_argument('--quant_step', type=float, default=1e-3)
    parser.add_argument(
        '--network_qat', action='store_true',
        help='对非 Grid 参数启用训练中模拟量化',
    )
    parser.add_argument('--network_quant_bits', type=int, default=8)
    parser.add_argument('--network_quant_start_epoch', type=int, default=30)
    parser.add_argument('--network_quant_freeze_epoch', type=int, default=270)

    parser.add_argument('--grid_levels', type=int, default=10)
    parser.add_argument('--grid_feat_dim', type=int, default=6)
    parser.add_argument('--base_resolution', type=int, default=20)
    parser.add_argument('--finest_resolution', type=int, default=100)
    parser.add_argument('--aspect_ratio', type=int, nargs=2, default=[16, 9])
    parser.add_argument('--time_scale', type=float, default=0.1)
    parser.add_argument('--pe_freq', type=int, default=14)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--base_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--fixed_res', type=int, nargs=2, default=[720, 1280])
    parser.add_argument('--dynamic_res', action='store_true')
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--max_scale', type=float, default=1.2)
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--out_dir', default='./output/compression')
    parser.add_argument('--exp_name', default='paper_entropy')
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--resume')
    parser.add_argument('--dump_images', action='store_true')
    parser.add_argument(
        '--loss_type',
        choices=(
            'L2', 'L1', 'SSIM', 'Fusion1', 'Fusion2', 'Fusion3', 'Fusion4',
            'Fusion5', 'Fusion6', 'Fusion7', 'Fusion8', 'Fusion9', 'Fusion10',
            'Fusion11', 'Fusion12',
        ),
        default='L2',
    )
    return parser


if __name__ == '__main__':
    train_compression(build_parser().parse_args())
