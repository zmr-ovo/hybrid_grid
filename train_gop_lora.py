import argparse
import hashlib
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from gop_lora import (
    GOPLoRAHybridGridNet,
    GOPVideoDataset,
    freeze_shared_parameters,
    gop_lora_layers,
    gop_local_coordinates,
    lora_parameters,
)
from model import DynamicVideoDataset, HybridGridNet
from train import seed_everything, seed_worker, setup_logging
from util import NervLoss, msssim_fn, psnr_fn


MODEL_CONFIG_KEYS = (
    'grid_levels', 'grid_feat_dim', 'base_resolution', 'finest_resolution',
    'aspect_ratio', 'time_scale', 'pe_freq', 'hidden_dim', 'gop_size',
)
ADAPTER_CONFIG_KEYS = MODEL_CONFIG_KEYS + ('rank', 'alpha')


@dataclass(frozen=True)
class GOPLoader:
    gop_index: int
    loader: DataLoader
    generator: torch.Generator


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    psnr: float
    msssim: float


@dataclass(frozen=True)
class Evaluation:
    psnr: float
    msssim: float
    per_gop: dict


def build_model(args, device):
    return HybridGridNet(
        grid_levels=args.grid_levels,
        grid_feat_dim=args.grid_feat_dim,
        base_resolution=args.base_resolution,
        finest_resolution=args.finest_resolution,
        aspect_ratio=tuple(args.aspect_ratio),
        time_scale=args.time_scale,
        pe_freq=args.pe_freq,
        hidden_dim=args.hidden_dim,
    ).to(device)


def make_datasets(args):
    fixed_res = tuple(args.fixed_res)
    train_base = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=None if args.dynamic_res else fixed_res,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        frame_interval=args.frame_interval,
    )
    val_base = DynamicVideoDataset(
        data_root=args.data_root,
        base_res=tuple(args.base_res),
        fixed_res=fixed_res,
        frame_interval=args.frame_interval,
    )
    train_dataset = GOPVideoDataset(train_base, args.gop_size)
    val_dataset = GOPVideoDataset(val_base, args.gop_size)
    if train_dataset.frame_indices != val_dataset.frame_indices:
        raise RuntimeError("训练集和验证集的帧映射不一致")
    return train_dataset, val_dataset


def make_gop_loaders(dataset, args, shuffle):
    entries = []
    for gop_index in range(dataset.num_gops):
        indices = dataset.gop_sample_indices(gop_index)
        if not indices:
            continue
        generator = torch.Generator().manual_seed(
            args.seed + gop_index + (0 if shuffle else dataset.num_gops)
        )
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        entries.append(GOPLoader(gop_index, loader, generator))
    return tuple(entries)


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
        logging.warning("检查点未保存随机状态，恢复结果可能不完全一致")
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def _validate_saved_config(saved, current, keys):
    if saved is None:
        raise ValueError("GOP 检查点缺少配置")
    mismatches = []
    for key in keys:
        if key not in saved:
            continue
        saved_value = saved[key]
        current_value = getattr(current, key)
        if isinstance(saved_value, list):
            saved_value = tuple(saved_value)
        if isinstance(current_value, list):
            current_value = tuple(current_value)
        if saved_value != current_value:
            mismatches.append(
                '{}: checkpoint={}, current={}'.format(
                    key, saved_value, current_value,
                )
            )
    if mismatches:
        raise ValueError("GOP 检查点配置不一致: " + '; '.join(mismatches))


def _validate_gop_metadata(checkpoint, dataset):
    if dataset is None:
        return
    saved = checkpoint.get('gop')
    expected = {
        'gop_size': dataset.gop_size,
        'num_gops': dataset.num_gops,
        'total_frames': dataset.total_frames,
    }
    if saved != expected:
        raise ValueError(
            "GOP 划分与检查点不一致: checkpoint={}, current={}".format(
                saved, expected,
            )
        )


def save_anchor_checkpoint(model, optimizer, epoch, best_val_psnr,
                           best_train_psnr, args, dataset, path):
    torch.save({
        'checkpoint_version': 1,
        'checkpoint_type': 'gop_anchor',
        'architecture': model.architecture,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_psnr': best_val_psnr,
        'best_train_psnr': best_train_psnr,
        'config': dict(vars(args)),
        'gop': {
            'gop_size': dataset.gop_size,
            'num_gops': dataset.num_gops,
            'total_frames': dataset.total_frames,
        },
        'random_state': _random_state(),
    }, path)


def load_anchor_checkpoint(path, model, device, args, optimizer=None,
                           restore_random=False, dataset=None):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("未找到锚点检查点: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('checkpoint_type') != 'gop_anchor':
        raise ValueError("检查点不是 GOP 锚点模型")
    if checkpoint.get('architecture') != model.architecture:
        raise ValueError("锚点模型架构与检查点不一致")
    _validate_saved_config(checkpoint.get('config'), args, MODEL_CONFIG_KEYS)
    _validate_gop_metadata(checkpoint, dataset)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if restore_random:
        _restore_random_state(checkpoint.get('random_state'))
    return (
        int(checkpoint.get('epoch', -1)) + 1,
        float(checkpoint.get('best_val_psnr', 0.0)),
        float(checkpoint.get('best_train_psnr', 0.0)),
    )


def adapter_state_dict(model):
    adapter_ids = {id(parameter) for parameter in lora_parameters(model)}
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if id(parameter) in adapter_ids
    }


def anchor_digest(model):
    adapter_ids = {id(parameter) for parameter in lora_parameters(model)}
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if id(parameter) in adapter_ids:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_adapter_state_dict(model, state_dict):
    parameters = dict(model.named_parameters())
    adapter_ids = {id(parameter) for parameter in lora_parameters(model)}
    expected = {
        name for name, parameter in parameters.items()
        if id(parameter) in adapter_ids
    }
    received = set(state_dict)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(
            "LoRA 参数不完整: missing={}, unexpected={}".format(
                missing, unexpected,
            )
        )
    with torch.no_grad():
        for name in sorted(expected):
            source = state_dict[name]
            target = parameters[name]
            if source.shape != target.shape:
                raise ValueError("LoRA 参数形状不一致: {}".format(name))
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def save_adapter_checkpoint(model, optimizer, epoch, best_val_psnr,
                            best_train_psnr, args, dataset, path):
    torch.save({
        'checkpoint_version': 1,
        'checkpoint_type': 'gop_decoder_lora',
        'architecture': model.architecture,
        'anchor_digest': anchor_digest(model),
        'epoch': epoch,
        'adapter_state_dict': adapter_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'best_val_psnr': best_val_psnr,
        'best_train_psnr': best_train_psnr,
        'config': dict(vars(args)),
        'gop': {
            'gop_size': dataset.gop_size,
            'num_gops': dataset.num_gops,
            'total_frames': dataset.total_frames,
        },
        'random_state': _random_state(),
    }, path)


def load_adapter_checkpoint(path, model, device, args, optimizer=None,
                            restore_random=False, dataset=None):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("未找到 LoRA 检查点: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('checkpoint_type') != 'gop_decoder_lora':
        raise ValueError("检查点不是 GOP Decoder LoRA")
    if checkpoint.get('architecture') != model.architecture:
        raise ValueError("GOP-LoRA 模型架构与检查点不一致")
    _validate_saved_config(checkpoint.get('config'), args, ADAPTER_CONFIG_KEYS)
    _validate_gop_metadata(checkpoint, dataset)
    if checkpoint.get('anchor_digest') != anchor_digest(model):
        raise ValueError("LoRA 检查点与当前锚点模型不匹配")
    load_adapter_state_dict(model, checkpoint['adapter_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if restore_random:
        _restore_random_state(checkpoint.get('random_state'))
    return (
        int(checkpoint.get('epoch', -1)) + 1,
        float(checkpoint.get('best_val_psnr', 0.0)),
        float(checkpoint.get('best_train_psnr', 0.0)),
    )


def set_learning_rate(optimizer, base_lr, epoch, batch_index, num_batches,
                      epochs, warmup_ratio):
    position = epoch + batch_index / num_batches
    warmup_epochs = epochs * warmup_ratio
    if warmup_epochs > 0 and position < warmup_epochs:
        multiplier = 0.1 + 0.9 * position / warmup_epochs
    else:
        span = max(epochs - warmup_epochs, 1.0)
        progress = min(max((position - warmup_epochs) / span, 0.0), 1.0)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = base_lr * multiplier
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    return learning_rate


def _forward(model, batch, device, adapter_stage):
    coords = batch['coords'].to(device, non_blocking=True)
    local_time = batch['gop_local_time']
    if adapter_stage:
        return model(coords, batch['gop_idx'], local_time)
    return model(gop_local_coordinates(coords, local_time))


def train_epoch(model, entries, optimizer, loss_fn, device, epoch, epochs,
                base_lr, warmup_ratio, log_interval, phase, seed):
    model.train()
    total_batches = sum(len(entry.loader) for entry in entries)
    loss_sum = 0.0
    psnr_sum = 0.0
    msssim_sum = 0.0
    samples_seen = 0
    global_batch = 0
    adapter_stage = isinstance(model, GOPLoRAHybridGridNet)

    for entry in entries:
        entry.generator.manual_seed(seed + epoch * 1009 + entry.gop_index)
        for batch_index, batch in enumerate(entry.loader):
            pixels = batch['pixels'].to(device, non_blocking=True)
            learning_rate = set_learning_rate(
                optimizer, base_lr, epoch, global_batch, total_batches,
                epochs, warmup_ratio,
            )
            prediction = _forward(model, batch, device, adapter_stage)
            loss = loss_fn(prediction, pixels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = pixels.size(0)
            loss_sum += loss.item() * batch_size
            psnr_sum += psnr_fn(prediction, pixels) * batch_size
            msssim_sum += msssim_fn(prediction, pixels, device) * batch_size
            samples_seen += batch_size
            if (batch_index % log_interval == 0 or
                    batch_index == len(entry.loader) - 1):
                logging.info(
                    "%s epoch [%d/%d] GOP %d batch [%d/%d] | "
                    "loss: %.6f | PSNR: %.2f dB | lr: %.3e",
                    phase,
                    epoch + 1,
                    epochs,
                    entry.gop_index,
                    batch_index + 1,
                    len(entry.loader),
                    loss.item(),
                    psnr_sum / samples_seen,
                    learning_rate,
                )
            global_batch += 1

    return EpochMetrics(
        loss_sum / samples_seen,
        psnr_sum / samples_seen,
        msssim_sum / samples_seen,
    )


@torch.no_grad()
def evaluate(model, entries, device, log_interval=50, save_dir=None):
    was_training = model.training
    model.eval()
    adapter_stage = isinstance(model, GOPLoRAHybridGridNet)
    total_psnr = 0.0
    total_msssim = 0.0
    total_samples = 0
    per_gop = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for entry in entries:
        gop_psnr = 0.0
        gop_msssim = 0.0
        gop_samples = 0
        for batch_index, batch in enumerate(entry.loader):
            pixels = batch['pixels'].to(device, non_blocking=True)
            prediction = _forward(model, batch, device, adapter_stage)
            batch_size = pixels.size(0)
            gop_psnr += psnr_fn(prediction, pixels) * batch_size
            gop_msssim += msssim_fn(prediction, pixels, device) * batch_size
            gop_samples += batch_size

            if save_dir:
                for item, frame_index in enumerate(batch['frame_idx']):
                    save_image(
                        prediction[item].detach().cpu(),
                        os.path.join(
                            save_dir, 'pred_{:05d}.png'.format(int(frame_index)),
                        ),
                    )
            if (batch_index % log_interval == 0 or
                    batch_index == len(entry.loader) - 1):
                logging.info(
                    "Val GOP %d [%d/%d] | PSNR: %.2f dB | MS-SSIM: %.4f",
                    entry.gop_index,
                    batch_index + 1,
                    len(entry.loader),
                    gop_psnr / gop_samples,
                    gop_msssim / gop_samples,
                )

        average_psnr = gop_psnr / gop_samples
        average_msssim = gop_msssim / gop_samples
        per_gop[entry.gop_index] = (average_psnr, average_msssim, gop_samples)
        total_psnr += gop_psnr
        total_msssim += gop_msssim
        total_samples += gop_samples

    model.train(was_training)
    return Evaluation(
        total_psnr / total_samples,
        total_msssim / total_samples,
        per_gop,
    )


def _should_evaluate(epoch, epochs, frequency):
    return (epoch + 1) % frequency == 0 or epoch == epochs - 1


def _log_evaluation(prefix, result):
    logging.info(
        "%s | FULL VIDEO PSNR: %.2f dB | MS-SSIM: %.4f",
        prefix, result.psnr, result.msssim,
    )
    for gop_index, values in sorted(result.per_gop.items()):
        logging.info(
            "%s | GOP %d | frames: %d | PSNR: %.2f dB | MS-SSIM: %.4f",
            prefix, gop_index, values[2], values[0], values[1],
        )


def train_anchor(args, model, train_entry, val_entry, optimizer, loss_fn,
                 device, writer, output_dir, dataset):
    start_epoch = 0
    best_val_psnr = 0.0
    best_train_psnr = 0.0
    if args.anchor_resume:
        start_epoch, best_val_psnr, best_train_psnr = load_anchor_checkpoint(
            args.anchor_resume, model, device, args, optimizer,
            restore_random=True, dataset=dataset,
        )
        logging.info("从 anchor epoch %d 恢复", start_epoch + 1)
        if start_epoch >= args.anchor_epochs:
            raise ValueError(
                "锚点训练已经完成；请使用 --stage adapter 和该锚点检查点"
            )

    for epoch in range(start_epoch, args.anchor_epochs):
        start = time.time()
        metrics = train_epoch(
            model, (train_entry,), optimizer, loss_fn, device, epoch,
            args.anchor_epochs, args.anchor_lr, args.anchor_warmup,
            args.log_interval, 'ANCHOR', args.seed,
        )
        best_train_psnr = max(best_train_psnr, metrics.psnr)
        writer.add_scalar('anchor/train_loss', metrics.loss, epoch + 1)
        writer.add_scalar('anchor/train_psnr', metrics.psnr, epoch + 1)
        writer.add_scalar(
            'anchor/best_train_psnr', best_train_psnr, epoch + 1,
        )
        writer.add_scalar('anchor/train_msssim', metrics.msssim, epoch + 1)
        logging.info(
            "ANCHOR Epoch %d | loss: %.6f | PSNR: %.2f dB | "
            "BEST TRAIN: %.2f dB | MS-SSIM: %.4f | time: %.2fs",
            epoch + 1, metrics.loss, metrics.psnr, best_train_psnr,
            metrics.msssim, time.time() - start,
        )

        if _should_evaluate(epoch, args.anchor_epochs, args.eval_freq):
            result = evaluate(model, (val_entry,), device, args.log_interval)
            writer.add_scalar('anchor/val_psnr', result.psnr, epoch + 1)
            writer.add_scalar('anchor/val_msssim', result.msssim, epoch + 1)
            if result.psnr > best_val_psnr:
                best_val_psnr = result.psnr
                save_anchor_checkpoint(
                    model, optimizer, epoch, best_val_psnr, best_train_psnr,
                    args, dataset,
                    os.path.join(output_dir, 'anchor_best.pth'),
                )
            _log_evaluation('ANCHOR VAL', result)

        if ((epoch + 1) % args.save_interval == 0 or
                epoch == args.anchor_epochs - 1):
            save_anchor_checkpoint(
                model, optimizer, epoch, best_val_psnr, best_train_psnr,
                args, dataset,
                os.path.join(output_dir, 'anchor_latest.pth'),
            )
    return best_train_psnr, best_val_psnr


def train_adapters(args, model, train_entries, val_entries, optimizer, loss_fn,
                   device, writer, output_dir, dataset):
    start_epoch = 0
    best_val_psnr = 0.0
    best_train_psnr = 0.0
    if args.adapter_resume:
        start_epoch, best_val_psnr, best_train_psnr = load_adapter_checkpoint(
            args.adapter_resume, model, device, args, optimizer,
            restore_random=True, dataset=dataset,
        )
        logging.info("从 adapter epoch %d 恢复", start_epoch + 1)
        if start_epoch >= args.adapter_epochs:
            raise ValueError("LoRA 训练已经达到 adapter_epochs")

    for epoch in range(start_epoch, args.adapter_epochs):
        start = time.time()
        metrics = train_epoch(
            model, train_entries, optimizer, loss_fn, device, epoch,
            args.adapter_epochs, args.adapter_lr, args.adapter_warmup,
            args.log_interval, 'ADAPTER', args.seed + 100000,
        )
        best_train_psnr = max(best_train_psnr, metrics.psnr)
        writer.add_scalar('adapter/train_loss', metrics.loss, epoch + 1)
        writer.add_scalar('adapter/train_psnr', metrics.psnr, epoch + 1)
        writer.add_scalar(
            'adapter/best_train_psnr', best_train_psnr, epoch + 1,
        )
        writer.add_scalar('adapter/train_msssim', metrics.msssim, epoch + 1)
        logging.info(
            "ADAPTER Epoch %d | loss: %.6f | PSNR: %.2f dB | "
            "BEST TRAIN: %.2f dB | MS-SSIM: %.4f | time: %.2fs",
            epoch + 1, metrics.loss, metrics.psnr, best_train_psnr,
            metrics.msssim, time.time() - start,
        )

        if _should_evaluate(epoch, args.adapter_epochs, args.eval_freq):
            result = evaluate(model, val_entries, device, args.log_interval)
            writer.add_scalar('adapter/val_psnr', result.psnr, epoch + 1)
            writer.add_scalar('adapter/val_msssim', result.msssim, epoch + 1)
            if result.psnr > best_val_psnr:
                best_val_psnr = result.psnr
                save_adapter_checkpoint(
                    model, optimizer, epoch, best_val_psnr, best_train_psnr,
                    args, dataset,
                    os.path.join(output_dir, 'adapter_best.pth'),
                )
            _log_evaluation('ADAPTER VAL', result)

        if ((epoch + 1) % args.save_interval == 0 or
                epoch == args.adapter_epochs - 1):
            save_adapter_checkpoint(
                model, optimizer, epoch, best_val_psnr, best_train_psnr,
                args, dataset,
                os.path.join(output_dir, 'adapter_latest.pth'),
            )
    return best_train_psnr, best_val_psnr


def _validate_args(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative")
    if args.gop_size < 1 or args.frame_interval < 1:
        raise ValueError("gop_size and frame_interval must be positive")
    if args.frame_interval > args.gop_size:
        raise ValueError("frame_interval must not exceed gop_size")
    if args.rank < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("rank and alpha must be positive")
    if args.anchor_epochs < 1 or args.adapter_epochs < 1:
        raise ValueError("anchor_epochs and adapter_epochs must be positive")
    if (not math.isfinite(args.anchor_lr) or args.anchor_lr <= 0 or
            not math.isfinite(args.adapter_lr) or args.adapter_lr <= 0):
        raise ValueError("learning rates must be positive")
    if not 0 <= args.anchor_warmup < 1 or not 0 <= args.adapter_warmup < 1:
        raise ValueError("warmup ratios must be in [0, 1)")
    if min(args.eval_freq, args.save_interval, args.log_interval) < 1:
        raise ValueError("log, save and eval intervals must be positive")
    if args.dynamic_res and args.batch_size != 1:
        raise ValueError("dynamic resolution requires batch_size=1")
    if args.stage in ('adapter', 'eval') and not args.anchor_checkpoint:
        raise ValueError("{} stage requires --anchor_checkpoint".format(
            args.stage,
        ))
    if args.anchor_resume and args.stage not in ('all', 'anchor'):
        raise ValueError("--anchor_resume is only valid for anchor training")
    if args.adapter_resume and args.stage != 'adapter':
        raise ValueError("--adapter_resume requires --stage adapter")


def run(args):
    _validate_args(args)
    seed_everything(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(args.out_dir, timestamp, args.exp_name)
    logger = setup_logging(output_dir)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("使用设备: %s", device)
    logger.info("训练阶段: %s", args.stage)
    logger.info("随机种子: %d", args.seed)
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)

    train_dataset, val_dataset = make_datasets(args)
    train_entries = make_gop_loaders(train_dataset, args, shuffle=True)
    val_entries = make_gop_loaders(val_dataset, args, shuffle=False)
    if len(train_entries) != train_dataset.num_gops:
        raise ValueError("帧采样导致部分 GOP 没有训练样本")
    logger.info(
        "视频帧: %d | GOP size: %d | GOP 数量: %d",
        train_dataset.total_frames,
        train_dataset.gop_size,
        train_dataset.num_gops,
    )

    anchor = build_model(args, device)
    loss_fn = NervLoss(args.loss_type, device)

    if args.stage in ('adapter', 'eval'):
        load_anchor_checkpoint(
            args.anchor_checkpoint, anchor, device, args,
            dataset=train_dataset,
        )
        logger.info("已加载锚点: %s", args.anchor_checkpoint)

    if args.stage in ('all', 'anchor'):
        anchor_optimizer = torch.optim.AdamW(
            anchor.parameters(), lr=args.anchor_lr,
            weight_decay=args.weight_decay,
        )
        train_anchor(
            args, anchor, train_entries[0], val_entries[0], anchor_optimizer,
            loss_fn, device, writer, output_dir, train_dataset,
        )
        if args.stage == 'anchor':
            writer.close()
            return
        best_anchor_path = os.path.join(output_dir, 'anchor_best.pth')
        load_anchor_checkpoint(
            best_anchor_path, anchor, device, args, dataset=train_dataset,
        )
        logger.info("LoRA 阶段使用最佳锚点: %s", best_anchor_path)

    if train_dataset.num_gops < 2:
        if args.stage == 'eval':
            result = evaluate(
                anchor, val_entries, device, args.log_interval,
                os.path.join(output_dir, 'eval') if args.dump_images else None,
            )
            _log_evaluation('EVAL', result)
            writer.close()
            return
        raise ValueError("LoRA 适配至少需要两个 GOP")

    model = GOPLoRAHybridGridNet(
        anchor, train_dataset.num_gops, args.rank, args.alpha,
    ).to(device)
    freeze_shared_parameters(model)
    adapter_parameters = lora_parameters(model)
    logger.info(
        "锚点参数: %d | 后续 GOP LoRA 参数: %d | adapters: %d",
        sum(parameter.numel() for parameter in anchor.parameters()) -
        sum(parameter.numel() for parameter in adapter_parameters),
        sum(parameter.numel() for parameter in adapter_parameters),
        model.num_adapters,
    )
    for name, layer in zip(model.injected_layers, gop_lora_layers(model)):
        logger.info(
            "LoRA layer: %s | effective rank: %d | alpha: %.4g | "
            "scaling: %.4g",
            name, layer.rank, layer.alpha, layer.scaling,
        )

    if args.stage == 'eval':
        if not args.adapter_checkpoint:
            raise ValueError("eval stage requires --adapter_checkpoint")
        load_adapter_checkpoint(
            args.adapter_checkpoint, model, device, args,
            dataset=train_dataset,
        )
        result = evaluate(
            model, val_entries, device, args.log_interval,
            os.path.join(output_dir, 'eval') if args.dump_images else None,
        )
        _log_evaluation('EVAL', result)
        writer.close()
        return

    optimizer = torch.optim.AdamW(
        adapter_parameters, lr=args.adapter_lr,
        weight_decay=args.weight_decay,
    )
    train_adapters(
        args,
        model,
        train_entries[1:],
        val_entries,
        optimizer,
        loss_fn,
        device,
        writer,
        output_dir,
        train_dataset,
    )
    final_result = evaluate(
        model, val_entries, device, args.log_interval,
        os.path.join(output_dir, 'final') if args.dump_images else None,
    )
    _log_evaluation('FINAL', final_result)
    writer.close()


def make_parser():
    parser = argparse.ArgumentParser(
        description="首 GOP 锚点 + 后续 GOP 独立 Decoder LoRA 训练",
    )
    parser.add_argument('-d', '--data_root', required=True)
    parser.add_argument(
        '--stage', choices=('all', 'anchor', 'adapter', 'eval'), default='all',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--gop_size', type=int, default=30)
    parser.add_argument('--anchor_epochs', type=int, default=300)
    parser.add_argument('--adapter_epochs', type=int, default=100)
    parser.add_argument('--anchor_lr', type=float, default=5e-3)
    parser.add_argument('--adapter_lr', type=float, default=1e-3)
    parser.add_argument('--anchor_warmup', type=float, default=0.2)
    parser.add_argument('--adapter_warmup', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=2.0)

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

    parser.add_argument('--out_dir', default='./output/gop_lora')
    parser.add_argument('--exp_name', default='gop_lora')
    parser.add_argument('--save_interval', type=int, default=25)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--anchor_resume')
    parser.add_argument('--adapter_resume')
    parser.add_argument('--anchor_checkpoint')
    parser.add_argument('--adapter_checkpoint')
    parser.add_argument('--dump_images', action='store_true')
    parser.add_argument(
        '--loss_type', default='L2',
        choices=(
            'L2', 'L1', 'SSIM', 'Fusion1', 'Fusion2', 'Fusion3', 'Fusion4',
            'Fusion5', 'Fusion6', 'Fusion7', 'Fusion8', 'Fusion9', 'Fusion10',
            'Fusion11', 'Fusion12',
        ),
    )
    return parser


if __name__ == '__main__':
    run(make_parser().parse_args())
