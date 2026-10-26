"""Evaluation and metric helpers for Scheme-A scalable training."""

from dataclasses import dataclass
from typing import Tuple

import torch

from util import msssim_fn, psnr_fn

from .losses import prefix_rate_distortion_loss


@dataclass(frozen=True)
class PrefixEvaluation:
    distortion: float
    total_loss: float
    psnr: float
    msssim: float
    rate_per_value: float
    grid_bits: float
    grid_bpp: float


@dataclass(frozen=True)
class PrefixBatchMetrics:
    distortion: float
    rate: float
    total: float
    weighted_total: float
    psnr: float
    msssim: float
    grid_bits: float


def prefix_names(count):
    if count == 3:
        return ('B', 'B+E1', 'Full')
    return tuple('Level{}'.format(index) for index in range(count))


def empty_metric_sums(count):
    return [
        {
            'distortion': 0.0,
            'rate': 0.0,
            'total': 0.0,
            'psnr': 0.0,
            'msssim': 0.0,
            'grid_bits': 0.0,
        }
        for _ in range(count)
    ]


def update_metric_sums(sums, output, losses, target, device, batch_size):
    for values, prefix, loss in zip(sums, output.prefixes, losses.prefixes):
        values['distortion'] += loss.distortion.item() * batch_size
        values['rate'] += loss.rate_per_value.item() * batch_size
        values['total'] += loss.total.item() * batch_size
        values['psnr'] += (
            psnr_fn(prefix.reconstruction, target) * batch_size
        )
        values['msssim'] += (
            msssim_fn(prefix.reconstruction, target, device) * batch_size
        )
        if prefix.rate is not None:
            values['grid_bits'] += prefix.rate.total_bits.item() * batch_size


def accumulate_prefix_gradients(model, coords, target, distortion_fn,
                                lambda_rate, distortion_weights,
                                rate_weights, quant_mode, device):
    """Backpropagate one prefix at a time and return detached batch metrics.

    Gradients accumulate in model parameters.  The caller must clear gradients
    before this function and call ``optimizer.step()`` afterwards.  Each
    prefix graph is released before the next prefix is constructed. In noise
    mode the RNG state is replayed, so overlapping Grid levels use the same
    quantization noise in every prefix.
    """
    count = model.hierarchy.num_quality_levels
    if len(distortion_weights) != count:
        raise ValueError(
            "distortion_weights must contain one value per prefix"
        )
    if len(rate_weights) != count:
        raise ValueError("rate_weights must contain one value per prefix")

    metrics = []
    joint_total = 0.0
    cpu_rng_state = None
    cuda_rng_state = None
    if quant_mode == 'noise':
        cpu_rng_state = torch.random.get_rng_state()
        if coords.is_cuda:
            cuda_rng_state = torch.cuda.get_rng_state(coords.device)
    for quality_level, (distortion_weight, rate_weight) in enumerate(zip(
        distortion_weights, rate_weights,
    )):
        if quality_level > 0 and cpu_rng_state is not None:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, coords.device)
        output = model.forward_prefix(coords, quality_level, quant_mode)
        loss = prefix_rate_distortion_loss(
            output,
            target,
            distortion_fn,
            lambda_rate,
            distortion_weight,
            rate_weight,
        )
        weighted_total = loss.total
        weighted_total.backward()
        weighted_value = weighted_total.detach().item()
        joint_total += weighted_value
        metrics.append(PrefixBatchMetrics(
            distortion=loss.distortion.detach().item(),
            rate=loss.rate_per_value.detach().item(),
            total=loss.total.detach().item(),
            weighted_total=weighted_value,
            psnr=psnr_fn(output.reconstruction, target),
            msssim=msssim_fn(output.reconstruction, target, device),
            grid_bits=(
                0.0 if output.rate is None
                else output.rate.total_bits.detach().item()
            ),
        ))
        del weighted_total, loss, output
    return tuple(metrics), joint_total


def update_metric_sums_from_batch(sums, metrics, batch_size):
    for values, item in zip(sums, metrics):
        for key in ('distortion', 'rate', 'total', 'psnr', 'msssim',
                    'grid_bits'):
            values[key] += getattr(item, key) * batch_size


def average_metric_sums(sums, samples):
    if samples < 1:
        raise ValueError("samples must be positive")
    return tuple({key: value / samples for key, value in item.items()}
                 for item in sums)


@torch.no_grad()
def evaluate_prefixes(model, loader, device, quant_mode, distortion_fn,
                      lambda_rate, distortion_weights, rate_weights,
                      total_video_pixels, log_interval=50, logger=None):
    """Evaluate every prefix using one deterministic quantization per batch."""
    from .losses import multi_prefix_rate_distortion_loss

    if log_interval < 1:
        raise ValueError("log_interval must be positive")
    was_training = model.training
    model.eval()
    sums = empty_metric_sums(model.hierarchy.num_quality_levels)
    names = prefix_names(model.hierarchy.num_quality_levels)
    samples = 0

    for batch_index, batch in enumerate(loader):
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)
        batch_size = pixels.size(0)
        output = model.forward_all(coords, quant_mode=quant_mode)
        losses = multi_prefix_rate_distortion_loss(
            output,
            pixels,
            distortion_fn,
            lambda_rate,
            distortion_weights,
            rate_weights,
        )
        update_metric_sums(
            sums, output, losses, pixels, device, batch_size,
        )
        samples += batch_size

        if (logger is not None and
                (batch_index % log_interval == 0 or
                 batch_index == len(loader) - 1)):
            averages = average_metric_sums(sums, samples)
            logger.info(
                "Val [%d/%d] | %s",
                batch_index + 1,
                len(loader),
                " | ".join(
                    "{} PSNR: {:.2f} dB rate/value: {:.4f}".format(
                        name, values['psnr'], values['rate'],
                    )
                    for name, values in zip(names, averages)
                ),
            )

    model.train(was_training)
    averages = average_metric_sums(sums, samples)
    return tuple(
        PrefixEvaluation(
            distortion=values['distortion'],
            total_loss=values['total'],
            psnr=values['psnr'],
            msssim=values['msssim'],
            rate_per_value=values['rate'],
            grid_bits=values['grid_bits'],
            grid_bpp=values['grid_bits'] / total_video_pixels,
        )
        for values in averages
    )
