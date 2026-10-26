"""Memory-efficient training helpers for the module-layered model."""

from dataclasses import dataclass
import math

import torch

from util import msssim_fn, psnr_fn


PREFIX_NAMES = ('B', 'B+E1', 'Full')


@dataclass(frozen=True)
class PrefixMetrics:
    distortion: float
    psnr: float
    msssim: float


def normalize_prefix_weights(weights, count=3):
    if not isinstance(weights, (tuple, list)) or len(weights) != count:
        raise ValueError('prefix_weights must contain one value per prefix')
    values = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('prefix weights must be finite and positive')
    total = sum(values)
    return tuple(value / total for value in values)


def accumulate_prefix_gradients(
    model,
    coords,
    target,
    distortion_fn,
    prefix_weights,
    device,
    metric_fn=msssim_fn,
):
    """Recompute and backpropagate each prefix before one optimizer step."""
    weights = normalize_prefix_weights(prefix_weights)
    metrics = []
    joint_loss = 0.0
    for quality_level, weight in enumerate(weights):
        output = model.forward_prefix(coords, quality_level)
        distortion = distortion_fn(output.reconstruction, target)
        weighted = weight * distortion
        weighted.backward()
        joint_loss += weighted.detach().item()
        metrics.append(PrefixMetrics(
            distortion=distortion.detach().item(),
            psnr=psnr_fn(output.reconstruction, target),
            msssim=metric_fn(output.reconstruction, target, device),
        ))
        del output, distortion, weighted
    return tuple(metrics), joint_loss


def empty_sums():
    return [
        {'distortion': 0.0, 'psnr': 0.0, 'msssim': 0.0}
        for _ in PREFIX_NAMES
    ]


def update_sums(sums, metrics, batch_size):
    for values, metric in zip(sums, metrics):
        values['distortion'] += metric.distortion * batch_size
        values['psnr'] += metric.psnr * batch_size
        values['msssim'] += metric.msssim * batch_size


def average_sums(sums, samples):
    if samples < 1:
        raise ValueError('samples must be positive')
    return tuple(
        {name: value / samples for name, value in item.items()}
        for item in sums
    )


@torch.no_grad()
def evaluate_prefixes(
    model,
    loader,
    device,
    distortion_fn,
    log_interval=50,
    logger=None,
):
    was_training = model.training
    model.eval()
    sums = empty_sums()
    samples = 0
    for batch_index, batch in enumerate(loader):
        coords = batch['coords'].to(device, non_blocking=True)
        pixels = batch['pixels'].to(device, non_blocking=True)
        batch_size = pixels.shape[0]
        metrics = []
        for quality_level in range(3):
            output = model.forward_prefix(coords, quality_level)
            metrics.append(PrefixMetrics(
                distortion=distortion_fn(
                    output.reconstruction, pixels,
                ).item(),
                psnr=psnr_fn(output.reconstruction, pixels),
                msssim=msssim_fn(output.reconstruction, pixels, device),
            ))
        update_sums(sums, metrics, batch_size)
        samples += batch_size

        if (logger is not None and
                (batch_index % log_interval == 0 or
                 batch_index == len(loader) - 1)):
            averages = average_sums(sums, samples)
            logger.info(
                'Val [%d/%d] | %s',
                batch_index + 1,
                len(loader),
                ' | '.join(
                    '{} PSNR: {:.2f} dB'.format(name, values['psnr'])
                    for name, values in zip(PREFIX_NAMES, averages)
                ),
            )
    model.train(was_training)
    return average_sums(sums, samples)
