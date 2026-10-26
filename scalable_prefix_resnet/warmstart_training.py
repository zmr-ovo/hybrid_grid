"""Sequential pure-distortion updates for Full-checkpoint warm starting.

The three prefix graphs are built and released one at a time.  Their weighted
gradients accumulate on the student before a single optimizer update.  A
frozen Additive-Skip teacher anchors the Full output to the known high-quality
solution without introducing rate, quantization or entropy objectives.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from util import msssim_fn, psnr_fn


@dataclass(frozen=True)
class PrefixDistortionMetrics:
    quality_level: int
    distortion: float
    psnr: float
    msssim: float


def validate_distortion_weights(weights):
    values = tuple(float(value) for value in weights)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("prefix_weights must contain three positive values")
    return values


def accumulate_warmstart_gradients(
    student,
    teacher,
    coords,
    target,
    prefix_weights=(0.25, 0.5, 1.0),
    anchor_weight=1.0,
    device=None,
    metric_fn=msssim_fn,
):
    """Backpropagate Base, Mid and Full sequentially.

    The caller must clear gradients before this function and execute the
    optimizer step afterwards.  The returned total is a detached scalar used
    only for logging.
    """
    weights = validate_distortion_weights(prefix_weights)
    anchor_weight = float(anchor_weight)
    if anchor_weight < 0:
        raise ValueError("anchor_weight must be non-negative")
    if device is None:
        device = coords.device

    with torch.no_grad():
        teacher_full = teacher(coords)

    metrics = []
    total_value = 0.0
    anchor_value = 0.0
    for quality_level, weight in enumerate(weights):
        reconstruction = student.forward_prefix(
            coords, quality_level,
        ).reconstruction
        distortion = F.mse_loss(reconstruction, target)
        objective = weight * distortion
        if quality_level == 2 and anchor_weight:
            anchor = F.mse_loss(reconstruction, teacher_full)
            objective = objective + anchor_weight * anchor
            anchor_value = anchor.detach().item()
        objective.backward()
        total_value += objective.detach().item()
        metrics.append(PrefixDistortionMetrics(
            quality_level=quality_level,
            distortion=distortion.detach().item(),
            psnr=psnr_fn(reconstruction, target),
            msssim=metric_fn(reconstruction, target, device),
        ))
        del reconstruction, distortion, objective

    return tuple(metrics), total_value, anchor_value
