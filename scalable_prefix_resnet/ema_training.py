"""EMA-balanced pure-distortion training for three scalable prefixes.

The raw distortions of Base, Mid and Full can differ by more than one order
of magnitude.  A fixed weighted sum therefore lets the largest raw error
dominate training.  This module tracks an exponential moving average (EMA)
of each distortion and derives detached, normalized weights whose expected
loss shares match a user-selected Base/Mid/Full ratio.
"""

from dataclasses import dataclass

import torch

from util import msssim_fn, psnr_fn


PREFIX_NAMES = ("Base", "Mid", "Full")


def validate_target_shares(shares):
    values = tuple(float(value) for value in shares)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("target_shares must contain three positive values")
    total = sum(values)
    return tuple(value / total for value in values)


class EMADistortionBalancer:
    """Convert distortion EMAs into normalized effective loss weights."""

    def __init__(self, target_shares=(0.2, 0.3, 0.5), decay=0.99, epsilon=1e-12):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.target_shares = validate_target_shares(target_shares)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.ema = None

    @property
    def initialized(self):
        return self.ema is not None

    def update(self, distortions):
        values = tuple(float(value) for value in distortions)
        if len(values) != 3 or any(value < 0 for value in values):
            raise ValueError("distortions must contain three non-negative values")
        if self.ema is None:
            self.ema = values
        else:
            self.ema = tuple(
                self.decay * average + (1.0 - self.decay) * value
                for average, value in zip(self.ema, values)
            )

    def weights(self):
        if self.ema is None:
            raise RuntimeError("EMA must be initialized before computing weights")
        raw = tuple(
            share / max(average, self.epsilon)
            for share, average in zip(self.target_shares, self.ema)
        )
        total = sum(raw)
        return tuple(value / total for value in raw)

    def state_dict(self):
        return {
            "target_shares": self.target_shares,
            "decay": self.decay,
            "epsilon": self.epsilon,
            "ema": self.ema,
        }

    def load_state_dict(self, state):
        shares = validate_target_shares(state["target_shares"])
        if any(abs(a - b) > 1e-12 for a, b in zip(shares, self.target_shares)):
            raise ValueError("checkpoint target shares do not match")
        if abs(float(state["decay"]) - self.decay) > 1e-12:
            raise ValueError("checkpoint EMA decay does not match")
        ema = state.get("ema")
        self.ema = None if ema is None else tuple(float(value) for value in ema)


@dataclass(frozen=True)
class BalancedPrefixMetrics:
    quality_level: int
    distortion: float
    psnr: float
    msssim: float
    weight: float
    contribution: float


@torch.no_grad()
def initialize_balancer(model, coords, target, distortion_fn, balancer):
    """Initialize the EMA from one batch without retaining three graphs."""
    distortions = []
    for quality_level in range(3):
        reconstruction = model.forward_prefix(
            coords, quality_level,
        ).reconstruction
        distortions.append(distortion_fn(reconstruction, target).item())
    balancer.update(distortions)


def accumulate_ema_prefix_gradients(
    model,
    coords,
    target,
    distortion_fn,
    balancer,
    device=None,
    metric_fn=msssim_fn,
):
    """Backpropagate Base, Mid and Full sequentially, then update the EMA.

    The caller clears gradients before this function and performs exactly one
    optimizer step afterwards.  Effective weights are computed from the EMA
    available before the current update, so they are detached from autograd.
    """
    if device is None:
        device = coords.device
    if not balancer.initialized:
        initialize_balancer(model, coords, target, distortion_fn, balancer)

    weights = balancer.weights()
    metrics = []
    distortions = []
    objective_value = 0.0
    for quality_level, weight in enumerate(weights):
        reconstruction = model.forward_prefix(
            coords, quality_level,
        ).reconstruction
        distortion = distortion_fn(reconstruction, target)
        objective = weight * distortion
        objective.backward()

        distortion_value = distortion.detach().item()
        contribution = objective.detach().item()
        distortions.append(distortion_value)
        objective_value += contribution
        metrics.append(BalancedPrefixMetrics(
            quality_level=quality_level,
            distortion=distortion_value,
            psnr=psnr_fn(reconstruction, target),
            msssim=metric_fn(reconstruction, target, device),
            weight=weight,
            contribution=contribution,
        ))
        del reconstruction, distortion, objective

    balancer.update(distortions)
    return tuple(metrics), objective_value
