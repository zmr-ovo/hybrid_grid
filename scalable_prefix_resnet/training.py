"""Training helpers for balanced single-operating-point sampling.

Only one prefix is evaluated and differentiated for each mini-batch.  Prefix
selection follows a deterministic round-robin sequence, which gives every
operating point the same number of updates over time without retaining or
combining three computation graphs for the same frame.
"""

from dataclasses import dataclass

import torch

from util import msssim_fn, psnr_fn


PREFIX_NAMES = ("Base", "Mid", "Full")


class BalancedPrefixScheduler:
    """Deterministic round-robin assignment of batches to prefixes."""

    def __init__(self, count=3, start_step=0):
        if count < 1:
            raise ValueError("count must be positive")
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        self.count = int(count)
        self.step = int(start_step)

    def next(self):
        quality_level = self.step % self.count
        self.step += 1
        return quality_level

    def state_dict(self):
        return {"count": self.count, "step": self.step}

    def load_state_dict(self, state):
        if int(state.get("count", -1)) != self.count:
            raise ValueError("prefix scheduler count does not match")
        step = int(state.get("step", -1))
        if step < 0:
            raise ValueError("invalid prefix scheduler step")
        self.step = step


@dataclass(frozen=True)
class SampleMetrics:
    quality_level: int
    distortion: float
    psnr: float
    msssim: float


def train_sample(
    model,
    coords,
    target,
    distortion_fn,
    quality_level,
    device,
    metric_fn=msssim_fn,
):
    """Backpropagate exactly one sampled operating point."""
    output = model.forward_prefix(coords, quality_level)
    distortion = distortion_fn(output.reconstruction, target)
    distortion.backward()
    return SampleMetrics(
        quality_level=quality_level,
        distortion=distortion.detach().item(),
        psnr=psnr_fn(output.reconstruction, target),
        msssim=metric_fn(output.reconstruction, target, device),
    )


def empty_sums():
    return [
        {"count": 0, "distortion": 0.0, "psnr": 0.0, "msssim": 0.0}
        for _ in PREFIX_NAMES
    ]


def update_sums(sums, metrics, batch_size):
    values = sums[metrics.quality_level]
    values["count"] += batch_size
    values["distortion"] += metrics.distortion * batch_size
    values["psnr"] += metrics.psnr * batch_size
    values["msssim"] += metrics.msssim * batch_size


def average_sums(sums):
    averages = []
    for values in sums:
        count = values["count"]
        if count:
            averages.append({
                "count": count,
                "distortion": values["distortion"] / count,
                "psnr": values["psnr"] / count,
                "msssim": values["msssim"] / count,
            })
        else:
            averages.append({
                "count": 0,
                "distortion": 0.0,
                "psnr": 0.0,
                "msssim": 0.0,
            })
    return tuple(averages)


@torch.no_grad()
def evaluate_prefixes(
    model,
    loader,
    device,
    distortion_fn,
    log_interval=50,
    logger=None,
):
    """Evaluate every prefix on every validation frame."""
    was_training = model.training
    model.eval()
    sums = [
        {"distortion": 0.0, "psnr": 0.0, "msssim": 0.0}
        for _ in PREFIX_NAMES
    ]
    samples = 0
    for batch_index, batch in enumerate(loader):
        coords = batch["coords"].to(device, non_blocking=True)
        target = batch["pixels"].to(device, non_blocking=True)
        batch_size = target.shape[0]
        for quality_level in range(len(PREFIX_NAMES)):
            output = model.forward_prefix(coords, quality_level)
            values = sums[quality_level]
            values["distortion"] += (
                distortion_fn(output.reconstruction, target).item() * batch_size
            )
            values["psnr"] += (
                psnr_fn(output.reconstruction, target) * batch_size
            )
            values["msssim"] += (
                msssim_fn(output.reconstruction, target, device) * batch_size
            )
        samples += batch_size

        if (logger is not None and
                (batch_index % log_interval == 0 or
                 batch_index == len(loader) - 1)):
            logger.info(
                "Val [%d/%d] | %s",
                batch_index + 1,
                len(loader),
                " | ".join(
                    "{} PSNR: {:.2f} dB".format(
                        name, sums[index]["psnr"] / samples,
                    )
                    for index, name in enumerate(PREFIX_NAMES)
                ),
            )

    model.train(was_training)
    return tuple(
        {name: value / samples for name, value in values.items()}
        for values in sums
    )
