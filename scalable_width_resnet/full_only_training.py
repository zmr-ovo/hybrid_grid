"""Pure Full-prefix backward helper for WidthScalableResNet diagnostics."""

from dataclasses import dataclass

from util import psnr_fn


@dataclass(frozen=True)
class FullMetrics:
    distortion: float
    psnr: float


def backward_full_only(model, coords, target, distortion_fn):
    """Backpropagate only quality level 2 and return its reconstruction."""
    output = model.forward_prefix(coords, quality_level=2)
    distortion = distortion_fn(output.reconstruction, target)
    distortion.backward()
    metrics = FullMetrics(
        distortion=distortion.detach().item(),
        psnr=psnr_fn(output.reconstruction, target),
    )
    return output.reconstruction, metrics
