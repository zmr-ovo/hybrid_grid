"""Small helpers for the Full-only capacity-bound experiment."""

from dataclasses import dataclass

from util import psnr_fn


@dataclass(frozen=True)
class FullOnlyBatch:
    distortion: float
    psnr: float


def backward_full_only(model, coords, target, distortion_fn):
    """Backpropagate only the Full prefix and return detached batch metrics."""
    output = model.forward_prefix(coords, quality_level=2)
    distortion = distortion_fn(output.reconstruction, target)
    distortion.backward()
    metrics = FullOnlyBatch(
        distortion=distortion.detach().item(),
        psnr=psnr_fn(output.reconstruction, target),
    )
    return output.reconstruction.detach(), metrics

