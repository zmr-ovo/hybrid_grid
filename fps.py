import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FpsBenchmarkResult:
    reconstruction_fps: float
    decoder_fps: float
    warmup_steps: int
    repeat_steps: int
    batch_size: int
    codec_encode_ms: Optional[float] = None
    codec_decode_ms: Optional[float] = None

    @property
    def paper_inference_fps(self):
        """论文推理 FPS：完整模型 forward，不包含 codec 时间。"""
        return self.reconstruction_fps


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _measure(callable_fn, device, warmup_steps, repeat_steps):
    for _ in range(warmup_steps):
        callable_fn()

    _synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat_steps):
        callable_fn()
    _synchronize(device)
    return time.perf_counter() - start


def _capture_decoder_inputs(model, coords):
    if not hasattr(model, 'decoder'):
        raise AttributeError("model must expose a decoder module")

    captured = {}

    def capture(_, inputs):
        captured['inputs'] = inputs

    handle = model.decoder.register_forward_pre_hook(capture)
    try:
        model(coords)
    finally:
        handle.remove()

    if 'inputs' not in captured:
        raise RuntimeError("decoder was not called during model forward")
    return captured['inputs']


@torch.no_grad()
def benchmark_fps(model, coords, warmup_steps=5, repeat_steps=20):
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1")
    if repeat_steps < 10:
        raise ValueError("repeat_steps must be at least 10")
    if coords.size(0) < 1:
        raise ValueError("benchmark batch must contain at least one frame")

    device = coords.device
    batch_size = coords.size(0)
    measured_frames = batch_size * repeat_steps
    was_training = model.training
    model.eval()

    try:
        decoder_inputs = _capture_decoder_inputs(model, coords)
        reconstruction_seconds = _measure(
            lambda: model(coords), device, warmup_steps, repeat_steps,
        )
        decoder_seconds = _measure(
            lambda: model.decoder(*decoder_inputs),
            device,
            warmup_steps,
            repeat_steps,
        )
    finally:
        model.train(was_training)

    return FpsBenchmarkResult(
        reconstruction_fps=measured_frames / max(reconstruction_seconds, 1e-12),
        decoder_fps=measured_frames / max(decoder_seconds, 1e-12),
        warmup_steps=warmup_steps,
        repeat_steps=repeat_steps,
        batch_size=batch_size,
    )
