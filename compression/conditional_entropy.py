import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossScaleConditionalEntropy(nn.Module):
    """Estimate one Grid level from the preceding lower-resolution level."""

    def __init__(self, channels, hidden_channels=None, init_scale=10.0,
                 min_scale=1e-3, likelihood_bound=1e-9):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if hidden_channels is None:
            hidden_channels = max(8, channels * 2)
        if (not isinstance(hidden_channels, int) or
                isinstance(hidden_channels, bool) or hidden_channels < 1):
            raise ValueError("hidden_channels must be a positive integer")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        if not math.isfinite(init_scale) or init_scale <= min_scale:
            raise ValueError("init_scale must be finite and greater than min_scale")
        if not 0 < likelihood_bound < 1:
            raise ValueError("likelihood_bound must be between 0 and 1")

        self.channels = channels
        self.min_scale = float(min_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.predictor = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels * 2, kernel_size=1),
        )
        output = self.predictor[-1]
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(output.bias)
        raw_scale = math.log(math.expm1(init_scale - min_scale))
        with torch.no_grad():
            output.bias[channels:].fill_(raw_scale)

    def _validate(self, name, values):
        if not torch.is_tensor(values):
            raise TypeError("{} must be a torch.Tensor".format(name))
        if not torch.is_floating_point(values):
            raise TypeError("{} must be floating point".format(name))
        if values.ndim != 4 or values.shape[-1] != self.channels:
            raise ValueError(
                "{} must have shape [X, Y, T, {}]".format(
                    name, self.channels,
                )
            )
        if values.numel() == 0:
            raise ValueError("{} must not be empty".format(name))
        if not torch.isfinite(values).all():
            raise ValueError("{} must contain only finite values".format(name))

    @staticmethod
    def _channel_first(values):
        return values.permute(3, 2, 1, 0).unsqueeze(0)

    @staticmethod
    def _grid_layout(values):
        return values.squeeze(0).permute(3, 2, 1, 0)

    def align_condition(self, condition, target_shape):
        """Trilinearly align [X,Y,T,C] condition values to a target Grid."""
        self._validate('condition', condition)
        if (not isinstance(target_shape, (tuple, list)) or
                len(target_shape) != 4):
            raise ValueError("target_shape must describe [X, Y, T, C]")
        if target_shape[-1] != self.channels:
            raise ValueError("target and condition channel counts must match")

        target_size = (target_shape[2], target_shape[1], target_shape[0])
        aligned = F.interpolate(
            self._channel_first(condition),
            size=target_size,
            mode='trilinear',
            align_corners=False,
        )
        return self._grid_layout(aligned)

    def forward(self, values, condition):
        """Return discrete logistic likelihoods for normalized symbols."""
        self._validate('values', values)
        self._validate('condition', condition)
        if values.device != condition.device:
            raise ValueError("values and condition must be on the same device")
        if values.dtype != condition.dtype:
            raise ValueError("values and condition must have the same dtype")

        aligned = self.align_condition(condition, values.shape)
        parameters = self.predictor(self._channel_first(aligned.float()))
        mean, raw_scale = parameters.chunk(2, dim=1)
        scale = F.softplus(raw_scale) + self.min_scale

        symbols = self._channel_first(values.float())
        upper = torch.sigmoid((symbols + 0.5 - mean) / scale)
        lower = torch.sigmoid((symbols - 0.5 - mean) / scale)
        likelihoods = (upper - lower).clamp(self.likelihood_bound, 1.0)
        return self._grid_layout(likelihoods).to(values.dtype)
