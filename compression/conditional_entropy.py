import math

import torch
from torch import nn
from torch.nn import functional as F

from .entropy_bottleneck import EntropyBottleneck


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


class _RecursiveEntropyStage(nn.Module):
    """Predict one level and fold it into a fixed-width cumulative context."""

    def __init__(self, channels, context_channels, hidden_channels,
                 init_scale, min_scale, likelihood_bound, update_context):
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels
        self.min_scale = min_scale
        self.likelihood_bound = likelihood_bound
        self.predictor = nn.Sequential(
            nn.Conv3d(context_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels * 2, kernel_size=1),
        )
        self.context_update = (
            nn.Sequential(
                nn.Conv3d(
                    context_channels + channels,
                    context_channels,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv3d(context_channels, context_channels, kernel_size=1),
            )
            if update_context else None
        )
        output = self.predictor[-1]
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(output.bias)
        raw_scale = math.log(math.expm1(init_scale - min_scale))
        with torch.no_grad():
            output.bias[channels:].fill_(raw_scale)

    @staticmethod
    def _channel_first(values):
        return values.permute(3, 2, 1, 0).unsqueeze(0)

    @staticmethod
    def _grid_layout(values):
        return values.squeeze(0).permute(3, 2, 1, 0)

    def _align_context(self, context, target_shape):
        target_size = (target_shape[2], target_shape[1], target_shape[0])
        return F.interpolate(
            self._channel_first(context.float()),
            size=target_size,
            mode='trilinear',
            align_corners=False,
        )

    def distribution_parameters(self, context, target_shape):
        """Predict discrete-logistic parameters without seeing target symbols."""
        aligned = self._align_context(context, target_shape)
        parameters = self.predictor(aligned)
        mean, raw_scale = parameters.chunk(2, dim=1)
        scale = F.softplus(raw_scale) + self.min_scale
        return self._grid_layout(mean), self._grid_layout(scale)

    def update(self, values, context):
        """Fold decoded symbols into the context used by the next level."""
        if self.context_update is None:
            return None
        aligned = self._align_context(context, values.shape)
        symbols = self._channel_first(values.float())
        return self._grid_layout(
            self.context_update(torch.cat((aligned, symbols), dim=1))
        )

    def forward(self, values, context):
        mean, scale = self.distribution_parameters(context, values.shape)

        symbols = values.float()
        upper = torch.sigmoid((symbols + 0.5 - mean) / scale)
        lower = torch.sigmoid((symbols - 0.5 - mean) / scale)
        likelihoods = (upper - lower).clamp(self.likelihood_bound, 1.0)
        return likelihoods.to(values.dtype), self.update(values, context)


class CumulativeCrossScaleEntropy(nn.Module):
    """Entropy model whose fixed-width state summarizes all decoded levels."""

    def __init__(self, channels, levels, context_channels=None,
                 hidden_channels=None, init_scale=10.0, min_scale=1e-3,
                 likelihood_bound=1e-9):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError("levels must be a positive integer")
        if context_channels is None:
            context_channels = max(8, channels)
        if hidden_channels is None:
            hidden_channels = max(8, channels * 2)
        if (not isinstance(context_channels, int) or
                isinstance(context_channels, bool) or context_channels < 1):
            raise ValueError("context_channels must be a positive integer")
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
        self.levels = levels
        self.context_channels = context_channels
        self.base_entropy = EntropyBottleneck(
            channels,
            init_scale=init_scale,
            likelihood_bound=likelihood_bound,
        )
        self.context_initializer = nn.Sequential(
            nn.Conv3d(channels, context_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(context_channels, context_channels, kernel_size=1),
        )
        self.stages = nn.ModuleList([
            _RecursiveEntropyStage(
                channels,
                context_channels,
                hidden_channels,
                init_scale,
                min_scale,
                likelihood_bound,
                index < levels - 2,
            )
            for index in range(levels - 1)
        ])

    def _validate_levels(self, values_by_level):
        if not isinstance(values_by_level, (tuple, list)):
            raise TypeError("values_by_level must be a tuple or list")
        if len(values_by_level) != self.levels:
            raise ValueError("values_by_level must match the configured levels")
        for index, values in enumerate(values_by_level):
            if not torch.is_tensor(values):
                raise TypeError("level {} must be a torch.Tensor".format(index))
            if not torch.is_floating_point(values):
                raise TypeError("level {} must be floating point".format(index))
            if values.ndim != 4 or values.shape[-1] != self.channels:
                raise ValueError(
                    "level {} must have shape [X, Y, T, {}]".format(
                        index, self.channels,
                    )
                )
            if values.numel() == 0 or not torch.isfinite(values).all():
                raise ValueError("level {} must be non-empty and finite".format(index))
        devices = {values.device for values in values_by_level}
        if len(devices) != 1:
            raise ValueError("all levels must be on the same device")

    @staticmethod
    def _channel_first(values):
        return values.permute(3, 2, 1, 0).unsqueeze(0)

    @staticmethod
    def _grid_layout(values):
        return values.squeeze(0).permute(3, 2, 1, 0)

    def initialize_context(self, base):
        """Create the recursive state from decoded base-level symbols."""
        return self._grid_layout(
            self.context_initializer(self._channel_first(base.float()))
        )

    def distribution_parameters(self, level, context, target_shape):
        """Return mean and scale for one non-base level."""
        if not isinstance(level, int) or isinstance(level, bool):
            raise TypeError("level must be an integer")
        if level < 1 or level >= self.levels:
            raise ValueError("level must identify a non-base configured level")
        return self.stages[level - 1].distribution_parameters(
            context, target_shape,
        )

    def update_context(self, level, values, context):
        """Update recursive state after a level has been decoded."""
        if level < 1 or level >= self.levels:
            raise ValueError("level must identify a non-base configured level")
        return self.stages[level - 1].update(values, context)

    def forward(self, values_by_level):
        self._validate_levels(values_by_level)
        base = values_by_level[0]
        likelihoods = [self.base_entropy(base)]
        context = self.initialize_context(base)
        for stage, values in zip(self.stages, values_by_level[1:]):
            likelihood, context = stage(values, context)
            likelihoods.append(likelihood)
        return tuple(likelihoods)
