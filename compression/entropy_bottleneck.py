import math

import torch
from torch import nn
from torch.nn import functional as F


class EntropyBottleneck(nn.Module):
    """Learn a fully factorized, per-channel distribution for one Grid level."""

    def __init__(
        self,
        channels,
        filters=(3, 3, 3),
        init_scale=10.0,
        likelihood_bound=1e-9,
    ):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not filters or any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in filters
        ):
            raise ValueError("filters must contain positive integers")
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("init_scale must be finite and positive")
        if not 0 < likelihood_bound < 1:
            raise ValueError("likelihood_bound must be between 0 and 1")

        self.channels = channels
        self.likelihood_bound = likelihood_bound
        widths = (1,) + tuple(filters) + (1,)
        scale = init_scale ** (1 / (len(widths) - 1))

        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for index in range(len(widths) - 1):
            matrix_init = math.log(math.expm1(1 / (scale * widths[index + 1])))
            self.matrices.append(nn.Parameter(torch.full(
                (channels, widths[index + 1], widths[index]),
                matrix_init,
            )))
            self.biases.append(nn.Parameter(torch.empty(
                channels, widths[index + 1], 1,
            ).uniform_(-0.5, 0.5)))
            if index < len(widths) - 2:
                self.factors.append(nn.Parameter(torch.zeros(
                    channels, widths[index + 1], 1,
                )))

    def _logits_cumulative(self, inputs):
        logits = inputs
        for index, (matrix, bias) in enumerate(zip(self.matrices, self.biases)):
            logits = torch.matmul(F.softplus(matrix), logits) + bias
            if index < len(self.factors):
                logits = logits + torch.tanh(self.factors[index]) * torch.tanh(logits)
        return logits

    def forward(self, values):
        if not torch.is_tensor(values):
            raise TypeError("values must be a torch.Tensor")
        if not torch.is_floating_point(values):
            raise TypeError("values must be a floating-point tensor")
        if values.numel() == 0:
            raise ValueError("values must not be empty")
        if values.ndim < 1 or values.shape[-1] != self.channels:
            raise ValueError(
                f"last dimension must contain {self.channels} channels"
            )
        if not torch.isfinite(values).all():
            raise ValueError("values must contain only finite values")

        original_shape = values.shape
        channel_first = values.float().reshape(-1, self.channels).transpose(0, 1)
        channel_first = channel_first.unsqueeze(1)

        lower = self._logits_cumulative(channel_first - 0.5)
        upper = self._logits_cumulative(channel_first + 0.5)
        sign = -torch.sign(lower + upper).detach()
        likelihoods = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        likelihoods = likelihoods.clamp(self.likelihood_bound, 1.0)

        return likelihoods.squeeze(1).transpose(0, 1).reshape(original_shape)
