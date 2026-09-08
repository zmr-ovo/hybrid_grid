import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


class GOPLoRALinear(nn.Module):
    """Use the anchor for GOP 0 and one independent adapter per later GOP."""

    def __init__(self, shared_linear, num_gops, rank, alpha=1.0):
        super().__init__()
        if not isinstance(shared_linear, nn.Linear):
            raise TypeError("shared_linear must be nn.Linear")
        self.num_gops = _positive_integer('num_gops', num_gops)
        self.rank = _positive_integer('rank', rank)
        if self.rank > min(
            shared_linear.in_features, shared_linear.out_features,
        ):
            raise ValueError("rank must not exceed the linear dimensions")
        if (isinstance(alpha, bool) or not isinstance(alpha, Real) or
                not math.isfinite(alpha) or alpha <= 0):
            raise ValueError("alpha must be finite and positive")

        self.shared = shared_linear
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.ParameterList()
        self.lora_b = nn.ParameterList()
        for _ in range(self.num_gops - 1):
            matrix_a = nn.Parameter(shared_linear.weight.new_empty(
                self.rank, shared_linear.in_features,
            ))
            matrix_b = nn.Parameter(shared_linear.weight.new_zeros(
                shared_linear.out_features, self.rank,
            ))
            nn.init.kaiming_uniform_(matrix_a, a=math.sqrt(5))
            self.lora_a.append(matrix_a)
            self.lora_b.append(matrix_b)

    @property
    def in_features(self):
        return self.shared.in_features

    @property
    def out_features(self):
        return self.shared.out_features

    def forward(self, inputs, gop_index):
        if not torch.is_tensor(inputs):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.shape[-1] != self.in_features:
            raise ValueError("inputs have an unexpected feature dimension")
        gop_index = self._normalize_gop_index(gop_index)

        base = self.shared(inputs)
        if gop_index == 0:
            return base

        adapter_index = gop_index - 1
        low_rank = F.linear(F.linear(inputs, self.lora_a[adapter_index]),
                            self.lora_b[adapter_index])
        return base + low_rank * self.scaling

    def shared_parameters(self):
        parameters = [self.shared.weight]
        if self.shared.bias is not None:
            parameters.append(self.shared.bias)
        return tuple(parameters)

    def lora_parameters(self):
        return tuple(self.lora_a) + tuple(self.lora_b)

    def gop_parameters(self, gop_index):
        gop_index = self._normalize_gop_index(gop_index)
        if gop_index == 0:
            return ()
        adapter_index = gop_index - 1
        return self.lora_a[adapter_index], self.lora_b[adapter_index]

    def _normalize_gop_index(self, gop_index):
        if torch.is_tensor(gop_index):
            if gop_index.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            ):
                raise TypeError("gop_index tensor must contain integers")
            if gop_index.numel() < 1:
                raise ValueError("gop_index tensor must not be empty")
            unique = torch.unique(gop_index.detach())
            if unique.numel() != 1:
                raise ValueError("one batch must contain exactly one GOP")
            gop_index = int(unique.item())
        elif not isinstance(gop_index, int) or isinstance(gop_index, bool):
            raise TypeError("gop_index must be an integer or integer tensor")

        if not 0 <= gop_index < self.num_gops:
            raise IndexError("gop_index is outside the available adapters")
        return gop_index

    def extra_repr(self):
        return (
            'in_features={}, out_features={}, num_gops={}, adapters={}, '
            'rank={}, alpha={}'
        ).format(
            self.in_features,
            self.out_features,
            self.num_gops,
            self.num_gops - 1,
            self.rank,
            self.alpha,
        )
