from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import parametrize


@dataclass(frozen=True)
class QuantizedParameterStorage:
    tensor_count: int
    parameter_count: int
    payload_bits: int
    metadata_bits: int


class FakeQuantizedParameter(nn.Module):
    """Unsigned affine fake quantization with a straight-through gradient."""

    def __init__(self, parameter, bits=8, axis=None, group='non_grid'):
        super().__init__()
        if (not torch.is_tensor(parameter) or
                not torch.is_floating_point(parameter)):
            raise TypeError("parameter must be a floating-point tensor")
        if (not isinstance(bits, int) or isinstance(bits, bool) or
                not 2 <= bits <= 16):
            raise ValueError("bits must be an integer between 2 and 16")
        if axis is not None and axis != 0:
            raise ValueError("only output-channel axis 0 is supported")
        if axis == 0 and parameter.ndim < 2:
            raise ValueError(
                "axis 0 requires a parameter with at least two dimensions"
            )

        qparam_shape = self._qparam_shape(parameter, axis)
        self.bits = bits
        self.axis = axis
        self.group = group
        self.parameter_count = parameter.numel()
        self.parameter_rank = parameter.ndim
        self.register_buffer('scale', parameter.new_ones(qparam_shape))
        self.register_buffer(
            'zero_point', torch.zeros(qparam_shape, dtype=torch.int32,
                                      device=parameter.device),
        )
        self.register_buffer(
            '_enabled', torch.tensor(False, device=parameter.device),
        )
        self.register_buffer(
            '_observer_enabled', torch.tensor(True, device=parameter.device),
        )
        self._update_qparams(parameter)

    @staticmethod
    def _qparam_shape(parameter, axis):
        if axis is None:
            return ()
        return (parameter.shape[0],) + (1,) * (parameter.ndim - 1)

    def _calculate_qparams(self, parameter):
        if self.axis is None:
            minimum = parameter.detach().amin()
            maximum = parameter.detach().amax()
        else:
            dimensions = tuple(range(1, parameter.ndim))
            minimum = parameter.detach().amin(dim=dimensions, keepdim=True)
            maximum = parameter.detach().amax(dim=dimensions, keepdim=True)

        zero = torch.zeros_like(minimum)
        minimum = torch.minimum(minimum, zero)
        maximum = torch.maximum(maximum, zero)
        qmax = 2 ** self.bits - 1
        value_range = maximum - minimum
        scale = torch.where(value_range > 0, value_range / qmax,
                            torch.ones_like(value_range))
        zero_point = torch.round(-minimum / scale).clamp(0, qmax).to(torch.int32)
        return scale, zero_point

    @torch.no_grad()
    def _update_qparams(self, parameter):
        scale, zero_point = self._calculate_qparams(parameter)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

    @property
    def enabled(self):
        return bool(self._enabled.item())

    @property
    def observer_enabled(self):
        return bool(self._observer_enabled.item())

    def configure(self, enabled, observer_enabled):
        self._enabled.fill_(bool(enabled))
        self._observer_enabled.fill_(bool(observer_enabled))

    def forward(self, parameter):
        if not self.enabled:
            return parameter
        if self.observer_enabled:
            self._update_qparams(parameter)

        qmax = 2 ** self.bits - 1
        zero_point = self.zero_point.to(parameter.dtype)
        quantized = torch.round(parameter / self.scale + zero_point)
        dequantized = (quantized.clamp(0, qmax) - zero_point) * self.scale
        return parameter + (dequantized - parameter).detach()

    def storage(self):
        qparam_count = self.scale.numel()
        # Per tensor: uint8 bits, int8 axis, uint8 rank and uint32 dimensions.
        header_bits = 24 + 32 * self.parameter_rank
        metadata_bits = header_bits + qparam_count * (32 + self.bits)
        return QuantizedParameterStorage(
            tensor_count=1,
            parameter_count=self.parameter_count,
            payload_bits=self.parameter_count * self.bits,
            metadata_bits=metadata_bits,
        )


def prepare_network_qat(model, bits=8, excluded_parameters=()):
    """Attach fake quantizers to every non-Grid trainable parameter."""
    excluded_ids = {id(parameter) for parameter in excluded_parameters}
    entropy_ids = {
        id(parameter) for parameter in model.entropy_models.parameters()
    }
    targets = []
    for _, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in excluded_ids:
                continue
            group = 'entropy' if id(parameter) in entropy_ids else 'non_grid'
            axis = 0 if parameter.ndim >= 2 else None
            targets.append((module, name, parameter, axis, group))

    for module, name, parameter, axis, group in targets:
        parametrize.register_parametrization(
            module,
            name,
            FakeQuantizedParameter(parameter, bits, axis, group),
        )
    return len(targets)


def iter_network_quantizers(model):
    for module in model.modules():
        if isinstance(module, FakeQuantizedParameter):
            yield module


def configure_network_qat(model, enabled, freeze=False):
    if freeze and not enabled:
        raise ValueError("frozen QAT must also be enabled")
    quantizers = tuple(iter_network_quantizers(model))
    if not quantizers:
        raise ValueError("model has not been prepared for network QAT")
    for quantizer in quantizers:
        quantizer.configure(enabled, enabled and not freeze)


def network_qat_state(model):
    quantizers = tuple(iter_network_quantizers(model))
    if not quantizers:
        return 'not prepared'
    if not any(quantizer.enabled for quantizer in quantizers):
        return 'disabled'
    if any(quantizer.observer_enabled for quantizer in quantizers):
        return 'calibrating'
    return 'frozen'


def network_qat_storage(model):
    grouped = {}
    for quantizer in iter_network_quantizers(model):
        current = grouped.setdefault(
            quantizer.group, QuantizedParameterStorage(0, 0, 0, 0),
        )
        item = quantizer.storage()
        grouped[quantizer.group] = QuantizedParameterStorage(
            tensor_count=current.tensor_count + item.tensor_count,
            parameter_count=current.parameter_count + item.parameter_count,
            payload_bits=current.payload_bits + item.payload_bits,
            metadata_bits=current.metadata_bits + item.metadata_bits,
        )
    return grouped
