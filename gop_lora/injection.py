import torch
from torch import nn

from model import Decoder, HybridGridNet, TemporalModulation

from .linear import GOPLoRALinear, HierarchicalGOPLoRALinear


def _lora_linear(linear, num_gops, rank, alpha):
    effective_rank = min(rank, linear.in_features, linear.out_features)
    effective_alpha = float(alpha) * effective_rank / rank
    return GOPLoRALinear(
        linear, num_gops, effective_rank, effective_alpha,
    )


def _hierarchical_lora_linear(linear, num_gops, rank, alpha,
                              common_rank, common_alpha):
    if (not isinstance(common_rank, int) or isinstance(common_rank, bool)
            or common_rank < 1):
        raise ValueError("common_rank must be a positive integer")
    effective_rank = min(rank, linear.in_features, linear.out_features)
    effective_alpha = float(alpha) * effective_rank / rank
    effective_common_rank = min(
        common_rank, linear.in_features, linear.out_features,
    )
    effective_common_alpha = (
        float(common_alpha) * effective_common_rank / common_rank
    )
    return HierarchicalGOPLoRALinear(
        linear,
        num_gops=num_gops,
        rank=effective_rank,
        alpha=effective_alpha,
        common_rank=effective_common_rank,
        common_alpha=effective_common_alpha,
    )


def hierarchical_lora_factory(common_rank, common_alpha):
    def make_linear(linear, num_gops, rank, alpha):
        return _hierarchical_lora_linear(
            linear, num_gops, rank, alpha,
            common_rank, common_alpha,
        )

    return make_linear


class GOPLoRAGate(nn.Module):
    """Route a gated Linear projection through one adapter per later GOP."""

    def __init__(self, shared_gate, num_gops, rank, alpha,
                 linear_factory=_lora_linear):
        super().__init__()
        if (not isinstance(shared_gate, nn.Sequential)
                or len(shared_gate) != 2
                or not isinstance(shared_gate[0], nn.Linear)):
            raise TypeError("shared_gate must be Linear followed by activation")
        self.linear = linear_factory(
            shared_gate[0], num_gops, rank, alpha,
        )
        self.activation = shared_gate[1]

    def forward(self, inputs, gop_index):
        return self.activation(self.linear(inputs, gop_index))


class GOPLoRATemporalModulation(nn.Module):
    """Temporal modulation whose two Linear layers use GOP adapters."""

    def __init__(self, shared_module, num_gops, rank, alpha,
                 linear_factory=_lora_linear):
        super().__init__()
        if not isinstance(shared_module, TemporalModulation):
            raise TypeError("shared_module must be TemporalModulation")
        self.first = linear_factory(
            shared_module.mlp[0], num_gops, rank, alpha,
        )
        self.activation = shared_module.mlp[1]
        self.second = linear_factory(
            shared_module.mlp[2], num_gops, rank, alpha,
        )
        self.norm = shared_module.norm

    def forward(self, inputs, coords, gop_index):
        batch, channels, _, _ = inputs.shape
        time = coords[:, 2, 0, 0].unsqueeze(1)
        hidden = self.activation(self.first(time, gop_index))
        gamma, beta = self.second(hidden, gop_index).chunk(2, dim=1)
        gamma = gamma.view(batch, channels, 1, 1)
        beta = beta.view(batch, channels, 1, 1)
        return self.norm(inputs) * gamma + beta


class GOPLoRADecoder(nn.Module):
    """Use the anchor Decoder for GOP 0 and route later-GOP adapters."""

    def __init__(self, shared_decoder, num_gops, rank, alpha,
                 linear_factory=_lora_linear):
        super().__init__()
        if not isinstance(shared_decoder, Decoder):
            raise TypeError("shared_decoder must be the paper Decoder")

        self.input_dim = shared_decoder.input_dim
        self.hidden_dim = shared_decoder.hidden_dim
        self.skip_layer = shared_decoder.skip_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.linear = nn.ModuleList([
            linear_factory(layer, num_gops, rank, alpha)
            for layer in shared_decoder.linear
        ])
        output_linear = shared_decoder.output_layer[0]
        self.output_layer = nn.ModuleList([
            linear_factory(output_linear, num_gops, rank, alpha),
            shared_decoder.output_layer[1],
        ])
        self.act = shared_decoder.act

    @property
    def effective_ranks(self):
        return tuple(
            layer.rank for layer in tuple(self.linear) + (self.output_layer[0],)
        )

    def forward(self, inputs, gop_index):
        hidden = inputs
        for index, layer in enumerate(self.linear):
            if index == self.skip_layer:
                hidden = torch.cat([inputs, hidden], dim=-1)
            hidden = self.act(layer(hidden, gop_index))

        output = self.output_layer[0](hidden, gop_index)
        return self.output_layer[1](output)


def inject_gop_lora(model, num_gops, rank, alpha=1.0, target='decoder',
                    linear_factory=_lora_linear):
    """Add independent later-GOP adapters to Decoder or every Linear layer."""
    if not isinstance(model, HybridGridNet):
        raise TypeError("model must be HybridGridNet")
    if target not in ('decoder', 'all_linear'):
        raise ValueError("target must be 'decoder' or 'all_linear'")
    if isinstance(model.decoder, GOPLoRADecoder):
        raise ValueError("GOP LoRA has already been injected")
    if not isinstance(model.decoder, Decoder):
        raise ValueError("model does not contain the paper Decoder")

    names = []
    if target == 'all_linear':
        model.gate_grid = GOPLoRAGate(
            model.gate_grid, num_gops, rank, alpha, linear_factory,
        )
        model.gate_pe = GOPLoRAGate(
            model.gate_pe, num_gops, rank, alpha, linear_factory,
        )
        model.time_mod = GOPLoRATemporalModulation(
            model.time_mod, num_gops, rank, alpha, linear_factory,
        )
        names.extend((
            'gate_grid.linear',
            'gate_pe.linear',
            'time_mod.first',
            'time_mod.second',
        ))

    model.decoder = GOPLoRADecoder(
        model.decoder, num_gops=num_gops, rank=rank, alpha=alpha,
        linear_factory=linear_factory,
    )
    names.extend(
        ['decoder.linear.{}'.format(index)
         for index in range(len(model.decoder.linear))]
        + ['decoder.output_layer.0']
    )
    return tuple(names)


def gop_lora_layers(model):
    return tuple(
        module for module in model.modules()
        if isinstance(module, GOPLoRALinear)
    )


def lora_parameters(model):
    return tuple(
        parameter
        for layer in gop_lora_layers(model)
        for parameter in layer.lora_parameters()
    )


def gop_parameters(model, gop_index):
    return tuple(
        parameter
        for layer in gop_lora_layers(model)
        for parameter in layer.gop_parameters(gop_index)
    )


def shared_parameters(model):
    lora_ids = {id(parameter) for parameter in lora_parameters(model)}
    return tuple(
        parameter for parameter in model.parameters()
        if id(parameter) not in lora_ids
    )


def freeze_shared_parameters(model):
    layers = gop_lora_layers(model)
    if not layers:
        raise ValueError("model does not contain GOP LoRA layers")
    for parameter in shared_parameters(model):
        parameter.requires_grad_(False)
    for parameter in lora_parameters(model):
        parameter.requires_grad_(True)


def gop_adaptation_parameters(model, gop_index):
    parameters = list(gop_parameters(model, gop_index))
    grid_residuals = getattr(model, 'grid_residuals', None)
    if grid_residuals is not None:
        parameters.extend(grid_residuals.gop_parameters(gop_index))
    return tuple(parameters)


def freeze_for_gop(model, gop_index):
    """Freeze the anchor and enable only one GOP's adaptation parameters."""
    parameters = gop_adaptation_parameters(model, gop_index)
    if not parameters:
        raise ValueError("the selected GOP has no adaptation parameters")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters
