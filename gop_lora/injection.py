import torch
from torch import nn

from model import Decoder, HybridGridNet

from .linear import GOPLoRALinear


class GOPLoRADecoder(nn.Module):
    """Use the anchor Decoder for GOP 0 and route later-GOP adapters."""

    def __init__(self, shared_decoder, num_gops, rank, alpha):
        super().__init__()
        if not isinstance(shared_decoder, Decoder):
            raise TypeError("shared_decoder must be the paper Decoder")

        self.input_dim = shared_decoder.input_dim
        self.hidden_dim = shared_decoder.hidden_dim
        self.skip_layer = shared_decoder.skip_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.linear = nn.ModuleList([
            GOPLoRALinear(layer, num_gops, rank, alpha)
            for layer in shared_decoder.linear
        ])
        output_linear = shared_decoder.output_layer[0]
        output_rank = min(
            rank, output_linear.in_features, output_linear.out_features,
        )
        output_alpha = self.alpha * output_rank / rank
        self.output_layer = nn.ModuleList([
            GOPLoRALinear(
                output_linear, num_gops, output_rank, output_alpha,
            ),
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


def inject_gop_lora(model, num_gops, rank, alpha=1.0, target='decoder'):
    """Add one independent adapter per later GOP to each Decoder layer."""
    if not isinstance(model, HybridGridNet):
        raise TypeError("model must be HybridGridNet")
    if target != 'decoder':
        raise ValueError("only target='decoder' is currently supported")
    if isinstance(model.decoder, GOPLoRADecoder):
        raise ValueError("GOP LoRA has already been injected")
    if not isinstance(model.decoder, Decoder):
        raise ValueError("model does not contain the paper Decoder")

    model.decoder = GOPLoRADecoder(
        model.decoder, num_gops=num_gops, rank=rank, alpha=alpha,
    )
    return tuple(
        ['decoder.linear.{}'.format(index)
         for index in range(len(model.decoder.linear))]
        + ['decoder.output_layer.0']
    )


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
