import torch
from torch import nn

from model import HybridGridNet

from .injection import inject_gop_lora


class GOPLoRAHybridGridNet(nn.Module):
    """Route a single shared Hybrid Grid model through GOP adapters."""

    architecture = 'hybrid_grid_gop_lora_v1'

    def __init__(self, shared_model, num_gops, rank, alpha=1.0):
        super().__init__()
        if not isinstance(shared_model, HybridGridNet):
            raise TypeError("shared_model must be HybridGridNet")

        self.injected_layers = inject_gop_lora(
            shared_model,
            num_gops=num_gops,
            rank=rank,
            alpha=alpha,
            target='decoder',
        )
        self.shared_model = shared_model
        self.num_gops = num_gops
        self.rank = rank
        self.alpha = float(alpha)

    def forward(self, coords, gop_index, grids=None):
        batch_size, _, height, width = coords.shape
        coords_hw = coords.permute(0, 2, 3, 1)

        grid_features = self.shared_model.grid_encoder(coords_hw, grids=grids)
        position_features = self.shared_model.pe_encoder(coords_hw)
        features = torch.cat([grid_features, position_features], dim=-1)

        gated_grid = self.shared_model.gate_grid(features) * grid_features
        gated_position = (
            self.shared_model.gate_pe(features) * position_features
        )
        fused = torch.cat([gated_grid, gated_position], dim=-1)
        modulated = self.shared_model.time_mod(
            fused.permute(0, 3, 1, 2), coords,
        ).permute(0, 2, 3, 1)

        flattened = modulated.reshape(batch_size * height * width, -1)
        rgb = self.shared_model.decoder(flattened, gop_index)
        return rgb.view(batch_size, height, width, 3).permute(0, 3, 1, 2)
