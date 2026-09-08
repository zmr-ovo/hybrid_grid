import torch
from torch import nn

from model import HybridGridNet

from .injection import inject_gop_lora


class GOPLoRAHybridGridNet(nn.Module):
    """Use GOP 0 as the anchor and route later GOPs through adapters."""

    architecture = 'hybrid_grid_anchor_gop_lora_v1'

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
        self.num_adapters = num_gops - 1
        self.rank = rank
        self.alpha = float(alpha)

    def forward(self, coords, gop_index, gop_local_time, grids=None):
        local_coords = self._local_coordinates(coords, gop_local_time)
        batch_size, _, height, width = local_coords.shape
        coords_hw = local_coords.permute(0, 2, 3, 1)

        grid_features = self.shared_model.grid_encoder(coords_hw, grids=grids)
        position_features = self.shared_model.pe_encoder(coords_hw)
        features = torch.cat([grid_features, position_features], dim=-1)

        gated_grid = self.shared_model.gate_grid(features) * grid_features
        gated_position = (
            self.shared_model.gate_pe(features) * position_features
        )
        fused = torch.cat([gated_grid, gated_position], dim=-1)
        modulated = self.shared_model.time_mod(
            fused.permute(0, 3, 1, 2), local_coords,
        ).permute(0, 2, 3, 1)

        flattened = modulated.reshape(batch_size * height * width, -1)
        rgb = self.shared_model.decoder(flattened, gop_index)
        return rgb.view(batch_size, height, width, 3).permute(0, 3, 1, 2)

    @staticmethod
    def _local_coordinates(coords, gop_local_time):
        if not torch.is_tensor(coords):
            raise TypeError("coords must be a torch.Tensor")
        if coords.ndim != 4 or coords.size(1) != 3:
            raise ValueError("coords must have shape [B, 3, H, W]")

        if torch.is_tensor(gop_local_time):
            if not torch.is_floating_point(gop_local_time):
                raise TypeError("gop_local_time tensor must be floating point")
            times = gop_local_time.detach().reshape(-1)
        elif isinstance(gop_local_time, (int, float)) and not isinstance(
            gop_local_time, bool,
        ):
            times = coords.new_tensor([gop_local_time])
        else:
            raise TypeError("gop_local_time must be a float or tensor")

        if times.numel() == 1:
            times = times.expand(coords.size(0))
        elif times.numel() != coords.size(0):
            raise ValueError("gop_local_time must contain one value per sample")
        if not torch.isfinite(times).all() or torch.any(times < 0) or torch.any(
            times > 1,
        ):
            raise ValueError("gop_local_time must be in [0, 1]")

        local_coords = coords.clone()
        local_coords[:, 2] = times.to(
            device=coords.device, dtype=coords.dtype,
        ).view(-1, 1, 1)
        return local_coords
