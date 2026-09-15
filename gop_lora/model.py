import torch
from torch import nn

from model import HybridGridNet

from .grid_residual import (
    GOPGridResiduals,
    GOPLowRankGridResiduals,
    GOPStructuredGridResiduals,
)
from .injection import (
    GOPLoRAGate,
    GOPLoRATemporalModulation,
    inject_gop_lora,
)


def gop_local_coordinates(coords, gop_local_time):
    """Return a coordinate tensor whose time channel uses GOP-local time."""
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


class GOPLoRAHybridGridNet(nn.Module):
    """Use GOP 0 as the anchor and route later GOPs through adapters."""

    architecture = 'hybrid_grid_anchor_gop_lora_v1'

    def __init__(self, shared_model, num_gops, rank, alpha=1.0,
                 lora_target='decoder'):
        super().__init__()
        if not isinstance(shared_model, HybridGridNet):
            raise TypeError("shared_model must be HybridGridNet")

        self.injected_layers = inject_gop_lora(
            shared_model,
            num_gops=num_gops,
            rank=rank,
            alpha=alpha,
            target=lora_target,
        )
        self.shared_model = shared_model
        self.num_gops = num_gops
        self.num_adapters = num_gops - 1
        self.rank = rank
        self.alpha = float(alpha)
        self.lora_target = lora_target
        self.architecture = (
            'hybrid_grid_anchor_gop_lora_all_linear_v1'
            if lora_target == 'all_linear'
            else 'hybrid_grid_anchor_gop_lora_v1'
        )

    def forward(self, coords, gop_index, gop_local_time, grids=None):
        local_coords = gop_local_coordinates(coords, gop_local_time)
        batch_size, _, height, width = local_coords.shape
        coords_hw = local_coords.permute(0, 2, 3, 1)

        grid_features = self.shared_model.grid_encoder(coords_hw, grids=grids)
        position_features = self.shared_model.pe_encoder(coords_hw)
        features = torch.cat([grid_features, position_features], dim=-1)

        grid_gate = self.shared_model.gate_grid
        position_gate = self.shared_model.gate_pe
        if isinstance(grid_gate, GOPLoRAGate):
            grid_weights = grid_gate(features, gop_index)
            position_weights = position_gate(features, gop_index)
        else:
            grid_weights = grid_gate(features)
            position_weights = position_gate(features)
        gated_grid = grid_weights * grid_features
        gated_position = (
            position_weights * position_features
        )
        fused = torch.cat([gated_grid, gated_position], dim=-1)
        time_modulation = self.shared_model.time_mod
        if isinstance(time_modulation, GOPLoRATemporalModulation):
            modulated = time_modulation(
                fused.permute(0, 3, 1, 2), local_coords, gop_index,
            )
        else:
            modulated = time_modulation(
                fused.permute(0, 3, 1, 2), local_coords,
            )
        modulated = modulated.permute(0, 2, 3, 1)

        flattened = modulated.reshape(batch_size * height * width, -1)
        rgb = self.shared_model.decoder(flattened, gop_index)
        return rgb.view(batch_size, height, width, 3).permute(0, 3, 1, 2)


class GOPGridResidualHybridGridNet(GOPLoRAHybridGridNet):
    """Combine Decoder LoRA with additive Grid residuals for later GOPs."""

    architecture = 'hybrid_grid_anchor_grid_residual_decoder_lora_v1'

    def __init__(self, shared_model, num_gops, adapted_gops, rank, alpha=1.0,
                 lora_target='decoder'):
        super().__init__(
            shared_model, num_gops, rank, alpha, lora_target,
        )
        self.architecture = (
            'hybrid_grid_anchor_grid_residual_all_linear_lora_v1'
            if lora_target == 'all_linear'
            else 'hybrid_grid_anchor_grid_residual_decoder_lora_v1'
        )
        grids = tuple(
            level.grid for level in self.shared_model.grid_encoder.levels
        )
        self.grid_residuals = GOPGridResiduals(
            grids, num_gops, adapted_gops,
        )

    def forward(self, coords, gop_index, gop_local_time, grids=None):
        if grids is None:
            grids = tuple(
                level.grid for level in self.shared_model.grid_encoder.levels
            )
        adapted_grids = self.grid_residuals(grids, gop_index)
        return super().forward(
            coords, gop_index, gop_local_time, grids=adapted_grids,
        )

    def grid_parameters(self, gop_index):
        return self.grid_residuals.gop_parameters(gop_index)


class GOPLowRankGridHybridGridNet(GOPLoRAHybridGridNet):
    """Combine Decoder LoRA with low-rank Grid residuals."""

    architecture = 'hybrid_grid_anchor_grid_lora_decoder_lora_v1'

    def __init__(self, shared_model, num_gops, adapted_gops, rank, alpha,
                 grid_rank, grid_alpha, grid_init='random',
                 lora_target='decoder'):
        super().__init__(
            shared_model, num_gops, rank, alpha, lora_target,
        )
        grids = tuple(
            level.grid for level in self.shared_model.grid_encoder.levels
        )
        self.grid_residuals = GOPLowRankGridResiduals(
            grids, num_gops, adapted_gops, grid_rank, grid_alpha,
            grid_init,
        )
        self.grid_rank = grid_rank
        self.grid_alpha = float(grid_alpha)
        self.grid_init = grid_init
        self.architecture = (
            'hybrid_grid_anchor_grid_lora_all_linear_lora_v1'
            if lora_target == 'all_linear'
            else 'hybrid_grid_anchor_grid_lora_decoder_lora_v1'
        )

    def forward(self, coords, gop_index, gop_local_time, grids=None):
        if grids is None:
            grids = tuple(
                level.grid for level in self.shared_model.grid_encoder.levels
            )
        adapted_grids = self.grid_residuals(grids, gop_index)
        return super().forward(
            coords, gop_index, gop_local_time, grids=adapted_grids,
        )

    def grid_parameters(self, gop_index):
        return self.grid_residuals.gop_parameters(gop_index)

    def grid_lora_levels(self, gop_index):
        return self.grid_residuals.gop_levels(gop_index)


class GOPStructuredGridHybridGridNet(GOPLoRAHybridGridNet):
    """Combine network LoRA with a structured 4D Grid residual."""

    architecture = 'hybrid_grid_anchor_structured_grid_lora_v1'

    def __init__(self, shared_model, num_gops, adapted_gops, rank, alpha,
                 grid_rank, grid_alpha, lora_target='decoder'):
        super().__init__(
            shared_model, num_gops, rank, alpha, lora_target,
        )
        grids = tuple(
            level.grid for level in self.shared_model.grid_encoder.levels
        )
        self.grid_residuals = GOPStructuredGridResiduals(
            grids, num_gops, adapted_gops, grid_rank, grid_alpha,
        )
        self.grid_rank = grid_rank
        self.grid_alpha = float(grid_alpha)
        self.architecture = (
            'hybrid_grid_anchor_structured_grid_all_linear_lora_v1'
            if lora_target == 'all_linear'
            else 'hybrid_grid_anchor_structured_grid_decoder_lora_v1'
        )

    def forward(self, coords, gop_index, gop_local_time, grids=None):
        if grids is None:
            grids = tuple(
                level.grid for level in self.shared_model.grid_encoder.levels
            )
        adapted_grids = self.grid_residuals(grids, gop_index)
        return super().forward(
            coords, gop_index, gop_local_time, grids=adapted_grids,
        )

    def grid_parameters(self, gop_index):
        return self.grid_residuals.gop_parameters(gop_index)

    def grid_lora_levels(self, gop_index):
        return self.grid_residuals.gop_levels(gop_index)
