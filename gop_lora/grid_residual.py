import math
from collections.abc import Iterable
from numbers import Real

import torch
from torch import nn


def _gop_index(value):
    if torch.is_tensor(value):
        if value.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise TypeError("gop_index tensor must contain integers")
        unique = torch.unique(value.detach())
        if unique.numel() != 1:
            raise ValueError("one batch must contain exactly one GOP")
        value = int(unique.item())
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("gop_index must be an integer or integer tensor")
    return value


class GridResidualSet(nn.Module):
    """One zero-initialized full Grid residual for a later GOP."""

    def __init__(self, grids):
        super().__init__()
        self.levels = nn.ParameterList([
            nn.Parameter(torch.zeros_like(grid)) for grid in grids
        ])


class LowRankGridResidualLevel(nn.Module):
    """LoRA-style residual for one Grid flattened as [positions, features]."""

    def __init__(self, grid, rank, alpha, initialization='random'):
        super().__init__()
        if not torch.is_tensor(grid) or not torch.is_floating_point(grid):
            raise TypeError("grid must be a floating-point tensor")
        if grid.ndim < 2:
            raise ValueError("grid must contain positions and features")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                or not math.isfinite(alpha) or alpha <= 0):
            raise ValueError("alpha must be finite and positive")
        if initialization not in ('random', 'anchor_pca'):
            raise ValueError(
                "initialization must be 'random' or 'anchor_pca'"
            )

        positions = grid.numel() // grid.shape[-1]
        self.grid_shape = tuple(grid.shape)
        self.rank = min(rank, positions, grid.shape[-1])
        self.alpha = float(alpha) * self.rank / rank
        self.scaling = self.alpha / self.rank
        self.initialization = initialization
        self.left = nn.Parameter(grid.new_zeros(positions, self.rank))
        self.right = nn.Parameter(grid.new_empty(
            self.rank, grid.shape[-1],
        ))
        self._initialize_right(grid)

    @torch.no_grad()
    def _initialize_right(self, grid):
        if self.initialization == 'random':
            nn.init.kaiming_uniform_(self.right, a=math.sqrt(5))
            return

        features = grid.detach().reshape(-1, grid.shape[-1]).float()
        features = features - features.mean(dim=0, keepdim=True)
        covariance = features.transpose(0, 1).matmul(features)
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -self.rank:].transpose(0, 1)
        self.right.copy_(basis.to(dtype=self.right.dtype))

    def forward(self, grid):
        if tuple(grid.shape) != self.grid_shape:
            raise ValueError("Grid shape does not match the low-rank residual")
        residual = torch.matmul(self.left, self.right).view(self.grid_shape)
        return grid + residual * self.scaling

    def lora_parameters(self):
        return self.left, self.right


class LowRankGridResidualSet(nn.Module):
    """One independent collection of low-rank residuals for a later GOP."""

    def __init__(self, grids, rank, alpha, initialization):
        super().__init__()
        self.levels = nn.ModuleList([
            LowRankGridResidualLevel(
                grid, rank, alpha, initialization,
            )
            for grid in grids
        ])


class StructuredGridResidualLevel(nn.Module):
    """Factor a 4D Grid residual into spatial, temporal and channel axes."""

    def __init__(self, grid, rank, alpha):
        super().__init__()
        if not torch.is_tensor(grid) or not torch.is_floating_point(grid):
            raise TypeError("grid must be a floating-point tensor")
        if grid.ndim != 4:
            raise ValueError("structured Grid LoRA requires an [X, Y, T, C] grid")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if (isinstance(alpha, bool) or not isinstance(alpha, Real)
                or not math.isfinite(alpha) or alpha <= 0):
            raise ValueError("alpha must be finite and positive")

        size_x, size_y, size_t, channels = grid.shape
        self.grid_shape = tuple(grid.shape)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.spatial = nn.Parameter(grid.new_zeros(rank, size_x, size_y))
        self.temporal = nn.Parameter(grid.new_empty(rank, size_t))
        self.channel = nn.Parameter(grid.new_empty(rank, channels))
        nn.init.kaiming_uniform_(self.temporal, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.channel, a=math.sqrt(5))

    def forward(self, grid):
        if tuple(grid.shape) != self.grid_shape:
            raise ValueError("Grid shape does not match the structured residual")
        residual = torch.einsum(
            'rxy,rt,rc->xytc',
            self.spatial,
            self.temporal,
            self.channel,
        )
        return grid + residual * self.scaling

    def lora_parameters(self):
        return self.spatial, self.temporal, self.channel


class StructuredGridResidualSet(nn.Module):
    """One collection of structured Grid residuals for a later GOP."""

    def __init__(self, grids, rank, alpha):
        super().__init__()
        self.levels = nn.ModuleList([
            StructuredGridResidualLevel(grid, rank, alpha)
            for grid in grids
        ])


class GOPGridResiduals(nn.Module):
    """Store independent additive Grid residuals for selected later GOPs."""

    def __init__(self, grids, num_gops, adapted_gops):
        super().__init__()
        grids = tuple(grids)
        if not grids:
            raise ValueError("grids must not be empty")
        if not isinstance(num_gops, int) or isinstance(num_gops, bool):
            raise TypeError("num_gops must be an integer")
        if num_gops < 2:
            raise ValueError("num_gops must be at least two")
        if not isinstance(adapted_gops, Iterable):
            raise TypeError("adapted_gops must be iterable")

        indices = tuple(sorted(set(adapted_gops)))
        if not indices:
            raise ValueError("adapted_gops must not be empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            or not 1 <= index < num_gops
            for index in indices
        ):
            raise ValueError("adapted_gops must contain valid later GOP indices")

        self.num_gops = num_gops
        self.adapted_gops = indices
        self.residuals = nn.ModuleDict({
            str(index): GridResidualSet(grids) for index in indices
        })

    def forward(self, grids, gop_index):
        grids = tuple(grids)
        gop_index = _gop_index(gop_index)
        if not 0 <= gop_index < self.num_gops:
            raise IndexError("gop_index is outside the available GOPs")
        if gop_index == 0:
            return grids

        key = str(gop_index)
        if key not in self.residuals:
            raise ValueError(
                "GOP {} has no Grid residual".format(gop_index)
            )
        residuals = self.residuals[key].levels
        if len(grids) != len(residuals):
            raise ValueError("Grid level count does not match the residual")
        if any(grid.shape != residual.shape
               for grid, residual in zip(grids, residuals)):
            raise ValueError("Grid shapes do not match the residual")
        return tuple(
            grid + residual for grid, residual in zip(grids, residuals)
        )

    def gop_parameters(self, gop_index):
        gop_index = _gop_index(gop_index)
        key = str(gop_index)
        if key not in self.residuals:
            if gop_index == 0:
                return ()
            raise ValueError(
                "GOP {} has no Grid residual".format(gop_index)
            )
        return tuple(self.residuals[key].levels)

    def parameters_for_all_gops(self):
        return tuple(
            parameter
            for gop_index in self.adapted_gops
            for parameter in self.gop_parameters(gop_index)
        )


class GOPLowRankGridResiduals(nn.Module):
    """Store LoRA-style Grid residuals for selected later GOPs."""

    def __init__(self, grids, num_gops, adapted_gops, rank, alpha,
                 initialization='random'):
        super().__init__()
        grids = tuple(grids)
        if not grids:
            raise ValueError("grids must not be empty")
        if not isinstance(num_gops, int) or isinstance(num_gops, bool):
            raise TypeError("num_gops must be an integer")
        if num_gops < 2:
            raise ValueError("num_gops must be at least two")

        indices = tuple(sorted(set(adapted_gops)))
        if not indices:
            raise ValueError("adapted_gops must not be empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            or not 1 <= index < num_gops
            for index in indices
        ):
            raise ValueError("adapted_gops must contain valid later GOP indices")

        self.num_gops = num_gops
        self.adapted_gops = indices
        self.initialization = initialization
        self.residuals = nn.ModuleDict({
            str(index): LowRankGridResidualSet(
                grids, rank, alpha, initialization,
            )
            for index in indices
        })

    def forward(self, grids, gop_index):
        grids = tuple(grids)
        gop_index = _gop_index(gop_index)
        if not 0 <= gop_index < self.num_gops:
            raise IndexError("gop_index is outside the available GOPs")
        if gop_index == 0:
            return grids

        key = str(gop_index)
        if key not in self.residuals:
            raise ValueError(
                "GOP {} has no Grid LoRA".format(gop_index)
            )
        levels = self.residuals[key].levels
        if len(grids) != len(levels):
            raise ValueError("Grid level count does not match Grid LoRA")
        return tuple(
            level(grid) for level, grid in zip(levels, grids)
        )

    def gop_parameters(self, gop_index):
        gop_index = _gop_index(gop_index)
        key = str(gop_index)
        if key not in self.residuals:
            if gop_index == 0:
                return ()
            raise ValueError("GOP {} has no Grid LoRA".format(gop_index))
        return tuple(
            parameter
            for level in self.residuals[key].levels
            for parameter in level.lora_parameters()
        )

    def gop_levels(self, gop_index):
        gop_index = _gop_index(gop_index)
        key = str(gop_index)
        if key not in self.residuals:
            raise ValueError("GOP {} has no Grid LoRA".format(gop_index))
        return tuple(self.residuals[key].levels)


class GOPStructuredGridResiduals(nn.Module):
    """Store structured 4D Grid LoRA residuals for later GOPs."""

    def __init__(self, grids, num_gops, adapted_gops, rank, alpha):
        super().__init__()
        grids = tuple(grids)
        if not grids:
            raise ValueError("grids must not be empty")
        if not isinstance(num_gops, int) or isinstance(num_gops, bool):
            raise TypeError("num_gops must be an integer")
        if num_gops < 2:
            raise ValueError("num_gops must be at least two")

        indices = tuple(sorted(set(adapted_gops)))
        if not indices:
            raise ValueError("adapted_gops must not be empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            or not 1 <= index < num_gops
            for index in indices
        ):
            raise ValueError("adapted_gops must contain valid later GOP indices")

        self.num_gops = num_gops
        self.adapted_gops = indices
        self.residuals = nn.ModuleDict({
            str(index): StructuredGridResidualSet(grids, rank, alpha)
            for index in indices
        })

    def forward(self, grids, gop_index):
        grids = tuple(grids)
        gop_index = _gop_index(gop_index)
        if not 0 <= gop_index < self.num_gops:
            raise IndexError("gop_index is outside the available GOPs")
        if gop_index == 0:
            return grids

        key = str(gop_index)
        if key not in self.residuals:
            raise ValueError(
                "GOP {} has no structured Grid LoRA".format(gop_index)
            )
        levels = self.residuals[key].levels
        if len(grids) != len(levels):
            raise ValueError("Grid level count does not match Grid LoRA")
        return tuple(level(grid) for level, grid in zip(levels, grids))

    def gop_parameters(self, gop_index):
        gop_index = _gop_index(gop_index)
        key = str(gop_index)
        if key not in self.residuals:
            if gop_index == 0:
                return ()
            raise ValueError(
                "GOP {} has no structured Grid LoRA".format(gop_index)
            )
        return tuple(
            parameter
            for level in self.residuals[key].levels
            for parameter in level.lora_parameters()
        )

    def gop_levels(self, gop_index):
        gop_index = _gop_index(gop_index)
        key = str(gop_index)
        if key not in self.residuals:
            raise ValueError(
                "GOP {} has no structured Grid LoRA".format(gop_index)
            )
        return tuple(self.residuals[key].levels)
