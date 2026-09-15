import math
from dataclasses import dataclass
from numbers import Real

from .injection import gop_parameters
from .linear import HierarchicalGOPLoRALinear
from .model import HierarchicalGOPHybridGridNet


def _positive_integer(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _positive_number(name, value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("{} must be finite and positive".format(name))
    return float(value)


def parameter_count(parameters):
    """Return the number of scalar values in a parameter collection."""
    return sum(parameter.numel() for parameter in parameters)


@dataclass(frozen=True)
class GOPAdapterParameterGroup:
    """Independent network and structured Grid LoRA for one later GOP."""

    gop_index: int
    network: tuple
    grid: tuple

    @property
    def parameters(self):
        return self.network + self.grid

    @property
    def network_count(self):
        return parameter_count(self.network)

    @property
    def grid_count(self):
        return parameter_count(self.grid)

    @property
    def total_count(self):
        return self.network_count + self.grid_count


@dataclass(frozen=True)
class AllGOPParameterGroups:
    """Complete parameter groups for the hierarchical all-GOP model."""

    shared: tuple
    common_network: tuple
    common_grid: tuple
    adapters: tuple

    def for_gop(self, gop_index):
        for adapter in self.adapters:
            if adapter.gop_index == gop_index:
                return adapter
        raise ValueError(
            "GOP {} has no independent adapter".format(gop_index)
        )

    @property
    def common(self):
        return self.common_network + self.common_grid

    @property
    def local_network(self):
        return tuple(
            parameter
            for adapter in self.adapters
            for parameter in adapter.network
        )

    @property
    def local_grid(self):
        return tuple(
            parameter
            for adapter in self.adapters
            for parameter in adapter.grid
        )

    @property
    def local(self):
        return tuple(
            parameter
            for adapter in self.adapters
            for parameter in adapter.parameters
        )

    @property
    def parameters(self):
        return self.shared + self.common + self.local

    @property
    def shared_count(self):
        return parameter_count(self.shared)

    @property
    def common_network_count(self):
        return parameter_count(self.common_network)

    @property
    def common_grid_count(self):
        return parameter_count(self.common_grid)

    @property
    def common_count(self):
        return self.common_network_count + self.common_grid_count

    @property
    def local_network_count(self):
        return parameter_count(self.local_network)

    @property
    def local_grid_count(self):
        return parameter_count(self.local_grid)

    @property
    def local_count(self):
        return self.local_network_count + self.local_grid_count

    @property
    def adaptation_count(self):
        return self.common_count + self.local_count

    @property
    def total_count(self):
        return self.shared_count + self.adaptation_count


@dataclass(frozen=True)
class TrainableLoRAParameters:
    """Network and Grid parameters enabled for one training stage."""

    network: tuple
    grid: tuple

    @property
    def parameters(self):
        return self.network + self.grid


def assemble_all_gop_model(shared_model, num_gops, rank, alpha,
                           grid_rank, grid_alpha, common_rank=None,
                           common_alpha=None, common_grid_rank=None,
                           common_grid_alpha=None):
    """Assemble the shared, common and independent GOP model hierarchy."""
    _positive_integer('num_gops', num_gops)
    if num_gops < 2:
        raise ValueError("num_gops must be at least two")
    _positive_integer('rank', rank)
    _positive_number('alpha', alpha)
    _positive_integer('grid_rank', grid_rank)
    _positive_number('grid_alpha', grid_alpha)

    common_rank = rank if common_rank is None else common_rank
    common_alpha = alpha if common_alpha is None else common_alpha
    common_grid_rank = (
        grid_rank if common_grid_rank is None else common_grid_rank
    )
    common_grid_alpha = (
        grid_alpha if common_grid_alpha is None else common_grid_alpha
    )
    _positive_integer('common_rank', common_rank)
    _positive_number('common_alpha', common_alpha)
    _positive_integer('common_grid_rank', common_grid_rank)
    _positive_number('common_grid_alpha', common_grid_alpha)

    return HierarchicalGOPHybridGridNet(
        shared_model=shared_model,
        num_gops=num_gops,
        rank=rank,
        alpha=alpha,
        grid_rank=grid_rank,
        grid_alpha=grid_alpha,
        common_rank=common_rank,
        common_alpha=common_alpha,
        common_grid_rank=common_grid_rank,
        common_grid_alpha=common_grid_alpha,
    )


def all_gop_parameter_groups(model):
    """Collect disjoint shared, common and independent GOP parameters."""
    if not isinstance(model, HierarchicalGOPHybridGridNet):
        raise TypeError("model must be HierarchicalGOPHybridGridNet")

    expected_gops = tuple(range(1, model.num_gops))
    if model.grid_residuals.adapted_gops != expected_gops:
        raise ValueError("model does not contain Grid LoRA for every later GOP")

    network_layers = tuple(
        module for module in model.modules()
        if isinstance(module, HierarchicalGOPLoRALinear)
    )
    common_network = tuple(
        parameter
        for layer in network_layers
        for parameter in layer.common_parameters()
    )
    common_grid = tuple(model.common_grid_parameters())
    adapters = tuple(
        GOPAdapterParameterGroup(
            gop_index=gop_index,
            network=tuple(gop_parameters(model, gop_index)),
            grid=tuple(model.grid_parameters(gop_index)),
        )
        for gop_index in expected_gops
    )
    local = tuple(
        parameter
        for adapter in adapters
        for parameter in adapter.parameters
    )
    adaptation = common_network + common_grid + local
    adaptation_ids = {id(parameter) for parameter in adaptation}
    if len(adaptation_ids) != len(adaptation):
        raise RuntimeError("LoRA parameter groups overlap")

    model_parameters = tuple(model.parameters())
    shared = tuple(
        parameter for parameter in model_parameters
        if id(parameter) not in adaptation_ids
    )
    groups = AllGOPParameterGroups(
        shared=shared,
        common_network=common_network,
        common_grid=common_grid,
        adapters=adapters,
    )
    grouped_ids = [id(parameter) for parameter in groups.parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("parameter groups overlap")
    if set(grouped_ids) != {id(parameter) for parameter in model_parameters}:
        raise RuntimeError("parameter groups do not cover the complete model")
    return groups


def _enable_only(model, selected):
    selected_ids = {id(parameter) for parameter in selected}
    if not selected_ids:
        raise ValueError("selected parameter group must not be empty")
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)


def configure_common_training(model):
    """Freeze the shared and local parameters, then enable common LoRA."""
    groups = all_gop_parameter_groups(model)
    selected = TrainableLoRAParameters(
        network=groups.common_network,
        grid=groups.common_grid,
    )
    _enable_only(model, selected.parameters)
    return selected


def configure_local_training(model, gop_index):
    """Enable only one later GOP's independent network and Grid LoRA."""
    groups = all_gop_parameter_groups(model)
    adapter = groups.for_gop(gop_index)
    selected = TrainableLoRAParameters(
        network=adapter.network,
        grid=adapter.grid,
    )
    _enable_only(model, selected.parameters)
    return selected
