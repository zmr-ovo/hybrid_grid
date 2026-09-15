import math
from dataclasses import dataclass
from numbers import Real

from .injection import gop_parameters
from .model import GOPStructuredGridHybridGridNet


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
    """Network and structured Grid LoRA parameters for one later GOP."""

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
    """Non-overlapping parameter groups for an assembled all-GOP model."""

    shared: tuple
    adapters: tuple

    def for_gop(self, gop_index):
        for adapter in self.adapters:
            if adapter.gop_index == gop_index:
                return adapter
        raise ValueError(
            "GOP {} has no independent adapter".format(gop_index)
        )

    @property
    def network(self):
        return tuple(
            parameter
            for adapter in self.adapters
            for parameter in adapter.network
        )

    @property
    def grid(self):
        return tuple(
            parameter
            for adapter in self.adapters
            for parameter in adapter.grid
        )

    @property
    def adaptation(self):
        return tuple(
            parameter
            for adapter in self.adapters
            for parameter in adapter.parameters
        )

    @property
    def parameters(self):
        return self.shared + self.adaptation

    @property
    def shared_count(self):
        return parameter_count(self.shared)

    @property
    def network_count(self):
        return parameter_count(self.network)

    @property
    def grid_count(self):
        return parameter_count(self.grid)

    @property
    def adaptation_count(self):
        return self.network_count + self.grid_count

    @property
    def total_count(self):
        return self.shared_count + self.adaptation_count


def assemble_all_gop_model(shared_model, num_gops, rank, alpha,
                           grid_rank, grid_alpha):
    """Attach independent network and structured Grid LoRA to every later GOP.

    GOP 0 uses the shared model directly. GOP 1 through ``num_gops - 1`` each
    receive their own all-Linear network LoRA and structured Grid LoRA.
    """
    _positive_integer('num_gops', num_gops)
    if num_gops < 2:
        raise ValueError("num_gops must be at least two")
    _positive_integer('rank', rank)
    _positive_number('alpha', alpha)
    _positive_integer('grid_rank', grid_rank)
    _positive_number('grid_alpha', grid_alpha)

    return GOPStructuredGridHybridGridNet(
        shared_model=shared_model,
        num_gops=num_gops,
        adapted_gops=range(1, num_gops),
        rank=rank,
        alpha=alpha,
        grid_rank=grid_rank,
        grid_alpha=grid_alpha,
        lora_target='all_linear',
    )


def all_gop_parameter_groups(model):
    """Collect complete, disjoint shared and per-GOP parameter groups."""
    if not isinstance(model, GOPStructuredGridHybridGridNet):
        raise TypeError(
            "model must be GOPStructuredGridHybridGridNet"
        )

    expected_gops = tuple(range(1, model.num_gops))
    if model.grid_residuals.adapted_gops != expected_gops:
        raise ValueError("model does not contain Grid LoRA for every later GOP")

    adapters = tuple(
        GOPAdapterParameterGroup(
            gop_index=gop_index,
            network=tuple(gop_parameters(model, gop_index)),
            grid=tuple(model.grid_parameters(gop_index)),
        )
        for gop_index in expected_gops
    )
    adaptation_ids = {
        id(parameter)
        for adapter in adapters
        for parameter in adapter.parameters
    }
    adaptation_size = sum(
        len(adapter.parameters) for adapter in adapters
    )
    if len(adaptation_ids) != adaptation_size:
        raise RuntimeError("adapter parameter groups overlap")

    model_parameters = tuple(model.parameters())
    shared = tuple(
        parameter for parameter in model_parameters
        if id(parameter) not in adaptation_ids
    )
    groups = AllGOPParameterGroups(shared=shared, adapters=adapters)
    if {id(parameter) for parameter in groups.parameters} != {
        id(parameter) for parameter in model_parameters
    }:
        raise RuntimeError("parameter groups do not cover the complete model")
    return groups
