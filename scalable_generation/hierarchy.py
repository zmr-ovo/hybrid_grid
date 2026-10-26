"""Grid-level grouping for deterministic progressive reconstruction.

A hierarchy divides ordered Grid levels into a base layer and one or more
enhancement layers.  Quality level ``k`` uses every group from zero through
``k``, so each quality level is a strict prefix of the next one.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridHierarchy:
    """An ordered, exhaustive partition of the model's Grid levels."""

    groups: tuple

    def __post_init__(self):
        normalized = tuple(tuple(group) for group in self.groups)
        if not normalized or any(not group for group in normalized):
            raise ValueError("groups must contain at least one non-empty group")
        if any(
            isinstance(level, bool) or not isinstance(level, int)
            for group in normalized
            for level in group
        ):
            raise TypeError("Grid level indices must be integers")

        flattened = tuple(level for group in normalized for level in group)
        if flattened != tuple(range(len(flattened))):
            raise ValueError(
                "groups must contain every Grid level once in ascending order"
            )
        object.__setattr__(self, 'groups', normalized)

    @property
    def num_levels(self):
        return sum(len(group) for group in self.groups)

    @property
    def num_quality_levels(self):
        return len(self.groups)

    def validate_quality_level(self, quality_level):
        if isinstance(quality_level, bool) or not isinstance(quality_level, int):
            raise TypeError("quality_level must be an integer")
        if not 0 <= quality_level < self.num_quality_levels:
            raise ValueError(
                "quality_level must be between 0 and {}".format(
                    self.num_quality_levels - 1,
                )
            )
        return quality_level

    def active_levels(self, quality_level):
        """Return the cumulative Grid prefix used at one quality level."""
        quality_level = self.validate_quality_level(quality_level)
        return tuple(
            level
            for group in self.groups[:quality_level + 1]
            for level in group
        )

    def mask(self, quality_level):
        """Return a tuple indicating which Grid levels are available."""
        active_count = len(self.active_levels(quality_level))
        return tuple(index < active_count for index in range(self.num_levels))


def default_hierarchy(num_levels):
    """Return the thesis default 7/2/1 hierarchy for a ten-level model."""
    if isinstance(num_levels, bool) or not isinstance(num_levels, int):
        raise TypeError("num_levels must be an integer")
    if num_levels != 10:
        raise ValueError(
            "the default 7/2/1 hierarchy requires exactly 10 Grid levels"
        )
    return GridHierarchy((tuple(range(7)), (7, 8), (9,)))
