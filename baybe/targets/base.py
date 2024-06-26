"""Base functionality for all BayBE targets."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field

from baybe.serialization import (
    SerialMixin,
    converter,
    get_base_structure_hook,
    unstructure_base,
)

if TYPE_CHECKING:
    from baybe.objective import SingleTargetObjective


@define(frozen=True)
class Target(ABC, SerialMixin):
    """Abstract base class for all target variables.

    Stores information about the range, transformations, etc.
    """

    name: str = field()
    """The name of the target."""

    def to_objective(self) -> SingleTargetObjective:
        """Create a single-task objective from the target."""
        from baybe.objectives.single import SingleTargetObjective

        return SingleTargetObjective(self)

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data into computational representation.

        The transformation depends on the target mode, e.g. minimization, maximization,
        matching, etc.

        Args:
            data: The data to be transformed.

        Returns:
            A dataframe containing the transformed data.
        """

    @abstractmethod
    def summary(self) -> dict:
        """Return a custom summarization of the target."""

    def __str__(self) -> str:
        return str(self.summary())


def _add_missing_type_hook(hook):
    """Adjust the structuring hook such that it auto-fills missing target types.

    Used for backward compatibility only and will be removed in future versions.
    """

    def added_type_hook(dict_, cls):
        if "type" not in dict_:
            warnings.warn(
                f"The target type is not specified for target '{dict_['name']}' and "
                f"thus automatically set to 'NumericalTarget'. "
                f"However, omitting the target type is deprecated and will no longer "
                f"be supported in future versions. "
                f"Therefore, please add an explicit target type.",
                DeprecationWarning,
            )
            dict_["type"] = "NumericalTarget"
        return hook(dict_, cls)

    return added_type_hook


# Register (un-)structure hooks
converter.register_structure_hook(
    Target, _add_missing_type_hook(get_base_structure_hook(Target))
)
converter.register_unstructure_hook(Target, unstructure_base)
