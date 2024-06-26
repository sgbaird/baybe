"""Base classes for all constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from attr import define, field
from attr.validators import min_len

from baybe.parameters import NumericalContinuousParameter
from baybe.serialization import (
    SerialMixin,
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.utils.numerical import DTypeFloatNumpy

if TYPE_CHECKING:
    from torch import Tensor


@define
class Constraint(ABC, SerialMixin):
    """Abstract base class for all constraints.

    Constraints use conditions and chain them together to filter unwanted entries from
    the search space.
    """

    # class variables
    # TODO: it might turn out these are not needed at a later development stage
    eval_during_creation: ClassVar[bool]
    """Class variable encoding whether the condition is evaluated during creation."""

    eval_during_modeling: ClassVar[bool]
    """Class variable encoding whether the condition is evaluated during modeling."""

    # Object variables
    parameters: list[str] = field(validator=min_len(1))
    """The list of parameters used for the constraint."""

    @parameters.validator
    def _validate_params(  # noqa: DOC101, DOC103
        self, _: Any, params: list[str]
    ) -> None:
        """Validate the parameter list.

        Raises:
            ValueError: If ``params`` contains duplicate values.
        """
        if len(params) != len(set(params)):
            raise ValueError(
                f"The given 'parameters' list must have unique values "
                f"but was: {params}."
            )

    def summary(self) -> dict:
        """Return a custom summarization of the constraint."""
        constr_dict = dict(
            Type=self.__class__.__name__, Affected_Parameters=self.parameters
        )
        return constr_dict

    @property
    def is_continuous(self) -> bool:
        """Boolean indicating if this is a constraint over continuous parameters."""
        return isinstance(self, ContinuousConstraint)

    @property
    def is_discrete(self) -> bool:
        """Boolean indicating if this is a constraint over discrete parameters."""
        return isinstance(self, DiscreteConstraint)


@define
class DiscreteConstraint(Constraint, ABC):
    """Abstract base class for discrete constraints.

    Discrete constraints use conditions and chain them together to filter unwanted
    entries from the search space.
    """

    # class variables
    eval_during_creation: ClassVar[bool] = True
    # See base class.

    eval_during_modeling: ClassVar[bool] = False
    # See base class.

    @abstractmethod
    def get_invalid(self, data: pd.DataFrame) -> pd.Index:
        """Get the indices of dataframe entries that are invalid under the constraint.

        Args:
            data: A dataframe where each row represents a particular parameter
                combination.

        Returns:
            The dataframe indices of rows where the constraint is violated.
        """


@define
class ContinuousConstraint(Constraint, ABC):
    """Abstract base class for continuous constraints."""

    # class variables
    eval_during_creation: ClassVar[bool] = False
    # See base class.

    eval_during_modeling: ClassVar[bool] = True
    # See base class.


@define
class ContinuousLinearConstraint(ContinuousConstraint, ABC):
    """Abstract base class for continuous linear constraints.

    Continuous linear constraints use parameter lists and coefficients to define
    in-/equality constraints over a continuous parameter space.
    """

    # object variables
    coefficients: list[float] = field()
    """In-/equality coefficient for each entry in ``parameters``."""

    rhs: float = field(default=0.0)
    """Right-hand side value of the in-/equality."""

    @coefficients.validator
    def _validate_coefficients(  # noqa: DOC101, DOC103
        self, _: Any, coefficients: list[float]
    ) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the number of coefficients does not match the number of
                parameters.
        """
        if len(self.parameters) != len(coefficients):
            raise ValueError(
                "The given 'coefficients' list must have one floating point entry for "
                "each entry in 'parameters'."
            )

    @coefficients.default
    def _default_coefficients(self):
        """Return equal weight coefficients as default."""
        return [1.0] * len(self.parameters)

    def _drop_parameters(
        self, parameter_names: Collection[str]
    ) -> ContinuousLinearConstraint:
        """Create a copy of the constraint with certain parameters removed.

        Args:
            parameter_names: The names of the parameter to be removed.

        Returns:
            The reduced constraint.
        """
        parameters = [p for p in self.parameters if p not in parameter_names]
        coefficients = [
            c
            for p, c in zip(self.parameters, self.coefficients)
            if p not in parameter_names
        ]
        return ContinuousLinearConstraint(parameters, coefficients, self.rhs)

    def to_botorch(
        self, parameters: Sequence[NumericalContinuousParameter], idx_offset: int = 0
    ) -> tuple[Tensor, Tensor, float]:
        """Cast the constraint in a format required by botorch.

        Used in calling ``optimize_acqf_*`` functions, for details see
        https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf

        Args:
            parameters: The parameter objects of the continuous space.
            idx_offset: Offset to the provided parameter indices.

        Returns:
            The tuple required by botorch.
        """
        import torch

        from baybe.utils.torch import DTypeFloatTorch

        param_names = [p.name for p in parameters]
        param_indices = [
            param_names.index(p) + idx_offset
            for p in self.parameters
            if p in param_names
        ]

        return (
            torch.tensor(param_indices),
            torch.tensor(self.coefficients, dtype=DTypeFloatTorch),
            np.asarray(self.rhs, dtype=DTypeFloatNumpy).item(),
        )


class ContinuousNonlinearConstraint(ContinuousConstraint, ABC):
    """Abstract base class for nonlinear constraints."""


# Register (un-)structure hooks
converter.register_unstructure_hook(Constraint, unstructure_base)
converter.register_structure_hook(Constraint, get_base_structure_hook(Constraint))
