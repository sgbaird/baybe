"""Functionality for managing search spaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import cast

import numpy as np
import pandas as pd
from attr import define, field

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
    validate_constraints,
)
from baybe.constraints.base import Constraint, ContinuousNonlinearConstraint
from baybe.parameters import SubstanceEncoding, TaskParameter
from baybe.parameters.base import Parameter
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.discrete import (
    MemorySize,
    SubspaceDiscrete,
    validate_simplex_subspace_from_config,
)
from baybe.searchspace.validation import validate_parameters
from baybe.serialization import SerialMixin, converter, select_constructor_hook
from baybe.telemetry import TELEM_LABELS, telemetry_record_value


class SearchSpaceType(Enum):
    """Enum class for different types of search spaces and respective compatibility."""

    DISCRETE = "DISCRETE"
    """Flag for discrete search spaces resp. compatibility with discrete search
    spaces."""

    CONTINUOUS = "CONTINUOUS"
    """Flag for continuous search spaces resp. compatibility with continuous
    search spaces."""

    EITHER = "EITHER"
    """Flag compatibility with either discrete or continuous, but not hybrid
    search spaces."""

    HYBRID = "HYBRID"
    """Flag for hybrid search spaces resp. compatibility with hybrid search spaces."""


@define
class SearchSpace(SerialMixin):
    """Class for managing the overall search space.

    The search space might be purely discrete, purely continuous, or hybrid.
    Note that created objects related to the computational representations of parameters
    (e.g., parameter bounds, computational dataframes, etc.) may use a different
    parameter order than what is specified through the constructor: While the
    passed parameter list can contain parameters in arbitrary order, the
    aforementioned objects (by convention) list discrete parameters first, followed
    by continuous ones.
    """

    discrete: SubspaceDiscrete = field(factory=SubspaceDiscrete.empty)
    """The (potentially empty) discrete subspace of the overall search space."""

    continuous: SubspaceContinuous = field(factory=SubspaceContinuous.empty)
    """The (potentially empty) continuous subspace of the overall search space."""

    def __str__(self) -> str:
        start_bold = "\033[1m"
        end_bold = "\033[0m"
        head_str = f"""{start_bold}Search Space{end_bold}
        \n{start_bold}Search Space Type: {end_bold}{self.type.name}"""

        # Check the sub space size to avoid adding unwanted break lines
        # if the sub space is empty
        discrete_str = f"\n\n{self.discrete}" if not self.discrete.is_empty else ""
        continuous_str = (
            f"\n\n{self.continuous}" if not self.continuous.is_empty else ""
        )
        searchspace_str = f"{head_str}{discrete_str}{continuous_str}"
        return searchspace_str.replace("\n", "\n ").replace("\r", "\r ")

    def __attrs_post_init__(self):
        """Perform validation and record telemetry values."""
        validate_parameters(self.parameters)
        validate_constraints(self.constraints, self.parameters)

        # Telemetry
        telemetry_record_value(TELEM_LABELS["COUNT_SEARCHSPACE_CREATION"], 1)
        telemetry_record_value(TELEM_LABELS["NUM_PARAMETERS"], len(self.parameters))
        telemetry_record_value(
            TELEM_LABELS["NUM_CONSTRAINTS"],
            len(self.constraints) if self.constraints else 0,
        )

    @classmethod
    def from_product(
        cls,
        parameters: Sequence[Parameter],
        constraints: Sequence[Constraint] | None = None,
        empty_encoding: bool = False,
    ) -> SearchSpace:
        """Create a search space from a cartesian product.

        In the search space, optional subsequent constraints are applied.
        That is, the discrete subspace becomes the (filtered) cartesian product
        containing all discrete parameter combinations while, analogously, the
        continuous subspace represents the (filtered) cartesian product of all
        continuous parameters.

        Args:
            parameters: The parameters spanning the search space.
            constraints: An optional set of constraints restricting the valid parameter
                space.
            empty_encoding: If ``True``, uses an "empty" encoding for all parameters.
                This is useful, for instance, in combination with random search
                strategies that do not read the actual parameter values, since it avoids
                the (potentially costly) transformation of the parameter values to their
                computational representation.

        Returns:
            The constructed search space.
        """
        # IMPROVE: The arguments get pre-validated here to avoid the potentially costly
        #   creation of the subspaces. Perhaps there is an elegant way to bypass the
        #   default validation in the initializer (which is required for other
        #   ways of object creation) in this particular case.
        validate_parameters(parameters)
        if constraints:
            validate_constraints(constraints, parameters)
        else:
            constraints = []

        discrete: SubspaceDiscrete = SubspaceDiscrete.from_product(
            parameters=[p for p in parameters if p.is_discrete],  # type:ignore[misc]
            constraints=[c for c in constraints if c.is_discrete],  # type:ignore[misc]
            empty_encoding=empty_encoding,
        )
        continuous: SubspaceContinuous = SubspaceContinuous(
            parameters=[p for p in parameters if p.is_continuous],  # type:ignore[misc]
            constraints_lin_eq=[  # type:ignore[misc]
                c
                for c in constraints
                if isinstance(c, ContinuousLinearEqualityConstraint)
            ],
            constraints_lin_ineq=[  # type:ignore[misc]
                c
                for c in constraints
                if isinstance(c, ContinuousLinearInequalityConstraint)
            ],
            constraints_nonlin=[
                c for c in constraints if isinstance(c, ContinuousNonlinearConstraint)
            ],
        )

        return SearchSpace(discrete=discrete, continuous=continuous)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        parameters: Sequence[Parameter],
    ) -> SearchSpace:
        """Create a search space from a specified set of parameter configurations.

        The way in which the contents of the columns are interpreted depends on the
        types of the corresponding parameter objects provided. For details, see
        :meth:`baybe.searchspace.discrete.SubspaceDiscrete.from_dataframe` and
        :meth:`baybe.searchspace.continuous.SubspaceContinuous.from_dataframe`.

        Args:
            df: A dataframe whose parameter configurations are used as
                search space specification.
            parameters: The corresponding parameter objects, one for each column
                in the provided dataframe.

        Returns:
            The created search space.

        Raises:
            ValueError: If the dataframe columns do not match with the parameters.
        """
        if {p.name for p in parameters} != set(df.columns.values):
            raise ValueError(
                "The provided dataframe columns must match exactly with the specified "
                "parameter names."
            )

        disc_params = [p for p in parameters if p.is_discrete]
        cont_params = [p for p in parameters if p.is_continuous]

        return SearchSpace(
            discrete=SubspaceDiscrete.from_dataframe(
                df[[p.name for p in disc_params]],
                disc_params,  # type:ignore[arg-type]
            ),
            continuous=SubspaceContinuous.from_dataframe(
                df[[p.name for p in cont_params]],
                cont_params,  # type:ignore[arg-type]
            ),
        )

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Return the list of parameters of the search space."""
        return (*self.discrete.parameters, *self.continuous.parameters)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return the constraints of the search space."""
        return (
            *self.discrete.constraints,
            *self.continuous.constraints_lin_eq,
            *self.continuous.constraints_lin_ineq,
            *self.continuous.constraints_nonlin,
        )

    @property
    def type(self) -> SearchSpaceType:
        """Return the type of the search space."""
        if self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.CONTINUOUS
        if not self.discrete.is_empty and self.continuous.is_empty:
            return SearchSpaceType.DISCRETE
        if not self.discrete.is_empty and not self.continuous.is_empty:
            return SearchSpaceType.HYBRID
        raise RuntimeError("This line should be impossible to reach.")

    @property
    def contains_mordred(self) -> bool:
        """Indicates if any of the discrete parameters uses ``MORDRED`` encoding."""
        return any(
            p.encoding is SubstanceEncoding.MORDRED for p in self.discrete.parameters
        )

    @property
    def contains_rdkit(self) -> bool:
        """Indicates if any of the discrete parameters uses ``RDKIT`` encoding."""
        return any(
            p.encoding is SubstanceEncoding.RDKIT for p in self.discrete.parameters
        )

    @property
    def param_bounds_comp(self) -> np.ndarray:
        """Return bounds as tensor."""
        return np.hstack(
            [self.discrete.param_bounds_comp, self.continuous.param_bounds_comp]
        )

    @property
    def task_idx(self) -> int | None:
        """The column index of the task parameter in computational representation."""
        try:
            # TODO [16932]: Redesign metadata handling
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
        except StopIteration:
            return None
        # TODO[11611]: The current approach has three limitations:
        #   1.  It matches by column name and thus assumes that the parameter name
        #       is used as the column name.
        #   2.  It relies on the current implementation detail that discrete parameters
        #       appear first in the computational dataframe.
        #   3.  It assumes there exists exactly one task parameter
        #   --> Fix this when refactoring the data
        return cast(int, self.discrete.comp_rep.columns.get_loc(task_param.name))

    @property
    def n_tasks(self) -> int:
        """The number of tasks encoded in the search space."""
        # TODO [16932]: This approach only works for a single task parameter. For
        #  multiple task parameters, we need to align what the output should even
        #  represent (e.g. number of combinatorial task combinations, number of
        #  tasks per task parameter, etc).
        try:
            task_param = next(
                p for p in self.parameters if isinstance(p, TaskParameter)
            )
            return len(task_param.values)

        # When there are no task parameters, we effectively have a single task
        except StopIteration:
            return 1

    @staticmethod
    def estimate_product_space_size(parameters: Iterable[Parameter]) -> MemorySize:
        """Estimate an upper bound for the memory size of a product space.

        Continuous parameters are ignored because creating a continuous subspace has
        no considerable memory footprint.

        Args:
            parameters: The parameters spanning the product space.

        Returns:
            The estimated memory size.
        """
        discrete_parameters = [p for p in parameters if p.is_discrete]
        return SubspaceDiscrete.estimate_product_space_size(discrete_parameters)  # type: ignore[arg-type]

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data from experimental to computational representation.

        This function can e.g. be used to transform data obtained from measurements.
        Continuous parameters are not transformed but included.

        Args:
            data: The data to be transformed. Must contain all specified parameters, can
                contain more columns.

        Returns:
            A dataframe with the parameters in computational representation.
        """
        # Transform subspaces separately
        df_discrete = self.discrete.transform(data)
        df_continuous = self.continuous.transform(data)

        # Combine Subspaces
        comp_rep = pd.concat([df_discrete, df_continuous], axis=1)

        return comp_rep


def validate_searchspace_from_config(specs: dict, _) -> None:
    """Validate the search space specifications while skipping costly creation steps."""
    # Validate product inputs without constructing it
    if specs.get("constructor", None) == "from_product":
        parameters = converter.structure(specs["parameters"], list[Parameter])
        validate_parameters(parameters)

        constraints = specs.get("constraints", None)
        if constraints:
            constraints = converter.structure(specs["constraints"], list[Constraint])
            validate_constraints(constraints, parameters)

    else:
        discrete_subspace_specs = specs.get("discrete", {})
        if discrete_subspace_specs.get("constructor", None) == "from_simplex":
            # Validate discrete simplex subspace
            _validation_converter = converter.copy()
            _validation_converter.register_structure_hook(
                SubspaceDiscrete, validate_simplex_subspace_from_config
            )
            _validation_converter.structure(discrete_subspace_specs, SubspaceDiscrete)
        else:
            # For all other types, validate by construction
            converter.structure(specs, SearchSpace)


# Register deserialization hook
converter.register_structure_hook(SearchSpace, select_constructor_hook)
