"""Tests for the continuous cardinality constraint."""

from itertools import combinations_with_replacement

import numpy as np
import pytest

from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace.core import SearchSpace


def _get_searchspace(
    n_parameters: int, min_cardinality: int, max_cardinality: int
) -> SearchSpace:
    """Create a unit-cube searchspace with cardinality constraint on all parameters."""
    parameters = [
        NumericalContinuousParameter(name=f"x_{i}", bounds=(0, 1))
        for i in range(n_parameters)
    ]
    constraints = [
        ContinuousCardinalityConstraint(
            parameters=[f"x_{i}" for i in range(n_parameters)],
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
        )
    ]
    searchspace = SearchSpace.from_product(parameters, constraints)
    return searchspace


def test_sampling():
    """
    Polytope sampling with cardinality constraints respects all involved constraints
    and produces distinct samples.
    """  # noqa
    N_PARAMETERS = 6
    MAX_NONZERO = 4
    MIN_NONZERO = 2
    N_POINTS = 20
    TOLERANCE = 1e-3

    parameters = [
        NumericalContinuousParameter(name=f"x_{i+1}", bounds=(0, 1))
        for i in range(N_PARAMETERS)
    ]
    params_equality = ["x_1", "x_2", "x_3", "x_4"]
    coeffs_equality = [0.9, 0.6, 2.8, 6.1]
    rhs_equality = 4.2
    params_inequality = ["x_1", "x_2", "x_5", "x_6"]
    coeffs_inequality = [4.7, 1.4, 4.6, 8.6]
    rhs_inequality = 1.3
    params_cardinality = ["x_1", "x_2", "x_3", "x_5"]
    constraints = [
        ContinuousLinearEqualityConstraint(
            parameters=params_equality, coefficients=coeffs_equality, rhs=rhs_equality
        ),
        ContinuousLinearInequalityConstraint(
            parameters=params_inequality,
            coefficients=coeffs_inequality,
            rhs=rhs_equality,
        ),
        ContinuousCardinalityConstraint(
            parameters=params_cardinality,
            max_cardinality=MAX_NONZERO,
            min_cardinality=MIN_NONZERO,
        ),
    ]
    searchspace = SearchSpace.from_product(parameters, constraints)

    samples = searchspace.continuous.sample_uniform(N_POINTS)

    # Assert that cardinality constraint is fulfilled
    n_nonzero = np.sum(~np.isclose(samples[params_cardinality], 0.0), axis=1)
    assert np.all(n_nonzero >= MIN_NONZERO) and np.all(n_nonzero <= MAX_NONZERO)

    # Assert that linear equality constraint is fulfilled
    assert np.allclose(
        np.sum(samples[params_equality] * coeffs_equality, axis=1),
        rhs_equality,
        atol=TOLERANCE,
    )

    # Assert that linear non-equality constraint is fulfilled
    assert (
        np.sum(samples[params_inequality] * coeffs_inequality, axis=1)
        .ge(rhs_inequality - TOLERANCE)
        .all()
    )

    # Assert that we obtain as many (unique!) samples as requested
    assert len(samples.drop_duplicates()) == N_POINTS


# Combinations of cardinalities to be tested
_cardinalities = sorted(combinations_with_replacement(range(0, 10), 2))


@pytest.mark.parametrize(
    "cardinalities", _cardinalities, ids=[str(x) for x in _cardinalities]
)
def test_random_recommender_with_cardinality_constraint(cardinalities):
    """
    Recommendations generated by a `RandomRecommender` under a cardinality constraint
    have the expected number of nonzero elements.
    """  # noqa
    N_PARAMETERS = 10
    BATCH_SIZE = 10
    min_cardinality, max_cardinality = cardinalities

    searchspace = _get_searchspace(N_PARAMETERS, min_cardinality, max_cardinality)
    recommender = RandomRecommender()
    recommendations = recommender.recommend(
        searchspace=searchspace,
        batch_size=BATCH_SIZE,
    )

    # Assert that cardinality constraint is fulfilled
    n_nonzero = np.sum(~np.isclose(recommendations, 0.0), axis=1)
    assert np.all(n_nonzero >= min_cardinality) and np.all(n_nonzero <= max_cardinality)

    # Assert that we obtain as many samples as requested
    assert len(recommendations) == BATCH_SIZE

    # If there are duplicates, they must all come from the case cardinality = 0
    assert np.all(recommendations[recommendations.duplicated()] == 0.0)
