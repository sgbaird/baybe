"""Hypothesis strategies for generating utility objects."""

import hypothesis.strategies as st
from hypothesis import assume

from baybe.utils.interval import Interval


@st.composite
def interval(
    draw: st.DrawFn,
    *,
    exclude_open: bool = False,
    exclude_half_open: bool = False,
    exclude_closed: bool = False,
):
    """Generate class:`baybe.utils.interval.Interval`."""
    assert not all(
        (exclude_open, exclude_half_open, exclude_closed)
    ), "At least one Interval type must be allowed."

    # Create interval from ordered pair of floats
    bounds = (
        st.tuples(st.floats(), st.floats()).map(sorted).filter(lambda x: x[0] < x[1])
    )
    interval = Interval.create(draw(bounds))

    # Filter excluded intervals
    if exclude_open:
        assume(not interval.is_open)
    if exclude_half_open:
        assume(not interval.is_half_open)
    if exclude_closed:
        assume(not interval.is_closed)

    return interval
