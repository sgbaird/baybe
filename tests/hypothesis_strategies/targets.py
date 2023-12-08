"""Hypothesis strategies for targets."""

import hypothesis.strategies as st
from attrs import NOTHING

from baybe.targets.enum import TargetMode
from baybe.targets.numerical import _VALID_TRANSFORM_MODES, NumericalTarget

from .utils import interval

target_name = st.text(min_size=1)
"""A strategy that generates target names."""


@st.composite
def numerical_target(draw: st.DrawFn):
    """Generate class:`baybe.targets.numerical.NumericalTarget`."""
    name = draw(target_name)
    mode = draw(st.sampled_from(TargetMode))
    bounds = draw(
        interval(exclude_half_open=True, exclude_open=mode is TargetMode.MATCH)
    )
    transform_mode = draw(st.none() | st.sampled_from(_VALID_TRANSFORM_MODES[mode]))

    # Explicitly trigger the attrs default method
    transform_mode = transform_mode if transform_mode is not None else NOTHING

    return NumericalTarget(
        name=name, mode=mode, bounds=bounds, transform_mode=transform_mode
    )


target = numerical_target()
"""A strategy that generates targets."""
