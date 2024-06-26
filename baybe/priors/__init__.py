"""Prior distributions.

The prior classes mimic classes from GPyTorch. For details on specification and
arguments see https://docs.gpytorch.ai/en/stable/priors.html.
"""

from baybe.priors.basic import (
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
    SmoothedBoxPrior,
)

__all__ = [
    "GammaPrior",
    "HalfCauchyPrior",
    "HalfNormalPrior",
    "LogNormalPrior",
    "NormalPrior",
    "SmoothedBoxPrior",
]
