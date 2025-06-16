# Understanding BayBE's MATCH Target Implementation

This document explains how BayBE implements "match" targets to get an objective as close as possible to a specific value X.

## Overview

BayBE's `TargetMode.MATCH` allows you to optimize towards a specific target value rather than simply minimizing or maximizing an objective. This is particularly useful in scenarios where:

- You want to hit a specific temperature (e.g., 50°C)
- You need a particular pH value (e.g., 7.0)
- You're targeting a specific reaction time (e.g., 2 hours)
- You want to match a reference measurement

## How It Works

### 1. Core Concept

The match target implementation uses a clever approach:

1. **Bounds Definition**: You specify bounds where the **midpoint** equals your desired target value
2. **Transformation Function**: BayBE applies a transformation that maps measured values to [0,1]
3. **Scoring**: The transformation gives score 1.0 at the target, lower scores elsewhere
4. **Integration**: This works seamlessly with BayBE's desirability framework

### 2. Setting Bounds

The key insight is that **the midpoint of your bounds becomes the target value**:

```python
# To target value 50:
tight_bounds = (45, 55)    # midpoint = 50, tolerance = ±5
wide_bounds = (30, 70)     # midpoint = 50, tolerance = ±20
```

The spread of the bounds controls how forgiving the optimization is around the target.

### 3. Transformation Functions

BayBE offers two transformation functions for MATCH mode:

#### Triangular Transformation (Default)
- **Shape**: Triangle with peak at bounds midpoint
- **Behavior**: Linear increase/decrease within bounds, zero outside
- **Use case**: When you want strict cutoffs outside acceptable range

#### Bell Transformation (Alternative)
- **Shape**: Gaussian bell curve centered at bounds midpoint
- **Behavior**: Smooth curve that gradually decreases, non-zero outside bounds
- **Use case**: When you want more forgiving behavior with smooth falloff

## Implementation Details

### Core Files

The implementation is spread across several key files:

1. **`baybe/targets/enum.py`**: Defines `TargetMode.MATCH` and `TargetTransformation` enums
2. **`baybe/targets/numerical.py`**: Contains `NumericalTarget` class with MATCH mode logic
3. **`baybe/targets/transforms.py`**: Implements `triangular_transform` and `bell_transform` functions

### Key Code Components

#### Transformation Function Selection
```python
_VALID_TRANSFORMATIONS: dict[TargetMode, Sequence[TargetTransformation]] = {
    TargetMode.MAX: (TargetTransformation.LINEAR,),
    TargetMode.MIN: (TargetTransformation.LINEAR,),
    TargetMode.MATCH: (TargetTransformation.TRIANGULAR, TargetTransformation.BELL),
}
```

#### Validation Logic
- MATCH mode **requires** bounded intervals (not infinite)
- Only TRIANGULAR and BELL transformations are allowed for MATCH mode
- Bounds cannot be degenerate (lower == upper)

#### Transform Application
```python
def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    # When a transformation is specified, apply it
    if self.transformation is not None:
        func = _get_target_transformation(self.mode, self.transformation)
        transformed = pd.DataFrame(
            func(data, *self.bounds.to_tuple()), index=data.index
        )
    # ... handle other cases
```

## Practical Usage Examples

### Basic Match Target

```python
from baybe.targets import NumericalTarget, TargetMode, TargetTransformation

# Target temperature of 50°C with tight tolerance
target = NumericalTarget(
    name="temperature",
    mode=TargetMode.MATCH,
    bounds=(45, 55),  # midpoint = 50°C (our target)
    transformation=TargetTransformation.TRIANGULAR
)
```

### Multiple Match Targets

```python
from baybe.objectives import DesirabilityObjective

# Optimize multiple properties simultaneously
objective = DesirabilityObjective(
    targets=[
        NumericalTarget("yield", mode="MAX", bounds=(0, 100)),
        NumericalTarget("temperature", mode="MATCH", bounds=(45, 55)),  # target 50°C
        NumericalTarget("pH", mode="MATCH", bounds=(6.5, 7.5)),        # target 7.0
    ],
    weights=[1.0, 2.0, 1.5],  # temperature most important
    scalarizer="GEOM_MEAN"
)
```

### Comparing Transformation Functions

```python
# Strict matching with sharp cutoffs
strict_target = NumericalTarget(
    name="pH",
    mode="MATCH",
    bounds=(6.8, 7.2),  # target pH = 7.0
    transformation="TRIANGULAR"  # zero score outside bounds
)

# Flexible matching with smooth falloff
flexible_target = NumericalTarget(
    name="pH", 
    mode="MATCH",
    bounds=(6.8, 7.2),  # target pH = 7.0
    transformation="BELL"  # gradual falloff outside bounds
)
```

## Mathematical Details

### Triangular Transformation
```python
def triangular_transform(arr, lower, upper):
    """
    Maps values to [0,1] with triangular shape:
    - Peak value 1.0 at midpoint = (lower + upper) / 2
    - Linear decrease to 0 at bounds
    - Zero outside bounds
    """
    mid = lower + (upper - lower) / 2
    # Linear increase from lower to mid
    # Linear decrease from mid to upper
    # Zero outside [lower, upper]
```

### Bell Transformation
```python
def bell_transform(arr, lower, upper):
    """
    Maps values to [0,1] with Gaussian bell shape:
    - Peak value 1.0 at mean = (lower + upper) / 2
    - Standard deviation = (upper - lower) / 2
    - Smooth falloff, non-zero outside bounds
    """
    mean = (lower + upper) / 2
    std = (upper - lower) / 2
    return exp(-((arr - mean)² / (2 * std²)))
```

## Tuning Guidelines

### Choosing Bounds

1. **Start Wide**: Begin with wider bounds, then tighten based on results
2. **Consider Noise**: Wider bounds help with measurement uncertainty
3. **Domain Knowledge**: Set bounds based on what's practically achievable
4. **Iterative Refinement**: Adjust based on experimental results

### Choosing Transformation

- **Use TRIANGULAR when**:
  - You have hard constraints (values outside bounds are unacceptable)
  - You want clear boundaries between acceptable/unacceptable regions
  - Your domain has natural cutoff points

- **Use BELL when**:
  - You prefer smooth, gradual preferences
  - Measurements have noise/uncertainty
  - You want some tolerance outside the nominal bounds

### Multi-Objective Considerations

- **Weights**: Higher weights for more critical match targets
- **Scalarizer**: 
  - `GEOM_MEAN`: More balanced, penalizes poor performance on any target
  - `MEAN`: More forgiving, allows compensation between targets

## Integration with BayBE Workflow

### In a Campaign

```python
from baybe import Campaign
from baybe.searchspace import SearchSpace

# Create campaign with match target
campaign = Campaign(
    searchspace=your_searchspace,
    objective=DesirabilityObjective(
        targets=[your_match_target, other_targets],
        weights=[2.0, 1.0, 1.0]
    )
)

# Use normally
recommendations = campaign.recommend(batch_size=5)
# ... run experiments, add measurements
campaign.add_measurements(results)
```

### With Desirability Framework

The match target seamlessly integrates with BayBE's desirability-based multi-objective optimization:

1. Each target (including match targets) gets transformed to [0,1]
2. Weights determine relative importance
3. Scalarizer combines into single objective
4. Bayesian optimization maximizes the combined desirability

## Common Patterns and Best Practices

### Pattern 1: Process Optimization
```python
# Optimize reaction conditions
targets = [
    NumericalTarget("yield", mode="MAX", bounds=(0, 100)),
    NumericalTarget("temperature", mode="MATCH", bounds=(48, 52)),  # target 50°C
    NumericalTarget("pressure", mode="MATCH", bounds=(0.8, 1.2)),  # target 1 atm
]
```

### Pattern 2: Quality Control
```python
# Match reference measurements
targets = [
    NumericalTarget("cost", mode="MIN", bounds=(0, 1000)),
    NumericalTarget("hardness", mode="MATCH", bounds=(45, 55)),    # match reference
    NumericalTarget("color_L", mode="MATCH", bounds=(78, 82)),     # match reference
]
```

### Pattern 3: Constrained Optimization
```python
# Optimize performance while maintaining specifications
targets = [
    NumericalTarget("performance", mode="MAX", bounds=(0, 100)),
    NumericalTarget("temp", mode="MATCH", bounds=(19, 21)),        # room temp ±1°C
    NumericalTarget("humidity", mode="MATCH", bounds=(48, 52)),    # 50% ±2%
]
```

## Summary

BayBE's MATCH target implementation is elegant and practical:

- **Simple Setup**: Just set bounds with midpoint = desired target
- **Flexible Control**: Adjust bounds spread to control tolerance  
- **Two Shapes**: Choose triangular (strict) or bell (smooth) preference
- **Seamless Integration**: Works naturally with multi-objective optimization
- **Domain Agnostic**: Works for any numerical target

This approach enables BayBE to handle real-world scenarios where you need to hit specific target values rather than just optimizing in one direction, making it particularly valuable for process optimization, quality control, and constrained optimization problems.