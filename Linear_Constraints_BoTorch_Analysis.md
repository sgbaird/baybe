# How Linear Constraints are Implemented from a BoTorch Perspective in BayBE

This document traces how linear constraints flow from BayBE's high-level constraint definitions through to BoTorch's low-level optimization routines, addressing the "zero volume" challenges that arise when linear constraints reduce the effective dimensionality of the search space.

## Overview

BayBE provides high-level linear constraint classes that are automatically converted to the specific format required by BoTorch's acquisition function optimization routines. This conversion happens through the `to_botorch()` method, which transforms BayBE's human-readable constraint definitions into the tensor-based format that BoTorch's optimizers expect.

## BayBE Constraint Definition

### Linear Constraint Classes

BayBE provides two main types of linear constraints:

1. **[`ContinuousLinearEqualityConstraint`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/constraints/continuous.py#L16-L26)** - For equality constraints: `∑ᵢ(xᵢ × cᵢ) = rhs`
2. **[`ContinuousLinearInequalityConstraint`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/constraints/continuous.py#L29-L41)** - For inequality constraints: `∑ᵢ(xᵢ × cᵢ) ≥ rhs`

Both inherit from [`ContinuousLinearConstraint`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/constraints/base.py#L121-L206), which contains the core constraint logic.

### Constraint Structure

Each linear constraint in BayBE consists of:
- **`parameters`**: `list[str]` - Names of the parameters involved in the constraint
- **`coefficients`**: `list[float]` - Coefficients for each parameter (defaults to 1.0 for all parameters)
- **`rhs`**: `float` - Right-hand side value (defaults to 0.0)

### Example BayBE Constraint Definition

```python
from baybe.constraints import ContinuousLinearEqualityConstraint, ContinuousLinearInequalityConstraint

# Equality constraint: x_1 + x_2 = 1.0
eq_constraint = ContinuousLinearEqualityConstraint(
    parameters=["x_1", "x_2"],
    coefficients=[1.0, 1.0], 
    rhs=1.0
)

# Inequality constraint: x_3 - 2*x_4 >= 0.5  
ineq_constraint = ContinuousLinearInequalityConstraint(
    parameters=["x_3", "x_4"],
    coefficients=[1.0, -2.0],
    rhs=0.5
)
```

## Constraint Conversion: `to_botorch()` Method

The crucial interface between BayBE and BoTorch is the [`to_botorch()`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/constraints/base.py#L175-L205) method in the `ContinuousLinearConstraint` class:

```python
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
```

### Return Format

The `to_botorch()` method returns a tuple `(Tensor, Tensor, float)` containing:

1. **Parameter indices** (`torch.Tensor`): Which decision variables are involved in the constraint
2. **Coefficients** (`torch.Tensor`): How each variable is weighted in the linear combination  
3. **Right-hand side** (`float`): The constraint bound value

### Example Conversion

For the constraint `x_1 + 2*x_3 >= 5` where:
- `parameters = ["x_1", "x_3"]`
- `coefficients = [1.0, 2.0]`
- `rhs = 5.0`

If `x_1` corresponds to index 0 and `x_3` to index 2 in the parameter space, `to_botorch()` returns:
- **indices**: `tensor([0, 2])`
- **coefficients**: `tensor([1.0, 2.0])`
- **rhs**: `5.0`

## Integration with BoTorch Optimization

### Where Constraints Are Passed to BoTorch

The converted constraints are passed to BoTorch's optimization functions in the [`BotorchRecommender`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/recommenders/pure/bayesian/botorch.py) class:

#### 1. Pure Continuous Spaces: [`_recommend_continuous()`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/recommenders/pure/bayesian/botorch.py#L127-L177)

```python
from botorch.optim import optimize_acqf

points, _ = optimize_acqf(
    acq_function=self._botorch_acqf,
    bounds=torch.from_numpy(subspace_continuous.param_bounds_comp),
    q=batch_size,
    num_restarts=5,
    raw_samples=10,
    equality_constraints=[
        c.to_botorch(subspace_continuous.parameters)
        for c in subspace_continuous.constraints_lin_eq
    ] or None,
    inequality_constraints=[
        c.to_botorch(subspace_continuous.parameters)
        for c in subspace_continuous.constraints_lin_ineq
    ] or None,
    sequential=self.sequential_continuous,
)
```

#### 2. Hybrid Spaces: [`_recommend_hybrid()`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/recommenders/pure/bayesian/botorch.py#L179-L288)

```python
from botorch.optim import optimize_acqf_mixed

points, _ = optimize_acqf_mixed(
    acq_function=self._botorch_acqf,
    bounds=torch.from_numpy(searchspace.param_bounds_comp),
    q=batch_size,
    num_restarts=5,
    raw_samples=10,
    fixed_features_list=fixed_features_list,
    equality_constraints=[
        c.to_botorch(
            searchspace.continuous.parameters,
            idx_offset=len(candidates_comp.columns),
        )
        for c in searchspace.continuous.constraints_lin_eq
    ] or None,
    inequality_constraints=[
        c.to_botorch(
            searchspace.continuous.parameters,
            idx_offset=num_comp_columns,
        )
        for c in searchspace.continuous.constraints_lin_ineq
    ] or None,
)
```

### BoTorch Parameter Format

BoTorch's [`optimize_acqf`](https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf) and related functions expect constraints in the format:

```python
equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]]
inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]]
```

Where each tuple `(A, b, c)` represents the constraint `A.T @ x = c` (equality) or `A.T @ x >= c` (inequality), with:
- `A`: Tensor of shape `(len(x), num_constraints)` - constraint coefficients
- `b`: Tensor of indices indicating which variables are involved  
- `c`: Float for the right-hand side value

**Note**: BayBE's format doesn't exactly match this description from BoTorch docs. Looking at the actual implementation, BayBE passes `(indices, coefficients, rhs)` which BoTorch processes accordingly.

## Handling the "Zero Volume" Challenge

The problem statement mentions challenges with linear constraints due to "zero volume" aspects - for example, a triangle embedded in 3D space having only 2 degrees of freedom.

### How BoTorch Addresses This

1. **Constraint-Aware Optimization**: BoTorch's optimization algorithms are designed to handle linear constraints by:
   - Using projected gradients that respect the constraint manifold
   - Sampling initial points that satisfy constraints
   - Employing constrained optimization algorithms (e.g., L-BFGS-B with bounds, interior-point methods)

2. **Feasible Region Focus**: The optimization focuses only on the feasible region defined by the intersection of all constraints, regardless of how this reduces the effective dimensionality.

3. **Acquisition Function Evaluation**: The acquisition function is only evaluated and optimized within the feasible region, avoiding issues with undefined regions.

### Example: Constraint Reducing Dimensionality

Consider a 3D optimization problem with the constraint `x + y + z = 1`. This constraint:
- Reduces the effective dimensionality from 3D to 2D (a plane in 3D space)
- Creates a "zero volume" in the sense that the feasible region has measure zero in the full 3D space
- Is handled by BoTorch by constraining optimization to lie exactly on this 2D manifold

The linear constraint `[1, 1, 1] @ [x, y, z] = 1` is passed to BoTorch as:
- **indices**: `tensor([0, 1, 2])` (all three variables)  
- **coefficients**: `tensor([1.0, 1.0, 1.0])` (equal weights)
- **rhs**: `1.0` (constraint value)

## Complete Example Flow

### 1. BayBE Constraint Definition
```python
from baybe.constraints import ContinuousLinearEqualityConstraint

# Create equality constraint: x_1 + x_2 = 1
constraint = ContinuousLinearEqualityConstraint(
    parameters=["x_1", "x_2"],
    coefficients=[1.0, 1.0],
    rhs=1.0
)
```

### 2. Constraint Conversion
```python
# In BotorchRecommender._recommend_continuous()
botorch_constraint = constraint.to_botorch(subspace_continuous.parameters)
# Returns: (tensor([0, 1]), tensor([1.0, 1.0]), 1.0)
```

### 3. BoTorch Optimization Call
```python
from botorch.optim import optimize_acqf

points, _ = optimize_acqf(
    acq_function=acquisition_function,
    bounds=parameter_bounds,
    q=batch_size,
    equality_constraints=[botorch_constraint],  # [(tensor([0, 1]), tensor([1.0, 1.0]), 1.0)]
    # ... other parameters
)
```

### 4. BoTorch Internal Processing
BoTorch processes the constraint `(tensor([0, 1]), tensor([1.0, 1.0]), 1.0)` to enforce that any candidate point `[x_1, x_2]` satisfies `x_1 + x_2 = 1.0` during optimization.

## Key Code Locations

1. **Constraint Definitions**: [`baybe/constraints/continuous.py`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/constraints/continuous.py)
2. **Base Constraint Logic**: [`baybe/constraints/base.py#L175-L205`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/constraints/base.py#L175-L205)
3. **BoTorch Integration**: [`baybe/recommenders/pure/bayesian/botorch.py`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/baybe/recommenders/pure/bayesian/botorch.py)
4. **Usage Example**: [`examples/Constraints_Continuous/linear_constraints.py`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/examples/Constraints_Continuous/linear_constraints.py)
5. **Tests**: [`tests/test_constraints_continuous.py`](https://github.com/sgbaird/baybe/blob/afc7efea454b42a0e97362b363505188d369f3e5/tests/test_constraints_continuous.py)

## Summary

BayBE's linear constraint implementation provides a clean abstraction layer over BoTorch's constraint handling:

1. **High-level Definition**: Users define constraints using parameter names, coefficients, and RHS values
2. **Automatic Conversion**: The `to_botorch()` method converts these to the index-based format BoTorch requires  
3. **Seamless Integration**: The `BotorchRecommender` automatically passes converted constraints to BoTorch's optimization functions
4. **Robust Constraint Handling**: BoTorch's internal algorithms handle the "zero volume" challenges by focusing optimization on the feasible constraint manifold

This design allows users to work with intuitive, mathematical constraint definitions while leveraging BoTorch's sophisticated constrained optimization capabilities under the hood.