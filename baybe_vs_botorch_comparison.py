#!/usr/bin/env python3
"""
BayBE vs BoTorch Constraint Implementation Comparison
====================================================

This script demonstrates how linear constraints are implemented in BayBE
versus the equivalent direct BoTorch implementation using the Hartmann6 function.

The comparison shows:
1. How BayBE provides high-level constraint abstractions
2. How constraints are converted to BoTorch format  
3. Direct BoTorch implementation with equivalent functionality
4. Performance comparison between both approaches

Run with: python baybe_vs_botorch_comparison.py
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
DIMENSION = 6  # Hartmann6 function
BATCH_SIZE = 3
N_ITERATIONS = 5
RANDOM_SEED = 42

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================================
# BAYBE IMPLEMENTATION
# ============================================================================

def demo_baybe_implementation():
    """
    Demonstrates how to use BayBE with linear constraints on Hartmann6.
    
    This uses BayBE's high-level constraint classes and shows how they
    automatically integrate with BoTorch optimization.
    """
    print("=" * 60)
    print("BayBE Implementation with Linear Constraints")
    print("=" * 60)
    
    try:
        from baybe import Campaign
        from baybe.constraints import (
            ContinuousLinearEqualityConstraint,
            ContinuousLinearInequalityConstraint,
        )
        from baybe.objectives import SingleTargetObjective
        from baybe.parameters import NumericalContinuousParameter
        from baybe.searchspace import SearchSpace
        from baybe.targets import NumericalTarget
        from baybe.utils.botorch_wrapper import botorch_function_wrapper
        from botorch.test_functions import Hartmann
        
        print("✓ All BayBE dependencies available")
        
        # Initialize Hartmann6 function
        hartmann = Hartmann(dim=DIMENSION)
        wrapped_function = botorch_function_wrapper(test_function=hartmann)
        bounds = hartmann.bounds
        
        print(f"✓ Hartmann{DIMENSION} function initialized")
        print(f"  Bounds: {bounds}")
        
        # Create parameters
        parameters = [
            NumericalContinuousParameter(
                name=f"x{i+1}",
                bounds=(bounds[0, i].item(), bounds[1, i].item()),
            )
            for i in range(DIMENSION)
        ]
        
        # Define linear EQUALITY constraints
        # Focus specifically on equality constraints as requested:
        # 1. x1 + 2*x6 = 1.0 (constraint on x1 and x6)
        # 2. x2 + x3 = 0.5 (constraint on x2 and x3)
        
        constraints = [
            ContinuousLinearEqualityConstraint(
                parameters=["x1", "x6"], 
                coefficients=[1.0, 2.0], 
                rhs=1.0,
                comment="x1 + 2*x6 = 1.0"
            ),
            ContinuousLinearEqualityConstraint(
                parameters=["x2", "x3"], 
                coefficients=[1.0, 1.0], 
                rhs=0.5,
                comment="x2 + x3 = 0.5"
            ),
        ]
        
        print(f"✓ Defined {len(constraints)} linear EQUALITY constraints:")
        for i, c in enumerate(constraints):
            print(f"  {i+1}. {getattr(c, 'comment', 'No description')}")
        
        # Create search space and objective
        searchspace = SearchSpace.from_product(
            parameters=parameters, 
            constraints=constraints
        )
        objective = SingleTargetObjective(
            target=NumericalTarget(name="hartmann_value", mode="MIN")
        )
        
        # Create campaign
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
        )
        
        print("✓ BayBE campaign created")
        
        # Demonstrate constraint conversion to BoTorch format
        print("\n--- Constraint Conversion to BoTorch Format ---")
        for i, constraint in enumerate(constraints):
            botorch_format = constraint.to_botorch(searchspace.continuous.parameters)
            indices, coeffs, rhs = botorch_format
            print(f"Constraint {i+1}:")
            print(f"  BayBE: {constraint.parameters} with coeffs {constraint.coefficients} {getattr(constraint, '__class__').__name__[18:]} {constraint.rhs}")
            print(f"  BoTorch: indices={indices.tolist()}, coeffs={coeffs.tolist()}, rhs={rhs}")
        
        # Run optimization iterations
        print(f"\n--- Running {N_ITERATIONS} Optimization Iterations ---")
        baybe_results = []
        
        for iteration in range(N_ITERATIONS):
            # Get recommendations
            recommendation = campaign.recommend(batch_size=BATCH_SIZE)
            
            # Evaluate function
            target_values = []
            for index, row in recommendation.iterrows():
                value = wrapped_function(*row.to_list())
                target_values.append(value)
            
            recommendation["hartmann_value"] = target_values
            campaign.add_measurements(recommendation)
            
            best_value = min(target_values)
            baybe_results.append(best_value)
            print(f"  Iteration {iteration+1}: Best value = {best_value:.6f}")
        
        print(f"✓ BayBE final best value: {min(baybe_results):.6f}")
        
        # Verify EQUALITY constraints are satisfied
        measurements = campaign.measurements
        print("\n--- Equality Constraint Verification ---")
        tolerance = 1e-3
        
        # Check constraint 1: x1 + 2*x6 = 1.0
        constraint1_values = measurements["x1"] + 2 * measurements["x6"]
        constraint1_satisfied = np.allclose(constraint1_values, 1.0, atol=tolerance)
        print(f"  x1 + 2*x6 = 1.0: {constraint1_satisfied} (mean = {constraint1_values.mean():.6f}, std = {constraint1_values.std():.6f})")
        
        # Check constraint 2: x2 + x3 = 0.5
        constraint2_values = measurements["x2"] + measurements["x3"]
        constraint2_satisfied = np.allclose(constraint2_values, 0.5, atol=tolerance)
        print(f"  x2 + x3 = 0.5: {constraint2_satisfied} (mean = {constraint2_values.mean():.6f}, std = {constraint2_values.std():.6f})")
        
        return baybe_results, measurements
        
    except ImportError as e:
        print(f"✗ BayBE dependencies not available: {e}")
        print("  This would normally run the full BayBE optimization")
        return None, None

# ============================================================================
# DIRECT BOTORCH IMPLEMENTATION  
# ============================================================================

def demo_botorch_implementation():
    """
    Demonstrates the equivalent direct BoTorch implementation.
    
    This shows how to manually implement the same constraint handling
    that BayBE does automatically.
    """
    print("\n" + "=" * 60)
    print("Direct BoTorch Implementation with Linear Constraints")
    print("=" * 60)
    
    try:
        import torch
        from botorch.test_functions import Hartmann
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_model
        from botorch.acquisition import UpperConfidenceBound
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        
        print("✓ BoTorch dependencies available")
        
        # Initialize Hartmann6 function
        hartmann = Hartmann(dim=DIMENSION)
        bounds = hartmann.bounds
        
        print(f"✓ Hartmann{DIMENSION} function initialized")
        
        # Define the same EQUALITY constraints in BoTorch format
        # Converting from BayBE format to BoTorch format manually
        
        # Constraint 1: x1 + 2*x6 = 1.0
        eq_constraint_1 = (
            torch.tensor([0, 5]),  # indices for x1, x6 (0-indexed)
            torch.tensor([1.0, 2.0]),  # coefficients
            1.0  # rhs
        )
        
        # Constraint 2: x2 + x3 = 0.5  
        eq_constraint_2 = (
            torch.tensor([1, 2]),  # indices for x2, x3 (0-indexed)
            torch.tensor([1.0, 1.0]),  # coefficients
            0.5  # rhs
        )
        
        inequality_constraints = []  # No inequality constraints - focusing on equality only
        equality_constraints = [eq_constraint_1, eq_constraint_2]
        
        print(f"✓ Defined EQUALITY constraints in BoTorch format:")
        print(f"  Inequality constraints: {len(inequality_constraints)} (none - focusing on equality)")
        print(f"  Equality constraints: {len(equality_constraints)}")
        
        for i, (indices, coeffs, rhs) in enumerate(equality_constraints):
            print(f"    Eq {i+1}: indices={indices.tolist()}, coeffs={coeffs.tolist()}, rhs={rhs}")
        
        # Initialize training data with a few feasible points
        # Generate initial points that satisfy all constraints
        initial_points = generate_feasible_initial_points(
            bounds, equality_constraints, inequality_constraints, n_points=2*BATCH_SIZE
        )
        
        train_X = initial_points
        train_Y = hartmann(train_X).unsqueeze(-1)
        
        print(f"✓ Generated {len(train_X)} initial feasible points")
        
        # Run optimization iterations
        print(f"\n--- Running {N_ITERATIONS} Optimization Iterations ---")
        botorch_results = []
        
        for iteration in range(N_ITERATIONS):
            # Fit GP model
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            
            # Create acquisition function
            acq_func = UpperConfidenceBound(model, beta=2.0)
            
            # Optimize acquisition function with constraints
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=5,
                raw_samples=20,
                equality_constraints=equality_constraints,
                inequality_constraints=inequality_constraints,
                sequential=True,
            )
            
            # Evaluate new points
            new_Y = hartmann(candidates).unsqueeze(-1)
            
            # Update training data
            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_Y])
            
            best_value = train_Y.min().item()
            botorch_results.append(best_value)
            print(f"  Iteration {iteration+1}: Best value = {best_value:.6f}")
        
        print(f"✓ BoTorch final best value: {min(botorch_results):.6f}")
        
        # Verify EQUALITY constraints are satisfied for all points
        print("\n--- Equality Constraint Verification ---")
        tolerance = 1e-3
        
        # Check constraint 1: x1 + 2*x6 = 1.0
        constraint1_values = train_X[:, 0] + 2 * train_X[:, 5]
        constraint1_satisfied = torch.allclose(constraint1_values, torch.tensor(1.0), atol=tolerance)
        print(f"  x1 + 2*x6 = 1.0: {constraint1_satisfied} (mean = {constraint1_values.mean():.6f}, std = {constraint1_values.std():.6f})")
        
        # Check constraint 2: x2 + x3 = 0.5
        constraint2_values = train_X[:, 1] + train_X[:, 2]
        constraint2_satisfied = torch.allclose(constraint2_values, torch.tensor(0.5), atol=tolerance)
        print(f"  x2 + x3 = 0.5: {constraint2_satisfied} (mean = {constraint2_values.mean():.6f}, std = {constraint2_values.std():.6f})")
        
        return botorch_results, train_X
        
    except ImportError as e:
        print(f"✗ BoTorch dependencies not available: {e}")
        print("  This would normally run the full BoTorch optimization")
        return None, None

def generate_feasible_initial_points(
    bounds: torch.Tensor, 
    equality_constraints: List[Tuple[torch.Tensor, torch.Tensor, float]],
    inequality_constraints: List[Tuple[torch.Tensor, torch.Tensor, float]],
    n_points: int = 10,
    max_attempts: int = 1000
) -> torch.Tensor:
    """
    Generate initial points that satisfy all linear EQUALITY constraints.
    
    Since we're focusing on equality constraints, we use a projection method
    to ensure generated points lie on the constraint manifold.
    """
    print(f"  Generating {n_points} feasible initial points for EQUALITY constraints...")
    
    # For equality constraints, we use projection rather than rejection sampling
    # since equality constraints define a lower-dimensional manifold
    return project_to_equality_constraints(bounds, equality_constraints, n_points)

def project_to_equality_constraints(
    bounds: torch.Tensor,
    equality_constraints: List[Tuple[torch.Tensor, torch.Tensor, float]], 
    n_points: int
) -> torch.Tensor:
    """
    Project random points to satisfy equality constraints.
    
    For multiple equality constraints, we solve them systematically:
    1. x1 + 2*x6 = 1.0 → solve for x6 given x1
    2. x2 + x3 = 0.5 → solve for x3 given x2
    """
    points = []
    
    for _ in range(n_points):
        point = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(bounds.shape[1])
        
        # Apply each equality constraint
        for constraint_idx, (indices, coeffs, rhs) in enumerate(equality_constraints):
            if len(indices) == 2:  # Simple case with 2 variables
                # For constraint: coeffs[0]*x[indices[0]] + coeffs[1]*x[indices[1]] = rhs
                # Solve for x[indices[1]]: x[indices[1]] = (rhs - coeffs[0]*x[indices[0]]) / coeffs[1]
                
                var1_idx, var2_idx = indices[0], indices[1]
                coeff1, coeff2 = coeffs[0], coeffs[1]
                
                # Calculate constrained variable value
                constrained_val = (rhs - coeff1 * point[var1_idx]) / coeff2
                
                # Clamp to bounds
                constrained_val = torch.clamp(
                    constrained_val, 
                    bounds[0, var2_idx], 
                    bounds[1, var2_idx]
                )
                point[var2_idx] = constrained_val
                
                # If constrained variable hit a bound, adjust the other variable
                if constrained_val == bounds[0, var2_idx] or constrained_val == bounds[1, var2_idx]:
                    adjusted_val = (rhs - coeff2 * constrained_val) / coeff1
                    adjusted_val = torch.clamp(
                        adjusted_val,
                        bounds[0, var1_idx],
                        bounds[1, var1_idx]
                    )
                    point[var1_idx] = adjusted_val
        
        points.append(point)
    
    return torch.stack(points)

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

def compare_implementations(baybe_results, botorch_results):
    """Compare the results from both implementations."""
    print("\n" + "=" * 60)
    print("Implementation Comparison")
    print("=" * 60)
    
    if baybe_results is None or botorch_results is None:
        print("Cannot compare - one or both implementations failed to run")
        return
    
    print("Convergence Comparison:")
    print(f"{'Iteration':<10} {'BayBE':<12} {'BoTorch':<12} {'Difference':<12}")
    print("-" * 50)
    
    for i, (baybe_val, botorch_val) in enumerate(zip(baybe_results, botorch_results)):
        diff = abs(baybe_val - botorch_val)
        print(f"{i+1:<10} {baybe_val:<12.6f} {botorch_val:<12.6f} {diff:<12.6f}")
    
    print(f"\nFinal Results:")
    print(f"  BayBE best:   {min(baybe_results):.6f}")
    print(f"  BoTorch best: {min(botorch_results):.6f}")
    print(f"  Difference:   {abs(min(baybe_results) - min(botorch_results)):.6f}")
    
    # Statistical comparison
    baybe_improvement = baybe_results[0] - min(baybe_results)
    botorch_improvement = botorch_results[0] - min(botorch_results)
    
    print(f"\nImprovement from initial:")
    print(f"  BayBE:   {baybe_improvement:.6f}")
    print(f"  BoTorch: {botorch_improvement:.6f}")

def analyze_constraint_handling():
    """Analyze how EQUALITY constraints are handled in both implementations."""
    print("\n" + "=" * 60)
    print("Linear EQUALITY Constraint Handling Analysis")
    print("=" * 60)
    
    print("""
BayBE Equality Constraint Handling:
----------------------------------
1. High-level API: Users define equality constraints using parameter names and coefficients
2. Automatic conversion: BayBE automatically converts constraints to BoTorch format
3. Integration: Constraints are seamlessly passed to BoTorch's optimize_acqf
4. Validation: BayBE validates constraint feasibility during search space creation

Example BayBE equality constraint:
    ContinuousLinearEqualityConstraint(
        parameters=["x1", "x6"], 
        coefficients=[1.0, 2.0], 
        rhs=1.0
    )
    # Represents: x1 + 2*x6 = 1.0

BoTorch Equality Constraint Handling:
------------------------------------
1. Low-level API: Users must provide constraints as (indices, coefficients, rhs) tuples
2. Manual conversion: Users must map parameter names to indices
3. Direct integration: Constraints passed directly to optimize_acqf as equality_constraints
4. Manual validation: Users responsible for ensuring constraint feasibility

Example BoTorch equality constraint:
    equality_constraints = [(
        torch.tensor([0, 5]),      # parameter indices for x1, x6
        torch.tensor([1.0, 2.0]),  # coefficients  
        1.0                        # rhs value (x1 + 2*x6 = 1.0)
    )]

Key Differences for Equality Constraints:
----------------------------------------
1. Abstraction Level: BayBE provides higher-level abstractions
2. Error Handling: BayBE has more built-in validation and error checking
3. Manifold Optimization: Both use BoTorch's constraint-aware optimization on reduced manifolds
4. Initial Point Generation: Both need to generate feasible starting points on constraint manifold
5. Performance: Both use the same underlying BoTorch equality constraint algorithms

Equality Constraint Conversion Process:
--------------------------------------
BayBE's to_botorch() method performs this conversion for equality constraints:

    def to_botorch(self, parameters, idx_offset=0):
        param_names = [p.name for p in parameters]
        param_indices = [param_names.index(p) + idx_offset for p in self.parameters]
        
        return (
            torch.tensor(param_indices),
            torch.tensor(self.coefficients),
            self.rhs
        )

Mathematical Challenge of Equality Constraints:
----------------------------------------------
Equality constraints reduce the dimensionality of the search space:
- Original space: 6D cube [0,1]^6
- With x1 + 2*x6 = 1.0: 5D manifold embedded in 6D space
- With x2 + x3 = 0.5: 4D manifold embedded in 6D space
- BoTorch handles this through constrained optimization algorithms

This shows how BayBE abstracts away the index mapping and manifold projection complexity.
""")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("BayBE vs BoTorch Linear Constraints Comparison")
    print("=" * 60)
    print(f"Function: Hartmann{DIMENSION}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    
    set_seeds(RANDOM_SEED)
    
    # Run BayBE implementation
    baybe_results, baybe_measurements = demo_baybe_implementation()
    
    # Run BoTorch implementation
    botorch_results, botorch_points = demo_botorch_implementation()
    
    # Compare results
    compare_implementations(baybe_results, botorch_results)
    
    # Analyze constraint handling approaches
    analyze_constraint_handling()
    
    print("\n" + "=" * 60)
    print("Summary: Linear EQUALITY Constraints")
    print("=" * 60)
    print("""
This comparison demonstrates equality constraint handling:

1. **Same Underlying Engine**: Both BayBE and direct BoTorch use identical 
   equality constraint algorithms from BoTorch's optimize_acqf function.

2. **Abstraction Benefits**: BayBE provides a more user-friendly interface
   for defining and managing linear EQUALITY constraints.

3. **Equivalent Performance**: Since BayBE uses BoTorch internally, 
   performance should be very similar between both approaches.

4. **Conversion Transparency**: BayBE's to_botorch() method provides a 
   clear conversion from high-level equality constraints to BoTorch format.

5. **Manifold Optimization**: Both approaches properly handle the reduced 
   dimensionality challenge by constraining optimization to the constraint manifold.

6. **Mathematical Insight**: Equality constraints like x1 + 2*x6 = 1.0 and 
   x2 + x3 = 0.5 reduce the 6D search space to a 4D manifold, which BoTorch
   handles through specialized constraint-aware optimization algorithms.

The key insight is that BayBE acts as a powerful abstraction layer over
BoTorch, providing the same equality constraint capabilities with a more
intuitive interface while maintaining full mathematical rigor.
""")

if __name__ == "__main__":
    main()