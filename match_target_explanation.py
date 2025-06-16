#!/usr/bin/env python3
"""
Demonstration of BayBE's MATCH Target Implementation

This script demonstrates how BayBE implements a "match" target to get an objective
as close as possible to a specific value X. It shows the key concepts without
requiring external dependencies.
"""

import math
from typing import List


def triangular_transform(value: float, lower: float, upper: float) -> float:
    """Map a value to the interval [0, 1] in a "triangular" fashion.

    The shape of the function is "triangular" in that is 0 outside a specified interval
    and linearly increases to 1 from both interval ends, reaching the value 1 at the
    center of the interval.

    Args:
        value: The value to be mapped.
        lower: The lower end of the triangle interval. Below, the mapped value is 0.
        upper: The upper end of the triangle interval. Above, the mapped value is 0.

    Returns:
        The transformed value.
    """
    if value < lower or value > upper:
        return 0.0
    
    mid = lower + (upper - lower) / 2
    
    if value <= mid:
        return (value - lower) / (mid - lower)
    else:
        return (upper - value) / (upper - mid)


def bell_transform(value: float, lower: float, upper: float) -> float:
    """Map a value to the interval [0, 1] in a "Gaussian bell" fashion.

    The shape of the function is "Gaussian bell curve", specified through the boundary
    values of the sigma interval. Reaches the maximum value of 1 at the interval center.

    Args:
        value: The value to be mapped.
        lower: The input value corresponding to the lower sigma interval boundary.
        upper: The input value corresponding to the upper sigma interval boundary.

    Returns:
        The transformed value.
    """
    mean = (lower + upper) / 2
    std = (upper - lower) / 2
    return math.exp(-((value - mean) ** 2) / (2.0 * std**2))


def demonstrate_match_target():
    """Demonstrate how BayBE's match target works."""
    
    print("=" * 70)
    print("BayBE MATCH Target Implementation Explained")
    print("=" * 70)
    
    print("""
HOW IT WORKS:
""")
    
    print("""1. CONCEPT:
   BayBE's MATCH mode allows you to target a specific value rather than
   simply minimizing or maximizing an objective.
   
2. IMPLEMENTATION:
   - You specify bounds where the MIDPOINT equals your desired match value
   - BayBE applies a transformation function to map measured values to [0,1]
   - The transformation gives score 1.0 at the match target, lower elsewhere
   - This integrates with BayBE's desirability framework for multi-objective optimization
""")
    
    # Example: We want to match a target value of 50
    target_value = 50
    print(f"\nEXAMPLE: Matching target value = {target_value}")
    print("-" * 50)
    
    # In BayBE, we specify bounds where the midpoint is our desired value
    
    # Example 1: Tight bounds (less forgiving)
    tight_bounds = (45, 55)  # midpoint = 50, spread = 10
    print(f"\nScenario A - Tight tolerance around target:")
    print(f"  bounds = {tight_bounds}")
    print(f"  midpoint = {(tight_bounds[0] + tight_bounds[1]) / 2} (this becomes our target)")
    print(f"  tolerance = ±{(tight_bounds[1] - tight_bounds[0]) / 2}")
    
    # Example 2: Wide bounds (more forgiving)  
    wide_bounds = (30, 70)  # midpoint = 50, spread = 40
    print(f"\nScenario B - Wide tolerance around target:")
    print(f"  bounds = {wide_bounds}")
    print(f"  midpoint = {(wide_bounds[0] + wide_bounds[1]) / 2} (this becomes our target)")
    print(f"  tolerance = ±{(wide_bounds[1] - wide_bounds[0]) / 2}")
    
    # Test values to evaluate
    test_values = [30, 40, 45, 48, 50, 52, 55, 60, 70]
    
    print(f"\nEVALUATING TEST VALUES: {test_values}")
    print("=" * 70)
    
    # Apply transformations
    print("\nTRIANGULAR TRANSFORMATION (DEFAULT FOR MATCH MODE)")
    print("-" * 60)
    print("Creates a triangle-shaped preference with peak at target")
    print()
    print(f"{'Value':<8} {'Tight Bounds':<12} {'Wide Bounds':<12} {'Interpretation'}")
    print("-" * 60)
    
    for val in test_values:
        tight_score = triangular_transform(val, *tight_bounds)
        wide_score = triangular_transform(val, *wide_bounds)
        
        if val == target_value:
            interp = "🎯 PERFECT MATCH"
        elif tight_score == 0:
            interp = "❌ Outside tight tolerance"
        elif tight_score < 0.5:
            interp = "⚠️  Poor match"
        elif tight_score < 0.9:
            interp = "✓ Good match"
        else:
            interp = "✓✓ Excellent match"
            
        print(f"{val:<8.0f} {tight_score:<12.3f} {wide_score:<12.3f} {interp}")
    
    print("\nBELL TRANSFORMATION (ALTERNATIVE FOR MATCH MODE)")
    print("-" * 60)
    print("Creates a smooth Gaussian preference curve")
    print()
    print(f"{'Value':<8} {'Tight Bounds':<12} {'Wide Bounds':<12} {'Interpretation'}")
    print("-" * 60)
    
    for val in test_values:
        tight_score = bell_transform(val, *tight_bounds)
        wide_score = bell_transform(val, *wide_bounds)
        
        if val == target_value:
            interp = "🎯 PERFECT MATCH"
        elif tight_score < 0.1:
            interp = "❌ Very poor match"
        elif tight_score < 0.5:
            interp = "⚠️  Poor match"
        elif tight_score < 0.8:
            interp = "✓ Good match"
        else:
            interp = "✓✓ Excellent match"
            
        print(f"{val:<8.0f} {tight_score:<12.3f} {wide_score:<12.3f} {interp}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS & PRACTICAL GUIDANCE")
    print("=" * 70)
    
    print(f"""
🎯 SETTING UP A MATCH TARGET:

1. CHOOSE YOUR TARGET VALUE:
   Example: You want temperature = 50°C

2. SET BOUNDS STRATEGICALLY:
   - Tight bounds (45, 55): Strict preference, steep penalties for deviation
   - Wide bounds (30, 70): Flexible preference, gradual penalties
   
   Rule: midpoint of bounds = your target value
   
3. CHOOSE TRANSFORMATION:
   - TRIANGULAR (default): Sharp cutoff outside bounds, linear inside
   - BELL: Smooth Gaussian curve, more forgiving outside bounds

📊 IN PRACTICE WITH BAYBE:

# Strict matching (temperature must be close to 50°C)
strict_target = NumericalTarget(
    name="temperature",
    mode=TargetMode.MATCH,
    bounds=(45, 55),  # midpoint = 50°C 
    transformation=TargetTransformation.TRIANGULAR
)

# Flexible matching (temperature should be around 50°C)
flexible_target = NumericalTarget(
    name="temperature", 
    mode=TargetMode.MATCH,
    bounds=(30, 70),  # midpoint = 50°C
    transformation=TargetTransformation.BELL
)

🔧 TUNING TIPS:

1. START WIDE: Begin with wider bounds, then tighten based on results
2. CONSIDER MEASUREMENT NOISE: Wider bounds help with noisy measurements  
3. MULTIPLE TARGETS: Use DesirabilityObjective to combine multiple match targets
4. DOMAIN KNOWLEDGE: Set bounds based on what's realistically achievable
""")
    
    print("\n" + "=" * 70)
    print("INTEGRATION WITH MULTI-OBJECTIVE OPTIMIZATION")
    print("=" * 70)
    
    print("""
BayBE's match targets work seamlessly with other objectives:

# Example: Optimize multiple properties simultaneously
objective = DesirabilityObjective(
    targets=[
        NumericalTarget("yield", mode="MAX", bounds=(0, 100)),
        NumericalTarget("temp", mode="MATCH", bounds=(45, 55)),  # target 50°C
        NumericalTarget("purity", mode="MAX", bounds=(80, 99)),
    ],
    weights=[1.0, 2.0, 1.0],  # temperature matching is most important
    scalarizer="GEOM_MEAN"
)

The system will:
✓ Maximize yield and purity  
✓ Keep temperature close to 50°C
✓ Balance trade-offs based on weights
""")


if __name__ == "__main__":
    demonstrate_match_target()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
BayBE's MATCH target implementation is both elegant and practical:

✅ SIMPLE SETUP: Just set bounds with midpoint = desired target
✅ FLEXIBLE CONTROL: Adjust bounds spread to control tolerance  
✅ TWO SHAPES: Choose triangular (strict) or bell (smooth) preference
✅ SEAMLESS INTEGRATION: Works naturally with multi-objective optimization
✅ DOMAIN AGNOSTIC: Works for any numerical target (temperature, pH, time, etc.)

This approach allows BayBE to handle real-world scenarios where you need
to hit specific target values rather than just optimizing in one direction.
""")