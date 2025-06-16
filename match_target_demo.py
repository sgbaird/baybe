#!/usr/bin/env python3
"""
Demonstration of BayBE's MATCH Target Implementation

This script demonstrates how BayBE implements a "match" target to get an objective
as close as possible to a specific value X. It shows the key concepts without
requiring the full BayBE installation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import ArrayLike


def triangular_transform(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """Map values to the interval [0, 1] in a "triangular" fashion.

    The shape of the function is "triangular" in that is 0 outside a specified interval
    and linearly increases to 1 from both interval ends, reaching the value 1 at the
    center of the interval.

    Args:
        arr: The values to be mapped.
        lower: The lower end of the triangle interval. Below, the mapped values are 0.
        upper: The upper end of the triangle interval. Above, the mapped values are 0.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    mid = lower + (upper - lower) / 2
    res = (arr - lower) / (mid - lower)
    res[arr > mid] = (upper - arr[arr > mid]) / (upper - mid)
    res[arr > upper] = 0.0
    res[arr < lower] = 0.0

    return res


def bell_transform(arr: ArrayLike, lower: float, upper: float) -> np.ndarray:
    """Map values to the interval [0, 1] in a "Gaussian bell" fashion.

    The shape of the function is "Gaussian bell curve", specified through the boundary
    values of the sigma interval. Reaches the maximum value of 1 at the interval center.

    Args:
        arr: The values to be mapped.
        lower: The input value corresponding to the lower sigma interval boundary.
        upper: The input value corresponding to the upper sigma interval boundary.

    Returns:
        A new array containing the transformed values.
    """
    arr = np.asarray(arr)
    mean = np.mean([lower, upper])
    std = (upper - lower) / 2
    res = np.exp(-((arr - mean) ** 2) / (2.0 * std**2))

    return res


def demonstrate_match_target():
    """Demonstrate how BayBE's match target works."""
    
    print("=" * 70)
    print("BayBE MATCH Target Demonstration")
    print("=" * 70)
    
    # Example: We want to match a target value of 50
    target_value = 50
    print(f"\nGoal: Match target value = {target_value}")
    
    # In BayBE, we specify bounds where the midpoint is our desired value
    # The spread controls how forgiving we are around the target
    
    # Example 1: Tight bounds (less forgiving)
    tight_bounds = (45, 55)  # midpoint = 50, spread = 10
    print(f"\nExample 1 - Tight bounds: {tight_bounds}")
    print(f"  Midpoint (match target): {(tight_bounds[0] + tight_bounds[1]) / 2}")
    print(f"  Spread: {tight_bounds[1] - tight_bounds[0]}")
    
    # Example 2: Wide bounds (more forgiving)
    wide_bounds = (30, 70)  # midpoint = 50, spread = 40
    print(f"\nExample 2 - Wide bounds: {wide_bounds}")
    print(f"  Midpoint (match target): {(wide_bounds[0] + wide_bounds[1]) / 2}")
    print(f"  Spread: {wide_bounds[1] - wide_bounds[0]}")
    
    # Test values to evaluate
    test_values = np.array([30, 40, 45, 48, 50, 52, 55, 60, 70])
    
    print(f"\nTest values: {test_values}")
    
    # Apply transformations
    print("\n" + "-" * 50)
    print("TRIANGULAR TRANSFORMATION")
    print("-" * 50)
    
    tight_triangular = triangular_transform(test_values, *tight_bounds)
    wide_triangular = triangular_transform(test_values, *wide_bounds)
    
    print(f"{'Value':<8} {'Tight':<8} {'Wide':<8}")
    print("-" * 24)
    for val, tight, wide in zip(test_values, tight_triangular, wide_triangular):
        print(f"{val:<8.0f} {tight:<8.3f} {wide:<8.3f}")
    
    print("\n" + "-" * 50)
    print("BELL TRANSFORMATION")
    print("-" * 50)
    
    tight_bell = bell_transform(test_values, *tight_bounds)
    wide_bell = bell_transform(test_values, *wide_bounds)
    
    print(f"{'Value':<8} {'Tight':<8} {'Wide':<8}")
    print("-" * 24)
    for val, tight, wide in zip(test_values, tight_bell, wide_bell):
        print(f"{val:<8.0f} {tight:<8.3f} {wide:<8.3f}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    print(f"""
1. MATCH TARGET CONCEPT:
   - Specify bounds where midpoint = desired match value
   - Transformation maps measurements to [0,1] with 1 = optimal
   - Value 1.0 achieved exactly at the match target ({target_value})

2. BOUNDS CONTROL TOLERANCE:
   - Tight bounds ({tight_bounds}): Less forgiving, steep penalty for deviation
   - Wide bounds ({wide_bounds}): More forgiving, gradual penalty for deviation

3. TRANSFORMATION SHAPES:
   - TRIANGULAR: Linear increase/decrease, harsh cutoff outside bounds
   - BELL: Smooth Gaussian curve, gradual falloff even outside bounds

4. PRACTICAL USAGE:
   In BayBE, you would create a target like:
   
   target = NumericalTarget(
       name="temperature",
       mode=TargetMode.MATCH,
       bounds=({tight_bounds[0]}, {tight_bounds[1]}),  # to match {target_value}
       transformation=TargetTransformation.TRIANGULAR  # or BELL
   )
""")


def create_visualization():
    """Create a visualization showing the transformation functions."""
    
    try:
        # Create test data
        x = np.linspace(20, 80, 1000)
        target_value = 50
        
        # Different bound scenarios
        tight_bounds = (45, 55)
        medium_bounds = (40, 60)
        wide_bounds = (30, 70)
        
        # Calculate transformations
        scenarios = [
            ("Tight (45-55)", tight_bounds),
            ("Medium (40-60)", medium_bounds),
            ("Wide (30-70)", wide_bounds)
        ]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'BayBE Match Target Transformations (Target Value = {target_value})', fontsize=14)
        
        # Triangular transformations
        ax1.set_title('Triangular Transformation')
        ax1.axvline(target_value, color='red', linestyle='--', alpha=0.7, label=f'Target = {target_value}')
        
        for name, bounds in scenarios:
            y = triangular_transform(x, *bounds)
            ax1.plot(x, y, label=f'{name}')
        
        ax1.set_xlabel('Measured Value')
        ax1.set_ylabel('Transformed Score [0-1]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bell transformations
        ax2.set_title('Bell Transformation')
        ax2.axvline(target_value, color='red', linestyle='--', alpha=0.7, label=f'Target = {target_value}')
        
        for name, bounds in scenarios:
            y = bell_transform(x, *bounds)
            ax2.plot(x, y, label=f'{name}')
        
        ax2.set_xlabel('Measured Value')
        ax2.set_ylabel('Transformed Score [0-1]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Comparison for tight bounds
        ax3.set_title('Comparison: Tight Bounds (45-55)')
        ax3.axvline(target_value, color='red', linestyle='--', alpha=0.7, label=f'Target = {target_value}')
        
        y_tri = triangular_transform(x, *tight_bounds)
        y_bell = bell_transform(x, *tight_bounds)
        
        ax3.plot(x, y_tri, label='Triangular', linewidth=2)
        ax3.plot(x, y_bell, label='Bell', linewidth=2)
        ax3.set_xlabel('Measured Value')
        ax3.set_ylabel('Transformed Score [0-1]')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Score table for specific values
        ax4.axis('off')
        test_vals = [40, 45, 48, 50, 52, 55, 60]
        tri_scores = triangular_transform(np.array(test_vals), *tight_bounds)
        bell_scores = bell_transform(np.array(test_vals), *tight_bounds)
        
        table_data = []
        table_data.append(['Value', 'Triangular', 'Bell'])
        table_data.append(['-----', '---------', '----'])
        for val, tri, bell in zip(test_vals, tri_scores, bell_scores):
            table_data.append([f'{val}', f'{tri:.3f}', f'{bell:.3f}'])
        
        table_text = '\n'.join([f'{row[0]:<8} {row[1]:<10} {row[2]:<8}' for row in table_data])
        ax4.text(0.1, 0.9, f'Score Comparison\n(Tight Bounds: {tight_bounds})\n\n{table_text}', 
                transform=ax4.transAxes, fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('/tmp/match_target_demo.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: /tmp/match_target_demo.png")
        
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization")
        print("Install matplotlib to see graphical demonstration")


if __name__ == "__main__":
    demonstrate_match_target()
    create_visualization()
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
BayBE's MATCH target implementation is elegant and intuitive:

1. Users specify the desired match value implicitly through bounds midpoint
2. Transformation functions convert raw measurements to optimization scores
3. The system naturally prefers values closer to the target
4. Bounds spread controls the tolerance around the target value

This approach integrates seamlessly with BayBE's desirability framework,
allowing multiple objectives to be combined while maintaining clear
target matching behavior.
""")