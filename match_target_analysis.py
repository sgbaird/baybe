#!/usr/bin/env python3
"""
Analysis of BayBE's MATCH target implementation to answer:
1. Does it distinguish between overshoot vs undershoot?
2. How do the transformations relate to MAE vs RMSE?
"""

import numpy as np
import matplotlib.pyplot as plt


def triangular_transform(arr, lower, upper):
    """Map values to [0, 1] in a triangular fashion (from BayBE source)."""
    arr = np.asarray(arr)
    mid = lower + (upper - lower) / 2
    res = (arr - lower) / (mid - lower)
    res[arr > mid] = (upper - arr[arr > mid]) / (upper - mid)
    res[arr > upper] = 0.0
    res[arr < lower] = 0.0
    return res


def bell_transform(arr, lower, upper):
    """Map values to [0, 1] in a Gaussian bell fashion (from BayBE source)."""
    arr = np.asarray(arr)
    mean = np.mean([lower, upper])
    std = (upper - lower) / 2
    res = np.exp(-((arr - mean) ** 2) / (2.0 * std**2))
    return res


def analyze_match_target_behavior():
    """Analyze and visualize the behavior of MATCH target transformations."""
    
    # Define target bounds (midpoint becomes the target)
    bounds = (45, 55)  # Target value is 50
    lower, upper = bounds
    target = (lower + upper) / 2
    
    # Create a range of values around the target
    values = np.linspace(35, 65, 1000)
    
    # Apply transformations
    triangular_scores = triangular_transform(values, lower, upper)
    bell_scores = bell_transform(values, lower, upper)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Transformation functions
    ax1.plot(values, triangular_scores, 'b-', linewidth=2, label='Triangular (similar to MAE)')
    ax1.plot(values, bell_scores, 'r-', linewidth=2, label='Bell (Gaussian-like)')
    ax1.axvline(x=target, color='k', linestyle='--', alpha=0.7, label=f'Target = {target}°C')
    ax1.axvline(x=lower, color='gray', linestyle=':', alpha=0.5, label=f'Bounds: [{lower}, {upper}]')
    ax1.axvline(x=upper, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Raw Value (e.g., Temperature °C)')
    ax1.set_ylabel('Desirability Score (0-1)')
    ax1.set_title('BayBE MATCH Target Transformations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)
    
    # Plot 2: Analysis of symmetry (overshoot vs undershoot)
    # Show values at specific deviations from target
    deviations = np.array([-10, -5, -2, 0, 2, 5, 10])
    test_values = target + deviations
    
    triangular_at_deviations = triangular_transform(test_values, lower, upper)
    bell_at_deviations = bell_transform(test_values, lower, upper)
    
    x_pos = np.arange(len(deviations))
    width = 0.35
    
    ax2.bar(x_pos - width/2, triangular_at_deviations, width, 
            label='Triangular', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, bell_at_deviations, width, 
            label='Bell', alpha=0.7, color='red')
    
    ax2.set_xlabel('Deviation from Target (°C)')
    ax2.set_ylabel('Desirability Score')
    ax2.set_title('Symmetry Analysis: Overshoot vs Undershoot')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{d:+d}' for d in deviations])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (dev, tri_score, bell_score) in enumerate(zip(deviations, triangular_at_deviations, bell_at_deviations)):
        if dev == 0:
            ax2.annotate(f'Perfect\nMatch', (i, tri_score + 0.05), 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/baybe/baybe/match_target_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze symmetry mathematically
    print("=== MATCH Target Analysis ===\n")
    
    print("1. OVERSHOOT vs UNDERSHOOT DISTINCTION:")
    print("   Answer: NO - Both transformations are symmetric around the target")
    print("   - Triangular: Same linear penalty for +5°C and -5°C deviation")
    print("   - Bell: Same Gaussian penalty for +5°C and -5°C deviation")
    print("   - The model cannot distinguish between being 5°C above vs 5°C below target\n")
    
    print("2. RELATIONSHIP TO LOSS FUNCTIONS:")
    print("   - Triangular ≈ MAE (Mean Absolute Error):")
    print("     * Linear penalty proportional to |actual - target|")
    print("     * Desirability = 1 - |actual - target| / tolerance")
    print("   - Bell ≈ Gaussian (related to RMSE concept):")
    print("     * Exponential penalty based on (actual - target)²")
    print("     * Desirability = exp(-0.5 * ((actual - target) / σ)²)")
    print("     * Not exactly RMSE, but uses squared differences\n")
    
    print("3. PRACTICAL IMPLICATIONS:")
    test_cases = [
        (40, "Far undershoot"),
        (48, "Close undershoot"), 
        (50, "Perfect match"),
        (52, "Close overshoot"),
        (60, "Far overshoot")
    ]
    
    print("   Value | Triangular | Bell     | Interpretation")
    print("   ------|------------|----------|---------------")
    for value, desc in test_cases:
        tri_score = triangular_transform(np.array([value]), lower, upper)[0]
        bell_score = bell_transform(np.array([value]), lower, upper)[0]
        print(f"   {value:5.0f} | {tri_score:10.3f} | {bell_score:8.3f} | {desc}")
    
    print(f"\n4. KEY INSIGHT:")
    print(f"   BayBE's MATCH targets are symmetric penalty functions:")
    print(f"   - Target value = midpoint of bounds = {target}°C")
    print(f"   - Both overshoot and undershoot get identical penalties")
    print(f"   - Model learns to optimize desirability, not directional bias")
    print(f"   - BoTorch receives only the transformed desirability scores (0-1)")
    

if __name__ == "__main__":
    analyze_match_target_behavior()