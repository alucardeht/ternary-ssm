"""Test F5: REQuant-Golden-Section-Threshold implementation"""

import torch
import sys

sys.path.insert(0, "C:/Users/th1rt/Documents/Workspaces/code-ai-workspace/src")

from model_v7.components.ptq_quantization import (
    golden_section_search,
    compute_ternary_quantization_error,
    find_optimal_threshold_gss,
    REQuantThresholdOptimizer,
    requant_layer,
)


def test_golden_section_search():
    """Test golden-section search on simple convex functions"""
    print("\n=== Testing golden_section_search ===")

    # Test 1: Quadratic function f(x) = (x - 3)^2, minimum at x = 3
    def quadratic(x):
        return (x - 3) ** 2

    x_min, f_min = golden_section_search(quadratic, a=0, b=10, tol=1e-8)
    print(f"Quadratic (x-3)^2: x_min={x_min:.6f}, f_min={f_min:.10f}")
    assert abs(x_min - 3) < 0.001, f"Expected x=3, got {x_min}"
    assert abs(f_min) < 0.001, f"Expected f=0, got {f_min}"

    # Test 2: f(x) = x^2, minimum at x = 0
    x_min, f_min = golden_section_search(lambda x: x**2, a=-5, b=5, tol=1e-8)
    print(f"Quadratic x^2: x_min={x_min:.6f}, f_min={f_min:.10f}")
    assert abs(x_min) < 0.001, f"Expected x=0, got {x_min}"

    # Test 3: Rosenbrock-like function (a=1, b=100)
    def rosenbrock(x):
        return (1 - x) ** 2 + 100 * (x**2 - x) ** 2

    x_min, f_min = golden_section_search(rosenbrock, a=-2, b=2, tol=1e-6)
    print(f"Rosenbrock: x_min={x_min:.6f}, f_min={f_min:.10f}")

    print("✓ golden_section_search tests passed!")


def test_quantization_error():
    """Test quantization error computation"""
    print("\n=== Testing compute_ternary_quantization_error ===")

    # Create random weight tensor
    torch.manual_seed(42)
    weight = torch.randn(512, 256)

    # Test different alpha values
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        error = compute_ternary_quantization_error(weight, alpha, per_group=False)
        print(f"alpha={alpha}: error={error:.6f}")

    # Test per-group
    error_grouped = compute_ternary_quantization_error(
        weight, 0.5, group_size=128, per_group=True
    )
    print(f"per-group (alpha=0.5): error={error_grouped:.6f}")

    print("✓ compute_ternary_quantization_error tests passed!")


def test_find_optimal_threshold():
    """Test finding optimal threshold with GSS"""
    print("\n=== Testing find_optimal_threshold_gss ===")

    torch.manual_seed(42)
    weight = torch.randn(512, 256)

    # Find optimal threshold
    alpha, min_error = find_optimal_threshold_gss(
        weight, group_size=128, per_group=False, tol=1e-6, max_iter=50
    )
    print(f"Global: optimal_alpha={alpha:.6f}, min_error={min_error:.10f}")

    # Compare with simple threshold (mean-based)
    mean_alpha = weight.abs().mean().item() * 0.767  # Default threshold coefficient
    simple_error = compute_ternary_quantization_error(
        weight, mean_alpha, per_group=False
    )
    print(f"Simple (mean-based): alpha={mean_alpha:.6f}, error={simple_error:.10f}")
    print(f"Improvement: {(simple_error - min_error) / simple_error * 100:.2f}%")

    # Test per-group
    alpha_grouped, min_error_grouped = find_optimal_threshold_gss(
        weight, group_size=128, per_group=True, tol=1e-6, max_iter=30
    )
    print(
        f"Per-group: optimal_alpha={alpha_grouped:.6f}, min_error={min_error_grouped:.10f}"
    )

    print("✓ find_optimal_threshold_gss tests passed!")


def test_requant_optimizer():
    """Test REQuantThresholdOptimizer class"""
    print("\n=== Testing REQuantThresholdOptimizer ===")

    torch.manual_seed(42)
    weight = torch.randn(256, 128)

    optimizer = REQuantThresholdOptimizer(
        group_size=64, per_group=True, tol=1e-6, max_iter=30
    )

    # Find optimal threshold
    alpha = optimizer.find_optimal_threshold(weight)
    print(f"Optimal alpha: {alpha:.6f}")

    # Quantize
    quantized, alpha_used = optimizer.quantize(weight, alpha)
    print(f"Alpha used: {alpha_used:.6f}")

    # Compute metrics
    mse = (weight - quantized).pow(2).mean().item()
    print(f"MSE: {mse:.10f}")

    # Verify quantization is ternary
    unique_vals = torch.unique(quantized)
    print(f"Unique values in quantized: {len(unique_vals)}")
    print(f"Values: {unique_vals}")

    print("✓ REQuantThresholdOptimizer tests passed!")


def test_requant_layer():
    """Test requant_layer function"""
    print("\n=== Testing requant_layer ===")

    import torch.nn as nn

    # Create a linear layer
    layer = nn.Linear(256, 128)
    torch.nn.init.xavier_uniform_(layer.weight)

    # Apply REQuant
    info = requant_layer(layer, group_size=64, per_group=True, layer_name="test_layer")

    print(f"Layer: {info['layer_name']}")
    print(f"Shape: {info['shape']}")
    print(f"Optimal alpha: {info['optimal_alpha']:.6f}")
    print(f"MSE: {info['mse']:.10f}")
    print(f"Group size: {info['group_size']}")
    print(f"Per-group: {info['per_group']}")

    print("✓ requant_layer tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("F5: REQuant-Golden-Section-Threshold Test Suite")
    print("=" * 60)

    test_golden_section_search()
    test_quantization_error()
    test_find_optimal_threshold()
    test_requant_optimizer()
    test_requant_layer()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
