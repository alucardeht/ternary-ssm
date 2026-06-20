"""Compute-Bound Regime Theorem for Ternary Kernels (R4-A5-M3).

Theorem: Ternary matvec is COMPUTE-BOUND (3.33 ops/byte) not memory-bound.
This reframes kernel optimization priorities:
  - Maximize addition ILP (instruction-level parallelism)
  - NOT: maximize prefetching/bandwidth
  - Optimal: 4-8 independent accumulator registers to saturate ALU ports

This module provides analysis tools and kernel design guidelines.
"""
import torch
from typing import Optional


def analyze_compute_intensity(weight_shape: tuple, activation_dtype: str = 'int8',
                               weight_packing: str = 'quinary') -> dict:
    """Analyze arithmetic intensity (ops/byte) for a ternary matmul layer.

    Args:
        weight_shape: (out_dim, in_dim)
        activation_dtype: 'int8' or 'float16'
        weight_packing: 'quinary' (5 trits/byte), 'binary' (4 trits/byte), 'naive' (1 trit/byte)

    Returns:
        dict with compute intensity analysis
    """
    out_dim, in_dim = weight_shape

    ops_per_output = in_dim

    if weight_packing == 'quinary':
        weight_bytes = in_dim / 5
    elif weight_packing == 'binary':
        weight_bytes = in_dim / 4
    else:
        weight_bytes = in_dim

    if activation_dtype == 'int8':
        act_bytes = in_dim
    else:
        act_bytes = in_dim * 2

    output_bytes = 4

    total_bytes = weight_bytes + act_bytes + output_bytes
    intensity = ops_per_output / total_bytes

    is_compute_bound_cpu = intensity > 0.5
    is_compute_bound_gpu = intensity > 10

    return {
        'ops_per_output': ops_per_output,
        'bytes_per_output': total_bytes,
        'arithmetic_intensity': intensity,
        'is_compute_bound_cpu': is_compute_bound_cpu,
        'is_compute_bound_gpu': is_compute_bound_gpu,
        'recommendation': 'Maximize ILP: use 4-8 independent accumulators' if is_compute_bound_cpu
                          else 'Memory-bound: maximize prefetching and bandwidth',
        'weight_packing': weight_packing,
        'activation_dtype': activation_dtype,
    }


def optimal_accumulator_count(simd_width: int = 256, element_bits: int = 32) -> int:
    """Compute optimal number of independent accumulators for ILP.

    For compute-bound ternary kernels, we want enough independent
    accumulators to saturate ALU ports without spilling to memory.

    Args:
        simd_width: SIMD register width in bits (128=SSE, 256=AVX2, 512=AVX512)
        element_bits: accumulator element size in bits

    Returns:
        Recommended number of accumulators
    """
    elements_per_register = simd_width // element_bits
    return max(4, elements_per_register * 2)


def estimate_theoretical_throughput(n: int, m: int, k: int,
                                     clock_ghz: float = 4.0,
                                     simd_width: int = 256,
                                     n_alu_ports: int = 2) -> dict:
    """Estimate theoretical throughput for ternary matmul.

    Args:
        n, m, k: matrix dimensions (output=n, input=k, batch=m)
        clock_ghz: processor clock speed
        simd_width: SIMD width in bits
        n_alu_ports: number of ALU ports that can execute add/sub

    Returns:
        dict with throughput estimates
    """
    total_ops = n * m * k
    elements_per_simd = simd_width // 32

    peak_gops = n_alu_ports * elements_per_simd * clock_ghz

    time_s = total_ops / (peak_gops * 1e9)
    throughput_gops = total_ops / time_s / 1e9 if time_s > 0 else 0

    return {
        'total_ops': total_ops,
        'peak_gops': peak_gops,
        'estimated_time_s': time_s,
        'achieved_throughput_gops': throughput_gops,
        'efficiency_pct': (throughput_gops / peak_gops * 100) if peak_gops > 0 else 0,
    }
