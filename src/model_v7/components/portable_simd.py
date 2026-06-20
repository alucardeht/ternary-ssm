"""
Portable SIMD kernel abstraction for ternary matmul.

Findings R4-A5-F6 (Highway), R4-A5-F7 (xsimd), R4-A5-F3 (auto-vectorization):
Documents the algorithm for portable C kernel implementation across multiple backends.
Provides reference implementations and diagnostic tools. Actual execution remains in
Python/PyTorch; this documents future C compilation targets.

Theory: Ternary matmul (add/subtract only) maps naturally to SIMD. Highway and xsimd
provide platform-independent abstractions that auto-compile to x86, ARM, RISC-V, and
WASM. Goal: single source code -> all platforms.

For ternary weights: zero_mask guides conditional add/subtract. Cache-friendly SoA layout
enables vectorization. Portable algorithms avoid intrinsics, rely on compiler auto-vec.
"""

import math
import torch


class PortableKernelSpec:
    """Specification for portable ternary kernel (platform-independent).

    Captures the algorithm in platform-independent form. Can generate pseudocode or
    validate against reference implementation. Used for planning C implementation.

    Attributes:
        name: Kernel name (e.g., 'ternary_gemv')
        description: Brief description
        loop_structure: Loop ordering (AP, OP, AP+OP)
        memory_layout: 'soa' (Structure-of-Arrays) or 'aos' (Array-of-Structures)
        alignment: Required alignment in bytes (usually 64)
        vectorization_hints: List of compiler hints for auto-vectorization
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.loop_structure = None
        self.memory_layout = None
        self.alignment = 64
        self.vectorization_hints = []

    def add_vectorization_hint(self, hint: str):
        """Record vectorization requirement.

        Args:
            hint: Hint string (e.g., 'restrict', 'aligned_loads', 'no_loop_carried_deps')
        """
        self.vectorization_hints.append(hint)


def reference_ternary_gemv(x: torch.Tensor, w_signs: torch.Tensor,
                            w_zero_mask: torch.Tensor,
                            scales: torch.Tensor) -> torch.Tensor:
    """Reference implementation of ternary GEMV using SoA layout.

    This is the PORTABLE algorithm that maps to:
    - Highway: HWY_ATTR void ternary_gemv(...)
    - xsimd: batch<int16_t> acc = ...
    - C auto-vectorizable: for (int i = 0; i < n; i++) acc += sign[i] * x[i]
    - WASM: v128_t acc = wasm_i16x8_add(...)

    Args:
        x: (batch, dim) or (dim,) activations
        w_signs: (out_dim, dim) int8 signs {-1, +1} (zero where weight=0)
        w_zero_mask: (out_dim, ceil(dim/8)) uint8 bitmask (1=nonzero)
        scales: (out_dim,) per-row scales

    Returns:
        (batch, out_dim) or (out_dim,) output tensor
    """
    has_batch = x.dim() > 1
    if has_batch:
        batch_size = x.shape[0]
    else:
        batch_size = 1
        x = x.unsqueeze(0)

    out_dim, dim = w_signs.shape
    output = torch.zeros((batch_size, out_dim), dtype=x.dtype, device=x.device)

    for batch_idx in range(batch_size):
        for out_idx in range(out_dim):
            acc = 0.0

            for in_idx in range(dim):
                byte_idx = in_idx // 8
                bit_idx = in_idx % 8

                if byte_idx < w_zero_mask.shape[1]:
                    is_nonzero = (w_zero_mask[out_idx, byte_idx].item() >> bit_idx) & 1
                    if is_nonzero:
                        sign = w_signs[out_idx, in_idx].item()
                        activation = x[batch_idx, in_idx].item()
                        if sign == 1:
                            acc = acc + activation
                        else:
                            acc = acc - activation

            scale = scales[out_idx].item()
            output[batch_idx, out_idx] = acc * scale

    return output.squeeze(0) if not has_batch else output


def verify_vectorization_opportunity(weight_shape: tuple) -> dict:
    """Analyze weight matrix shape for SIMD vectorization potential.

    Reports optimal tile sizes, required padding, expected speedups, and
    recommended loop structure. Helps inform C kernel design.

    Args:
        weight_shape: (out_dim, in_dim) tensor shape

    Returns:
        dict with vectorization analysis
    """
    out_dim, in_dim = weight_shape

    simd_widths = {
        'avx2_i8': 32,
        'avx512_i8': 64,
        'neon_i8': 16,
        'rvv_i8': (128, 2048),
        'wasm_i8': 16,
    }

    analysis = {
        'out_dim': out_dim,
        'in_dim': in_dim,
        'total_ops': out_dim * in_dim,
    }

    for arch, width in simd_widths.items():
        if isinstance(width, tuple):
            width = width[0]
        vf = width
        analysis[f'{arch}_vector_factor'] = width
        analysis[f'{arch}_ops_per_cycle'] = width // 8

    l1_cache_kb = 32
    l1_elements = (l1_cache_kb * 1024) // 2
    optimal_tile_n = min(in_dim, l1_elements // out_dim)
    analysis['optimal_tile_n_for_l1'] = optimal_tile_n
    analysis['optimal_tile_m'] = min(out_dim, 32)

    if out_dim > 2 * in_dim:
        analysis['recommended_dataflow'] = 'AP'
    elif in_dim > 2 * out_dim:
        analysis['recommended_dataflow'] = 'OP'
    else:
        analysis['recommended_dataflow'] = 'OP'

    return analysis


def estimate_cross_platform_throughput(weight_shape: tuple,
                                        architectures: list = None) -> dict:
    """Estimate throughput on various architectures.

    Uses simplified roofline model: throughput = min(compute-bound, memory-bound).

    Architectures: x86-AVX2 (16 i8 ops/cycle), ARM-NEON (8 i8/cycle),
                   RISC-V-RVV (4-32 i8/cycle depending on VLEN),
                   WASM (8 i8/cycle), CPU scalar (1 i8/cycle)

    Args:
        weight_shape: (out_dim, in_dim)
        architectures: List of architecture names. If None, estimate all major ones.

    Returns:
        dict mapping architecture -> estimated throughput (elements/cycle)
    """
    if architectures is None:
        architectures = ['avx2', 'arm_neon', 'rvv_256', 'wasm', 'scalar']

    out_dim, in_dim = weight_shape

    arch_specs = {
        'avx2': {
            'simd_width': 32,
            'i8_ops_per_cycle': 16,
            'memory_bw_bytes_per_cycle': 8,
        },
        'arm_neon': {
            'simd_width': 16,
            'i8_ops_per_cycle': 8,
            'memory_bw_bytes_per_cycle': 4,
        },
        'rvv_256': {
            'simd_width': 256,
            'i8_ops_per_cycle': 32,
            'memory_bw_bytes_per_cycle': 8,
        },
        'wasm': {
            'simd_width': 128,
            'i8_ops_per_cycle': 8,
            'memory_bw_bytes_per_cycle': 2,
        },
        'scalar': {
            'simd_width': 1,
            'i8_ops_per_cycle': 1,
            'memory_bw_bytes_per_cycle': 1,
        },
    }

    results = {}
    for arch in architectures:
        if arch not in arch_specs:
            continue

        spec = arch_specs[arch]
        compute_throughput = spec['i8_ops_per_cycle']

        bytes_per_output = in_dim + 1
        memory_throughput = spec['memory_bw_bytes_per_cycle'] // bytes_per_output

        throughput = min(compute_throughput, memory_throughput)
        results[arch] = {
            'compute_throughput': compute_throughput,
            'memory_throughput': memory_throughput,
            'estimated_elements_per_cycle': throughput,
            'estimated_ops_per_cycle': throughput * in_dim,
        }

    return results


def recommend_loop_structure(weight_shape: tuple) -> str:
    """Recommend optimal loop structure (AP vs OP) based on shape.

    AP (Activation-Persistent): Keep input x in registers, stream W.
      - Best for tall W (many outputs, few inputs): out_dim >> in_dim
      - Example: 768 x 256 (proj-out)

    OP (Output-Persistent): Keep accumulators in registers, stream both x and W.
      - Best for wide W (few outputs, many inputs): in_dim >> out_dim
      - Example: 256 x 768 (proj-in)

    Args:
        weight_shape: (out_dim, in_dim)

    Returns:
        'AP' or 'OP'
    """
    out_dim, in_dim = weight_shape
    ratio = out_dim / in_dim

    if ratio > 2.0:
        return 'AP'
    elif ratio < 0.5:
        return 'OP'
    else:
        return 'OP'


class CKernelGenerator:
    """Generate C source code targeting different SIMD backends.

    Produces portable C that auto-vectorizes with GCC/Clang flags.
    Separate backends for Highway, xsimd, and plain C with compiler hints.

    This is a planning tool - actual compilation happens externally.
    """

    @staticmethod
    def generate_c_reference() -> str:
        """Generate plain C reference implementation (auto-vectorizable).

        Returns:
            C source code as string
        """
        return """
#include <stdint.h>
#include <string.h>

void ternary_gemv_reference(
    const float *restrict x,      /* (in_dim,) input */
    const int8_t *restrict w_signs,   /* (out_dim, in_dim) weight signs */
    const uint8_t *restrict w_mask,   /* (out_dim, ceil(in_dim/8)) zero mask */
    float *restrict y,            /* (out_dim,) output */
    const float *restrict scales, /* (out_dim,) scales */
    int out_dim, int in_dim
) {
    /* Compiler hints for auto-vectorization */
    #pragma omp simd collapse(2)
    for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
        float acc = 0.0f;

        for (int in_idx = 0; in_idx < in_dim; ++in_idx) {
            int byte_idx = in_idx >> 3;
            int bit_idx = in_idx & 7;
            int is_nonzero = (w_mask[out_idx * ((in_dim + 7) >> 3) + byte_idx]
                              >> bit_idx) & 1;
            int sign = w_signs[out_idx * in_dim + in_idx];

            /* Branchless multiply: sign * x[in_idx] only if nonzero */
            acc += (float)(is_nonzero * sign) * x[in_idx];
        }

        y[out_idx] = acc * scales[out_idx];
    }
}
"""

    @staticmethod
    def generate_highway_kernel() -> str:
        """Generate Highway (portable SIMD) implementation.

        Returns:
            C++ source using google/highway
        """
        return """
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

HWY_ATTR void ternary_gemv_highway(
    const float *x,
    const int8_t *w_signs,
    const uint8_t *w_mask,
    float *y,
    const float *scales,
    int out_dim, int in_dim
) {
    const hn::ScalableTag<int8_t> di8;
    const hn::ScalableTag<float> df;
    const size_t N8 = hn::Lanes(di8);

    for (int out_idx = 0; out_idx < out_dim; ++out_idx) {
        auto acc_v = hn::ZeroExtendVector(df, hn::Zero(di8));

        for (int in_idx = 0; in_idx < in_dim; in_idx += N8) {
            auto signs = hn::Load(di8, w_signs + out_idx * in_dim + in_idx);
            auto x_vals = hn::LoadU(df, x + in_idx);

            /* Ternary: conditional add/subtract based on sign */
            auto products = hn::Mul(hn::ConvertTo(df, signs), x_vals);
            acc_v = hn::Add(acc_v, products);
        }

        float acc_scalar = hn::ReduceSum(df, acc_v);
        y[out_idx] = acc_scalar * scales[out_idx];
    }
}
"""
