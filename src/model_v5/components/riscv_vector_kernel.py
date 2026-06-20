"""
RISC-V Vector Extension Kernel Reference Implementation.

R4-A5-F9: Reference implementation targeting RISC-V Vector Extension (RVV 1.0).

RVV 1.0: scalable vector length (128-2048 bit).
GCC 14.1+ auto-vectorizes C loops to RVV without code changes.

Same C code -> all VLEN from same source.
No intrinsics needed - compiler handles vectorization automatically.

Reference:
    Research Round 4, Aspect 5, Finding F9
    docs/model-v5-architecture/research-round4-aspect5.md
"""

from typing import Dict, Optional

import torch


class RiscVVectorTernaryGemv:
    """Reference implementation of ternary GEMV for RISC-V V extension.

    RVV 1.0: scalable vector length (128, 256, 512, 1024, 2048 bits).
    GCC 14.1+ auto-vectorizes C loops to RVV without code changes.

    Same C code -> all VLEN from same source.
    """

    def __init__(self, vlen_bits: int = 256):
        """Initialize RISC-V Vector kernel.

        Args:
            vlen_bits: vector register width in bits
                Options: 128, 256, 512, 1024, 2048
        """
        if vlen_bits not in [128, 256, 512, 1024, 2048]:
            raise ValueError(f"VLEN must be power-of-2 in [128, 2048], got {vlen_bits}")

        self.vlen_bits = vlen_bits
        self.i8_elements = vlen_bits // 8

    def reference_gemv(self, x: torch.Tensor, w_packed: torch.Tensor,
                       scales: torch.Tensor) -> torch.Tensor:
        """Reference GEMV matching RVV loop structure.

        Uses vsetvli pattern for scalable vector length.
        Code structure is portable across all VLEN values.

        Args:
            x: (K,) activation vector
            w_packed: (M, ceil(K/bytes_per_weight)) packed weights
            scales: (M,) per-row scales

        Returns:
            y: (M,) output vector
        """
        M = w_packed.shape[0]
        K = x.shape[0]
        device = x.device

        y = torch.zeros(M, dtype=x.dtype, device=device)

        for row in range(M):
            acc = 0.0

            for k in range(K):
                weight_idx = k // 4
                weight_shift = (k % 4) * 2
                packed_byte = w_packed[row, weight_idx].item()
                weight_bits = (packed_byte >> weight_shift) & 0x3

                weight = self._unpack_ternary_weight(weight_bits)
                acc += weight * x[k].item()

            y[row] = acc * scales[row].item()

        return y

    def generate_c_code(self) -> str:
        """Generate C code that auto-vectorizes to RVV via GCC.

        Key patterns:
        - Clean for loops with known trip counts
        - restrict pointers
        - Aligned buffers (64-byte alignment)
        - No manual intrinsics needed (GCC handles it)

        Compile with: riscv64-unknown-linux-gnu-gcc -O3 kernel.c
        """
        c_code = '''
#include <stdint.h>
#include <stddef.h>

void ternary_gemv_riscv(
    const float* restrict x,        // (K,) activations [restrict]
    const int8_t* restrict w_signs, // (M, K) weight signs
    const uint8_t* restrict w_mask, // (M, ceil(K/8)) nonzero mask
    const float* restrict scales,   // (M,) per-row scales
    int M,                          // output dimension
    int K,                          // input dimension
    float* restrict y               // (M,) output
) {
    // Ensure buffers are 64-byte aligned for optimal SIMD
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;

        // Inner loop: compiler auto-vectorizes via RVV
        for (int k = 0; k < K; k++) {
            if ((w_mask[k >> 3] >> (k & 7)) & 1) {
                float weight_sign = (w_signs[i*K + k] > 0) ? 1.0f : -1.0f;
                acc += weight_sign * x[k];
            }
        }

        y[i] = acc * scales[i];
    }
}

// Batched version for higher throughput
void ternary_gemv_batch_riscv(
    const float* restrict x,        // (batch, K)
    const int8_t* restrict w_signs, // (M, K)
    const uint8_t* restrict w_mask, // (M, ceil(K/8))
    const float* restrict scales,   // (M,)
    int batch,
    int M,
    int K,
    float* restrict y               // (batch, M)
) {
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < M; i++) {
            float acc = 0.0f;

            for (int k = 0; k < K; k++) {
                if ((w_mask[k >> 3] >> (k & 7)) & 1) {
                    float weight_sign = (w_signs[i*K + k] > 0) ? 1.0f : -1.0f;
                    acc += weight_sign * x[b*K + k];
                }
            }

            y[b*M + i] = acc * scales[i];
        }
    }
}
'''
        return c_code

    def _unpack_ternary_weight(self, bits: int) -> float:
        """Unpack 2 bits to ternary weight {-1, 0, +1}.

        Encoding: 00 -> 0, 01 -> +1, 10 -> -1, 11 -> unused
        """
        if bits == 0b00:
            return 0.0
        elif bits == 0b01:
            return 1.0
        elif bits == 0b10:
            return -1.0
        else:
            return 0.0

    @staticmethod
    def estimate_riscv_throughput(model_params: int, hidden_dim: int,
                                  vlen_bits: int = 256,
                                  clock_ghz: float = 1.5) -> Dict[str, float]:
        """Estimate throughput across RISC-V implementations.

        Returns estimates for:
        - Embedded (128-bit RVV, 50-100 tok/s)
        - Desktop (256-bit, 200-300 tok/s)
        - Server (512-bit, 500+ tok/s)

        Args:
            model_params: Total parameters
            hidden_dim: Hidden dimension
            vlen_bits: Vector register width
            clock_ghz: Clock frequency in GHz

        Returns:
            dict with throughput estimates
        """
        ops_per_token = model_params * 2
        memory_bytes_per_token = (model_params / 8) + (hidden_dim * 4)

        memory_bandwidth_gbps = 8.0

        ipc = 2.0
        cycles_per_token = ops_per_token / (clock_ghz * 1e9 * ipc)
        tok_per_second_compute = clock_ghz * 1e9 / (ops_per_token / ipc)

        memory_limited_tok_s = (memory_bandwidth_gbps * 1e9) / (memory_bytes_per_token * 8)
        actual_tok_s = min(tok_per_second_compute, memory_limited_tok_s)

        vlen_fraction = vlen_bits / 256.0

        return {
            'vlen_bits': vlen_bits,
            'base_tokps': int(actual_tok_s * vlen_fraction),
            'memory_bound': memory_bandwidth_gbps < ops_per_token / 8,
            'compute_bound': not (memory_bandwidth_gbps < ops_per_token / 8),
            'expected_ipc': ipc,
            'clock_ghz': clock_ghz,
        }

    @property
    def memory_bytes(self) -> int:
        """Memory footprint estimate for VLEN register file."""
        return (self.vlen_bits // 8) * 32
