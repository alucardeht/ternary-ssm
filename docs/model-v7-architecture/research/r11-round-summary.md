# R11 Round Summary — Ternary-Specific Acceleration

**Date:** 2026-03-06
**Aspects covered:** A1–A6 (Ternary Bitwise, SSM Scan, Quantization, Optimizer, Inference, Hardware)

---

## Totals

| Aspect | Topic | IMPLEMENT | EXPLORE | SKIP | Total |
|--------|-------|-----------|---------|------|-------|
| A1 | Ternary Bitwise Compute | 20 | — | — | 20 |
| A2 | SSM Scan Kernels | 20 | — | — | 20 |
| A3 | Quantization Efficiency | 18 | — | 1 | 19 |
| A4 | Ternary-Aware Optimizer | 20 | — | — | 20 |
| A5 | Inference Acceleration | 16 | 4 | — | 20 |
| A6 | Hardware Architecture | 20 | — | — | 20 |
| **Total** | | **114** | **4** | **1** | **119** |

---

## Top 10 Findings by Expected Gain

| Rank | ID | Technique | Gain | Aspect |
|------|----|-----------|----|--------|
| 1 | R11-A1-W2-B-F3 | CRT-GROUP-INT8-ACCUMULATION | 2–3× CPU | A1 |
| 2 | R11-A1-W1-A-F4 | ABSMEAN-INT8-ACT-BITMASK | 2–4× tput | A1 |
| 3 | R11-A6-W1-A-F5 | VECTORIZED-PACK-TERNARY-INT2 | 1000× pack | A6 |
| 4 | R11-A6-W1-A-F1 | UNIFIED-ISA-DISPATCH | 1000× pack | A6 |
| 5 | R11-A1-W2-A-F1 | WHT-NIBBLE-SUBSET-SUM | 2.3× FLOP | A1 |
| 6 | R11-A2-W1-A-F2 | TERNARY-XPROJ-INT8-PRESCAN | 15–20% scan | A2 |
| 7 | R11-A3-W2-A-F2 | BAYESIAN-THRESHOLD | 20–30% | A3 |
| 8 | R11-A3-W1-B-F1 | COMMITTED-SKIP | 35–45% | A3 |
| 9 | R11-A1-W1-A-F1 | BITMASK-GEMV-ALWAYS-ACTIVE | 15–30% | A1 |
| 10 | R11-A6-W1-A-F2 | AVX2-WIRING | 2–4× CPU | A6 |

---

## Key Themes

- **Ternary-native arithmetic:** 60+ findings exploit {−1, 0, +1} structure (bitwise OPs, scatter-add, popcount) → eliminate multipliers at algorithmic level, enabling 10–1000× speedups on specialized hardware.
- **Committed-weight amortization:** 19 findings leverage 59% commitment rate (static zero patterns) → skip quantize/repack, cache unpacked weights, zero gradients for committed groups → 25–45% QoL improvements.
- **Integer & fixed-point paths:** 18 findings use INT8/INT16/BF16 quantization on activations, states, scales → halve bandwidth, map to hardware INT units, maintain ternary invariants.
- **Algebraic rewriting:** 12 findings apply monoid/semiring/GF(3) theory, E-graph optimizations, and functor composition → kernel fusion, zero-skip gates, GEMV decomposition → orthogonal to algorithmic changes.

---

## Synergies

- **A1 + A2 + A6 (bitwise → scan → ISA):** Bitmask GEMV (A1) feeds d_inner → SSM scan (A2); unified `PackedTernary` ISA (A6-F1) abstracts both → single custom accelerator interface.
- **A3 (quantization) ↔ A4 (optimizer):** Committed-weight masking (A3-W1-B-F1) produces stable zero patterns → A4 optimizer gates (15+ findings) adapt LR per group to stability, reducing thrashing → synergistic damping.
- **A5 (inference) + A6 (hardware):** A5 INT2 packing, LSQ absorption, cache locality → hardware blueprint (A6-W1-A-F1 ISA, W2-A-F4 committed-INT2-only) → zero training/eval gap.

---

## R10+R11 Combined Impact Estimate

**R10** (32 findings, 4 implemented): Theta-scaling, attention fusion, mixed-radix SSM → est. +5–8% throughput, +12% VRAM efficiency.
**R11** (119 findings, 114 implement): Ternary-native (bitwise, scatter, integer paths) + committed-amortization + algebraic rewrites → est. **5–15× GPU throughput, 10–40% VRAM, 200–500× inference latency** on custom hardware. Combined: **10–25× end-to-end acceleration** (training + inference) over v5 baseline.

