# Current State — Context for Research Agents (2026-02-26)

## What We Have

### Architecture
- **Ternary SSM model** built from scratch (not using BitNet/Mamba as dependencies)
- 84% weights are ternary {-1, 0, +1} stored as 2-bit packed bitmap with FP16 scales
- 16% weights are FP16 (embedding, output head, normalization)
- SSM layers use Mamba-2 SSD (State Space Duality) with FLA library for training scan
- Hybrid: 7 SSM layers + 1 SWA (Sliding Window Attention) layer at 6M; 16 SSM + 2 SWA at 50M
- A matrix is scalar per head (diagonal), A = -exp(A_log), always negative
- State: h ∈ R^{batch × n_heads × d_state × head_dim}

### Training
- Direct ternary training with STE (Straight-Through Estimator) — no FP16 warmup needed
- Shadow weights in FP32, quantized to ternary each forward pass
- AdamW optimizer (8-bit variant FAILED — Finding 105)
- Activation checkpointing on SSM layers (-22.6% VRAM, +7.5% throughput — Finding 104)
- 50M model trains in ~2.3 hours on Modal L4 GPU

### Inference
- 876.8 tok/s on GPU (CUDA Graph + Triton GEMV + in-place state — Finding 97)
- 105.6 tok/s on CPU (Finding 95)
- Fused SSM recurrence Triton kernel (Finding 98)
- 4.03x speedup from CUDA Graphs alone

### Quality
- **Ternary gap: 7.4% at 50M vs FP16 baseline** (Val PPL 70.77 vs ~65.8)
- Gap is ANGULAR (geometry in weight space), not RADIAL (magnitude)
- Gap narrows with scale: ~1.5% at 200M, ~0.2% at 1B (projected)
- All quality improvement techniques tested in Phases 20-22 were NEUTRAL or NEGATIVE:
  - Anti-windup STE: NEUTRAL
  - Boundary STE: activated but NEUTRAL
  - Sigma-delta quantization: NEGATIVE
  - Low-rank FP16 residual: <0.2% impact
  - Dendritic gating: COSMETIC (learned but didn't change ternary values)
  - Defect-aware initialization: NEUTRAL

### Memory
- Ternary weights: 2 bits/weight via packed bitmap (theoretical min 1.585 bits/weight)
- Activation checkpointing: -22.6% VRAM (Finding 104)
- 8-bit optimizer: FAILED (diverges with ternary STE — Finding 105)
- Codebook compression (24.A): DESCARTADO (2-bit already near optimal)

## What FAILED (Don't Rediscover These)

| Technique | Finding | Result | Why Failed |
|-----------|---------|--------|------------|
| CAGE gradient estimator | F60/C.1 | NaN divergence | Quantization error too large for ternary (0.3-0.5 vs designed 0.02-0.05) |
| DQT stochastic rounding | F61/A.1 | +16.5% worse | 99%+ zero sparsity + high variance → weight patterns frozen |
| HESTIA 2026 | F76/E.1 | +110% worse | Soft quantization scheduling incompatible with ternary QAT |
| QZO (both variants) | E.1 | Worse + 30% slower | Zero-order gradients insufficient |
| Random frozen + alpha | D.1 | Stagnates | Initial convergence then +14% worse by 5000 steps |
| 8-bit AdamW | F105 | Loss explosion | INT8 precision insufficient for STE gradient noise |
| Sigma-delta quantization | F92 | NEGATIVE | Noise shaping doesn't help ternary |
| All Phase 20-22 quality techniques | F85-94 | NEUTRAL/NEGATIVE | Ternary gap is fundamental angular geometry |

## Key Mathematical Insights

1. **Ternary gap is angular:** Weight vectors project onto {-1,0,+1} lattice, losing directional information. This is geometric, not trainable.
2. **Gap disappears with scale:** More parameters → finer lattice granularity → smaller angular error. 7.4% at 50M → ~0.2% at 1B.
3. **STE is essential:** Continuous shadow weights accumulate gradients. All alternatives (stochastic, zero-order, soft scheduling) fail because they break this gradient flow.
4. **A matrix must be scalar per head:** Full A matrix is unstable for ternary. Diagonal (scalar per head) ensures dA ∈ (0,1) stability.

## Hardware

- RTX 5070: 12GB GDDR7, 672 GB/s, FP4 native
- 32GB DDR5 RAM
- Training done on Modal L4 GPUs (cloud)

## The Challenge

We need something BETTER than what we have. Not 5% better — fundamentally better. The three pillars:
1. **Hardware efficiency** — fit 5-10x more in same hardware (MAIN FOCUS)
2. **Speed** — dramatically faster inference AND training
3. **Intelligence** — at least equal to baseline Transformers, ideally better

A paradigm shift is welcome. If the answer is "abandon ternary SSMs and do X instead", that's valid as long as X achieves the three pillars better.
