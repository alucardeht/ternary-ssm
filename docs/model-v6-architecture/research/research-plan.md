> **CORREÇÃO 2026-03-02:** Esta pesquisa foi feita INCORRETAMENTE.
> Devia ter sido 3 waves com 2-3 agentes em paralelo cada (research-reader → math-paradigm-breaker → paradigm-breaker).
> Ver: `.workspace/memories/research-workflow-v6.md` para o workflow correto.

# v6 Research Plan — Training Speed Focus

**Date:** 2026-03-02
**Status:** Task 1 Complete - Aspects Defined

---

## 6 Research Aspects (Training Speed Focus)

### Aspect 1: Stateless Energy-Based Training
- FOCO: Replace gradient-based optimizer with energy minimization per batch
- GATE: Energy selection quality, speedup vs Adam, gradient proxy comparison
- WAVE1: "Energy-based training discrete optimization stateless 2025"

### Aspect 2: Ternary Backward Pass Optimization
- FOCO: CUDA kernels fusing STE into backward pass
- GATE: Kernel fusion speedup, backward vs current PyTorch, memory bandwidth
- WAVE1: "CUDA kernel fusion STE backward pass 2025"

### Aspect 3: Commitment-Driven Progressive Freezing
- FOCO: Automatic layer-wise commitment, skip backward for frozen
- GATE: Optimal threshold, freeze prediction, quality degradation
- WAVE1: "Progressive layer freezing neural network training"

### Aspect 4: Data Pipeline Parallelization
- FOCO: Prefetch + async CPU-GPU transfer overlap
- GATE: Optimal prefetch depth, pinned memory, tokenization bottleneck
- WAVE1: "DataLoader prefetch async GPU transfer 2025"

### Aspect 5: Sign-Only Optimizer
- FOCO: Sign-accumulator instead of full momentum
- GATE: Convergence vs Adam, optimal window, integer-only arithmetic
- WAVE1: "Sign SGD optimizer ternary weight training"

### Aspect 6: Hybrid Precision Strategies
- FOCO: Layer-wise precision (FP16 for embeddings, ternary for SSM)
- GATE: Which layers FP16, precision per layer type, dynamic adjustment
- WAVE1: "Mixed precision layer-wise training optimization"

---

## Priority (Expected Impact)

| Aspect | Speedup | Memory | Priority |
|--------|---------|--------|----------|
| 5: Sign Optimizer | 3.3× | 12× | P0 |
| 3: Layer Freezing | 6× | 0× | P0 |
| 1: Energy Training | 3.3× | 28× | P1 |
| 2: Backward Optimization | 2-3× | 0× | P1 |
| 4: Data Pipeline | 1.2-1.5× | 0× | P2 |
| 6: Hybrid Precision | 1.5-2× | 2× | P2 |

**Target:** 10-15× speedup over v5

---

## Round 2 Deep Dives

### Aspect 1: Energy-Based Optimizer
- Threshold formula: Δ = α × mean(|w|), α ∈ [0.7, 0.8]
- Reference: TWN (Li 2016), TTQ (Zhu 2016), GAQAT (2024)

### Aspect 2: Ternary Backward
- PyTorch autograd.Function pattern for custom backward
- STE backward = identity (grad_output passed through)
- No Triton backward fusion available

### Aspect 3: Progressive Freezing
- AutoFreeze: n_ell = ||ḡ(t) - ḡ(t-1)||_2 / ||ḡ(t)||_2
- Optimal threshold: 50th percentile (median)
- SSM: freeze A-matrix early (preserves memory structure)

### Aspect 4: Data Pipeline
- Pool-based BatchSampler: random chunks, sort each chunk
- Dynamic padding: torch.nn.utils.rnn.pad_sequence
- Gradient accumulation for memory efficiency

### Aspect 5: Sign-Only Optimizer
- SignSGD formula: m_t = β * m_{t-1} + sign(g_t)
- Memory: same as SGD momentum (1 buffer per tensor)
- Custom optimizer implementation needed

### Aspect 6: Hybrid Precision
- PyTorch autocast + GradScaler
- custom_fwd/custom_bwd for custom ops
- FP16 for matmul, FP32 for softmax/loss
