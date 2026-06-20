# Research Findings — Organized by Mathematical Principle

**Purpose:** All ~142 research findings reorganized by MATHEMATICAL PRINCIPLE instead of by round/technique. This is the basis for principled analysis — understand WHY techniques work, then derive solutions for our specific architecture.

**Created:** 2026-02-27
**Context:** After v3 "lego" failure (25 techniques plugged, 3.5% success rate), this document reorganizes findings to enable proper mathematical analysis.

## Our Architecture (Context for Analysis)

- **Weights:** Ternary {-1, 0, +1} with per-group FP16 scales (group_size=128)
- **Training:** STE (Straight-Through Estimator) with FP32 shadow weights, AdamW
- **Architecture:** Hybrid SSM (87.5%) + Sliding Window Attention (12.5%), 24 layers
- **Current gap:** 7.4% PPL vs FP16 baseline
- **Current speed:** 189.8 tok/s GPU (CUDA Graph), 84.1 tok/s CPU
- **Training speed:** ~10,700 tok/s (50M scale)

## What FAILED in v3 (lego approach)

| Technique | Approach | Result | Why it failed |
|-----------|----------|--------|---------------|
| STEWS ensemble | Plugged as flag | +1.4% loss | STE error is systematic, not random — averaging doesn't help |
| MoTE | Plugged as flag | +2.2% loss | Expert routing adds overhead without quality gain at 6M |
| Angular+Boundary reg | Plugged as flag | +3.4% loss | Regularization too aggressive for training dynamics |
| Deferred Shadow | Plugged as flag | +6.2% loss | Memory savings come at quality cost |
| Full Integration | All flags on | +19.3% loss | Negative interaction between techniques |
| torch.compile | Enabled | -94.9% throughput | SSM sequential scan incompatible |

## What WORKED (and why)

| Technique | Result | Mathematical reason it worked |
|-----------|--------|-------------------------------|
| GaLore | -0.9% loss | Low-rank gradient projection reduces optimizer memory without losing gradient direction quality |
| QAD distillation | -50% loss (synthetic) | Teacher provides soft targets that smooth discrete loss landscape |
| mmap streaming | -99.97% load memory | Pure I/O optimization, no model interaction |
| XOR-Delta | 1.53x compression | Exploits discrete structure (XOR on 2-bit encoding is exact) |
| CUDA Graph fix | 72.5→189.8 tok/s | Bug fix, not technique — graph capture amortized over calls |

## Principle Families (17 identified)

### P1: Gradient Estimation & STE Improvement
**Core problem:** STE provides biased gradients (treats discrete quantization as identity). This bias wastes ~50% of optimization budget.

Findings: F32 (ReSTE), F62 (STE Decomposition), F54 (Natural Gradient), F60 (Second-Order)

**Key mathematical insight:** STE gradient error has two components — BIAS (systematic, from threshold discontinuity) and VARIANCE (stochastic, from discrete rounding). Current STE addresses neither. Techniques that decompose and correct each component separately have highest potential.

**What to derive:** A gradient estimator that (1) corrects bias near thresholds specifically, (2) reduces variance via importance weighting, while (3) maintaining STE's computational simplicity.

### P2: Information Compression & Representation
**Core problem:** Ternary has 1.58 bits/weight but Shannon entropy of useful weight distributions is ~1.4 bits. Current encoding wastes ~0.18 bits.

Findings: F19 (Spectral Error Compensation), F45 (Arithmetic Coding), F25 (Entropy-Coded Bus), F64 (Information Bottleneck)

**Key mathematical insight:** The representation gap isn't just about bits — it's about WHERE information is lost. Spectral analysis shows ternary loses high-frequency weight components. Compensating specifically in the lost spectral band is more efficient than increasing overall precision.

**What to derive:** Minimal spectral corrector that recovers the specific information lost by ternary quantization, without increasing weight precision.

### P3: Sparse Computation & Kernel Optimization
**Core problem:** Ternary multiplication is conditional negation (not arithmetic multiply), but current kernels treat it as standard GEMV.

Findings: F1 (Masked Reduction), F2 (Mega-Kernel), F3 (CUDA Graph), F109 (POPCOUNT), F27 (Circuit Compilation), F6 (ELUT+SIMD)

**Key mathematical insight:** For W ∈ {-1,0,+1}: y = Wx = W_+·x - W_-·x where W_+ and W_- are binary masks. This transforms GEMV into binary masking + accumulation. POPCOUNT on bit-packed weights gives Hamming-weight accumulation in O(1) per 64 elements.

**What to derive:** Unified inference kernel that exploits the add/subtract structure across CPU (SIMD), GPU (integer cores), and edge (WASM) without hardware-specific code.

### P4: Quality Gap Closure (Training)
**Core problem:** 7.4% PPL gap vs FP16. Only 0.9% closed by GaLore.

Findings: F13 (Multi-Codebook K=2), F14 (Tequila Deadzone), F15 (LOTION), F16 (Evolutionary Polish), F17 (Self-Distillation), F29 (QAD), F16 (QAT Scaling Laws)

**Key mathematical insight:** The 7.4% gap decomposes into:
- ~30% representation gap (ternary can't represent f* exactly)
- ~20% optimization gap (STE gradient bias)
- ~40-60% training duration gap (QAT needs 1.5-2x more steps)
- ~10% interaction effects

Each component needs a DIFFERENT solution. Plugging everything at once causes negative interaction.

**What to derive:** Principled decomposition of the 7.4% gap into addressable components, with ONE targeted solution per component, tested individually.

### P5: Hardware Mapping
**Core problem:** Current kernels don't exploit hardware acceleration paths designed for low-precision compute.

Findings: F4 (FP4 Tensor Cores), F5 (INT8 Pipeline), F7 (Compiled Binary), F27 (NPU)

**Key mathematical insight:** Ternary {-1,0,+1} is a strict subset of FP4 E2M1. It's also representable as INT2. Hardware has dedicated paths for FP4 (988 TOPS on RTX 5070) and INT8 (tensor cores, NPUs). Current implementation uses FP16 paths — wasting 4-8x hardware capability.

**What to derive:** Determine optimal hardware precision path (FP4 vs INT8 vs dedicated ternary) per compute stage.

### P6: Ternary-Unique Structural Properties
**Core problem:** Most optimization techniques are designed for continuous weights. Ternary's DISCRETE structure enables things impossible with FP16.

Findings: F29 (Hash Decoding), F30 (Zero-Cost Binary Draft), F31 (GF(3) Algebra), F10 (XOR-Delta), F8 (Codebook Redundancy), F18 (Ensemble Diversity)

**Key mathematical insight:** Ternary weights live in a Galois Field GF(3). This gives ALGEBRAIC properties: exact composition, exact delta encoding, finite group structure for enumeration. FP16 weights live in a continuous approximation space with no such algebraic structure.

**What to derive:** Novel applications of GF(3) properties — exact adapter composition, algebraic weight search, finite neighborhood exploration.

### P7: Distillation & Knowledge Transfer
Findings: F29 (QAD), F18 (Learnable Regularizer), F61 (Manifold Alignment), F17 (Self-Distillation)

### P8: Architectural Capacity
Findings: F20 (Routing Mamba MoE), F14 (FPQ), F56 (Residual Corrector)

### P9: Training Dynamics & Scheduling
Findings: F16 (QAT Scaling Laws), F30 (Cooldown Fusion), F31 (SAM), F34 (Curriculum)

### P10: Memory Hierarchy & Serving
Findings: F21 (Inverted Memory), F22 (LLM OS), F24 (Persistent DAG), F11 (Layer Streaming)

### P11: SSM-Specific Optimization
Findings: F9 (SSD Checkpointing), F12 (Adaptive State Precision), F17 (Hybrid Ratio), F103 (d_state=64)

### P12: Edge & Mobile Deployment
Findings: F26 (WebGPU), F27 (NPU), F28 (Edge Mesh)

### P13: Ensemble & Test-Time Compute
Findings: F18 (Ternary Ensemble), F19 (Test-Time Arbitrage), F101 (STEWS inference)

### P14: Layer-wise & Channel-wise Adaptation
Findings: F33 (Layerwise LR), F59 (QA-BatchNorm), F38 (Meta-Quantization), F55 (Mixed-Precision)

### P15: Optimizer Memory
Findings: F104 (GaLore), F105 (8-bit/offload), F43 (Amortized Decay)

### P16: Compilation & Code Generation
Findings: F107 (Selective torch.compile), F108 (Data Loading)

### P17: Weight Generation & Compression
Findings: F42 (Self-Generating), F47 (Content-Addressed), F48 (Delta-Sigma)

## Analysis Status

- [ ] P1 analyzed by math-paradigm-breaker
- [ ] P2 analyzed by math-paradigm-breaker
- [ ] P3 analyzed by math-paradigm-breaker
- [ ] P4 analyzed by math-paradigm-breaker
- [ ] P5 analyzed by paradigm-breaker
- [ ] P6 analyzed by paradigm-breaker
- [ ] P7-P17 quick assessment pending
