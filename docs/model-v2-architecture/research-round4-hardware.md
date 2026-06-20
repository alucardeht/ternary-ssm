# Research Round 4 — Architecture & Hardware Co-design

## Overview

This document tracks research findings on hardware co-design, custom accelerators, and deployment strategies for ternary SSM models discovered in Research Cycle 4 (Feb 2026).

**DEDUPLICATION NOTE:** Findings that overlap with Round 3 (research-round3-velocity.md) or Round 4 Aspects 1-3 are EXCLUDED. TOM, TerEffic, T-MAC, BitNet 2B4T, Sherry already documented.

## NEW FINDINGS (Round 4, Aspect 4)

### Wave 1: Web Research (2 researchers, Feb 2026)

#### HARDWARE ACCELERATORS (NEW only)

**F1: TCN-CUTIE — 22nm Fabricated Ternary ASIC** (MEDIUM)
- Fabricated silicon (not just design): 1036 TOp/s/W peak, 12.2mW at 0.5V
- RISC-V SoC integration, 22nm FDX technology
- Validates extreme energy efficiency of ternary in real silicon
- Relevance: proves ternary ASIC viability with measured numbers, not projections

**F2: Sparsity-Aware ROM Density Analysis** (HIGH)
- At 70% zero-bit ratio: 15.0 MB/mm² (5.2x denser than standard ROM)
- At 95% sparsity: 25.3 MB/mm²
- Our 50M model = ~6.25MB → fits in <0.5mm² of sparsity-aware ROM
- Logic-based synthesis: zero-value bits tied to ground = zero area cost
- Common sub-expression elimination further reduces gates
- Game-changer: entire model on-chip in tiny silicon area

**F3: Hybrid ROM-SRAM Architecture** (MEDIUM-HIGH)
- Base model in ROM (immutable, dense), adapters in SRAM (tunable)
- Ternary QLoRA adapters reuse same compute units
- Solves "ROM can't update" — base rarely changes, apps change frequently
- On-device personalization without rewriting ROM

#### COMPILATION & OPTIMIZATION (NEW)

**F4: PT2-LLM Post-Training Ternarization** (HIGH)
- Training-FREE ternary quantization competitive with 2-bit PTQ
- Asymmetric Ternary Quantizer (ATQ) with Iterative Ternary Fitting (ITF)
- Activation-aware Grid Alignment (AGA): align output, not weights
- Structural Similarity Reordering (SSR): cluster similar columns → less variance
- LLaMA-7B: 11.39 PPL vs 14.58 for 2-bit Slim-LLM
- Relevance: SSR column reordering directly applicable to our weight matrices

**F5: OpenVINO BitNet b1.58 Support** (MEDIUM)
- Intel OpenVINO 2025.4 adds experimental BitNet support
- Hugging Face integration for ternary models
- Targets Intel NPUs and CPUs (our Intel 225F has 13 TOPS NPU)
- Validates ternary as production-ready with commercial tooling

**F6: Spectra 1.1 — Ternary Scaling Laws** (HIGH)
- Establishes scaling laws SPECIFICALLY for ternary LMs
- Ternary follows predictable compute-optimal scaling
- Aligns with our Ternary Gap Theorem (gap closes with scale)
- Provides theoretical foundation for 200M/1B scaling plans
- Efficient inference techniques and architectural guidelines

**F7: SSM Kernel Characterization Study** (MEDIUM)
- Analyzes SSM kernel performance with long context on GPUs
- O(n) memory vs Transformer O(n²) confirmed at kernel level
- Kernel fusion opportunities for selective scan operations
- Hybrid SSM-Transformer performance profiling

#### HARDWARE-AWARE TRAINING (NEW)

**F8: Noise-Injection Training for Analog Hardware** (HIGH)
- Inject 3-5% Gaussian noise to weights during forward pass
- Makes network robust to hardware imperfections (PCM/RRAM)
- Weight clipping: [-α·σ_W, +α·σ_W], α=2.0-2.5
- ResNet-32 CIFAR-10: <0.2% accuracy drop after PCM transfer
- MANDATORY for any analog/memristive hardware deployment
- Simple to implement: add noise during forward, not backward

**F9: Adaptive Batch Norm for Hardware Drift Compensation** (MEDIUM)
- Periodically recalibrate BN parameters to correct activation drift
- Compensates for memristor conductance changes over time
- Maintains >93.5% accuracy over 1 day vs 92.6% without
- Digital-only computation (no hardware rewrite needed)

**F10: 3:4 Structured Sparsity Training (Sherry detail)** (HIGH)
- Arenas mechanism: annealing residual synapses prevent weight trapping
- 3:4 pattern (3 nonzero per 4) → 1.25-bit packing (power-of-two aligned)
- 25% bit savings + 10% CPU speedup vs standard ternary
- Our model has ~40% natural sparsity → Sherry regularizes to exact 25% zeros
- Addresses known ternary training collapse (weight trapping) issue

#### DEPLOYMENT & EDGE (NEW)

**F11: RRAM CIM Scaling Analysis** (MEDIUM)
- Ternary more resilient to wire parasitics than binary in analog CIM
- BUT 40% energy penalty vs binary due to ADC complexity (3 states vs 2)
- Variable step-size ADC + calibration reduces ADC requirements
- Key: digital GPU deployment has NO analog penalty — we benefit from ternary expressiveness

**F12: Ternary-Input Binary-Weight CNN Accelerator** (MEDIUM)
- Inverse of our approach: ternary activations + binary weights
- Validates mixed-precision ternary as viable
- Could explore ternary activations for FFN outputs (8x activation memory reduction)
- Dataflow insights for mixed-precision matmul transferable

**F13: STePs — Non-Learnable Ternary Filters** (LOW)
- Static ternary filters from Local Binary Patterns (Haar features)
- 40-80% trainable parameter reduction
- Different paradigm (static vs learned) — not applicable to SSM dynamics
- Interesting comparison: our learned ternary vs static ternary patterns

#### KEY PATTERNS (Wave 1)
- Sparsity-aware ROM (F2) enables sub-mm² on-chip storage for 50M models
- Post-training ternarization (F4) now competitive — from-scratch training not always needed
- Ternary scaling laws (F6) validated — our gap theorem confirmed independently
- Hardware training (F8) simple: add 3-5% noise during forward pass
- Structured sparsity (F10) solves bit-alignment + training collapse simultaneously
- Digital GPU > analog CIM for ternary (F11) — no conversion overhead
- Commercial tooling emerging (F5) — ternary going mainstream

---

### Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

#### TERNARY COMPUTE PRIMITIVE FORMULATIONS

| # | Technique | Speedup | Viability |
|---|-----------|---------|-----------|
| M1 | Binary Rank Factorization (BRF) | 2-5x | LOW-MEDIUM |
| M2 | Alpha-Beta Early Termination (ABET) | 1.5-5x | MEDIUM-HIGH |
| M3 | Inclusion-Exclusion Basis (IEBC) | 2-4x | MEDIUM |
| M4 | Temporal Delta Accumulation (TDA) | 5-20x | HIGH |
| M5 | Fixed-Point Domain Accumulation (FPDA) | 1.5-3x CPU | MEDIUM-HIGH |
| M6 | Entropy-Coded Weight Streaming (ECWS) | 1.2x | MEDIUM-HIGH |
| M7 | Activation-Manifold Centroids (AMCP) | 2-10x | LOW-MEDIUM |
| M8 | Holographic Reduced Representations | Unknown | SPECULATIVE |

**M4: Temporal Delta Accumulation (TDA)** — TOP RECOMMENDATION
- y(t) = y(t-1) + W*(x(t) - x(t-1)): incremental GEMV via linearity
- SSM guarantees smooth state evolution → delta is SPARSE (10-20% nonzero entries)
- Sort columns by |x_j| descending → process high-impact first
- Periodic full recomputation prevents error accumulation
- 5-20x per token depending on delta sparsity
- Composes with ALL other speedup techniques (TDA on delta, then WPH/SCS on sparse)
- Uniquely suited to autoregressive SSM inference (temporal correlation)

**M2: Alpha-Beta Early Termination (ABET)**
- Process columns by |x_j| descending, maintain running bounds per row
- Exit row when |partial_sum| >> remaining_potential (bound guaranteed)
- For gating outputs (sigmoid): sign is sufficient → exit after ~20% of columns
- First ~20% columns account for ~80% output magnitude (heavy-tailed activations)
- 1.5-5x depending on layer type (gating layers benefit most)

**M5: Fixed-Point Domain Accumulation (FPDA)** — CPU-specific
- Convert x to INT32 before accumulation: ternary add/sub becomes pure integer
- On AVX-512: INT16 = 32 lanes vs FP32 = 16 lanes (2x SIMD width)
- INT add latency 1 cycle vs FP add 3-4 cycles
- Net: up to 6x faster inner loop on x86 CPUs
- ARM NEON: FP16 has more lanes than INT32, so FPDA loses there

**M6: Entropy-Coded Weight Streaming (ECWS)**
- Block-5 encoding: 5 trits in 8 bits (3^5=243 ≤ 256)
- 1.6 bits/entry vs 2.0 bits/entry standard = 20% bandwidth reduction
- LUT decode: 243 entries, fits L1 cache, ~0.2 cycles/entry overhead
- GEMV is memory-bandwidth-bound → 20% less bandwidth ≈ 20% speedup

**M3: Inclusion-Exclusion Basis Compression (IEBC)**
- Decompose all row masks into boolean combinations of t basis sets
- Precompute Sigma_S = sum over intersections via Mobius transform: O(t*2^t)
- Each row evaluated as signed sum of precomputed values: O(2^t) per row
- For t=10: total O(M*1024) vs O(M*768) ≈ similar. Viable only if t≤8.

#### SSM-HARDWARE CO-OPTIMIZATION FORMULATIONS

| # | Technique | Impact | Viability |
|---|-----------|--------|-----------|
| M9 | Block-Circulant Ternary (BCT) | 10.7x | MEDIUM |
| M10 | Block Floating-Point (BFP-k) | 2-8x | HIGH |
| M11 | Hamiltonian Co-Training (HHC) | Variable | MEDIUM-HIGH |
| M12 | Trellis-Coded Weights (TCW) | 1.5-2x | MEDIUM |
| M13 | Spectral Sparsity Training (SST) | 13-18x | MEDIUM |
| M14 | Locality-Sensitive Row Factorization (LSRF) | 5-15x | HIGH |
| M15 | Dyadic-Rational SSM Ring (DRSR) | 2-4x | HIGH |

**M14: LSRF** — TOP RECOMMENDATION (ZERO quality loss)
- Cluster rows by Hamming distance, compute shared centroid GEMV + sparse corrections
- EXACT computation — zero quality impact
- For K=32 clusters, s_avg=0.1: 7.1x speedup
- Combines MULTIPLICATIVELY with SCS/RTDC: estimated 21x combined
- Post-training optimization: no training changes needed
- Measurable immediately on existing trained model

**M10: Block Floating-Point (BFP-k)** — Full Integer Pipeline
- Groups of g activations share exponent; mantissas become INT-k
- BFP-8: ~0.5% PPL loss, 2x SIMD throughput
- On RTX 5070 tensor cores: INT8 = 988 TOPS vs FP16 = 123 TFLOPS = 8x peak
- Combined with ternary: fully integer pipeline (no FP in entire inference)
- Enables Intel NPU (13 TOPS INT8, currently idle)

**M15: Dyadic-Rational SSM Ring (DRSR)** — Integer-Only SSM
- Constrain A entries to Z[1/2] = {p/2^q}: a_i = 1 - 2^{-k}
- State update: h = h - (h >> k) + ternary_select(x) — ZERO multiplications
- Progressive training: quantize A from q=4 to q=8 during training
- Enables deployment on integer-only hardware (MCU, DSP, NPU, FPGA)
- Related to M6/Discretized Decay from Round 3, but with ring-theoretic foundation

**M9: Block-Circulant Ternary (BCT)**
- Constrain W to block-circulant: each block defined by first row
- GEMV becomes cyclic convolution → NTT in O(n*log(b)/b)
- For b=64: 10.7x speedup. NTT is integer-only (modular arithmetic)
- Quality concern: circulant constraint may cost 3-8% PPL

**M13: Spectral Sparsity Training (SST)**
- Regularize weights for sparse DFT representation
- GEMV in frequency domain: O(n*log(n) + m*K) for K-sparse spectra
- For K=32: 18.3x speedup
- Ternary vectors naturally have few sign transitions → natural spectral sparsity
- Quality cost uncertain: 1-5% PPL depending on lambda

**M11: Hamiltonian Co-Training (HHC)** — Novel Training Paradigm
- Training as Hamiltonian dynamics: H = Loss + lambda*HW_cost is CONSERVED
- Symplectic integrator guarantees bounded quality-hardware tradeoff
- HW cost function: block pattern entropy + row Hamming distance + sparsity variance
- Produces hardware-friendly weight patterns by construction

**BONUS: Information-Theoretic Speed Limit**
- 50M model on RTX 5070 (672 GB/s): T_max = 53,760 tok/s (2-bit encoding)
- Current measured: 876.8 tok/s = 1.6% of theoretical maximum
- Gap from: FP16 activation traffic, kernel overhead, Python loops, non-weight memory
- With BFP-8 activations: activation traffic negligible at 50M scale

#### KEY PATTERNS (Wave 2)
- TDA (M4) exploits temporal correlation unique to autoregressive inference — highest-value single technique
- LSRF (M14) provides 5-15x with ZERO quality loss — purely computational
- BFP-k (M10) + DRSR (M15) together eliminate ALL floating-point from inference
- "Full Integer Stack": LSRF + BFP + DRSR = 15-45x, approaching theoretical limit
- Current engine at 1.6% of bandwidth limit — massive room for optimization

---

### Wave 3: Paradigm Breakers (2 agents, Feb 2026)

#### RADICAL HARDWARE APPROACHES

| # | Technique | Key Insight | Viability |
|---|-----------|-------------|-----------|
| P1 | Photonic Phase-Encoded Inference | Light interference computes ternary GEMV | SPECULATIVE |
| P2 | Event-Driven Neuromorphic SSM | SSM recurrence = LIF neuron | MEDIUM |
| P3 | HDC Subsystem Replacement | POPCOUNT-based LM head, router, cache | HIGH |
| P4 | Stochastic Bit-Stream Computing | MUX/NOT/wire circuit, precision knob | MEDIUM |

**P1: Photonic Phase-Encoded Ternary Inference** — SPECULATIVE
- Encode ternary weights as SLM phase states: {-1, 0, +1} → {π, blocked, 0}
- Light interference naturally computes dot products (additions = free)
- One SLM pass = one GEMV. Speed-of-light computation.
- Ternary is IDEAL for photonics: only 3 states needed (binary needs continuous modulation)
- Challenge: SLM refresh rate limits throughput to ~100 kHz (still fast for per-token)
- Status: lab demonstrations exist for binary; ternary is natural extension

**P2: Event-Driven Neuromorphic SSM** — MEDIUM VIABILITY
- Mathematical equivalence: SSM diagonal recurrence h[i] = a[i]*h[i] + b[i]*x IS a leaky integrate-and-fire (LIF) neuron
- Map directly to neuromorphic hardware (Intel Loihi 2, SpiNNaker 2)
- Ternary projections = spike routing tables (hardwired, no computation)
- Power: <1W for entire model (neuromorphic chips designed for this)
- Challenge: batch inference on neuromorphic hardware is wasteful (designed for event-driven)
- Best for: always-on edge devices, continuous inference, IoT

**P3: HDC Subsystem Replacement** — HIGH PRIORITY
- Replace expensive subsystems with Hyperdimensional Computing (HDC) using ternary vectors:
  - LM head: bind token embedding with position → POPCOUNT similarity → top-k (eliminates 32K logit computation)
  - Router: HDC similarity for expert selection (replaces learned gating network)
  - KV cache keys: holographic reduced representation, O(1) retrieval
- Ternary binding: element-wise multiply (trit × trit = trit)
- Bundling: majority vote (natural for ternary)
- Synergy with Phase 30 POPCOUNT binary decomposition

**P4: Stochastic Bit-Stream Computing** — MEDIUM VIABILITY
- Encode activations as Bernoulli bit-streams: p(1) = (x+1)/2
- Ternary weight application: MUX gate (select/invert/zero stream)
- Accumulation = count bits = POPCOUNT
- Precision-throughput knob: longer streams = more precision, more time
- Circuit: MUX + NOT + wire. No adder, no multiplier.
- For 8-bit precision: 256-cycle stream. For 4-bit: 16-cycle stream.
- Error: stochastic, uncorrelated → averages out over dimensions

#### RADICAL DEPLOYMENT APPROACHES

| # | Technique | Key Insight | Viability |
|---|-----------|-------------|-----------|
| P5 | GF(3) Erasure Coding | Reed-Solomon over GF(3), censorship-resistant | MEDIUM |
| P6 | Photonic SLM Inference | Speed-of-light GEMV, macro-scale | SPECULATIVE |
| P7 | Metamaterial Wave Computing | Physical material IS the model | FAR-FUTURE |
| P8 | Sovereign Executable | Model compiled to ADD/SUB instructions | HIGH |

**P5: GF(3) Erasure Coding ("Holographic Model")** — MEDIUM VIABILITY
- Ternary weights live naturally in GF(3) = {0, 1, 2} (map: {-1,0,+1} → {2,0,1})
- Apply Reed-Solomon erasure coding over GF(3): encode k shards into n shards
- Any k-of-n shards reconstruct model. Individual shards = random noise (information-theoretic security)
- Use case: censorship-resistant model distribution. No single shard reveals model.
- Only works for TERNARY models (GF(3) arithmetic). Binary models need GF(2) (less interesting).
- Practical: 10-of-15 coding → 50% overhead, tolerates 5 lost shards

**P6: Photonic SLM Inference** — SPECULATIVE
- Same principle as P1 but at macro deployment scale
- Spatial Light Modulator (SLM) array programs ternary weight matrix
- Input beam modulated by activation values → lens focuses → detector reads output
- One optical pass = one matrix-vector multiply at speed of light
- Challenge: precision limited by optical noise (~6-8 effective bits for activations)
- 50M model: ~30 SLM panels in cascade, room-scale optical bench

**P7: Metamaterial Wave Computing** — FAR-FUTURE
- Fabricate physical metamaterial whose electromagnetic scattering IS the ternary GEMV
- Material structure encodes weights permanently (like ROM but physics-based)
- Input = RF/microwave signal, output = scattered signal
- Energy: only the input signal (picojoules per inference)
- Ternary is ideal: {-1,0,+1} maps to {reflect, absorb, transmit}
- Current state: proof-of-concept for simple operations (2-4 element "networks")

**P8: Sovereign Executable** — HIGH PRIORITY
- Compile ternary model into straight-line program: only ADD, SUB, MOV instructions
- No library dependencies, no runtime, no Python, no framework
- Entire model = single statically-linked binary (~12MB for 50M params)
- Execution: load binary, feed tokens, get output. Works on bare metal.
- Ternary weights enable this: no MUL instruction needed (MUL is the complex one)
- Implications: run on ANY CPU ever made (even 8-bit microcontrollers), impossible to "ban" or restrict
- Synergy with Phase 23 inference engine (already uses add/subtract kernels)

#### KEY PATTERNS (Wave 3)
- Photonic computing is natural fit for ternary (3 phase states vs continuous modulation for FP)
- HDC subsystem replacement (P3) is highest-value: POPCOUNT-based, implementable NOW
- Sovereign executable (P8) aligns with existing inference engine architecture
- Neuromorphic SSM (P2) reveals deep mathematical connection: SSMs ARE spiking neural networks
- GF(3) erasure coding (P5) is unique to ternary — no other quantization enables this

---

## Aspect 4 Summary: Architecture & Hardware Co-design

**Total findings:** 13 web research (F1-F13) + 15 mathematical (M1-M15) + 8 paradigm shifts (P1-P8) = **36 findings**

**Top 5 by estimated impact:**
1. **M4 Temporal Delta Accumulation (TDA)** — 5-20x via temporal correlation exploitation (unique to autoregressive)
2. **M14 LSRF Row Clustering** — 5-15x with ZERO quality loss (purely computational optimization)
3. **P8 Sovereign Executable** — Model as bare-metal binary, zero dependencies, any CPU
4. **M10 BFP-k Integer Pipeline** — Unlock INT8 tensor cores (988 TOPS on RTX 5070)
5. **P3 HDC Subsystem Replacement** — POPCOUNT-based LM head/router, implementable now

**Immediate actions (can implement now):**
- M14 (LSRF): Post-training row clustering, zero risk, significant speedup
- P8 (Sovereign Executable): Extend Phase 23 engine to standalone binary
- P3 (HDC LM Head): Replace linear LM head with POPCOUNT similarity search
- M10 (BFP-k): Integer activation quantization for tensor core access

**Full Integer Inference Stack (combining findings):**
- LSRF (M14) + BFP-k (M10) + DRSR (M15) + TDA (M4) = 15-60x theoretical speedup
- Approaches information-theoretic bandwidth limit (53,760 tok/s)
- All floating-point eliminated from inference pipeline

---

## Round 4 Complete Summary

| Aspect | Findings | Key Highlight |
|--------|----------|---------------|
| 1. Training & Scaling | 39 | Proxy-task transfer, phase-space compression |
| 2. Inference Deployment | 42 | Speculative decoding for SSMs, WASM deployment |
| 3. Long-Context & Reasoning | 44 | Barrington-SSM (TC⁰→NC¹), holographic memory |
| 4. Hardware Co-design | 36 | TDA 5-20x, full integer stack, sovereign executable |
| **TOTAL** | **161** | **Across 12 waves of research** |

**Cross-Aspect Synergies:**
- TDA (Aspect 4) + Speculative Decoding (Aspect 2) = multiplicative speedup
- Barrington-SSM (Aspect 3) + DRSR (Aspect 4) = integer-only enhanced reasoning
- Proxy-task Transfer (Aspect 1) + BFP-k (Aspect 4) = faster training on integer hardware
- HDC Subsystem (Aspect 4) + Holographic Memory (Aspect 3) = unified ternary algebra framework
