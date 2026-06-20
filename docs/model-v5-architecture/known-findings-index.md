# v5 Known Findings Index — Deduplication Reference

## Purpose
Master list of ALL findings documented in v5 research.
Every research agent MUST receive this list to avoid re-discovering known findings.

## Format
- ID: [Round]-[Aspect]-[Type]-[Number]
  - Type: F (web), M (math), P (paradigm)
- Example: R1-A1-F3 = Round 1, Aspect 1, Web Finding 3

## Findings

### Round 1, Aspect 1: Extreme Sparse Activation
- v5-R1-A1-F1: MoTE — Mixture of Ternary Experts (2.3 GB for 7B)
- v5-R1-A1-F2: FlashMoE — SSD/NVMe Expert Offloading (64GB → ~1GB DRAM)
- v5-R1-A1-F3: BuddyMoE — Expert Redundancy Exploitation
- v5-R1-A1-F4: DeepSeek MLA — 93x KV Cache Compression
- v5-R1-A1-F5: Ultra-Sparse MoE (1000+ Experts, K=1)
- v5-R1-A1-F6: PowerInfer — Hot/Cold Neuron Split (Production)
- v5-R1-A1-F7: DejaVu — Activation Prediction
- v5-R1-A1-F8: R-Sparse — Rank-Aware Rotation Sparsity (Training-Free)
- v5-R1-A1-F9: TEAL — Training-Free Magnitude Sparsity
- v5-R1-A1-F10: TC-MoE — Ternary Expert Choice Routing
- v5-R1-A1-F11: MoR — Mixture of Recursions
- v5-R1-A1-F12: Early-Exit LLMs (REJECTED — no RAM savings)
- v5-R1-A1-F13: Dynamic Input Pruning (REJECTED — bandwidth only)

### Round 1, Aspect 1 Wave 2: Math Paradigm-Breakers
- v5-R1-A1-M1: Information-Theoretic Lower Bound (SSM+MoE = 130-160 MB active)
- v5-R1-A1-M2: Sparsity-Quality Phase Transition (2-5% critical threshold)
- v5-R1-A1-M3: Ternary Expert Compression Hierarchy (0.016-2.0 bits/weight)
- v5-R1-A1-M4: NVMe Expert Loading Math (hierarchical cache ~290 MB)
- v5-R1-A1-M5: State Compression (SSM 16 MB vs Attention 1 GB FATAL)
- v5-R1-A1-M6: Grand Synthesis Architecture (285-540 MB proven)
- v5-R1-A1-M7: Optimal Bit Allocation via Water-Filling
- v5-R1-A1-M8: Kronecker-Factored Ternary (8-32x compression)
- v5-R1-A1-M9: Ternary Residual Coding (info-theoretically optimal, TOP)
- v5-R1-A1-M10: Product Quantization of Ternary Vectors (30-50% extra)
- v5-R1-A1-M11: Spectral Ternary Encoding (8x compression + speedup)
- v5-R1-A1-M12: Binary+Sparse Hybrid for FFN (2-10x faster)
- v5-R1-A1-M13: Circulant Ternary with NTT (100-500x, complex)
- v5-R1-A1-M14: Boolean Decomposition (SPECULATIVE)

### Round 1, Aspect 1 Wave 3: Paradigm-Breakers
- v5-R1-A1-P1: RATR — Retrieval-Augmented Tiny Reasoner (HIGH, 270 MB)
- v5-R1-A1-P2: Computation-as-Lookup / Product Quantized Inference (MEDIUM, 164 MB, 10K+ tok/s)
- v5-R1-A1-P3: Dendritic Trees / Decision Tree Neurons (MEDIUM, 95 MB, 5K-12K tok/s)
- v5-R1-A1-P4: HDC Ternary Composition Module (MEDIUM as component, 48 MB)
- v5-R1-A1-P5: Sparse Distributed Memory with Ternary Addressing (LOW standalone)
- v5-R1-A1-P6: Frequency-Domain NTT Processing (LOW standalone)
- v5-R1-A1-P7: Bandwidth-Intelligence Bound + 4 Architecture Proposals Comparison
- v5-R1-A1-P8: Proposal A — Dense Ternary Tower (2.85B, 793 MB, MMLU 48-55%, $0)
- v5-R1-A1-P9: Proposal C — Mega-MoE Deep Tower (42.4B, 555 MB, MMLU 63-70%, $3K-8K)
- v5-R1-A1-P10: Proposal D — Kronecker-MoE Hybrid (28B, 513 MB, MMLU 50-58%, $1K-4K)

## Cross-Reference: Known from v2/v3/v4

These findings from previous versions are ALREADY KNOWN. Do NOT re-discover:
- Ternary quantization ({-1,0,+1}) — extensively tested, validated
- Mamba/Mamba-2 SSM — our base architecture
- BitNet, BitNet b1.58 — reference material
- GaLore — tested, marginal gain (-0.9%)
- STEWS, MoTE, AngBound, DeferredShadow — all tested, all neutral/negative
- STE (Straight-Through Estimator) — our gradient method, validated
- CAGE, DQT, QZO, HESTIA — all tested, all failed
- Speculative decoding (Medusa, PEARL) — speedup trick, not RAM reduction
- LLaDA, Mercury 2 — diffusion LMs, validated but different paradigm
- T-MAC, TerEffic, BitNet.cpp — ternary inference engines
- Slender-Mamba — ternary SSM validation (COLING 2025)
- Reversible coupling — ~50% VRAM savings (training only)
- Prediction skip / token sparsity — 2-3x inference speed
- Hadamard-Domain Diagonal SSM — O(d log d) mixing
- All ~170 findings from v4 research (see docs/model-v4-architecture/)

## Aspect 2: Natively RAM-Efficient Architectures (Wave 1)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A2-F1 | Universal Reasoning Model (Cross-Layer Sharing) | Web | PROMISING |
| v5-R1-A2-F2 | Recursive Inference Scaling (RINS) | Web | PROMISING |
| v5-R1-A2-F3 | Basis Sharing (Cross-Layer Basis Vectors) | Web | PROMISING |
| v5-R1-A2-F4 | Fine-Grained MoE (DeepSeek V3/Qwen3) | Web | PROMISING |
| v5-R1-A2-F5 | Inference-Time Compute Scaling (R1 paradigm) | Web | PROMISING |
| v5-R1-A2-F6 | Over-Training Small Models (Llama 3) | Web | PROMISING |
| v5-R1-A2-F7 | xLSTM 7B (Production SSM) | Web | PROMISING |
| v5-R1-A2-F8 | MDL/Kolmogorov Complexity Objectives | Web | NEEDS-MORE-WORK |

## Aspect 2: Natively RAM-Efficient Architectures (Wave 2 — Math)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A2-M1 | Ternary Monoid Sharing | Math | NEEDS-MORE-WORK |
| v5-R1-A2-M2 | Information-Theoretic RAM Floor | Math | PROMISING |
| v5-R1-A2-M3 | Kolmogorov/Hamming Sharing Topology | Math | PROMISING |
| v5-R1-A2-M4 | Recursive Ternary SSM Fixed-Point | Math | PROMISING |
| v5-R1-A2-M5 | GF(3) Basis Sharing | Math | PROMISING |
| v5-R1-A2-M6 | Spectral Ternary/Hadamard Decomposition | Math | SPECULATIVE |
| v5-R1-A2-M7 | Expert-Selective SSM (ES-SSM) | Math | PROMISING |
| v5-R1-A2-M8 | Optimal RAM Budget Allocation | Math | PROMISING |
| v5-R1-A2-M9 | Ternary MoE-SSM Scaling Law | Math | PROMISING |
| v5-R1-A2-M10 | MoE Decouples Intelligence from Speed | Math | PROMISING |
| v5-R1-A2-M11 | NVMe Scheduling Bound | Math | PROMISING |
| v5-R1-A2-M12 | Multiplicative Efficiency Composition | Math | PROMISING |
| v5-R1-A2-M13 | State-Expert SSM (SE-SSM) | Math | SPECULATIVE |

## Aspect 2: Natively RAM-Efficient Architectures (Wave 3 — Paradigm)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A2-P1 | Algorithmic Weights (Procedural Generation) | Paradigm | SPECULATIVE |
| v5-R1-A2-P2 | Entropy-Gated Dual-Path | Paradigm | PROMISING |
| v5-R1-A2-P3 | Ternary Cellular Automata Layers | Paradigm | MOONSHOT |
| v5-R1-A2-P4 | Resonant Oscillator SSM (Zero-Param A) | Paradigm | PROMISING |
| v5-R1-A2-P5 | Attractor-Based Knowledge Storage | Paradigm | SPECULATIVE |
| v5-R1-A2-P6 | Cyclic Ternary Automaton (CTA) | Paradigm | MOONSHOT |
| v5-R1-A2-P7 | PLA Logic Minimization | Paradigm | PROMISING |
| v5-R1-A2-P8 | Inverted Index Sublinear Inference | Paradigm | PROMISING |
| v5-R1-A2-P9 | Energy-Based SSM (EB-SSM) | Paradigm | SPECULATIVE |
| v5-R1-A2-P10 | Kronecker-Ternary Decomposition | Paradigm | PROMISING |

## Aspect 3: Compute-on-Demand & Adaptive Architectures (Wave 1)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A3-F1 | FlexGen Block-Scheduled Offloading | Web | PROMISING |
| v5-R1-A3-F2 | Speculative Decoding (EAGLE-3 + Draft) | Web | PROMISING |
| v5-R1-A3-F3 | Layer Streaming on Unified Memory | Web | PROMISING |
| v5-R1-A3-F4 | Mixture of Depths (MoD) | Web | PROMISING |
| v5-R1-A3-F5 | Phase-Level Early Exit (AdaInfer) | Web | PROMISING |
| v5-R1-A3-F6 | Mamba-Shedder SSM Pruning | Web | PROMISING |
| v5-R1-A3-F7 | First-Finish Search (FFS) | Web | PROMISING |
| v5-R1-A3-F8 | HALT-CoT Adaptive Early Stopping | Web | PROMISING |

## Aspect 3: Compute-on-Demand & Adaptive Architectures (Wave 2 — Math)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A3-M1 | State Delta Norm Bound (Ternary Operator Spectrum) | Math | PROMISING |
| v5-R1-A3-M2 | Ternary State Stabilization Phase Transition (EXACT exit) | Math | PROMISING |
| v5-R1-A3-M3 | Entropy-Gated State Update (Information Geometry) | Math | PROMISING |
| v5-R1-A3-M4 | Depth-Width Capacity Theorem (Ternary Networks) | Math | PROMISING |
| v5-R1-A3-M5 | Hamming Distance Router (POPCOUNT Skip Detection) | Math | PROMISING |
| v5-R1-A3-M6 | MDL Routing (Kolmogorov-Optimal Depth) | Math | PROMISING |
| v5-R1-A3-M7 | Spectral Decay of Ternary Operators | Math | NEEDS-MORE-WORK |
| v5-R1-A3-M8 | Minimum Viable Ternary Router (Single Vector) | Math | PROMISING |
| v5-R1-A3-M9 | Ternary Speculative Decoding Agreement Zone | Math | PROMISING |
| v5-R1-A3-M10 | Optimal Memory Hierarchy Allocation (Waterfalling) | Math | PROMISING |
| v5-R1-A3-M11 | Layer Streaming Crossover Theorem (5.1x proof) | Math | PROMISING |
| v5-R1-A3-M12 | Expert Prefetching (Markov Routing Prediction) | Math | PROMISING |
| v5-R1-A3-M13 | Fundamental Ternary Throughput Ceiling (8x Theorem) | Math | PROMISING |
| v5-R1-A3-M14 | Unified Memory Zero-Copy Theorem (11x advantage) | Math | PROMISING |
| v5-R1-A3-M15 | SSM State as Expert Cache Predictor | Math | PROMISING |
| v5-R1-A3-M16 | Holographic MoE Expert Compression (128x, MOONSHOT) | Math | SPECULATIVE |

## Aspect 3: Compute-on-Demand & Adaptive Architectures (Wave 3 — Paradigm)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A3-P1 | Ternary Butterfly Factorization (157x projection speedup) | Paradigm | PROMISING |
| v5-R1-A3-P2 | Resonant SSM with Ternary Spectral Gating | Paradigm | PROMISING |
| v5-R1-A3-P3 | Seed-Generated Random Ternary Projections (35% params gone) | Paradigm | PROMISING |
| v5-R1-A3-P4 | Permutation-Based Ternary Weight Sharing (6.7-30x) | Paradigm | PROMISING |
| v5-R1-A3-P5 | SSM-Guided Ternary Weight Flipping (Self-Modifying) | Paradigm | SPECULATIVE |
| v5-R1-A3-P6 | Instruction-Stream Compilation (Weights = Code) | Paradigm | PROMISING |
| v5-R1-A3-P7 | Periodic State Automaton + Heterogeneous Sparsity | Paradigm | PROMISING |
| v5-R1-A3-P8 | Bitplane POPCOUNT Engine (4-32x throughput) | Paradigm | PROMISING |
| v5-R1-A3-P9 | Collapsed Path Network (Layer Fusion) | Paradigm | NEEDS-MORE-WORK |
| v5-R1-A3-P10 | Moment-Bypass Routing (13x faster routing) | Paradigm | PROMISING |

## Aspect 4: Alternative Knowledge Representations (Wave 1)

| ID | Name | Wave | Verdict |
|----|------|------|---------|
| v5-R1-A4-F1 | StructGPT (LLM as Query Generator over Databases) | Web | PROMISING |
| v5-R1-A4-F2 | Memory Layers at Scale (Sparse Retrieval Replaces FFN) | Web | PROMISING |
| v5-R1-A4-F3 | RETRO (Retrieval-Enhanced, 25x ratio) | Web | PROMISING |
| v5-R1-A4-F4 | kNN-LMs "Great Memory, Shallow Reasoning" | Web | NEEDS-MORE-WORK |
| v5-R1-A4-F5 | MemoryLLM (Self-Updatable External Memory) | Web | PROMISING |
| v5-R1-A4-F6 | Memory-Augmented LLMs Ecosystem (MemLoRA, MMAG) | Web | PROMISING |
| v5-R1-A4-F7 | SpikeGPT (260M Spiking LM, 22x fewer ops) | Web | PROMISING |
| v5-R1-A4-F8 | SDNN Sigma-Delta Conversion (6% MACs, ANN→SNN) | Web | PROMISING |
| v5-R1-A4-F9 | LUT-NN (Lookup Table Inference) | Web | NEEDS-MORE-WORK |

## Aspect 4: Alternative Knowledge Representations (Wave 2 — Math)

| ID | Title | Rating | Source |
|----|-------|--------|--------|
| v5-R1-A4-M1 | Rate-Distortion Pareto Frontier (Parametric vs Retrieval) | HIGH | Math-PB |
| v5-R1-A4-M2 | Hamming-Distance Ternary Retrieval (d=128, 10^8 entries) | HIGH | Math-PB |
| v5-R1-A4-M3 | SSM State as Sufficient Statistic for Retrieval | HIGH | Math-PB |
| v5-R1-A4-M4 | Spiking-Ternary Information Capacity (retrieval NECESSARY) | MEDIUM | Math-PB |
| v5-R1-A4-M5 | Compositional TPR in Ternary | MEDIUM | Math-PB |
| v5-R1-A4-M6 | Neural Switchboard (7700x sparsity) | MEDIUM | Math-PB |
| v5-R1-A4-M7 | Z_3 Holographic Memory (mod-3 convolution) | SPECULATIVE | Math-PB |
| v5-R1-A4-M8 | Rate-Distortion Knowledge Partition Theorem (multi-tier) | HIGH | Math-PB |
| v5-R1-A4-M9 | Fused Multi-Layer Ternary LUT Algebra | MEDIUM-HIGH | Math-PB |
| v5-R1-A4-M10 | Dual-Channel SSM + External Memory Capacity | HIGH | Math-PB |
| v5-R1-A4-M11 | Ternary-Spike Integer Pipeline Theorem (zero FP) | HIGH | Math-PB |
| v5-R1-A4-M12 | Query Generator Minimum Size (~50M ternary) | MEDIUM | Math-PB |
| v5-R1-A4-M13 | Ternary Holographic SSM (TH-SSM, ~800 bindings) | MEDIUM-HIGH | Math-PB |
| v5-R1-A4-M14 | GF(3) Knowledge Graph Algebra (POPCOUNT retrieval) | HIGH | Math-PB |
| v5-R1-A4-M15 | NTT Ordered Binding (non-commutative, mod 257) | SPECULATIVE | Math-PB |

### Aspect 4: Alternative Knowledge Representations (Wave 3 — Paradigm-Breakers)

| ID | Title | Rating | Source |
|----|-------|--------|--------|
| v5-R1-A4-P1 | Ternary Unification Machine (TUM) — inference as logical unification | HIGH | PB |
| v5-R1-A4-P2 | GF(3) Append-Only Knowledge Log (TAOL) — versioned ternary diffs | HIGH | PB |
| v5-R1-A4-P3 | All-Ternary Resonance Cascade (ATRC) — sigma-delta ternary activations | MEDIUM | PB |
| v5-R1-A4-P4 | Ternary Legislature (TL) — voting theory framework | MEDIUM | PB |
| v5-R1-A4-P5 | Metabolic Ternary Network (MTN) — usage-driven weight fatigue | MEDIUM | PB |
| v5-R1-A4-P6 | Stigmergic Trace Computation (STC) — pheromone working memory | HIGH | PB |
| v5-R1-A4-P7 | Self-Modifying Ternary Programs (SMTP) — Hebbian in-inference | MEDIUM | PB |
| v5-R1-A4-P8 | Epigenetic Ternary Modulation (ETM) — input-dependent weight silencing | MEDIUM-HIGH | PB |
| v5-R1-A4-P9 | Ternary Rule Automata (TRA) — cellular automaton layers, O(d) | LOW-MEDIUM | PB |
| v5-R1-A4-P10 | Ternary Rewriting Systems (TRS) — pattern-replacement inference | LOW | PB |

## Aspect 5: Memory-Efficient Training & Inference (Wave 1)

| ID | Title | Rating | Source |
|----|-------|--------|--------|
| v5-R1-A5-F1 | AdaPM — Adaptive Partial Momentum (90% momentum reduction) | PROMISING | ICLR 2026 |
| v5-R1-A5-F2 | Q-Adam-mini — 8× optimizer memory reduction (INT8 momentum) | PROMISING | ICML 2025 |
| v5-R1-A5-F3 | CompAct — Activation compression via random projections | PROMISING | 2024-2025 |
| v5-R1-A5-F4 | TriTera — Ternary LLMs at 1.2T tokens (data > params scaling) | PROMISING | ACL 2025 |
| v5-R1-A5-F5 | Joint Reversible-Quantized Optimizer (16.2× savings) | PROMISING | IEEE 2025 |
| v5-R1-A5-F6 | ZenFlow — Gradient priority offloading (10% on GPU, 90% CPU) | PROMISING | Feb 2026 |
| v5-R1-A5-F7 | MHA-to-GQA Procrustes (87.5% KV reduction) | PROMISING | EMNLP 2025 |
| v5-R1-A5-F8 | EvolKV — Evolutionary KV budget (98.5% reduction) | PROMISING | EMNLP 2025 |
| v5-R1-A5-F9 | ChunkKV — Semantic chunk compression (+26% throughput) | PROMISING | Feb 2025 |
| v5-R1-A5-F10 | BitNet.cpp ELUT — Element-wise LUT (6.25× vs FP16) | PROMISING | ACL 2025 |
| v5-R1-A5-F11 | Hybrid SSM-Transformer Industry Convergence (Jamba, Bamba, Nemotron-H) | PROMISING | 2025-2026 |
| v5-R1-A5-F12 | Gated Attention — NeurIPS 2025 Best Paper (sigmoid gate) | PROMISING | NeurIPS 2025 |
| v5-R1-A5-F13 | Sub-1GB Mobile: deep-thin architecture wins (Meta consensus) | PROMISING | Meta 2026 |
| v5-R1-A5-F14 | KV-Cache 3-bit Quantization (5× cache compression) | PROMISING | 2025-2026 |

### Aspect 5: Memory-Efficient Training & Inference (Wave 2 — Math Paradigm-Breakers)

| ID | Title | Rating | Source |
|----|-------|--------|--------|
| v5-R1-A5-M1 | Threshold-Distance Optimizer (TDO) — 5.05x reduction | HIGH | Math-PB |
| v5-R1-A5-M2 | SPRT Optimizer — 5.33x, provably optimal | MEDIUM | Math-PB |
| v5-R1-A5-M3 | Ternary Gradient Checkpointing Advantage — 3x cheaper recompute | HIGH | Math-PB |
| v5-R1-A5-M4 | Ternary Plateau Sparsity (80% gradients skippable) | HIGH | Math-PB |
| v5-R1-A5-M5 | Ternary Depth-Width Theorem ((2/3)^L gradient SNR) | MEDIUM | Math-PB |
| v5-R1-A5-M6 | Binary Gradient Accumulation Pipeline (integer backward) | HIGH | Math-PB |
| v5-R1-A5-M7 | Ternary Chinchilla Scaling Law (150:1 data/param) | MEDIUM | Math-PB |
| v5-R1-A5-M8 | Shadow-Free Training via Combinatorial CD | SPECULATIVE | Math-PB |
| v5-R1-A5-M9 | Ternary SSM Memory Trichotomy (oscillators > accumulators) | HIGH | Math-PB |
| v5-R1-A5-M10 | Attention Allocation Information Gap (10-15% not 8%) | HIGH | Math-PB |
| v5-R1-A5-M11 | GF(3)-SSM — Finite Field SSM (10x state reduction) | MEDIUM | Math-PB |
| v5-R1-A5-M12 | Two-Tier Ternary State T3S (4.7x compression, self-correcting) | HIGH | Math-PB |
| v5-R1-A5-M13 | Fused LUT-SSM (C×A is ternary, weight streaming) | MEDIUM | Math-PB |
| v5-R1-A5-M14 | Streaming Memory Certificate (5B model in 18.4 MB) | HIGH | Math-PB |
| v5-R1-A5-M15 | Spectral Folding Attention Replacement (SFAR) | SPECULATIVE | Math-PB |
| v5-R1-A5-M16 | Convolutional Ternary SSM (CT-SSM, O(d log d) attention replacement) | MEDIUM | Math-PB |

### Aspect 5: Memory-Efficient Training & Inference (Wave 3 — Paradigm-Breakers)

| ID | Name | Type | Verdict |
|----|------|------|---------|
| v5-R1-A5-P1 | Potts Model Parallel Tempering (55× training memory reduction) | paradigm | HIGH impact, HIGH difficulty |
| v5-R1-A5-P2 | Hypernetwork Weight Compilation (168× reduction, 4M generator) | paradigm | MEDIUM-HIGH impact, MEDIUM difficulty |
| v5-R1-A5-P3 | Block-SPSA Zeroth-Order Training (35× reduction, no backprop) | paradigm | MEDIUM impact, MEDIUM difficulty |
| v5-R1-A5-P4 | Forward-Forward Layer-Streaming (636× reduction, 132 MB peak) | paradigm | HIGH impact, HIGH difficulty |
| v5-R1-A5-P5 | Discrete Score-Matching Weight Diffusion (56× reduction) | paradigm | HIGH impact, HIGH difficulty |
| v5-R1-A5-P6 | Micro-Stripe Pipelined Inference (852 KB working memory) | paradigm | HIGH impact, LOW difficulty |
| v5-R1-A5-P7 | Ternary Sign Sufficiency (sign+scale activations) | paradigm | MEDIUM impact, MEDIUM difficulty |
| v5-R1-A5-P8 | Incremental Logit Refinement (eliminates output head, 9 MB) | paradigm | HIGH impact, HIGH difficulty |
| v5-R1-A5-P9 | Hash-Indexed Inference (165 KB, inference as lookup) | paradigm | EXTREMELY HIGH impact, HIGH difficulty |
| v5-R1-A5-P10 | Trie-Compressed Suffix Memory (143 KB, inference as retrieval) | paradigm | EXTREMELY HIGH impact, VERY HIGH difficulty |

## Aspect 6: Training Speed & Throughput (Wave 1)

| ID | Name | Type | Verdict |
|----|------|------|---------|
| v5-R1-A6-F1 | EDGC Entropy-Driven Dynamic Gradient Compression (46% comm reduction) | web | PROMISING |
| v5-R1-A6-F2 | TAGC Transformer-Aware Gradient Compression (15% training accel) | web | PROMISING |
| v5-R1-A6-F3 | PacTrain Pruning + Adaptive Sparse Gradient Compression | web | NEEDS-MORE-WORK |
| v5-R1-A6-F4 | WSM Warmup-Free Learning Rate Schedule (10-20% step savings) | web | PROMISING |
| v5-R1-A6-F5 | Muon Optimizer for Pretraining | web | NEEDS-MORE-WORK |
| v5-R1-A6-F6 | Critical Batch Size Scaling Laws | web | NEEDS-MORE-WORK |
| v5-R1-A6-F7 | Hardware-Aware Kernel Fusion (2.3-10.56x speedup) | web | PROMISING |
| v5-R1-A6-F8 | Curriculum Learning & Data Pipeline Optimization (10-30% token reduction) | web | PROMISING |
| v5-R1-A6-F9 | FOGZO First-Order-Guided Zeroth-Order Training (796x vs pure ZO) | web | PROMISING |
| v5-R1-A6-F10 | QZO-Scale Quantized ZO via Scale Perturbation (18x memory reduction) | web | PROMISING |
| v5-R1-A6-F11 | SRigL Structured N:M Sparsity Training (3.4x CPU speedup at 90%) | web | PROMISING |
| v5-R1-A6-F12 | Phased Specialization for SSM-Attention Hybrids | web | NEEDS-MORE-WORK |
| v5-R1-A6-F13 | ZO Subspace Gradient Projection | web | NEEDS-MORE-WORK |

### Aspect 6, Wave 2 — Math Paradigm-Breakers (Training Speed & Throughput)
| ID | Name | Type | Verdict |
|----|------|------|---------|
| v5-R1-A6-M1 | FING Fisher Information Natural Gradient (1.5-2.5x convergence) | math | HIGH feasibility |
| v5-R1-A6-M2 | TSCS Ternary SSM Segmented Cumsums (4-8x scan speedup) | math | HIGH feasibility |
| v5-R1-A6-M3 | STCD Shadow-Free Ternary Coordinate Descent (2-2.5x) | math | MEDIUM feasibility |
| v5-R1-A6-M4 | TCBS Ternary Critical Batch Scheduling (1.3-1.8x) | math | HIGH feasibility |
| v5-R1-A6-M5 | IBTP Integer Backward Training Pipeline INT8 (1.3-1.8x) | math | HIGH feasibility |
| v5-R1-A6-M6 | FPDV Flip Potential Data Valuation (1.15-1.6x) | math | MEDIUM feasibility |
| v5-R1-A6-M7 | GRCF Gradient Recycling via Flip Coherence (1.3-1.7x) | math | HIGH feasibility |
| v5-R1-A6-M8 | WAHD Walsh-Hadamard Sublinear Gradient (moonshot) | math | SPECULATIVE |
| v5-R1-A6-M9 | TPC Ternary Prefix-Sum Collapse (=M2, independently derived) | math | HIGH feasibility |
| v5-R1-A6-M10 | FDTC Fourier-Domain Ternary Convolution (2-5x) | math | HIGH feasibility |
| v5-R1-A6-M11 | FPIT Full Pipeline Integer Training (3.5x, extends M5) | math | HIGH feasibility |
| v5-R1-A6-M12 | TMA Ternary Momentum Accumulator INT8 counters (1.5-2x) | math | MEDIUM feasibility |
| v5-R1-A6-M13 | SLP Speculative Layer Pipeline ternary correction (2-4x) | math | MEDIUM feasibility |
| v5-R1-A6-M14 | SGFM Sparse Gradient Flip-Margin Detection (1.2x standalone) | math | HIGH feasibility |
| v5-R1-A6-M15 | THPT Ternary Hebbian Perturbation Training no backprop (2-3x) | math | SPECULATIVE |
| v5-R1-A6-M16 | MMSIOR Spectral Initialization + Online Refinement (2.5x) | math | SPECULATIVE |

#### Wave 3 — Paradigm Proposals
- **v5-R1-A6-P1:** LWPI — Layer-Wise Potts Inference (5-10× as init, Potts spin glass initialization)
- **v5-R1-A6-P2:** HBT — Holographic Boundary Training (moonshot, lower-dimensional boundary encoding)
- **v5-R1-A6-P3:** SGTS — Surrogate-Guided Ternary Search (10-100×, surrogate predicts flip outcomes)
- **v5-R1-A6-P4:** RATL — Resonant Annealing for Ternary Lattice (2-5× polish, oscillating temperature)
- **v5-R1-A6-P5:** TWCCP — Constraint-Completion Propagation (10-24×, SAT-style constraint training)
- **v5-R1-A6-P6:** BTG — Backward-Ternary GEMV (1.27-1.5×, backward uses add/subtract too)
- **v5-R1-A6-P7:** PTC — Persistent Ternary Cache (incremental update, skip re-quantization)
- **v5-R1-A6-P8:** MSL — Megakernel Fused SSM Layer (2-4×, 78× bandwidth reduction)
- **v5-R1-A6-P9:** CTRI — Convolutional Training, Recurrent Inference (3×, FFT parallel training)
- **v5-R1-A6-P10:** AGA — Algebraic Gradient Accumulation (7.1× memory reduction, INT16 flip accumulators)

## Round 1 COMPLETE — 215 Total Findings

| Aspect | Topic | Web | Math | Paradigm | Total |
|--------|-------|-----|------|----------|-------|
| A1 | Extreme Sparse Activation | ~12 | ~15 | ~10 | 37 |
| A2 | Natively RAM-Efficient Architectures | ~8 | ~13 | ~10 | 31 |
| A3 | Compute-on-Demand & Adaptive Architectures | ~8 | ~16 | ~10 | 34 |
| A4 | Alternative Knowledge Representations | ~9 | ~15 | ~10 | 34 |
| A5 | Memory-Efficient Training & Inference | 14 | 16 | 10 | 40 |
| A6 | Training Speed & Throughput | 13 | 16 | 10 | 39 |
| **Total** | | | | | **215** |
