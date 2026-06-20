# Known Findings Index — Model v4

## Purpose
Master deduplication list. ALL research agents MUST read this before searching.
If a finding is listed here, do NOT re-discover it.

## Format
Each finding has: `ID | Title | Source | Round | Status`

## Findings from v2/v3 (CARRY-OVER — do NOT re-research)

These are established results from model v2/v3 that inform but should not be re-discovered:

| ID | Title | Source | Status |
|----|-------|--------|--------|
| v3-carry-1 | Ternary quantization viable for all layers | Phase 10 validation | PROVEN |
| v3-carry-2 | SSM provides O(1) memory per token | Phase 2 architecture | PROVEN |
| v3-carry-3 | RLEC rank=16 best quality technique (-5.91% PPL) | Phase A optimization | PROVEN |
| v3-carry-4 | STE equilibrium fragile — orthogonal techniques only | Phase A analysis | PROVEN |
| v3-carry-5 | C-level optimizations irrelevant at <1B scale | Prerequisite measurements | PROVEN |
| v3-carry-6 | Direct ternary training works (no FP16 warmup needed) | Phase 10.B.1/B.2 | PROVEN |
| v3-carry-7 | GF(3) factorization not viable (full rank matrices) | Prerequisite measurements | PROVEN |
| v3-carry-8 | SVD biclustering not viable (energy dispersed) | Prerequisite measurements | PROVEN |

## Model v4 Findings — Round 1

| ID | Title | Source | Round.Aspect | Status |
|----|-------|--------|-------------|--------|
| v4-R1-A1-F1 | LLaDA — First Diffusion LM Matching AR Quality | arXiv Feb 2025 | R1.A1 | PROMISING |
| v4-R1-A1-F2 | Mercury 2 — 5x Faster Diffusion LM | Inception Labs Feb 2026 | R1.A1 | PROMISING |
| v4-R1-A1-F3 | Discrete Diffusion Scaling Laws | ETH Zurich Dec 2024 | R1.A1 | PROMISING |
| v4-R1-A1-F4 | Jacobi Forcing — Parallel Causal Decoding | arXiv Dec 2025 | R1.A1 | PROMISING |
| v4-R1-A1-F5 | One-Step Continuous Denoising | arXiv Feb 2026 | R1.A1 | PROMISING |
| v4-R1-A1-F6 | Diffusion Provably Optimal Parallel Samplers | NeurIPS 2025 | R1.A1 | PROMISING |
| v4-R1-A1-F7 | Medusa — Multi-Head Speculative Decoding | ICML 2024 | R1.A1 | PROMISING (trick) |
| v4-R1-A1-F8 | PEARL — Adaptive Speculative Decoding | ICLR 2025 | R1.A1 | PROMISING (trick) |
| v4-R1-A1-F9 | Meta Multi-Token Prediction | Meta 2024 | R1.A1 | PROMISING (training) |
| v4-R1-A1-F10 | SDTT Self-Distillation Through Time | ICLR 2025 | R1.A1 | NEEDS-MORE-WORK |
| v4-R1-A1-F11 | DLMs Struggle with Truly Parallel Decoding | arXiv Feb 2026 | R1.A1 | NEEDS-MORE-WORK |
| v4-R1-A1-F12 | CMLM Non-Autoregressive Transformer | arXiv 2024 | R1.A1 | REJECTED |

## Model v4 Math Analysis — Round 1

| ID | Title | Source | Round.Aspect | Priority |
|----|-------|--------|-------------|----------|
| v4-R1-A1-M1 | Confidence-Based Error Self-Correction in Diffusion | Math-breaker 1 | R1.A1.W2 | HIGH |
| v4-R1-A1-M2 | 160x Bandwidth Reduction (Ternary + Parallel Tokens) | Math-breaker 1 | R1.A1.W2 | HIGH |
| v4-R1-A1-M3 | Fully Ternary Diffusion in Z_3 Space | Math-breaker 1 | R1.A1.W2 | MEDIUM |
| v4-R1-A1-M4 | Time-Conditional RLEC with Step Amortization | Math-breaker 1 | R1.A1.W2 | HIGH |
| v4-R1-A1-M5 | Ternary-Optimal Slow-Start-Fast-Finish Schedule | Math-breaker 1 | R1.A1.W2 | MEDIUM |
| v4-R1-A1-M6 | Weight-Space Diffusion (Input-Conditioned Weights) | Math-breaker 1 | R1.A1.W2 | SPECULATIVE |
| v4-R1-A1-M7 | Bidirectional SSM for Diffusion Backbone | Math-breaker 1 | R1.A1.W2 | MEDIUM |
| v4-R1-A1-M8 | Bidirectional SSM via Forward-Backward Fusion | Math-breaker 2 | R1.A1.W2 | HIGH |
| v4-R1-A1-M9 | State-Space Diffusion (2000x Dimensionality Reduction) | Math-breaker 2 | R1.A1.W2 | MEDIUM |
| v4-R1-A1-M10 | Jacobi-SSM Parallel Generation | Math-breaker 2 | R1.A1.W2 | HIGH |
| v4-R1-A1-M11 | Absorbing-State Markov Chain on SSM States | Math-breaker 2 | R1.A1.W2 | MEDIUM |
| v4-R1-A1-M12 | Info-Theoretic Bounds — SSMs Optimal for Denoising | Math-breaker 2 | R1.A1.W2 | HIGH |
| v4-R1-A1-M13 | Contraction-Expansion Architecture | Math-breaker 2 | R1.A1.W2 | SPECULATIVE |
| v4-R1-A1-M14 | Spectral Diffusion via SSM Eigenstructure | Math-breaker 2 | R1.A1.W2 | MEDIUM-HIGH |

## Model v4 Paradigm-Breaker Analysis — Round 1

| ID | Title | Source | Round.Aspect | Feasibility |
|----|-------|--------|-------------|-------------|
| v4-R1-A1-P1 | Memory-Computation Separation (Tiny Core + External Memory) | PB1 | R1.A1.W3 | MEDIUM-HIGH |
| v4-R1-A1-P2 | Ternary Energy-Based Potts Model | PB1 | R1.A1.W3 | MEDIUM |
| v4-R1-A1-P3 | Predictive Coding with Ternary Error Propagation | PB1 | R1.A1.W3 | MEDIUM |
| v4-R1-A1-P4 | Hyperdimensional Computing with Ternary Algebra | PB1 | R1.A1.W3 | MEDIUM |
| v4-R1-A1-P5 | Kolmogorov Compression Machine / Finite-State Model | PB1 | R1.A1.W3 | LOW-MEDIUM |
| v4-R1-A1-P6 | Asynchronous Cortical Columns (Micro-MoE) | PB1 | R1.A1.W3 | MEDIUM-HIGH |
| v4-R1-A1-P7 | Thermodynamic Ternary Computation (Equilibrium Propagation) | PB1 | R1.A1.W3 | LOW-MEDIUM |
| v4-R1-A1-P8 | ANS-Native Generation (Compression=Generation Duality) | PB2 | R1.A1.W3 | HIGH |
| v4-R1-A1-P9 | Retrieval-Augmented Ternary Router | PB2 | R1.A1.W3 | MEDIUM-HIGH |
| v4-R1-A1-P10 | Ternary Potts + Parallel Gibbs Generation | PB2 | R1.A1.W3 | MEDIUM |
| v4-R1-A1-P11 | Ternary Matrix Product State / Tensor Network | PB2 | R1.A1.W3 | MEDIUM |
| v4-R1-A1-P12 | Codebook Memory + Incremental Compilation | PB2 | R1.A1.W3 | HIGH |
| v4-R1-A1-P13 | Bidirectional SSM + Jacobi Fixed-Point Generation | PB2 | R1.A1.W3 | MEDIUM |
| v4-R1-A1-P14 | Ternary Kolmogorov-Arnold SSM (KAN) | PB2 | R1.A1.W3 | LOW-MEDIUM |

## Model v4 Findings — Round 1, Aspect 2

| ID | Title | Source | Round.Aspect | Status |
|----|-------|--------|-------------|--------|
| v4-R1-A2-F1 | Symbolic Knowledge Distillation (100x Compression) | PMC 2025 | R1.A2 | PROMISING |
| v4-R1-A2-F2 | LightRAG (10x Token Reduction via KG) | Oct 2024 | R1.A2 | PROMISING |
| v4-R1-A2-F3 | Paths-over-Graph (PoG) KG Reasoning | ACM WWW 2025 | R1.A2 | PROMISING |
| v4-R1-A2-F4 | ReKnoS Super-Relations Framework | OpenReview 2025 | R1.A2 | NEEDS-MORE-WORK |
| v4-R1-A2-F5 | Hybrid SyGuS + LLM Program Synthesis | Edinburgh 2025 | R1.A2 | PROMISING |
| v4-R1-A2-F6 | SOAR Self-Improving Synthesis | arXiv Jul 2025 | R1.A2 | NEEDS-MORE-WORK |
| v4-R1-A2-F7 | ReMindRAG Train-Free KG Traversal | OpenReview 2025 | R1.A2 | PROMISING |
| v4-R1-A2-F8 | Neurosymbolic AI Industrial (Amazon Vulcan/Rufus) | 2025-2026 | R1.A2 | PROMISING |
| v4-R1-A2-F9 | Infini-Attention (Bounded Memory Infinite Context) | Google 2024 | R1.A2 | PROMISING |
| v4-R1-A2-F10 | GraphRAG (Microsoft KG-based RAG) | Microsoft 2024 | R1.A2 | NEEDS-MORE-WORK |
| v4-R1-A2-F11 | Hybrid RAG Search (RRF Dense+Sparse) | SIGIR 2009→2025 | R1.A2 | PROMISING |
| v4-R1-A2-F12 | Self-RAG + Corrective RAG | 2023-2024 | R1.A2 | PROMISING |
| v4-R1-A2-F13 | RDR2 Document Structure-Aware RAG | EMNLP 2025 | R1.A2 | PROMISING |
| v4-R1-A2-F14 | Activation Caching + Hybrid KV Caching | arXiv Jan 2025 | R1.A2 | PROMISING |
| v4-R1-A2-F15 | RETRO Follow-ups (2023-2025) | Multiple | R1.A2 | NEEDS-MORE-WORK |
| v4-R1-A2-F16 | RAG Survey 2025 State of Field | arXiv May 2025 | R1.A2 | CONTEXTUAL |

## Model v4 Math Analysis — Round 1, Aspect 2

| ID | Title | Source | Round.Aspect | Priority |
|----|-------|--------|-------------|----------|
| v4-R1-A2-M1 | Holographic Ternary Memory (HTM) via HDC | MB1 | R1.A2.W2 | MEDIUM |
| v4-R1-A2-M2 | Rank-1 Dynamics Modulation (Multiplicative Knowledge) | MB1 | R1.A2.W2 | HIGH |
| v4-R1-A2-M3 | ECC Addressing via GF(3) Syndrome Decoding | MB1 | R1.A2.W2 | HIGH |
| v4-R1-A2-M4 | POPCOUNT Attention with STE (Differentiable Ternary Retrieval) | MB1 | R1.A2.W2 | HIGH |
| v4-R1-A2-M5 | Compressed Sensing Knowledge Recovery | MB1 | R1.A2.W2 | MEDIUM |
| v4-R1-A2-M6 | Product-Quantized Ternary Codebooks | MB1 | R1.A2.W2 | HIGH |
| v4-R1-A2-M7 | Coupled SSM-Memory Dynamical System (Ternary NTM) | MB1 | R1.A2.W2 | SPECULATIVE |
| v4-R1-A2-M8 | Bayesian Bit-Shift Compressive Memory | MB2 | R1.A2.W2 | HIGH |
| v4-R1-A2-M9 | Resonant Interference Memory (RIM) | MB2 | R1.A2.W2 | MEDIUM |
| v4-R1-A2-M10 | Ternary Hopfield with Integer LUT Softmax | MB2 | R1.A2.W2 | HIGH |
| v4-R1-A2-M11 | Signed-Permutation Gated SSM (Negation Gate) | MB2 | R1.A2.W2 | MEDIUM |
| v4-R1-A2-M12 | Ternary Sparse Block CAM (T-CAM) | MB2 | R1.A2.W2 | HIGH |
| v4-R1-A2-M13 | Hierarchical Resonant-Bayesian Ternary Memory (HRBTM) | MB2 | R1.A2.W2 | MEDIUM-HIGH |
| v4-R1-A2-M14 | Ternary ECC Memory via GF(3) Codes (Vector→Matrix Capacity) | MB2 | R1.A2.W2 | SPECULATIVE |

## Model v4 Paradigm-Breaker Analysis — Round 1, Aspect 2

| ID | Title | Source | Round.Aspect | Feasibility |
|----|-------|--------|-------------|-------------|
| v4-R1-A2-P1 | Holographic Interference Memory | PB1 | R1.A2.W3 | MEDIUM-HIGH |
| v4-R1-A2-P2 | Ternary Microcode Processor | PB1 | R1.A2.W3 | MEDIUM |
| v4-R1-A2-P3 | Dual Representation (Rules + Exceptions) | PB1 | R1.A2.W3 | HIGH |
| v4-R1-A2-P4 | Constraint Propagation / 3-SAT Knowledge | PB1 | R1.A2.W3 | MEDIUM |
| v4-R1-A2-P5 | Context-Conditional Weight Generation | PB1 | R1.A2.W3 | MEDIUM |
| v4-R1-A2-P6 | Cellular Automata Knowledge (48-byte Rules) | PB1 | R1.A2.W3 | LOW |
| v4-R1-A2-P7 | Topological Knowledge Encoding (Zero-Pattern) | PB1 | R1.A2.W3 | MEDIUM |
| v4-R1-A2-P8 | Predictive Coding Anti-Memory | PB2 | R1.A2.W3 | HIGH |
| v4-R1-A2-P9 | Sparse Ternary Weight Overlay (XOR) | PB2 | R1.A2.W3 | MEDIUM-HIGH |
| v4-R1-A2-P10 | Holographic Ternary Superposition (BIND/BUNDLE) | PB2 | R1.A2.W3 | MEDIUM |
| v4-R1-A2-P11 | Register-File SSM (d_state in CPU Registers) | PB2 | R1.A2.W3 | HIGH |
| v4-R1-A2-P12 | Resonant Population Memory (Ternary Oscillators) | PB2 | R1.A2.W3 | LOW-MEDIUM |
| v4-R1-A2-P13 | Shared Ternary Knowledge Fabric | PB2 | R1.A2.W3 | MEDIUM |
| v4-R1-A2-P14 | Discrete Automaton State (576-byte Transitions) | PB2 | R1.A2.W3 | HIGH |

## Model v4 Findings — Round 1, Aspect 3

| ID | Title | Source | Round.Aspect | Status |
|----|-------|--------|-------------|--------|
| v4-R1-A3-F1 | Transformers as Energy-Based Hopfield Models | arXiv Jan 2025 | R1.A3 | PROMISING |
| v4-R1-A3-F2 | Intel Loihi 2 SNN for LLMs (145x Energy) | arXiv Feb 2025 | R1.A3 | HIGHLY PROMISING |
| v4-R1-A3-F3 | IBM Analog In-Memory Computing (PCM 14nm) | IBM PASC 2025 | R1.A3 | PROMISING |
| v4-R1-A3-F4 | AIMC Attention for LLMs (ReRAM) | Nature CS Jan 2025 | R1.A3 | PROMISING |
| v4-R1-A3-F5 | Mixture-of-Depths (MoD) Dynamic Compute | DeepMind Apr 2024 | R1.A3 | PROMISING |
| v4-R1-A3-F6 | MoD Routing Advances (A-MoD + Router-Tuning) | ICLR/EMNLP 2025 | R1.A3 | PROMISING |
| v4-R1-A3-F7 | Contextual Sparsity (DejaVu + Polar 2.2x) | ICML 2023 / NeurIPS 2025 | R1.A3 | HIGHLY PROMISING |
| v4-R1-A3-F8 | LrcSSM — Liquid Resistance-Capacitance SSM | NeurIPS 2025 | R1.A3 | PROMISING |
| v4-R1-A3-F9 | Neural ODEs for Continuous-Depth Language | Chen 2018 + 2025 | R1.A3 | NEEDS-MORE-WORK |
| v4-R1-A3-F10 | KAT — Kolmogorov-Arnold Transformer | ICLR 2025 (Oral) | R1.A3 | PROMISING |
| v4-R1-A3-F11 | PETRA — Parallel Reversible Training | ICLR 2025 (Spotlight) | R1.A3 | PROMISING |
| v4-R1-A3-F12 | RevFFN — Reversible MoE Fine-Tuning (49% VRAM) | arXiv Dec 2025 | R1.A3 | PROMISING |

## Model v4 Math Analysis — Round 1, Aspect 3

| ID | Title | Source | Round.Aspect | Priority |
|----|-------|--------|-------------|----------|
| v4-R1-A3-M1 | Ternary Hopfield Energy Landscape | MB1 | R1.A3.W2 | MEDIUM |
| v4-R1-A3-M2 | Predictive Spike-Coded SSM (2-3x Speedup) | MB1 | R1.A3.W2 | HIGH |
| v4-R1-A3-M3 | Neuromorphic Temporal Coding + Bit Shifts | MB1 | R1.A3.W2 | HIGH |
| v4-R1-A3-M4 | Coding-Theoretic State Matrix (MDS Codes) | MB1 | R1.A3.W2 | HIGH |
| v4-R1-A3-M5 | Equilibrium Propagation Training (No STE) | MB1 | R1.A3.W2 | MEDIUM-HIGH |
| v4-R1-A3-M6 | GF(3) Cellular Automaton SSM (12.7x Compression) | MB1 | R1.A3.W2 | MEDIUM |
| v4-R1-A3-M7 | Ternary Information-Geometric Routing | MB2 | R1.A3.W2 | HIGH |
| v4-R1-A3-M8 | Walsh-KAN with Ternary Coefficients (POPCOUNT-Only) | MB2 | R1.A3.W2 | HIGH |
| v4-R1-A3-M9 | Exact Reversible Ternary Coupling (Zero Error) | MB2 | R1.A3.W2 | HIGH |
| v4-R1-A3-M10 | Diagonal Ternary SSM (Gradient Independent of T) | MB2 | R1.A3.W2 | HIGH |
| v4-R1-A3-M11 | Entropy-Calibrated Early Exit (32x Compound) | MB2 | R1.A3.W2 | HIGH |
| v4-R1-A3-M12 | Ternary Residual Lattice Z_{3^k} | MB2 | R1.A3.W2 | SPECULATIVE |

## Model v4 Paradigm-Breaker Analysis — Round 1, Aspect 3

| ID | Title | Source | Round.Aspect | Feasibility |
|----|-------|--------|-------------|-------------|
| v4-R1-A3-P1 | Tropical SSM (min/add Semiring) | PB1 | R1.A3.W3 | MEDIUM-HIGH |
| v4-R1-A3-P2 | Standing Wave Language Model | PB1 | R1.A3.W3 | MEDIUM |
| v4-R1-A3-P3 | Ballot Machine (Width-for-Depth) | PB1 | R1.A3.W3 | MEDIUM |
| v4-R1-A3-P4 | Constraint Propagation Engine (SAT) | PB1 | R1.A3.W3 | MEDIUM |
| v4-R1-A3-P5 | Frequency-Domain WHT-SSM (10,000x Reduction) | PB1 | R1.A3.W3 | MEDIUM-HIGH |
| v4-R1-A3-P6 | Quasicrystal Growth | PB1 | R1.A3.W3 | LOW-MEDIUM |
| v4-R1-A3-P7 | Hierarchical Ternary Codec (123x) | PB1 | R1.A3.W3 | MEDIUM |
| v4-R1-A3-P8 | "Surprise Processor" Unified Architecture (~160x) | PB2 | R1.A3.W3 | HIGH |
| v4-R1-A3-P9 | "Ternary Finite Machine" Unified Architecture (~16-50x) | PB2 | R1.A3.W3 | VERY HIGH |
| v4-R1-A3-P10 | Conflict Resolutions + Implementation Path | PB2 | R1.A3.W3 | SYNTHESIS |

### Round 1, Aspect 4: Ultra-Efficient SSM State (Wave 1 — Web)

| ID | Finding | Category | Verdict |
|----|---------|----------|---------|
| v4-R1-A4-F1 | Mamba-3 MIMO formulation (ICLR 2026) | SSM Architecture | PROMISING |
| v4-R1-A4-F2 | RWKV-7 Goose dynamic state evolution | SSM Architecture | PROMISING |
| v4-R1-A4-F3 | xLSTM exponential gating + matrix memory (NeurIPS 2024) | SSM Architecture | PROMISING |
| v4-R1-A4-F4 | S²Mamba word-importance sparsity + ASIC 35-97x | SSM Compression | PROMISING |
| v4-R1-A4-F5 | HGRN-2 hierarchical gated recurrence | SSM Architecture | NEEDS-MORE-WORK |
| v4-R1-A4-F6 | Slender-Mamba ternary SSM (COLING 2025) | Ternary Validation | CRITICAL |
| v4-R1-A4-F7 | Bi-Mamba binary {-1,+1} SSM | Quantized SSM | PROMISING |
| v4-R1-A4-F8 | TVMamba ternary weights + activations | Ternary Validation | PROMISING |
| v4-R1-A4-F9 | Hybrid SSM-Attention 10-30% optimal mix | Architecture Design | PROMISING |
| v4-R1-A4-F10 | Mamba-X systolic array 2.3x speed 11.5x energy | Hardware Acceleration | PROMISING |
| v4-R1-A4-F11 | Parallel scan optimizations up to 4.5x throughput | Training Speed | PROMISING |
| v4-R1-A4-F12 | QS4D quantized SSM for analog compute-in-memory | Analog/Neuromorphic | NEEDS-MORE-WORK |
| v4-R1-A4-F13 | SNN+SSM hybrids + neuromorphic LLMs sub-25 nJ/tok | Neuromorphic | PROMISING |
| v4-R1-A4-F14 | Griffin/Hawk + eMamba + TR-mamba2attn (minor) | SSM Variants | NEEDS-MORE-WORK |

### Round 1, Aspect 4: Ultra-Efficient SSM State (Wave 2 — Math)

| ID | Finding | Category | Verdict |
|----|---------|----------|---------|
| v4-R1-A4-M1 | Hadamard-Domain Diagonal SSM O(d log d) global mixing | SSM Math | PROMISING |
| v4-R1-A4-M2 | Hyperoctahedral Group SSM signed permutation transitions | SSM Math | PROMISING |
| v4-R1-A4-M3 | Algebraic Closure Theorem — monomial matrices maximal | Foundational | PROMISING |
| v4-R1-A4-M4 | Complex-Ternary Diagonal GF(9) 4 timescales | SSM Math | PROMISING |
| v4-R1-A4-M5 | Ternary Monoid Scan OR+XOR 16-32x gate reduction | Training Speed | PROMISING |
| v4-R1-A4-M6 | Block-Decomposed Scan zero-boundary 64x gate reduction | Training Speed | PROMISING |
| v4-R1-A4-M7 | Gradient Death/Preservation design equations for p_0 | Training Theory | PROMISING |
| v4-R1-A4-M8 | Adaptive State Precision 16-bit provably sufficient | Memory/Efficiency | PROMISING |
| v4-R1-A4-M9 | Optimal State Dimension 10.1x ternary advantage | Theory | PROMISING |
| v4-R1-A4-M10 | Token Sparsity Bounds 3-4x from non-amplification | Efficiency | PROMISING |
| v4-R1-A4-M11 | Ternary Scaling Law η(N) representation efficiency | Scaling | PROMISING |
| v4-R1-A4-M12 | Exact Finite-Difference Gradient for A-matrix | Training | NEEDS-MORE-WORK |
| v4-R1-A4-M13 | Syndrome-Carry + Spectral Init + Wavelet (consolidated) | Training | NEEDS-MORE-WORK |
| v4-R1-A4-M14 | Negative: Matrix Memory = d Parallel SSMs | Architecture | REJECTED |

### Round 1, Aspect 4: Ultra-Efficient SSM State (Wave 3 — Paradigm)

| ID | Finding | Category | Verdict |
|----|---------|----------|---------|
| v4-R1-A4-P1 | Symplectic Integrator State det(J)=1 gradient preservation | Radical Architecture | PROMISING |
| v4-R1-A4-P2 | Game-Theoretic DEQ ternary equilibrium adaptive compute | Radical Architecture | PROMISING |
| v4-R1-A4-P3 | CRT Modular State exact arithmetic with error correction | Radical Architecture | PROMISING |
| v4-R1-A4-P4 | Pushdown Stack State context-free language recognition | Radical Architecture | NEEDS-MORE-WORK |
| v4-R1-A4-P5 | LSM-Tree Hierarchical State multi-timescale database | Radical Architecture | NEEDS-MORE-WORK |
| v4-R1-A4-P6 | "Granite Foundation" unified safe architecture ~10x | Unified Architecture | PROMISING |
| v4-R1-A4-P7 | "Velocity Demon" unified speed architecture ~79x | Unified Architecture | PROMISING |
| v4-R1-A4-P8 | "Pragmatic Machine" balanced RECOMMENDED ~30-60x | Unified Architecture | PROMISING |
| v4-R1-A4-P9 | "Walsh Machine" moonshot 107KB core in L2 cache | Unified Architecture | NEEDS-MORE-WORK |
| v4-R1-A4-P10 | Component Hierarchy Levels 0-4 implementation roadmap | Synthesis | PROMISING |
