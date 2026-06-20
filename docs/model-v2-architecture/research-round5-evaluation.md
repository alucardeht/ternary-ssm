# Research Round 5 — Emergent Capabilities & Evaluation

## Overview

This document tracks research findings on evaluation methodologies, emergent capabilities, and capability boundaries for ternary SSM models discovered in Research Cycle 5, Aspect 6 (Feb 2026).

## Wave 1: Web Research (2 researchers, Feb 2026)

### SSM Evaluation & Long-Context

**F1: State Rank Stratification in Linear Attention SSMs** (arXiv 2602.02195, Feb 2026)
- SSM heads bifurcate into HIGH-RANK (saturate at bound) and LOW-RANK (maintain low-dimensional manifold)
- Low-rank heads are INDISPENSABLE for retrieval (removing drops NIAH 93.8%→46.9%)
- High-rank heads show redundancy (removing only drops to 90.6%)
- State quality ≠ state rank — structured low-rank > high-rank for retrieval

**F2: Negative Eigenvalues Unlock State-Tracking** (ICLR 2025)
- Expanding eigenvalue range from [0,1] to [-1,1] enables parity, modular counting
- Finite precision LRNNs with only positive eigenvalues CANNOT solve parity
- Mamba/DeltaNet with negative eigenvalues improve code/math at 1.3B scale
- TERNARY {-1,0,+1} naturally provides negative values — potential inherent advantage

**F3: Retrieval-Aware Distillation for Hybrids** (arXiv 2602.11374, Feb 2026)
- Only 10 "Gather-and-Aggregate" attention heads (2% of 512) needed for retrieval
- Retaining 10 heads recovers >95% teacher performance (SWDE 27.7%→71.1%)
- SSM state dimension can reduce 64→8 once attention handles G&A
- Validates hybrid architecture — sparse attention + dense SSM is optimal

**F4: LAMB Token Filtering for SSM Long-Context** (ACL 2025)
- Training-free method using attention patterns to guide SSM token filtering
- Up to 30.35% improvement over SOTA on long-context benchmarks
- SSM degradation at long context is NOT fundamental — about token selection

**F5: LongBench Pro — Effective Context Length Metric** (Jan 2026)
- ECL metric: longest input where performance stays within ε of short-context baseline
- Effective context typically LESS than half the nominal window
- U-shaped attention ("lost-in-the-middle"), counting/aggregation <50% at 128K
- Standard perplexity does NOT reveal long-context weaknesses

**F6: SSM Retrieval Gap is Gather-and-Aggregate Deficit**
- SSM underperformance localized to specific G&A mechanism, not holistic deficit
- Hybrid models with 10 G&A attention heads close the retrieval gap
- SSM-only fails on SWDE, KV-Retrieval, GSM8K (multi-hop math)

**F7: Perplexity vs Capabilities Divergence** (arXiv 2601.01237, 2026)
- SSMs can achieve LOWER perplexity but still lag on retrieval tokens
- Perplexity improvements don't translate to proportionate capability gains
- SSMs excel at local dependencies, fail at sparse long-range dependencies
- WARNING: our 7.4% perplexity gap may UNDERESTIMATE capability gaps

**F8: NoLiMa Benchmark — Beyond Literal Matching** (Feb 2025)
- Massive performance drop at 32K when tasks require paraphrasing/reasoning
- Standard needle-in-haystack with exact match is insufficient
- Need semantic retrieval, counting/aggregation, temporal reasoning

**F9: Oolong Benchmark — Aggregation Bottleneck** (Nov 2025)
- 128K-1.3M tokens, atomized classification + aggregation with distributional stats
- Frontier models <50% accuracy at 128K on aggregation tasks
- Aggregation requires SSM state to accumulate structured information (counts, distributions)
- O(1) SSM memory may struggle vs attention for accumulation tasks

**F10: Pause-Tuning for Lost-In-The-Middle** (Feb 2025)
- Inject <PAUSE> tokens, fine-tune attention bias for recalibration
- +10% EM at 64K tokens. Could explore SSM state reset at paragraph boundaries

**F11: Dynamic Chunking with Question-Aware Selection** (Jun 2025)
- Semantic similarity-based adaptive chunking + MLP predicts chunk answerability
- +20-29% F1 over fixed-chunk baselines up to 256K tokens
- Preprocessing could help SSMs — route chunks to SSM vs attention dynamically

**F12: Mamba-3 Improved Discretization** (ICLR 2025)
- Improved discretization, complex dynamics, MIMO updates
- Complex dynamics may relate to F2 negative eigenvalues — broader eigenvalue space

**F13: TVMamba — Ternary Quantization for Visual Mamba** (OpenReview 2025)
- Ternary quantization of Visual Mamba for edge inference
- Quinary-to-ternary annealing + frequency-aware routing
- Direct evidence ternary SSMs are viable and actively researched

**F14: Quantizing Small-Scale SSMs for Edge AI** (arXiv 2506.12480, Jun 2025)
- State matrix MORE SENSITIVE to quantization than input/output projections
- Small-scale SSMs can be quantized for edge with careful state precision handling

**F15: SSM State Dimension Ablation** (Retrieval-Aware paper, Table 4)
- d_state=8 achieves 90% coverage on retrieval when attention handles G&A
- d_state=4 drops to ~81% coverage
- Our d_state=16 is 2x the minimum viable — reasonable but may need increase for ternary

### Ternary Capability Boundaries

**F16: BitNet b1.58 2B4T Performance Parity** (Microsoft, Apr 2025)
- Native 1-bit training on 4T tokens achieves PARITY with full-precision 2-3B models
- GSM8K 58.38% (beats Qwen2.5 1.5B's 56.79%) — math reasoning PRESERVED
- HumanEval+ 38.4% vs Qwen2.5 50.6% — coding 24% DEGRADED
- ARC, HellaSwag, WinoGrande, PIQA, MMLU all competitive

**F17: Capability Hierarchy in Ternary Models**
- MOST PRESERVED: Knowledge retrieval (TriviaQA, BoolQ) — simple lookup
- WELL PRESERVED: Mathematical reasoning (GSM8K 58.38%) — structured, decomposable
- MODERATELY PRESERVED: Language understanding (HellaSwag, WinoGrande) — pattern matching
- MOST DEGRADED: Coding (HumanEval+ -24%) — compositional, abstract reasoning

**F18: Native Training Outperforms Post-Quantization** (BitNet 2B4T)
- Native 1-bit OUTPERFORMS INT4 PTQ (GPTQ, AWQ) despite lower bitwidth (1.58 vs 4)
- Memory: BitNet 0.4GB vs INT4 ~1.0GB vs FP16 4.8GB
- Validates "train from scratch" vs "quantize after" approach

**F19: Reliability Scaling Laws for Quantized LLMs** (OpenReview)
- Performance scales monotonically with bits (more = better)
- Reliability shows NONLINEAR trends — 4-bit peak
- Quantization ENHANCES robustness to character/word-level perturbations
- Ternary (1.58-bit) sacrifices reliability for efficiency

**F20: Layer-Wise Capability Distribution** ("Beyond Benchmarks" paper)
- Early layers: knowledge encoding (PRESERVED in ternary)
- Mid layers: attribute extraction (VULNERABLE to forgetting)
- Late layers: reasoning/prediction (MOST DEGRADED)
- Suggests selective precision: ternary early, higher precision late?

**F21: Energy and Throughput Metrics** (BitNet benchmarks)
- Memory: 0.4GB vs 4.8GB FP16 = 12x reduction
- Throughput: 8.9x higher, 11x larger batch size possible
- Energy: 0.028J vs 0.17J per inference = 6x reduction
- ARM: 1.37-5.07x speedup. x86: 2.37-6.17x speedup

**F22: Evaluation Beyond Perplexity** ("Beyond Benchmarks" paper)
- Perplexity FAILS to measure domain knowledge (measures linguistic capability only)
- Token-level attribution rate better predictor of domain knowledge
- Completion-based domain-specific benchmarks with deterministic pipeline
- Benchmark-contamination-free evaluation using raw corpus directly

### KEY PATTERNS (Wave 1)
1. **Perplexity is insufficient** — need task-specific benchmarks (retrieval, reasoning, code, aggregation)
2. **SSM retrieval gap is localized** — G&A heads (2% of attention) close the gap completely
3. **State quality ≠ state rank** — low-rank structured states better than high-rank saturated
4. **Ternary may unlock state-tracking** — negative eigenvalues {-1} enable parity/counting
5. **Coding is most degraded capability** — -24% in ternary, compositional reasoning suffers most
6. **Native training > PTQ** — ternary from scratch beats INT4 post-quantization

## Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

### Evaluation Mathematics (M1-M6)

**M1: Fisher Information Dimension of SSM States**
- SSM state h_t defines implicit distribution over continuations p(x_{t+1}|h_t)
- Fisher Information Matrix: F_ij(h) = E[-∂²log p(x|h)/∂h_i ∂h_j]
- EFFECTIVE INFORMATION DIMENSION = participation ratio of F eigenvalues
- Measures how many state dimensions are STATISTICALLY relevant (not just linearly independent)
- Different from rank: low-rank state can be high-information; high-rank can be low-information
- Ternary prediction: fewer but MORE informative dimensions (discrete routing concentrates info)

**M2: Persistent Homology of State Trajectories**
- SSM state traces trajectory in R^d_state — has topological features (loops, voids, holes)
- Compute Betti numbers via Topological Data Analysis: β₀ (modes), β₁ (loops), β₂ (voids)
- Persistence diagrams show ROBUST vs NOISE topological features
- Ternary SSM: finite group structure of A → inherent PERIODICITY → cleaner topology
- Compare float vs ternary persistence diagrams (Wasserstein distance = computational strategy distance)
- Reveals GEOMETRIC STRUCTURE of computation invisible to accuracy metrics

**M3: Oscillatory Information Capacity**
- Decompose A into real Jordan form; for complex eigenvalue pairs (α ± iβ):
- FREQUENCY = β/(2π), DECAY RATE = α
- OIC = Σ_k |β_k| × (-1/α_k) — frequency × time-constant for all oscillatory modes
- Ternary: negative eigenvalues give exact period-2 oscillations (FREE, not learned)
- If ternary SSMs are WORSE at coding despite period-2 oscillations → coding needs MORE than alternation
- Measures capacity to track periodic/quasi-periodic patterns weighted by persistence

**M4: Grassmannian Angular Decomposition**
- Hidden representations at each layer span k-dimensional subspace of R^d_model
- Subspace = point on Grassmannian Gr(k, d_model)
- For ternary vs float processing SAME input: compute PRINCIPAL ANGLES between subspaces
- Decompose angular gap per layer per task → identify WHERE and WHY gap arises
- SVD of capability matrix across models defines "capability modes" (orthogonal dimensions)
- Angular gap decomposes into contributions from each capability mode

**M5: Automata-Theoretic State Probing**
- SSM with ternary A = weighted finite automaton over integers
- Transition matrices generate finitely generated subgroup of GL(d, Z)
- Zariski closure determines representational capacity
- SYNTHETIC PROBING: test formal language hierarchy:
  - Regular: (a^n b^n for fixed n), Context-free: Dyck languages, Counter: a^n b^n (unbounded)
- Measure: acceptance accuracy, generalization radius, state fidelity (linear probe from h_t to true automaton state)
- Connects algebraic structure of ternary matrices to formal language theory

**M6: Rényi Information Spectrum**
- Generalize Shannon entropy to order-α Rényi entropy: H_α(X) = (1/(1-α)) log Σ p_i^α
- α→1: Shannon, α→∞: min-entropy, α→0: max-entropy (Hartley)
- Compute Rényi mutual information I_α(h_t; x_{1:t}) across spectrum of α values
- Creates "information fingerprint" — how information is distributed across probability levels
- Ternary SSM: expect DIFFERENT Rényi spectrum (concentrated at specific α values)
- Calibrate against automata probing (M5) for controlled ground truth

### Emergent Capability Mathematics (M7-M12)

**M7: Integer Kernel Exactness**
- Ternary A, B → convolution kernel C·A^k·B has INTEGER entries
- Integer arithmetic is EXACT — zero numerical error in temporal mixing
- FP16 SSMs accumulate floating-point errors over long sequences
- Ternary advantage GROWS with sequence length for exact computation tasks
- Prediction: ternary outperforms FP16 on counting/matching at very long sequences
- The impulse response kernel is error-free: no rounding, no drift

**M8: Kronecker Nilpotency Theorem** — FUNDAMENTAL RESULT
- THEOREM: If A ∈ Z^{d×d} (integer matrix) with all |eigenvalues| < 1, then A is NILPOTENT
- Proof: by Kronecker's theorem — algebraic integers with all conjugates in unit disk are roots of unity or 0
- Strictly contracting integer matrix → all eigenvalues = 0 → A^k = 0 for some k ≤ d
- IMPLICATION: pure ternary SSM has HARD FINITE MEMORY of at most d tokens per layer
- For d_state=16: max memory = 16 tokens per SSM layer. With L layers: effective = 16×L
- Fundamentally different from FP16 (eigenvalue 0.999 → soft exponential decay, theoretically infinite)
- PREDICTS capability hierarchy: coding needs VERY long memory → degrades most
- Note: if A = α·M (ternary M, learned scalar α), nilpotency may not apply. Architecture-dependent.

**M9: Symmetric Threshold Circuit Complexity** — EXPLAINS CAPABILITY HIERARCHY
- Ternary layer + ReLU: y_i = ReLU(Σ_{j∈S⁺} x_j - Σ_{j∈S⁻} x_j + b_i)
- Every ternary neuron = SYMMETRIC threshold gate (unit weights only)
- Circuit class: STC^0 (symmetric threshold circuits) — STRICT SUBSET of TC^0
- STC^0 contains: parity, majority, MOD-k for all k
- TC^0 (FP16) additionally contains: weighted majority, graduated comparisons
- Single FP16 layer (TC^0) can compute what requires Ω(log n) depth in ternary (STC^0)
- CODING degrades because: type inference needs GRADUATED attention (3x importance vs 1x impossible)
- Model compensates with DEPTH (each SSM step adds a layer) but less efficient
- Asymmetric thresholds (where weight magnitudes matter) are the GAP between STC^0 and TC^0

**M10: Discrete Capability Phase Transitions via Flip Graph**
- Weight space {-1,0,1}^N is discrete → loss Φ(W) changes in JUMPS, not smoothly
- FLIP GRAPH: G = (V,E) where V = all ternary configs, E connects 1-flip neighbors
- Capability EMERGES when small # of weight flips causes Φ to cross threshold ("cliff" in flip graph)
- EMERGENCE WIDTH = number of coordinated flips needed to unlock capability
- Small width → sudden emergence (phase transition); Large width → gradual emergence
- Math: SMALL emergence width (few strategic flips unlock arithmetic)
- Coding: LARGE emergence width (many coordinated flips across layers)
- Ternary gap at scale S = how many flips needed vs how many "flip events" occur during training

**M11: Modular Shadow Computation** — UNIQUE TO INTEGER MODELS
- Since A ∈ Z^{d×d}, the map h → Ah (mod p) is well-defined for any prime p
- SSM SIMULTANEOUSLY computes p-shadows: h_t^(p) = A·h_{t-1}^(p) + B·x_t^(p) (mod p)
- p-shadow is finite automaton with p^d states
- Chinese Remainder Theorem: sufficient p-shadows recover full integer state
- Negative eigenvalues: -1 mod 2 = 1 (no oscillation in 2-shadow), but -1 mod 3 = 2 (generates Z₃*)
- Hierarchy of modular views: mod-2 (parity), mod-3 (ternary-aligned), mod-5,7,... (finer)
- UNIQUE to integer (ternary) matrices — FP16 has no meaningful modular reduction
- Advantage: divisibility, periodicity detection, hashing, checksums
- Disadvantage: continuous optimization, interpolation (need real arithmetic)

**M12: Capability Constellation Topology** — SPECULATIVE
- Capability space C: each axis = one capability measure (PPL, math, code, knowledge)
- c(W) ∈ C maps weights to capability point. For ternary: Γ_N = {c(W) : W ∈ {-1,0,1}^N} is FINITE
- FP16: image is smooth manifold. Ternary: image is finite point cloud
- Pareto frontier: ternary = discrete point set; FP16 = smooth surface
- As N grows, ternary Pareto frontier densifies and approaches FP16 frontier
- CONJECTURE: d_H(Γ_N^ternary, Γ_N^FP16) = O(N^{-1/dim(C)}) — polynomial convergence
- Voronoi tessellation of continuous weight space → each cell maps to one ternary config
- Phase transitions occur when Voronoi cells merge/split during training
- Connects capability emergence to topological changes in discrete loss landscape

### KEY PATTERNS (Wave 2)
1. **Nilpotency constrains memory** — pure integer contraction → hard finite memory of d tokens (M8)
2. **STC^0 ⊊ TC^0 explains hierarchy** — ternary = symmetric thresholds, FP16 = general thresholds (M9)
3. **Integer arithmetic is exact** — ternary SSM has ZERO numerical error in state evolution (M7)
4. **Modular shadows are unique** — only integer models compute mod-p arithmetic natively (M11)
5. **Phase transitions are discrete** — capabilities emerge via coordinated weight flips (M10)
6. **Topology reveals strategy** — persistent homology of state trajectories = computational fingerprint (M2)

### Cross-Pollination (Wave 2)
- **M1 (Fisher) + M4 (Grassmannian):** Fisher metric induces Riemannian metric on Grassmannian → curvature of capability manifold
- **M2 (Topology) + M3 (Oscillatory):** Oscillatory modes create LOOPS; persistence measures their robustness
- **M5 (Automata) + M6 (Rényi):** Formal language probes provide ground truth for calibrating Rényi spectrum
- **M7 (Integer) + M11 (Modular):** Integer exactness enables modular shadows; together predict counting/periodicity advantage
- **M8 (Nilpotency) + M9 (STC^0):** Both constrain capability → combined prediction is tighter than either alone
- **M10 (Flip Graph) + M12 (Constellation):** Flip graph = discrete landscape, constellation = capability-space view of same geometry

## Wave 3: Paradigm Breakers (2 agents, Feb 2026)

### Radical Evaluation Paradigms (P1-P4)

**P1: Lattice-Geometric Capability Index (LGCI)**
- Measure HOW ROBUST capability is in discrete weight space (not just performance level)
- Flip sensitivity spectrum: for each weight, measure task-specific loss change on flip (-1→0→+1)
- Basin volume: random walk on flip graph — expected steps before task degradation
- Degeneracy gap: Hamming distance to nearest equally-good alternative configuration
- Narrow spectrum = distributed (robust); heavy-tailed = concentrated (fragile)
- ANSWERS: is 7.4% gap because lattice point is far from optimal (trainable) or lattice too sparse (architectural)?
- Coding: expect SMALL basin (fragile), Knowledge: expect LARGE basin (robust)

**P2: Directed Information Channel Capacity (DICC)**
- Treat SSM as communication channel with memory — measure ARCHITECTURAL capacity, not trained performance
- Capacity with feedback: C_fb = lim (1/n) max I(X^n → Y^n) (directed information)
- Nilpotent ternary A → finite-state channel with ≤ 3^d_state states
- Upper bound: C_fb ≤ log(3) ≈ 1.585 bits per state dimension per step
- FP16 with |λ|=0.99: C_fb ~ Σ log(1/(1-|λᵢ|²)) via water-filling — MUCH larger
- SEPARATES architectural gap from optimization gap — the SINGLE most important question
- If C_fb_ternary/C_fb_fp16 ≈ 0.926 → gap is architectural; if ≈ 1.0 → gap is trainable

**P3: Defect Field Theory (DFT)** — CONDENSED MATTER PHYSICS ANALOG
- Ternary weight matrix IS a crystal; quantization residuals ARE its defects
- Defect types: VACANCY (W_c≠0, W_t=0 → info loss), INTERSTITIAL (W_c≈0, W_t=±1 → noise), SIGN (opposite signs → interference)
- Defect-defect spatial autocorrelation C_D(r) reveals structure:
  - ξ << 1: point defects (random, benign, average out via CLT)
  - ξ ~ 10-100: line defects (correlated, accumulate, problematic)
  - ξ ~ layer width: grain boundaries (structural, critical)
- Scaling predictions: point defects ∝ O(1/N), dislocations ∝ O(1/√N), grain boundaries ∝ O(N^{-1/3})
- Measured gap scaling (7.4%→1.5%→0.2%) can be FIT to determine dominant defect type
- Layer-specific diagnosis: sign defects → need OptRot; vacancies → need larger groups; interstitials → need better thresholding

**P4: Computational Szilard Engine Efficiency (CSEE)** — THERMODYNAMIC ANALOG
- Model is thermodynamic engine: receives entropy (tokens), extracts work (predictions)
- Work extracted: W_t = H(x_{t+1}) - H(x_{t+1}|h_t) (predictive information gained)
- Dissipation: Q_info_t = I(x_{1:t-1}; h_{t-1}) - I(x_{1:t}; h_t) (historical info lost)
- Efficiency: η_t = W_t / |Q_info_t| — how much prediction per bit of memory erasure
- Landauer bound: ternary has NONZERO MINIMUM dissipation (k × log₂(3) per trit change)
- FP16 Landauer floor → 0 (continuous, in-principle reversible)
- Token-level diagnostic: η_t varies per token → pinpoints specific linguistic phenomena where ternary wastes compute
- Scale prediction: Landauer floor FIXED but W_t grows → efficiency improves with scale

### Unique Ternary Capabilities (P5-P8)

**P5: Proof-Carrying Inference (Cryptographic Computation Certificates)** — ONLY TERNARY
- IEEE 754 is NON-ASSOCIATIVE: (a+b)+c ≠ a+(b+c) → FP16 outputs are hardware-dependent
- Ternary integer arithmetic is EXACT and DETERMINISTIC on any hardware
- Computation certificate: (hash(M), hash(x), hash(f_M(x)), Merkle proof of intermediate states)
- Verifier recomputes selected states → any discrepancy proves tampering
- FP16 verification IMPOSSIBLE: no "correct" output to verify against (hardware-dependent cloud)
- Applications: regulated AI (medical/legal), model tampering detection, AI output provenance
- Per-group FP16 scales: fix to dyadic rationals (k/2^n) for full exactness
- Creates entirely new market category: CRYPTOGRAPHICALLY VERIFIABLE AI

**P6: Algebraic Model Composition (Ternary Weight Algebra)** — ONLY TERNARY
- {-1,0,+1} is CLOSED under multiplication (trit × trit = trit) — FP16 is NOT (overflow)
- Operations: AND (W1⊙W2), OR (union of non-zero paths), XOR (symmetric difference), NOT (-W)
- AND semantics: keep only routing paths where BOTH models agree. Sparsity increases (42%→66%)
- Zero-shot domain adaptation: AND(base_model, domain_model) = shared-knowledge model
- Model debugging: XOR(checkpoint_A, checkpoint_B) = exact diff of routing changes
- Weight-space ensemble: sign(Σ W_k) = majority-vote model. Result is ternary. Exact.
- FP16 merging (SLERP, TIES, DARE) is heuristic; ternary composition is ALGEBRAIC and EXACT

**P7: Complete Circuit Tracing (Exact Mechanistic Interpretability)** — HIGHEST IMPACT
- 42% zero weights → path survival through L layers: P = 0.58^L
  - L=8 (6M): 1.5% paths survive. L=24 (50M): 0.00034% survive
- FP16: 100% of paths carry signal (no exact zeros) → intractable
- Surviving path sign = product of L values from {-1,+1} → either +1 or -1
- influence(j→i) = (n_positive - n_negative) × Π(group_scales) × x_j — EXACT integer computation
- SSM state trace: a_k=+1 accumulates, a_k=-1 oscillates, a_k=0 resets → complete causal map
- Applications: safety-critical explanations, bias detection (provably no path from protected attribute), adversarial robustness certification, model surgery
- FP16 interpretability (activation patching, SHAP) is APPROXIMATE; ternary is EXACT

**P8: FHE-Native Inference (Practical Private AI)** — EXTREMELY HIGH IMPACT
- FHE cost hierarchy: addition (1x) < plaintext×ciphertext (2-3x) < ciphertext×ciphertext (10-50x)
- Ternary layer: y_i = Σ(±x_j) — PURE addition/subtraction of encrypted values. ZERO multiplications
- FP16 layer: 262,144 plaintext-ciphertext multiplications per layer (d_model²)
- Ternary: ~152,000 additions (1x cost) + 4 scale multiplications (group scales only)
- Per-layer speedup: ~20-100x under FHE
- Bootstrapping: FP16 needs every 3-4 layers; ternary every 15-20 (only for nonlinearities)
- Total inference speedup: estimated 50-500x under FHE
- Applications: medical AI on encrypted patient data, financial AI on proprietary data, sovereign AI
- Makes ternary SSM the ONLY architecture viable for practical real-time FHE inference

### KEY PATTERNS (Wave 3)
1. **Discreteness is a FEATURE** — exact computation enables verification (P5), tracing (P7), privacy (P8), composition (P6)
2. **Gap decomposition possible** — architectural vs trainable (P2), defect type analysis (P3), thermodynamic floor (P4)
3. **Path sparsity is exponential** — 0.58^L means at 24 layers, 99.9997% of paths are provably zero (P7)
4. **New market categories** — verifiable AI (P5), algebraically composable models (P6), practical FHE inference (P8)
5. **Ternary weights are NOT degraded floats** — they are exact integers enabling capabilities impossible for continuous weights

### Cross-Wave Synergies (Aspect 6)
- **F2 (Negative Eigenvalues) + M3 (Oscillatory Capacity):** Negative eigenvalues from F2 directly power the oscillatory modes measured by M3
- **F1 (State Rank) + M1 (Fisher Information):** Low-rank but high-Fisher-info heads = the important ones (both say same thing differently)
- **M8 (Nilpotency) + P2 (Channel Capacity):** Nilpotency bounds memory → directly limits channel capacity
- **M9 (STC^0) + P7 (Circuit Tracing):** STC^0 = what ternary circuits CAN compute; P7 = trace WHICH circuits actually fire
- **P5 (Verification) + P7 (Tracing):** Prove output is untampered AND explain exactly why it was produced
- **P5 (Verification) + P8 (FHE):** Encrypted inference + output certificates = complete trust stack
- **F16-F18 (BitNet parity) + M10 (Phase Transitions):** BitNet's scale-dependent parity = evidence of capability phase transitions in flip graph
- **P3 (Defect Field) + P4 (Szilard):** Defect type maps to thermodynamic cost type (erasure/noise/interference)

---

## Aspect 6 Summary: Emergent Capabilities & Evaluation

**Total findings:** 22 web research (F1-F22) + 12 mathematical (M1-M12) + 8 paradigm shifts (P1-P8) = **42 findings**

**Top 5 by estimated impact:**
1. **P7 Complete Circuit Tracing** — exact mechanistic interpretability (0.00034% path survival at 24L)
2. **P8 FHE-Native Inference** — 50-500x FHE speedup, enables practical private AI
3. **M8 Kronecker Nilpotency Theorem** — integer contraction → nilpotent → hard d-token memory limit
4. **P5 Proof-Carrying Inference** — cryptographically verifiable AI outputs (impossible for FP16)
5. **M9 STC^0 Circuit Complexity** — explains WHY coding degrades most (symmetric vs general thresholds)

**Key meta-insight:**
> Ternary weights are NOT degraded floats. They are EXACT INTEGERS enabling:
> - Deterministic computation → cryptographic verification (P5)
> - Massive path sparsity → exact interpretability (P7)
> - Addition-only linear ops → practical FHE inference (P8)
> - Algebraic closure → model composition algebra (P6)
> - Integer kernel → zero numerical error (M7)
> - Modular reduction → native mod-p computation (M11)

**Immediate actions (can implement now):**
- P7 (Circuit Tracing): Pure software diagnostic, no training changes, immediate value
- P3 (Defect Field): Weight analysis only (no model evaluation needed), ~30 min
- P1 (LGCI): 1000 random weight flips, ~2 hours on RTX 5070
- M3 (Oscillatory Capacity): Eigendecomposition of A matrices (already available)
