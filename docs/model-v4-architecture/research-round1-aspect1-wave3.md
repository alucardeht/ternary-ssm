# Round 1, Aspect 1: Wave 3 — Paradigm-Breaker Analysis

## Overview
Two paradigm-breakers explored radical alternatives beyond the current diffusion+SSM candidate. 14 radical paradigms/combinations identified. Several have HIGH feasibility and could provide 100-2500x efficiency gains.

---

## PB1: Radical Paradigm Alternatives (Beyond Diffusion+SSM)

### v4-R1-A1-P1: Memory-Computation Separation (MEDIUM-HIGH)
- **Concept:** Split model into tiny ternary SSM "reasoning core" (~100-500M params, ~125MB) + massive ternary associative memory (~10B entries, ~1.2GB). Core learns HOW to think, memory stores WHAT.
- **Mechanism:** Core generates query vectors, memory retrieves facts via Hamming distance (XOR + POPCOUNT). Non-gradient memory updates: wrong fact → add correct (key, value) pair.
- **Advantage:** 100M core + 10B memory = knowledge of 100B model in ~1.3GB total. Knowledge updates without retraining.
- **Risk:** Interface between reasoning and memory may bottleneck quality. Ternary Hamming retrieval may lack semantic nuance.
- **Min experiment:** Add 1M-entry ternary memory to 50M SSM. If PPL drops >5% with zero core size increase, paradigm validated.

### v4-R1-A1-P2: Ternary Energy-Based Potts Model (MEDIUM)
- **Concept:** Abandon generative model. Define energy function E(input, output) using ternary weights (3-state Potts model). Generation = finding minimum-energy output configuration via combinatorial optimization.
- **Mechanism:** E = -(count agreeing features) + (count disagreeing features). Pure counting, no FP. Local search: for each position, try all token substitutions, pick lowest delta_E.
- **Advantage:** All positions optimized simultaneously. Natural variable compute. Connection to Ising/Potts hardware (D-Wave, Digital Annealer).
- **Risk:** EBMs historically struggle with text quality. Local minima traps. No production EBM LM at frontier quality.
- **Min experiment:** Reformulate 6M SSM as EBM. 128 output candidates, 20 steps local search. If within 20% PPL of AR, paradigm merits scaling.

### v4-R1-A1-P3: Predictive Coding with Ternary Error Propagation (MEDIUM)
- **Concept:** Each layer maintains persistent belief state. Only PREDICTION ERRORS propagate between layers. Ternary quantization maps near-zero errors to exactly 0 → >90% of communication is ZERO.
- **Mechanism:** Top-down prediction, bottom-up error, belief update. All ternary. "Infinite effective depth" via iteration (like Deep Equilibrium Models). LOCAL Hebbian learning rules — no backpropagation needed.
- **Advantage:** O(1) memory for training. >90% sparsity in inter-layer communication. Variable compute per input difficulty. Ideal for neuromorphic hardware.
- **Risk:** Convergence speed unknown for language. Not demonstrated beyond CIFAR-100 scale. Ternary requantization may accumulate error.
- **Min experiment:** Modify 6M SSM to weight-tied layers (DEQ style). Run 10 iterations. Measure convergence + sparsity of prediction errors.

### v4-R1-A1-P4: Hyperdimensional Computing with Ternary Algebra (MEDIUM)
- **Concept:** Replace matrix multiplication entirely with element-wise operations on d=10,000+ dimensional ternary vectors. Only three ops: binding (element-wise product), bundling (majority vote), permutation (free pointer shift). All O(d), not O(d^2).
- **Mechanism:** Sequence encoding via shift+bundle. Prediction via ONE ternary matrix multiply. Token retrieval via Hamming distance (625 POPCOUNT ops for d=10K).
- **Advantage:** Encoding O(n*d) instead of O(n*d^2*L) = 320,000x fewer ops. Single matrix W is the ONLY learned component. Model = ~20MB.
- **Risk:** HDC not demonstrated for language generation at scale. Representational capacity may be insufficient for LLM-quality semantics.
- **Min experiment:** Encode WikiText via HDC (d=10K), train single ternary matrix. If PPL within 2x of 8-layer SSM, validates HDC encoding.

### v4-R1-A1-P5: Kolmogorov Compression Machine / Finite-State Model (LOW-MEDIUM)
- **Concept:** A ternary SSM with d-dim state IS a finite automaton with 3^d states. Build transition table EXPLICITLY as lookup rather than computing via matrix multiply. Hierarchical lookup or decision tree approximation.
- **Mechanism:** Split state into K blocks. Each block has independent lookup table (3^(d/K) entries). For d=512, K=32: each table ~43M entries, total ~350MB. O(K) lookups instead of O(d^2) matmul.
- **Advantage:** O(K) or O(d*log(d)) per token instead of O(d^2). 56x speedup for d=512.
- **Risk:** Block-independent lookup loses cross-block interactions. Compressibility of learned ternary transitions unknown.
- **Min experiment:** Extract transition function from trained SSM. Measure compressibility. If >80% compressible, paradigm viable.

### v4-R1-A1-P6: Asynchronous Cortical Columns (MEDIUM-HIGH)
- **Concept:** Ensemble of 100-1000 tiny specialized ternary SSM circuits (~500K params each) that communicate via shared ternary state vector. Columns activate ONLY when relevant. Router selects 3-10 active columns per token.
- **Mechanism:** Each column reads relevant slice of shared state, produces update delta. Updates summed, requantized. ~5/500 active = 100x less compute.
- **Advantage:** Natural specialization. Modular knowledge (add columns for new domains). Asynchronous execution. Inspectable reasoning (which columns activated).
- **Risk:** Router must learn relevance without supervision. Dead expert problem amplified. Communication bottleneck through shared state.
- **Min experiment:** Replace 18 homogeneous layers with 100 tiny columns + router. If avg active <10 and quality within 15%, paradigm works.

### v4-R1-A1-P7: Thermodynamic Ternary Computation (LOW-MEDIUM)
- **Concept:** Reformulate ternary SSM as Potts physical system. Energy landscape E(h) = -sum w_ij h_i h_j. Inference = relaxation to thermal equilibrium. Training via equilibrium propagation (LOCAL Hebbian rules, proven to compute exact gradients).
- **Advantage:** No backpropagation. O(1) training memory. Natural connection to Ising hardware. Variable depth via iteration count.
- **Risk:** Not demonstrated beyond CIFAR-10. Frustration may slow convergence. Phase transitions may cause training instability.
- **Min experiment:** Tiny Potts model (d=256), character-level LM. Test equilibrium propagation convergence.

---

## PB2: Extreme Multiplicative Combinations

### v4-R1-A1-P8: ANS-Native Generation — Compression=Generation Duality (HIGH)
- **Concept:** Generation IS decompression. Replace softmax sampling, top-p, temperature with Asymmetric Numeral Systems (ANS) decoding. Mathematically proven optimal. EMBRACES sequential SSM nature instead of fighting it with diffusion.
- **Mechanism:** Same SSM forward pass produces P(x_t | context). ANS decode: table lookup + integer division → exact sample from P. No FP in generation step.
- **Advantage:** Samples from EXACT distribution (zero approximation). Deterministic (same seed = same output). No temperature/top-p tuning. SSM stays unidirectional (natural form). Dual-use: same model compresses AND generates.
- **Multiplicative gains:** 8x (ternary memory) × 50x (POPCOUNT compute) × 1.5x (no sampling overhead) × 2x (dual-use) = ~1200x theoretical, ~150-500x realistic.
- **Critical insight:** "The project's diffusion candidate FIGHTS the SSM architecture. SSMs are sequential. Diffusion is parallel. ANS EMBRACES the sequential nature."
- **Risk:** ANS table reconstruction per step is O(V) = 50K ops. Delta updates can mitigate.
- **Min experiment:** Existing 50M SSM + ANS decode vs standard sampling. Compare quality + speed.

### v4-R1-A1-P9: Retrieval-Augmented Ternary Router (MEDIUM-HIGH)
- **Concept:** 5M ternary router (~1.2MB, fits L1 cache) + terabyte activation database. Router approximates context, database corrects with precomputed precise activations.
- **Multiplicative gains:** L1 cache router (100x bandwidth) × POPCOUNT hash (1 cycle) × unlimited knowledge scale from database.
- **Risk:** SSD random read latency (~10μs) may bottleneck at >100K tok/s. Novel contexts degrade to router-only quality.
- **Min experiment:** Train 5M router, store h_t vectors during training. Compare router-alone vs router+database vs 50M full model.

### v4-R1-A1-P10: Ternary Potts + Parallel Gibbs Generation (MEDIUM)
- **Concept:** Energy landscape over (context, output) with ternary couplings. Parallel Gibbs sampling: all T positions updated simultaneously per iteration. Converges in 3-8 iterations.
- **Multiplicative gains:** 50x (POPCOUNT) × T/5 (parallel positions) × 2x (integer pipeline) = ~10,000x for T=500.
- **Risk:** Training EBMs is notoriously difficult. Gibbs mixing for complex language distributions unproven.

### v4-R1-A1-P11: Ternary Matrix Product State / Tensor Network (MEDIUM)
- **Concept:** Replace layers of neurons with tensor network (MPS). Each token selects a r×r ternary matrix. Product of selected matrices encodes ALL correlations. Bond dimension r controls expressiveness.
- **Model:** V×r×r = 50K×128×128 = ~800M ternary params = 200MB. NO attention needed.
- **Multiplicative gains:** 8x (ternary) × ∞ (no O(n²) attention) × 50x (POPCOUNT) × 5x (parallel tree contraction) = ~2000x for long sequences.
- **Risk:** MPS training at language scale unvalidated. "Mostly 1D" assumption about language correlations may be wrong.

### v4-R1-A1-P12: Codebook Memory + Incremental Compilation (HIGH)
- **Concept:** Track frequent (context_state, output_distribution) pairs. Cache direct mappings. At inference, check codebook FIRST → O(1) lookup. Only fall through to full SSM for novel inputs.
- **Hit rate:** ~40-60% for natural language (common phrases, grammar). Model gets FASTER with use.
- **Multiplicative gains:** 50x (POPCOUNT for misses) × 2x (codebook saves 50% of compute) = ~100x blended.
- **Risk:** Codebook may grow too large. Distribution shift makes entries stale.
- **Min experiment:** Run 50M SSM on test set, store state_hash→logits dict. Second pass: measure hit rate + KL divergence.

### v4-R1-A1-P13: Bidirectional SSM + Jacobi Fixed-Point Generation (MEDIUM)
- **Concept:** Refinement of diffusion candidate. Use model's OWN predictions as fixed-point iteration (not separate denoising process). Start all-mask, iterate until convergence. SSM contractiveness guarantees convergence.
- **Multiplicative gains:** 50x (POPCOUNT) × T/10 (parallel) = ~2500x for T=500.
- **Risk:** Bidirectional SSM breaks O(1)-memory-per-token. Training bidirectional ternary SSM unvalidated.

### v4-R1-A1-P14: Ternary Kolmogorov-Arnold SSM (LOW-MEDIUM)
- **Concept:** Replace fixed activation functions with learnable ternary-parameterized B-splines on edges (KAN). Each spline = piecewise constant via ternary coefficients.
- **Advantage:** KAN shows 100x parameter efficiency for smooth functions. Spline eval = table lookup (10x faster than SiLU).
- **Risk:** KAN + ternary + SSM + language = completely unexplored territory. Ternary splines may be too coarse.

---

## Recommended Priority Paths

### PATH A: Sequential Optimization (HIGH feasibility, 150-1000x)
**ANS-Native Generation (P8) + Codebook Caching (P12)**
- SSM stays unidirectional (proven, natural)
- ANS replaces softmax (mathematically optimal)
- Codebook accelerates frequent patterns
- Can be added to existing 50M model with ZERO architectural changes
- Timeline: 2-4 weeks proof of concept

### PATH B: Parallel Generation (MEDIUM feasibility, 2500x)
**Bidirectional SSM + Jacobi (P13) + Energy Model (P10)**
- Parallel generation via iterative convergence
- Ternary Potts energy formulation natural for integer arithmetic
- Requires bidirectional SSM (new capability)
- Timeline: 1-2 months

### PATH C: Paradigm Shift (MEDIUM feasibility, potentially unlimited)
**Memory-Computation Separation (P1) + Cortical Columns (P6) + HDC (P4)**
- Tiny specialized columns + massive external memory + HDC encoding
- O(d) per token instead of O(d^2 × L) = 32,000x theoretical
- Timeline: 3-6 months

### PATH D: Long-term Moonshot (LOW-MEDIUM feasibility)
**Tensor Network MPS (P11) + Predictive Coding (P3)**
- Exponential compression of correlations
- No backpropagation, O(1) training memory
- Timeline: 6-12 months

## Key Insight from PB2
> "The project's candidate Ternary Bidirectional SSM + Masked Diffusion + Jacobi Iteration is FIGHTING ITSELF. SSMs are sequential. Diffusion is parallel. Making SSMs bidirectional to serve diffusion's needs adds complexity to solve a problem that arithmetic coding already solves. ANS EMBRACES the sequential nature. The compression=generation duality is the NATURAL formulation for a sequential model like an SSM."
