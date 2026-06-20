# Round 1, Aspect 3: Radical Architecture Departures — Wave 3 (Paradigm-Breakers)

## PB1 — Radical Alternatives Beyond Neural Networks

### v4-R1-A3-P1: Tropical SSM — Language as Shortest-Path Computation
- **Concept:** Replace (multiply, add) semiring with (min, add) TROPICAL semiring. Forward pass = shortest-path computation through weighted graph. With ternary edges {-1,0,+1}, every transition adds/subtracts/zero cost.
- **Key Math:** h_i[t] = min(min_j(A_ij + h_j[t-1]), min_k(B_ik + x_k[t])). Output = argmin(cost), not argmax(probability). Decoding IS Viterbi. Tropical matrix composition = O(d^3 log L) for L layers.
- **Operations:** INTEGER COMPARE (min), INTEGER ADD, ARGMIN. Zero multiplication. Even cheaper than POPCOUNT.
- **Advantage:** Exact polynomial-time algorithms. Tropical polynomials are piecewise-linear = formally verifiable.
- **Feasibility:** MEDIUM-HIGH
- **Risk:** min operation selects one input (not blend) — may need wider networks.

### v4-R1-A3-P2: Standing Wave Language Model — Bidirectional Resonance
- **Concept:** Model is a resonant cavity, not a pipeline. Input/output are boundary conditions. Answer = standing wave pattern from constructive interference of forward + backward waves.
- **Key Math:** F_l = W_fwd*F_{l-1} + inject(x,l); B_l = W_bwd*B_{l+1} + inject(prompt,l). Intensity I_l = |F_l + B_l|.
- **Generation:** Initialize blank outputs. Run forward+backward sweeps. Positions where wave "collapses" = output tokens. Natural adaptive compute (uncertain positions get more iterations).
- **Advantage:** Parallel generation. Time-reversible (editing = changing boundary conditions). Ternary weights = crystal lattice with band gaps.
- **Feasibility:** MEDIUM
- **Risk:** Unknown convergence iterations. Language may not decompose into frequency bands.

### v4-R1-A3-P3: Ballot Machine — Width Replaces Depth Entirely
- **Concept:** 100K independent 1D voting agents replace 24-layer deep network. Each agent: ternary projection + ternary state mode {integrate/alternate/reset} + ternary vote. Output = majority vote.
- **Key Math:** Reservoir computing (Jaeger/Maass) + JL lemma for ternary projections. Three state modes = PID decomposition (proven complete for control systems).
- **Hardware:** Embarrassingly parallel. 100K agents fit in 400KB (L1 cache). All POPCOUNT + integer add.
- **Self-modification:** Track per-agent accuracy. Mute consistently wrong agents at inference time.
- **Feasibility:** MEDIUM
- **Risk:** Width-for-depth may need massive width. 100K × (d+V) = ~5B ternary params = 1.2GB.

### v4-R1-A3-P4: Constraint Propagation Engine — Language as SAT
- **Concept:** Weight W_{ij}=+1 means features i,j must agree. W_{ij}=-1 means disagree. W_{ij}=0 = unrelated. Generation = find assignment satisfying maximum constraints. Uses SAT propagation (deterministic, fast), not sampling.
- **Key Math:** Energy E(s) = -sum W_{ij}*s_i*s_j (weighted MAX-SAT). Unit propagation in O(V+E). If constraint graph is 2-SAT structure: LINEAR TIME generation.
- **Advantage:** Deterministic for "easy" text. Natural uncertainty quantification. Formal verifiability. Backtracking for self-correction.
- **Feasibility:** MEDIUM
- **Risk:** MAX-SAT is NP-hard in general. Language may not have enough "constrainty" structure.

### v4-R1-A3-P5: Frequency-Domain WHT-SSM (10,000x Compute Reduction)
- **Concept:** Process tokens in Walsh-Hadamard frequency domain instead of time domain. SSM convolution becomes POINTWISE in WHT domain. With ternary A={-1,0,+1}, the SSM kernels have SPARSE WHT representations.
- **Key Math:** WHT(integrator kernel [1,1,1...]) = single nonzero coefficient (DC). WHT(alternator [-1,1,-1...]) = single coefficient (R_1). WHT: O(T log T) additions ONLY (no multiply). Pointwise ternary filtering: O(T) sign flips.
- **Total cost:** O(L*T*log T) additions + O(L*T) sign flips. For L=24, T=1024: ~240K operations. Standard SSM: 6.4B operations. **26,000x reduction.**
- **CRITICAL INSIGHT:** Making A ternary is the LINCHPIN. It unlocks 10,000x compute reduction AND gradient independence (M10). Quality cost of 3 eigenvalues vs continuous spectrum is tiny compared to computational gain.
- **Feasibility:** MEDIUM-HIGH
- **Risk:** Three kernel types (DC, alternating, passthrough) may limit temporal resolution. Empirical question.

### v4-R1-A3-P6: Quasicrystal Growth — Crystallographic Generation
- **Concept:** Tokens are "tiles" with ternary edge markings. Weight matrix defines matching rules. Generation = crystal growth satisfying local matching constraints.
- **Score:** POPCOUNT(W_embed[k] XNOR b[t]) — count matching edges.
- **Advantage:** Defect detection (grammatical errors = lattice strain). Phase transitions for temperature control. Principled creativity via controlled defects.
- **Feasibility:** LOW-MEDIUM
- **Risk:** Highly speculative. Connection between matching rules and language unproven.

### v4-R1-A3-P7: Hierarchical Ternary Codec — Multi-Resolution Generation
- **Concept:** Model = codec, not predictor. Decompose meaning into hierarchy: topic → paragraph → sentence → phrase → token. Each level predicted from above with TERNARY RESIDUALS.
- **Key Math:** If 70% predictable at each level, 4 levels: effective computation = 0.3^4 = 0.81% of full. **123x reduction** from hierarchy alone.
- **Advantage:** Anytime generation (coarse first, refine progressively). Semantic control at each level. Extreme compression.
- **Feasibility:** MEDIUM
- **Risk:** Language's hierarchical decomposition is semantic, not geometric. Sparsity may not reach 70%/level.

## PB2 — Unified Architecture Synthesis

### v4-R1-A3-P8: "The Surprise Processor" (~160x Speedup)
- **Components:** M10 (diagonal SSM) + M2 (predictive spike) + M7 (three-valued routing) + M4 (MDS B matrix) + M11 (early exit)
- **How it works:** Token arrives → predict from state → if predictable (70%), skip expensive layers. If surprising, route through layers. Three-valued routing selects skip/compute/suppress per layer. Early exit when confident.
- **Compound speedup:** 0.3 (surprising tokens) × 0.5 (layer routing) × 0.5 (early exit) × 16 (ternary) ≈ **160x** (conservative ~80x for unpredictable text)
- **Weakest link:** Predictability varies by domain. Degrades GRACEFULLY — never worse than 16x baseline.
- **Feasibility:** HIGH — safest high-efficiency option

### v4-R1-A3-P9: "The Ternary Finite Machine" (~16-50x, Simplest)
- **Components:** M10 (diagonal SSM) + M4 (MDS B matrix) + M8 (Walsh-KAN) + M9 (exact reversible)
- **How it works:** No adaptive computation. Pure stack: diagonal ternary SSM state + MDS-coded input projection + Walsh-KAN nonlinearity + reversible coupling for O(1) memory.
- **Speed:** ~16x raw ternary, potentially ~30-50x with hardware optimization. CPU: ~1,887 tok/s at 50M.
- **Weakest link:** Walsh-KAN expressiveness with ternary coefficients. May need N=256 terms or small FP16 residual.
- **Feasibility:** VERY HIGH — zero tensions between components, simplest possible architecture.

### v4-R1-A3-P10: Key Conflict Resolutions + Implementation Path
- **MDS vs Diagonal SSM:** MDS in B matrix (information preservation), diagonal in A (gradient stability). COMPATIBLE — different roles.
- **Reversible vs Early Exit:** Reversible for TRAINING (all layers, O(1) memory). Early exit for INFERENCE (variable layers). No conflict.
- **Predictive Skip vs Layer Routing:** ORTHOGONAL axes. M2 = spatial sparsity (which features). M7 = depth sparsity (which layers). MULTIPLY: 0.3 × 0.5 = 15% compute.
- **M10 is universal:** Appears in ALL 7 unified architectures. Strongest stability guarantee for ANY recurrent model.
- **Implementation path:** Phase 1: Validate Walsh-KAN (1 week) → Phase 2: Build Arch 7 "Ternary Finite Machine" (2-3 weeks) → Phase 3: Add predictive coding M2 (1-2 weeks) → Phase 4: Add routing+exit = Arch 2 "Surprise Processor" (2-3 weeks)

## Key Insights — Aspect 3, Wave 3

1. **WHT-SSM (P5) is potentially the biggest single finding**: 10,000-26,000x compute reduction by processing in Walsh-Hadamard frequency domain. Requires ternary A (already justified by M10).
2. **"Surprise Processor" (P8) is the safest path to ~160x speedup** — all components are individually validated, degrades gracefully.
3. **"Ternary Finite Machine" (P9) should be built FIRST** — simplest, zero conflicts, VERY HIGH viability.
4. **Tropical SSM (P1) provides alternative PARADIGM** — min/add replaces mul/add, Viterbi-native decoding.
5. **All conflicts are RESOLVED**: MDS in B + diagonal A, reversible training + early exit inference, orthogonal sparsity axes.
6. **M10 (diagonal ternary SSM) is the universal foundation** — appears in every viable unified architecture.
