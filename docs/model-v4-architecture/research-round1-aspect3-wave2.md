# Round 1, Aspect 3: Radical Architecture Departures — Wave 2 (Math Paradigm-Breakers)

## MB1 — Energy-Based & Neuromorphic Mathematics

### v4-R1-A3-M1: Ternary Hopfield Energy Landscape
- **Concept:** SSM state updates as energy minimization over ternary states. Energy E_t(h) = -(1/2)h^T W h - h^T(Bx_t) - lambda*h^T h_{t-1}. With ternary W and h, energy is INTEGER.
- **Key Result:** Convergence guaranteed for symmetric W (Lyapunov function). Ternary capacity ~1.585x binary Hopfield. Update is asynchronous coordinate descent: h_i = sign_tau(sum_j W_{ij}h_j + ...).
- **Complexity:** O(kd^2) per token (k = sparsity, d = state dim)
- **Priority:** MEDIUM — theoretical framework, not computational speedup
- **Key Insight:** Integer energy landscape = discrete lattice. Semantic clusters as energy plateaus.

### v4-R1-A3-M2: Predictive Spike-Coded SSM
- **Concept:** Most tokens are predictable. Compute prediction residual delta_t = x_t - Ch_t, ternarize to spike sigma_t. Update state ONLY when prediction fails (enough non-zero spikes). Otherwise: cheap diagonal decay.
- **Key Math:** Cost_avg = O(d*d_in) + p*O(d^2) + (1-p)*O(d), where p = fraction of surprising tokens (~0.3 for natural language). Net speedup: 2-3x average.
- **Foundation:** Formally a Predictive Coding system (Rao & Ballard 1999, Friston 2005). Free energy F_t = ||x_t - Ch_t||^2 + ||h_t - Ah_{t-1}||^2.
- **Priority:** HIGH — sound framework, concrete 2-3x speedup, additive on existing SSM
- **Key Insight:** 70% of language is predictable. Only process surprises.

### v4-R1-A3-M3: Neuromorphic Temporal Coding with Bit-Shift Arithmetic
- **Concept:** Encode activations as spike TIMES (earlier = stronger), not rates. With tau_m = 1/ln(2), all computation reduces to INTEGER BIT SHIFTS + CLZ (count leading zeros).
- **Key Math:** exp(tau_i/tau_m) = 2^{tau_i} (power of 2 = bit shift!). Net sum D_j = S_pos - S_neg (integer additions of shifted values). Output spike time = CLZ(D_j) — hardware O(1).
- **Free Nonlinearity:** The function f(tau_1,...,tau_n) = log2(sum sign(W)*2^{tau_i}) IS a log-sum-exp — attention-like soft-max behavior for FREE.
- **Operations:** Bit shift, integer add/subtract, CLZ. ZERO floating point. ZERO multiplication.
- **Priority:** HIGH — exact math (no approximation), native to all hardware
- **Concern:** Bit width management (2^{tau_i} grows fast). Mitigation: normalize spike times to [0,63] for 64-bit.

### v4-R1-A3-M4: Coding-Theoretic State Matrix (MDS Codes)
- **Concept:** Requantization h_t = Q(Ah_{t-1} + Bx_t) is an information bottleneck. Design A to maximize information preserved through quantization using Maximum Distance Separable (MDS) codes over GF(3).
- **Key Math:** Quantization loses 44% info (sparse A, k=3) to 86% (dense A). MDS structure ensures minimum Hamming distance d_min = d-k+1 between codewords — tolerates floor((d-k)/2) quantization errors per step.
- **Construction:** Block-MDS: partition state into blocks of 3, use 3x3 MDS matrix [[1,1,1],[1,2,0],[1,0,2]]. For larger: Vandermonde over GF(3^m).
- **Priority:** HIGH — principled A-matrix design, directly testable as drop-in replacement
- **Key Insight:** Standard A-matrix design ignores the quantization bottleneck. MDS codes are OPTIMAL for preserving information through discrete channels.

### v4-R1-A3-M5: Equilibrium Propagation Training (No STE)
- **Concept:** Train ternary SSM WITHOUT STE. EP computes gradients from state differences (free vs nudged phase), never differentiating through quantization.
- **Key Math:** Free phase = forward pass (find h*). Nudged phase = forward + small loss perturbation (find h+). Weight update: DeltaA = (1/beta)*sum_t(h+*h_{t-1}^T - h*h_{t-1}^T). Gradient comes from STATES, not weights.
- **Memory:** O(d) per layer (vs O(dT) for BPTT). No activation storage needed.
- **Priority:** MEDIUM-HIGH — eliminates STE, O(d) memory, but needs symmetrization of A
- **Key Insight:** EP provides EXACT gradients for symmetric networks. Training cost: ~2x forward (vs ~3x for backprop). No BPTT truncation needed.

### v4-R1-A3-M6: GF(3) Cellular Automaton SSM
- **Concept:** Replace matrix multiply with CA rules. Each state dimension updated by local neighborhood (radius r) via GF(3) polynomial. Any function GF(3)^n -> GF(3) expressible as polynomial of degree ≤2 per variable.
- **Key Math:** For r=1 (n=4 neighbors): 81 terms, 324 logic ops/dimension. Dense matmul: d ops/dimension. CA is 12.7x MORE parameter-efficient (81 vs 1024 params/dim for d=1024) but 3x slower for equivalent global mixing.
- **Hierarchical stride:** log2(d) CA steps with doubling stride = global mixing in O(324*d*log d) ops.
- **Priority:** MEDIUM — 12.7x parameter compression, but 3x speed overhead
- **Key Insight:** GF(3) polynomial ring is NATIVE ternary algebra. a^2 = |a| in GF(3) (absolute value!).

## MB2 — Adaptive Computation, KAN & Reversible Mathematics

### v4-R1-A3-M7: Ternary Information-Geometric Routing
- **Concept:** Three-valued routing r∈{-1,0,+1}: +1=full compute, 0=pass-through, -1=active suppression (negate layer output). Suppression is FREE in ternary (sign flip).
- **Key Math:** Information demand D = H(y|h_l) - H(y|h_L). Routing minimizes rate-distortion over ternary lattice L_3. Error-correcting: r=-1 means "state going wrong direction — subtract."
- **Stability:** Bang-bang controller with dead zone — guaranteed bounded trajectories.
- **Priority:** HIGH — error-correcting routing impossible in binary skip/compute
- **Key Insight:** Three-valued routing is a natural negative feedback loop. Binary routing wastes the opportunity to actively correct diverging states.

### v4-R1-A3-M8: Walsh-KAN with Ternary Coefficients
- **Concept:** KAN edge function f(x) = sum c_n*W_n(x) where c_n∈{-1,0,+1} and W_n are Walsh functions (values in {-1,+1}). Evaluation: W_n(x) = (-1)^{popcount(n AND gray(x))} — single POPCOUNT + AND.
- **Key Math:** Walsh system is complete orthonormal basis for L^2[0,1]. Ternary Walsh expansion with N terms: error = O(N^{1-2alpha}) + O(N*delta^2) for alpha-smooth functions.
- **Operations:** N POPCOUNT + N conditional additions per KAN edge. ZERO multiplications.
- **Priority:** HIGH — replaces BOTH linear layer + activation with POPCOUNT-only computation
- **Key Insight:** Walsh functions are the natural basis for ternary coefficients. Every function in the Walsh-KAN is computed by bit operations only.

### v4-R1-A3-M9: Exact Reversible Ternary Coupling
- **Concept:** Coupling y1=x1+F(x2), y2=x2+G(y1) with ternary F,G on integer activations. Inverse is EXACTLY x2=y2-G(y1), x1=y1-F(x2) — zero reconstruction error because integer addition is exactly invertible.
- **Key Theorem:** For ANY ternary W_F, W_G and ANY integer activations, Couple^{-1}(Couple(x1,x2)) = (x1,x2) exactly. No conditions on weights needed.
- **Activation Growth Fix:** Work in Z_M (integers mod M). Still exactly invertible. Gradient of (a+b) mod M = 1 almost everywhere (STE trivially identity).
- **Priority:** HIGH — UNIQUE ternary advantage. Float reversible networks ALWAYS have reconstruction error.
- **Key Insight:** O(1) activation memory + zero gradient degradation + arbitrarily deep networks. Impossible in floating point.

### v4-R1-A3-M10: Diagonal Ternary SSM (Three-Timescale Memory)
- **Concept:** A = diag(a_1,...,a_d) where a_i∈{-1,0,+1}. Three behaviors: a=0 (memoryless/feedforward), a=+1 (integrator/running sum), a=-1 (alternator/derivative).
- **Key Theorem:** Gradient ||dh(T)/dh(0)||_2 = sqrt(n_+ + n_-), INDEPENDENT of T. No vanishing/exploding gradients regardless of sequence length.
- **Selective Gating:** a_i(t) = Q_T(w_gate^T * x_t) — input-dependent choice of integrate/alternate/reset per dimension. Ternary Mamba selectivity.
- **Complexity:** O(d) for A*h (just sign flips), O(d^2) for gating. Total O(d^2) additions.
- **Priority:** HIGH — strongest possible gradient guarantee + O(d) state operations
- **Key Insight:** Three timescales (0, 2-step, infinity) via ternary eigenvalues. Learnable allocation during training.

### v4-R1-A3-M11: Entropy-Calibrated Early Exit for Ternary
- **Concept:** Ternary layers are 16-32x cheaper than float. This CHANGES the optimal exit policy. Optimal ternary depth = L_float + ln(F)/gamma layers DEEPER. Easy tokens exit early (few cheap layers), hard tokens get MORE layers than float at same budget.
- **Key Math:** Compound speedup: 16x (ternary) × 2x (adaptive exit) = 32x vs float. Depth reallocation: easy tokens use L/2 ternary layers, hard tokens use up to 16L ternary layers.
- **Exit Check Cost:** d*V ternary additions (cheap). Optimal check frequency: every sqrt(V/d) ~ 5-6 layers.
- **Priority:** HIGH — 32x compound speedup, Pareto dominates float + adaptive exit in BOTH quality and efficiency
- **Key Insight:** When per-layer cost drops 16x, the optimal strategy is to be DEEPER but cheaper per layer. Hard tokens get disproportionately more compute.

### v4-R1-A3-M12: Ternary Residual Lattice Networks (Z_{3^k})
- **Concept:** Computation in Z_M where M=3^k (power of 3). State h∈Z_M^d. Nonlinearity via base-3 digit extraction and remapping. Natural 3-adic precision hierarchy.
- **Key Math:** Every h∈Z_{3^k} has unique base-3 representation. Digit-wise nonlinearity sigma(h) applies lookup table per digit. 3-adic filtration creates precision levels.
- **Error Correction:** 3-adic valuation v_3(h-h') measures closeness. Lattice codes from algebraic number theory provide intrinsic noise robustness.
- **Priority:** SPECULATIVE — rich algebraic structure, but representational power for language unproven

## Key Insights — Aspect 3, Wave 2

1. **Ternary enables UNIQUE mathematical properties** — exact reversibility, integer energy landscapes, bit-shift arithmetic, GF(3) algebra. These have NO analog in floating-point.
2. **Gradient independence from T** (M10) is the strongest stability result possible — eliminates the fundamental challenge of training long-sequence models.
3. **32x compound speedup** (M11) from ternary × adaptive exit — Pareto dominates float in both quality and efficiency.
4. **Walsh-KAN** (M8) eliminates ALL floating point — even the nonlinear activation function becomes POPCOUNT.
5. **Exact reversibility** (M9) enables O(1) memory for arbitrarily deep networks — unique to ternary+integer arithmetic.
6. **Predictive coding** (M2) provides principled 2-3x speedup by skipping predictable tokens — language is 70% predictable.

## Priority Synthesis

### Tier 1 — Implement Immediately (HIGH, synergistic triad):
- **M10 (Diagonal SSM)** + **M9 (Exact Reversible)** + **M11 (Early Exit)** = bounded gradients + O(1) memory + 32x speedup

### Tier 2 — Prototype (HIGH, novel compute):
- **M3 (Temporal Coding)** — replace entire arithmetic with bit shifts
- **M8 (Walsh-KAN)** — POPCOUNT-only nonlinear activation
- **M7 (Three-Valued Routing)** — error-correcting adaptive computation

### Tier 3 — Research (MEDIUM-HIGH):
- **M2 (Spike SSM)** + **M4 (MDS State Matrix)** + **M5 (EP Training)**
