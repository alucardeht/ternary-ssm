# Research Round 6 — Training Velocity: Precision Flow & Structural Optimization

## Overview

This document tracks research findings on training velocity techniques for ternary SSM models discovered in Research Cycle 6, Aspect 2 (Feb 2026).

## Wave 1: Web Research (2 researchers, Feb 2026)

### Precision Flow & Quantization-Aware Training

**F1: InfiR2 Hybrid-Granularity FP8 Training** (arXiv 2509.22536v3, 2025)
- Hybrid FP8 quantization: block-wise for weights (E4M3), token-wise for activations, UE8M0 scaling
- Power-of-2 rounding for scale factors ensures hardware efficiency
- 22% faster training, 14% memory reduction, 19% throughput increase vs BF16
- Directly applicable to ternary: mixed-granularity quantization + proper scaling proven lossless

**F2: Smooth-SwiGLU Outlier Prevention** (arXiv 2409.12517v2, Intel/Technion, 2025)
- Per-channel scaling on SwiGLU linear branch prevents outlier amplification
- Weight alignment phenomenon: with L2 regularization, gated weight pairs align over time
- Correlation→1 causes quadratic outlier growth after 200B+ tokens
- Enables stable FP8 training to 2 trillion tokens (20x previous limit), ~34% throughput gain
- Directly relevant: ternary models use gated activations, same alignment risk applies

**F3: FP8 Optimizer Moments Quantization** (arXiv 2409.12517v2, 2025)
- First successful quantization of BOTH Adam moments to FP8 (m: E5M2, v: E4M3)
- Delayed scaling with per-tensor factors
- Reduces optimizer memory 2x vs FP16, 4x vs FP32
- For ternary training: optimizer states dominate VRAM — FP8 moments enable larger batches

**F4: STE + Zeroth-Order Gradient Augmentation** (arXiv 2510.23926, NeurIPS 2025)
- Augments STE with zeroth-order estimates (finite differences) for better gradient approximation
- Hybrid first-order (STE) + zeroth-order estimator reduces quantization gradient error
- Improves convergence speed AND final accuracy for quantized training
- Directly improves our core STE technique for ternary weights

**F5: Activation Compression for Memory-Bound Training** (ActNN/GACT, referenced 2025)
- Compress activations stored for backward pass using quantization + lossless encoding
- Decompress on-the-fly during backprop
- 2-4x activation memory reduction, enables larger batch sizes
- Critical for our VRAM-bound SSM training: batch=64→96+ without OOM

**F6: Ternary Encoding Optimization** (BitNet analysis, 2026)
- Encode {-1,0,+1} as {10,00,01} to maximize zero-bit ratio (94% vs 50% for INT2)
- Zero-masking skips zero-gradient computations during weight updates
- Reduces effective memory traffic by ~50%
- Combined with sparsity-aware training: skip entire zero-weight rows in gradient accumulation

**F7: Weight Alignment Phenomenon in Gated Activations** (Fishman et al., 2025)
- FUNDAMENTAL training dynamic: with L2 reg, SwiGLU weight pairs (w1, w2) align over time
- At stationary point: either w1→w2 or w1→-w2
- Causes quadratic outlier amplification after extended training
- Monitor weight correlation during ternary training; apply smoothing if alignment detected
- Explains late-training instabilities in quantized models

### Structured Compression & Training Optimization

**F8: LayerLock — Progressive Layer Freezing** (arXiv 2509.10156, ICCV 2025)
- Progressively freeze layers shallow→deep based on convergence order
- Observation: shallow layers converge FIRST (same in ViT, likely SSMs)
- Accelerates training by reducing gradient computation in frozen layers
- For ternary SSM: freeze converged layers to reduce STE overhead, focus capacity on deeper layers

**F9: Progressive Data Dropout** (arXiv 2505.22342v2, May 2025)
- Progressively drop training samples each epoch (difficulty-based or random schedule)
- Final epoch uses full dataset as "revision" — Schedule-Matched Random Dropout (SMRD) best
- Reduces effective epochs to 12.4% of baseline, +4.82% accuracy improvement
- Zero architecture changes. Apply to any training pipeline including ternary SSM

**F10: Gumbel Noise + STE for Discrete Networks** (NeurIPS 2025, poster 117069)
- Inject Gumbel noise with straight-through estimator for discrete weight networks
- Provides implicit Hessian regularization — provably reduces discretization gap
- 4.5x faster training, 98% reduction in discretization gap, 100% unused gate elimination
- Directly applicable to ternary STE: Gumbel noise during forward pass, STE in backward

**F11: Critical Batch Size Theory** (arXiv 2505.23971, NeurIPS 2025)
- Empirical method to determine optimal batch size for LLM training
- Too-large batches harm token efficiency (diminishing returns)
- Provides theory explaining why batch=64 was our optimum at 50M scale
- Can predict optimal batch size at larger scales (200M, 1B)

**F12: WSD Cooldown Optimization** (arXiv 2508.01483, Aug 2025)
- Higher β₂ during cooldown phase improves final performance
- "River valley" loss landscape theory explains nontraditional WSD loss curves
- Cooldown shape reveals bias-variance tradeoff
- Applicable to ternary training: optimize final training phase for better weight consolidation

**F13: COMP Hybrid-Granularity Structured Pruning** (arXiv 2501.15255, Jan 2025)
- Layer-grained + neuron-grained pruning with matrix condition-based importance
- 6.13% better than LLM-Pruner at 20% pruning ratio, 80% memory reduction
- Post-training: no retraining required
- For ternary SSM: identify which layers tolerate aggressive quantization via condition number

**F14: Meta-Quantization for Learned Codebooks** (ICML 2025, poster 43509)
- Hyper-network generates quantization dictionary, main network trains with it
- Two-level meta-learning: outer loop optimizes codebook, inner loop trains model
- Better codebook utilization than fixed dictionaries
- Could learn optimal ternary thresholds per layer instead of using global α scaling

### KEY PATTERNS (Wave 1)
1. **FP8 ecosystem maturing** — full-stack FP8 (weights + activations + optimizer) now validated (F1, F3)
2. **Gated activation alignment** — fundamental training instability affects ALL gated models (F2, F7)
3. **Progressive methods dominate** — freezing (F8), dropout (F9), precision (F12) all progressive
4. **STE improvements converging** — zeroth-order augmentation (F4) + Gumbel noise (F10) both improve discrete training
5. **Memory is king** — activation compression (F5) + optimizer FP8 (F3) together give 4-6x VRAM reduction
6. **Post-training optimizations underexplored** — structured pruning (F13) + meta-quantization (F14) can compress after training

## Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

### Training Acceleration Mathematics (M1-M6)

**M1: Flip-Score Coordinate Descent (FSCD)**
- After each STE backward pass, compute first-order loss change for every possible ternary flip
- Flip score: s_{ij}(q→q') = (dL/dy_i) · α · (q'-q) · x_j — reuses cached backprop quantities
- Top-k flips executed directly on Q(W), complementing standard STE shadow weight updates
- Flip-rate schedule: ρ_t = ρ_0·(1-t/T)² — start aggressive (1%), decay to pure STE
- Cost: O(mn) per layer (dominated by existing matmul, ~10% overhead)
- Expected: 1.4-2.3x wall-clock speedup. ~70 lines of code.

**M2: Categorical Natural Gradient (CNG)**
- Reparameterize each weight as categorical distribution over {-1,0,+1} with logits θ
- Fisher information matrix for categorical: F = diag(p) - pp^T (closed-form 2×2 inverse)
- Natural gradient: Δθ = -η·F⁻¹·∇θL — rare categories get AMPLIFIED updates
- Gumbel-Softmax with temperature annealing (τ: 1.0→0.1) for differentiable sampling
- Expected: 2-4x fewer steps (natural gradient is Cramer-Rao optimal), 1.3x cost per step
- Net: 1.5-3x speedup. ~110 lines. 1.5x memory (2 logits per weight).

**M3: Ternary Langevin Dynamics with Accumulated Potential (TLDAP)**
- ELIMINATES shadow weights entirely — ternary weights as primary representation
- Potential field φ (FP16) accumulates gradient evidence: φ ← γ·φ + α·g_i·x_j·Q_{ij}
- Flip probability via Boltzmann: p(q→q') = min(1, exp(-Δφ/T_t))
- Temperature schedule: T_t = T_0·max(ε, 1-t/T) — Metropolis-Hastings acceptance
- Memory: Q (2 bits) + φ (FP16, 2 bytes) = 2.25 bytes/weight vs 10 bytes (4.4x reduction)
- SPECULATIVE: convergence rate unknown. Could be 0.5-3x. Needs empirical validation.

**M4: Boundary-Attentive Gradient Reweighting (BAGR)**
- STE gradient is uniform — wastes budget on weights deep inside Voronoi cells
- Boundary attention: a(w) = exp(-d(w)²/(2σ²)) where d(w) = distance to nearest boundary
- Near boundary (a≈1): full STE gradient. Far from boundary (a≈0): only sign(gradient) (SignSGD)
- Rationale: interior weights only need direction, not magnitude (any movement within cell is useless)
- σ schedule: σ_t = σ_0/(1+t/T) — narrow over time to concentrate at boundaries
- Expected: 1.3-2x fewer steps, ~0% overhead. ~35 lines. **Most efficient per line of code.**

**M5: Analytic Surrogate Gradient via Kernel Smoothing (ASGKS)** — BEST THEORETICAL VALUE
- Replace STE gradient (dQ/dW=1) with exact derivative of smooth kernel-smoothed Q
- Q_τ(w) = Σ_k k·K_τ(w,k·α) where K_τ is Gaussian softmax over ternary codebook
- Properties: Q_τ→Q as τ→0; Q_τ(k·α)=k exactly; infinitely differentiable
- Key: dQ_τ/dw = 0 at ternary values (correct!), MAXIMAL at boundaries (correct!)
- Mathematical INVERSE of STE: concentrates gradient where it matters
- Temperature: τ_t = τ_0·(τ_f/τ_0)^{t/T}. Cost: 3 exp() per weight (<0.1% overhead)
- Expected: 1.5-3x fewer steps. ~45 lines. **Highest expected value.**

**M6: Variance-Reduced Ternary Gradient (VRTG)**
- Every K steps: compute exact flip-losses for m sampled weights in last 2 layers
- Exact flip-loss for last layer: Delta_L = CrossEntropy(logits + α·(q'-q)·x_j·e_i) - L
- Control variate: g_CV = g_STE - β·(g_STE[S] - g_exact[S])
- Variance reduction: Var(g_CV) = Var(g_STE)·(1-ρ²) where ρ = Corr(g_STE, g_exact)
- Cost: O(m·V) every K steps. For m=1000, V=50K, K=10: amortized 0.05% overhead
- Expected: 1.2-1.8x. ~90 lines. Negligible overhead, guaranteed improvement.

### Training Efficiency Mathematics (M7-M12)

**M7: Boundary-Proximal Sparse Optimizer (BPSO)**
- Only weights NEAR quantization boundaries need full Adam optimizer state
- Partition: Active set A (d<ε, full AdamW) vs Dormant set D (d≥ε, INT8 sign accumulator)
- Dormant update: c ← clamp(c + sign(g), -127, +127). Promotion when |c| > T_promote
- Active fraction f ≈ 0.1-0.3 empirically → Memory: f·12 + (1-f)·1 bytes/param
- For f=0.2: 3.75x memory reduction. For f=0.1: 5.7x reduction
- Implementation: ~80 lines. HIGH viability.

**M8: Ternary Transition Voting (TTV)** — PARADIGM SHIFT
- ELIMINATE shadow weights + Adam entirely. Replace with vote counters.
- Per weight: s ∈ {-1,0,+1} (2 bits) + V ∈ [-127,+127] (INT8) = 1.25 bytes total
- Vote: V ← clamp(V + sign(g), -127, +127). Transition when |V| > threshold T
- T schedule: T(t) = T_init·(1+t/T_warmup) — increasing T = LR decay equivalent
- Memory: 1.25 bytes vs 12 bytes = **9.6x reduction**
- Zero floating-point optimizer computation. Only integer addition + comparison.
- Convergence: sequential probability ratio test (SPRT) — provably optimal decision time
- MEDIUM viability — highest risk but paradigm-shifting potential.

**M9: Bayesian Ternary State Estimation (BTSE)**
- Each weight: latent categorical s ∈ {-1,0,+1} with posterior π = (log p₋₁, log p₀, log p₊₁)
- Bayesian update: log p_s ← log p_s - γ·g·(s - s_current)
- Weight = MAP estimate: s = argmax_s π(s)
- Temperature: γ(t) = γ_0·(1+t/T) — posterior sharpens over training
- Memory: 2 FP16 values = 4 bytes/param (3x reduction). INT8: 2 bytes (6x reduction)
- Compute: 2 multiply-adds per param vs Adam's 5 = ~60% fewer optimizer FLOPs
- Theoretically elegant: Bayesian bandits achieve optimal explore-exploit tradeoff

**M10: Compressed Shadow via Ternary Residual Coding (CTRC)**
- Decompose shadow: w = c_{s} + r where s = ternary state, r = bounded residual
- Residual r stored as INT8 with stochastic rounding (preserves expected value)
- First moment m: INT8 with per-group scale. Second moment v: Adafactor-style row+col factors
- Memory: ~2.31 bytes/param vs 12 bytes = **5.2x reduction**
- Conservative approach: each component is proven. Combination is novel for ternary.
- HIGH viability — lowest risk of all efficiency approaches.

**M11: Reversible SSM Activation Recovery (RSAR)** — ORTHOGONAL TO OPTIMIZER SAVINGS
- SSM recurrence h_t = A_t⊙h_{t-1} + B_t·x_t is INVERTIBLE when A_t ≠ 0
- Inverse: h_{t-1} = (h_t - B_t·x_t) / A_t — elementwise division (diagonal A)
- Recover intermediate states during backward instead of storing them
- Error amplification: (1/a_min)^K per checkpoint interval K
- For a_min=0.99, K=32: amplification = 1.38 (acceptable)
- Memory: 32x reduction in SSM activation storage (768MB → 24MB at 50M scale)
- Compute: ~doubles SSM backward (small fraction of total training)
- HIGH viability. ORTHOGONAL to all optimizer compression techniques.

**M12: Token-Selective Gradient Accumulation (TSGA)**
- Select top-p fraction of tokens by loss for weight gradient computation
- Ternary decisions depend on gradient SIGN, not magnitude → biased estimation acceptable
- Sign agreement: P(sign(G_sel) = sign(G)) ≥ 1 - exp(-2|S|μ²/σ²) — higher for high-loss tokens
- For p=0.5: 50% reduction in weight gradient matmul → ~15-20% total training speedup
- Importance weight correction only needed for continuous α updates, not ternary decisions
- MORE effective for ternary than continuous: discrete target space tolerates magnitude bias

### KEY PATTERNS (Wave 2)
1. **Boundary awareness is THE key insight** — M1, M4, M5, M7 all exploit ternary boundary structure
2. **Shadow weights are overkill** — M3, M8 eliminate them entirely; M7, M9, M10 compress massively
3. **Ternary decisions need signs, not magnitudes** — M4 (SignSGD interior), M8 (voting), M12 (biased OK)
4. **Information-theoretic gap is 60x** — current 96 bits/param vs theoretical 1.58 bits needed
5. **Orthogonal savings stack** — M11 (activations) + M7/M10 (optimizer) + M12 (compute) are independent
6. **Kernel-smoothed gradient is mathematically superior to STE** — M5 provides correct zero at ternary values

## Wave 3: Paradigm Breakers (2 agents, Feb 2026)

### Radical Training Paradigms (P1-P4)

**P1: Gradient-Informed Potts Annealing**
- Ternary weight config is a q=3 Potts spin glass with loss as Hamiltonian
- Gradient-informed proposals: weights with large |g| disagreeing with current state → proposed flips
- Metropolis-Hastings acceptance: P(accept) = min(1, exp(-ΔL/T)) with temperature annealing
- At T>0, Markov chain can accept WORSE moves to escape STE local minima
- Convergence to global minimum guaranteed under logarithmic cooling T(t) = C/log(1+t)
- Best as hybrid: STE for first 80%, Potts annealing for final 20% refinement
- Minimum experiment: modify discrete_polish_v2.py to add temperature + Metropolis criterion

**P2: Rate-Distortion Bound — THE META-QUESTION**
- Compute Shannon's D(R) at R=1.585 bits/weight for actual weight distributions
- Blahut-Arimoto algorithm: ~50 lines, iteratively optimizes encoder/decoder mapping
- Answers: is ternary gap a TRAINING problem or REPRESENTATION problem?
- If current distortion ≈ D(R): no algorithm can close the gap → focus on scaling
- If current distortion >> D(R): room for better training exists → pursue P1/P3/P4
- HIGHEST PRIORITY: ~50 lines of code, 5 min compute, enormous theoretical insight
- Also enables successive refinement quantization (multi-pass error correction)

**P3: Boundary-Focused Evolutionary Forward Search — NO BACKPROP**
- Eliminate backpropagation entirely using forward-mode gradient estimation
- Only boundary weights (10-30% of total) need perturbation → O(B) not O(N)
- K forward passes with TERNARY perturbations (flip actual ternary states, not FP32)
- WHY different from QZO (which failed): QZO perturbed continuous shadow weights (most don't change Q(W)); P3 perturbs discrete states directly (every perturbation changes computation)
- Vote: accumulate directional signals from K perturbations, flip when convinced
- Memory: NO backward pass → no activation caching (saves ~50% training memory)
- Embarrassingly parallel: K forward passes independent → GPU-friendly

**P4: Layerwise Ternary Target Propagation — SSM EXACT INVERSES**
- Each layer gets LOCAL objective: "find ternary W such that Q(W)·X ≈ Target"
- Targets propagated backward using APPROXIMATE INVERSES of each layer
- SSM inverse is EXACT: h_t = (h_{t+1} - B·x_t) / A_t (diagonal A, elementwise division)
- Eliminates STE bias accumulation: no gradient passes through 27 quantizers
- Local ternary regression at each layer: well-structured combinatorial problem
- Attacks angular gap directly: target propagation optimizes for DIRECTIONAL ALIGNMENT
- Per-layer parallelism: all 28 layers optimized simultaneously
- SSM exact inverse is UNIQUE advantage — prior target propagation work lacked this

### Ternary-Native Training Paradigms (P5-P8)

**P5: Ternary Gradient Voting (TGV) — NO SHADOW WEIGHTS**
- Replace FP32 shadow weights + AdamW with integer vote counters
- Per weight: 3 INT8 counters C = [c₋₁, c₀, c₊₁] + 2-bit ternary state = 26 bits total
- Each step: gradient sign votes for preferred state transition
- Exponential decay (0.999) + threshold T for flipping: natural learning rate equivalent
- Memory: 26 bits vs 98 bits per param = **73% reduction** (162MB vs 612MB at 50M)
- Mathematical justification: CLT → vote majority converges to sign(E[g]) (sufficient for ternary)
- WHY different from DQT (which failed): DQT = stochastic rounding (99% zeros); TGV = deterministic voting (weight stays until convinced)
- Pure integer training pipeline: ternary forward + integer vote accumulation

**P6: Analytic Ternary Construction (ATC) — BUILD WEIGHTS, DON'T TRAIN THEM**
- Compute optimal continuous weights via Wiener filter: W* = Y^T·X·(X^T·X)⁻¹
- Project onto ternary lattice with Hessian-weighted rounding
- Layerwise construction with error propagation through already-constructed layers
- Cost: 9 seconds total vs 2.3 hours for STE training (at 50M scale)
- Even if mediocre alone: provides dramatically better initialization than random Xavier
- Combined with 500 steps STE fine-tuning: potentially 10x training speedup
- WHY uniquely ternary: continuous weights → Wiener filter IS the answer; ternary → projection is where the art lies

**P7: Ternary Hypernetwork — COMPRESSED WEIGHT GENERATION**
- Small ternary classifier H(layer, row, col) → {-1, 0, +1} generates all main model weights
- Factored variant: row_embed · col_embed → sign → ternary (low-rank generation)
- 50-2000x compression of stored parameters
- Each H parameter change updates CORRELATED blocks in main model simultaneously
- WHY uniquely ternary: predicting {-1,0,+1} is 3-class CLASSIFICATION (easy), not continuous regression
- Transfer learning: trained H generates weights for DIFFERENT main model sizes
- Viability: MEDIUM — expressiveness question is genuinely open

**P8: Ternary Generational Bootstrap (Ouroboros) — SELF-DISTILLATION**
- Train M₀ with standard STE. Use M₀'s soft predictions to train M₁. Iterate.
- Loss: (1-α)·CE(M_n(x), y) + α·KL(M_n/T, M_{n-1}/T)·T²
- WHY uniquely ternary: teacher and student share SAME computational basis → zero quantization gap
- FP16→ternary distillation: student can NEVER replicate teacher. Ternary→ternary: student CAN.
- Soft labels carry MORE information than hard labels (teacher uncertainty, inter-class structure)
- Each generation: different random init (diverse basin exploration) + accumulated knowledge
- Gap closure: if each gen closes fraction r: gap_n = (1-r)^n · gap_0
- Conservative (r=0.1, 5 gen): 7.4% → 4.4%. Optimistic (r=0.2, 5 gen): 7.4% → 2.4%
- VERY HIGH viability: 1 day implementation, uses existing infrastructure, ~30 lines change

### KEY PATTERNS (Wave 3)
1. **Stop treating ternary as degraded continuous** — it's a DISCRETE COMBINATORIAL problem (P1, P3, P5)
2. **Rate-distortion theory answers THE meta-question** — training vs representation limit (P2)
3. **SSM architecture uniquely enables target propagation** — exact inverses unavailable elsewhere (P4)
4. **Ternary self-distillation has zero quantization gap** — uniquely advantaged over FP16 teachers (P8)
5. **Weights can be CONSTRUCTED, not just searched** — analytic construction bypasses optimization (P6)
6. **Integer-only training is feasible for ternary** — votes, not gradients (P5)

### Cross-Wave Synergies (Training Velocity)
- **P2 (Rate-Distortion) first** — answers whether other techniques CAN help before investing effort
- **P6 (Construction) + P8 (Bootstrap)** — construct M₀ analytically, then bootstrap to improve
- **P5 (Voting) + P8 (Bootstrap)** — each generation trains with 73% less memory via voting
- **M5 (ASGKS) + M1 (FSCD)** — kernel-smoothed gradient for continuous updates + flip-score for discrete corrections
- **M11 (RSAR) + M7 (BPSO)** — reversible SSM activations + boundary-sparse optimizer = orthogonal savings
- **P4 (Target Prop) + M2 (CNG)** — local objectives solved with natural gradient on categorical manifold
- **F10 (Gumbel STE) + M5 (ASGKS)** — Gumbel noise injection into kernel-smoothed gradient

---

## Aspect 2 Summary: Training Velocity

**Total findings:** 14 web research (F1-F14) + 12 mathematical (M1-M12) + 8 paradigm shifts (P1-P8) = **34 findings**

**Top 5 by estimated impact:**
1. **P2 Rate-Distortion Bound** — answers THE meta-question, ~50 lines, 5 min compute (DO FIRST)
2. **M5 ASGKS Kernel-Smoothed Gradient** — 1.5-3x fewer steps, <0.1% overhead, ~45 lines
3. **P8 Ternary Generational Bootstrap** — progressive gap closure, VERY HIGH viability, ~30 lines
4. **M11 RSAR Reversible SSM** — 32x SSM activation memory (768→24MB), HIGH viability
5. **M8 TTV Ternary Transition Voting** — 9.6x optimizer memory, paradigm shift if it works

**Implementation priority (by effort/impact ratio):**
1. **P2** (50 lines, theoretical insight, DO IMMEDIATELY)
2. **M4 BAGR** (35 lines, 1.3-2x, zero risk)
3. **M5 ASGKS** (45 lines, 1.5-3x, low risk)
4. **P8 Bootstrap** (30 lines loss change + 100 lines soft label gen, progressive improvement)
5. **M7 BPSO** (80 lines, 3.75-5.7x memory, HIGH viability)

**Information-theoretic insight:** Current training uses 96 bits/param vs theoretical minimum 1.58 bits. The 60x gap means MASSIVE room for optimizer compression. Shadow weights are the single biggest waste.
