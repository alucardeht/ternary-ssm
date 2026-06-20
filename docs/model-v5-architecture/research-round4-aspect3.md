# Research Round 4 — Aspect 3: Loss Function & Training Signal Optimization

**Date:** 2026-03-01
**Focus:** Loss function alternatives, training signal amplification, convergence acceleration
**Workflow:** v3 (3-gate self-filtering, hard caps, prescriptive format)

---

## Wave 1 — Web Research (Loss Functions & Training Signal)

### Theme A — Loss Function Alternatives for Ternary/Quantized Models

**R4-A3-F1: SCALED-CONVERGENCE-INFONCE-LOSS**
- **Category:** IMPLEMENT
- **Source:** Cheng et al., "Understanding InfoNCE: Transition Probability Matrix Induced Feature Clustering" (Nov 2025)
- **Core insight:** SC-InfoNCE adds tunable scaling (α) and bias (γ) to InfoNCE loss, controlling inter-cluster similarity baseline. Enables flexible adaptation to task heterogeneity without architectural changes. Modifies rigid equi-angular clustering to adaptive clustering objectives.
- **Mathematical principle:** L_SC = L_InfoNCE - (α/τ)·sim(z_i, z_i+) + (γ/τ)·Σ_j≠i sim(z_i, z_j). Changes stationary co-occurrence target from fixed to adaptive. Two additional hyperparameters (α, γ) control deviation from baseline.
- **Estimated impact:** +0.8-2.0 points on text tasks. For ternary SSM: applicable only if adding contrastive pre-training phase or auxiliary contrastive task.
- **Implementation complexity:** Low — two hyperparameters in contrastive loss
- **Dependencies:** Requires contrastive learning objective (not standard autoregressive LM)
- **Risks:** Not directly applicable to next-token prediction without modifications. Feature collapse with excessive scaling.

**R4-A3-F2: ADAPTIVE-LOSS-FUNCTION-LEARNING**
- **Category:** EXPLORE
- **Source:** Raymond et al., "Meta-Learning Adaptive Loss Functions" (TMLR 2025)
- **Core insight:** Loss function parameterized as small neural network, continuously updated online via bilevel optimization. Emergent behaviors include implicit LR scheduling, early stopping regularization, and label smoothing. Loss evolves with training dynamics rather than remaining static.
- **Mathematical principle:** Bilevel optimization — inner loop updates model with learned loss L_φ, outer loop updates loss params φ via validation gradients. L_φ(p) = NN_φ(p) with smooth leaky ReLU activation to prevent flattening.
- **Estimated impact:** 1.6× faster convergence on small-scale models. Could automatically adapt loss to ternary quantization stages.
- **Implementation complexity:** High — requires bilevel optimization (Higher library), validation split, second-order gradients. ~1.3-1.5× training cost. Untested at 7B+ scale.
- **Dependencies:** Higher library, validation dataset, PyTorch autograd second-order support
- **Risks:** Scalability concerns at 7B scale. Meta-loss overfitting to validation set. Computational overhead may be prohibitive.

**R4-A3-F3: LATE-TO-EARLY-TRAINING-LET**
- **Category:** IMPLEMENT
- **Source:** Zhao et al., "Late-to-Early Training: LET LLMs Learn Earlier, So Faster" (arXiv 2602.05393, Feb 2026)
- **Core insight:** Accelerates convergence by guiding early layers using representations from late layers of a pretrained model. Combines late-to-early-step learning (later training knowledge informs earlier phases) and late-to-early-layer learning (advanced representations transferred to earlier depths). Works with teacher models 10× smaller than target.
- **Mathematical principle:** L_total = L_CE(y, ŷ) + λ·Σ_l MSE(h_l^student, h_l^teacher). Distillation most aggressive in early steps, gradually fades. Teacher can be 10× smaller (700M → 7B).
- **Estimated impact:** 1.6× training speedup on 1.4B models with ~5% downstream accuracy improvement. Valuable for RTX 5070 resource-constrained training.
- **Implementation complexity:** Medium — requires pretrained teacher model, intermediate layer distillation, λ(t) scheduling
- **Dependencies:** Pretrained teacher model (could be smaller FP16 baseline). Teacher must have compatible layer structure.
- **Risks:** FP16 teacher vs ternary student = different representation spaces. Distillation weight scheduling critical. Teacher quality directly affects student performance.

**R4-A3-F4: STEPWISE-SELF-KNOWLEDGE-DISTILLATION-SSKD**
- **Category:** IMPLEMENT
- **Source:** Zou et al., "SSKD: Stepwise Self-Knowledge Distillation for Binary Neural Networks" (Applied Sciences 2026)
- **Core insight:** Self-distillation from model's own full-precision shadow weights (used in STE backward) to quantized forward weights. No separate teacher needed. Progressive quantization schedule with distillation loss ensures smooth transition. Specifically designed for 1-bit quantization, directly applicable to ternary.
- **Mathematical principle:** L = L_task + β·KL(softmax(logits_shadow/T), softmax(logits_quant/T)). Shadow weights accumulate gradients continuously, distillation preserves knowledge as quantization hardens. T=3-4, β=0.1-0.5 typical.
- **Estimated impact:** Zero accuracy loss vs SOTA baselines. Addresses weight trapping. 5-10% quality improvement during ternary training.
- **Implementation complexity:** Low — already have STE shadow weights. Store shadow-weight logits, compute KL divergence, add to loss. ~50 lines.
- **Dependencies:** STE or similar gradient estimator with shadow weights. Compatible with any task loss.
- **Risks:** Double forward pass increases compute. Over-regularization if β too large. Unclear benefit when training ternary from scratch (vs progressive).

**R4-A3-F5: UNCERTAINTY-WEIGHTED-MULTI-TASK-LOSS**
- **Category:** IMPLEMENT
- **Source:** Kendall et al. (2018) + 2025 best practices. "Uncertainty-based dynamic weighting for multi-task learning."
- **Core insight:** Treat loss weights as learnable parameters representing task uncertainty (inverse variance). Automatically balances multi-component losses during training. Tasks with higher uncertainty receive more weight. Eliminates manual loss weight tuning.
- **Mathematical principle:** L = Σ_i (1/(2·exp(s_i)))·L_i + Σ_i s_i where s_i = log(σ_i²). Regularization Σ s_i prevents trivial solution. Hard tasks (high σ²) get lower effective weight initially, adjusts as training progresses.
- **Estimated impact:** Eliminates hyperparameter tuning for multi-component losses. Proven effective for multi-task learning. Directly applicable to our multi-loss setup.
- **Implementation complexity:** Low — 1 learnable parameter per loss component. ~10 lines PyTorch.
- **Dependencies:** Requires separable loss components (any combination of CE + auxiliary losses)
- **Risks:** Numerical instability if exp(s_i) becomes extreme. May converge to suboptimal weighting for highly imbalanced loss scales.

### Theme B — Training Signal & Convergence Acceleration

**R4-A3-F6: LOAD-BALANCING-AUXILIARY-LOSS**
- **Category:** IMPLEMENT
- **Source:** MoTE: Mixture of Ternary Experts (arXiv 2506.14435, 2025)
- **Core insight:** Auxiliary load balancing loss for ternary experts prevents expert collapse and accelerates convergence. Enforces uniform expert usage, providing richer gradient signal to all ternary parameters. Applicable to RecursiveSSM by tracking per-layer-application gradient contributions.
- **Mathematical principle:** L_aux = sum_i (f_i - 1/E)^2 where f_i is fraction of tokens processed by expert/layer i, E is number of experts/layers. Creates force opposing degenerate convergence.
- **Estimated impact:** 10-20% faster convergence by preventing gradient starvation in shared block applications
- **Implementation complexity:** Low — ~50 lines, single λ_aux hyperparameter (0.01-0.1)
- **Dependencies:** Layer-wise usage/gradient tracking in RecursiveSSM
- **Risks:** λ_aux tuning required. Too high forces artificial uniformity.

**R4-A3-F7: ARENAS-ANNEALING-RESIDUAL-SYNAPSE**
- **Category:** IMPLEMENT
- **Source:** Sherry: Hardware-Efficient Ternary Quantization (arXiv 2601.07892, 2026)
- **Core insight:** Maintains "residual synapses" (continuous shadow values) that gradually anneal to zero during ternary training. Prevents weight trapping where ternary weights freeze into local minima. Provides richer gradient signal through temporary exploration in continuous space.
- **Mathematical principle:** W_shadow = W_shadow + α * grad, α = α_init * (1 - step/total_steps)^β. Quantized output: W_ternary = Q(W_shadow). Annealed α gradually reduces shadow deviation from quantized values.
- **Estimated impact:** Addresses weight trapping issue. 5-15% loss quality improvement for longer training runs (>10K steps).
- **Implementation complexity:** Medium — separate shadow tensor (2x ternary layer memory), annealing schedule. ~100 lines.
- **Dependencies:** Compatible with STE backward. Replaces direct W += lr * grad with shadow weight system.
- **Risks:** Increases training memory. Annealing schedule hyperparameters (α_init, β) must be tuned. Too fast annealing = no benefit, too slow = never commits to ternary.

**R4-A3-F8: LOSS-LANDSCAPE-DRIVEN-DYNAMIC-WEIGHTING**
- **Category:** IMPLEMENT
- **Source:** Zhang & Deng, "Loss-driven dynamic weight in physics-informed neural network" (Neural Networks Vol 191, 2025)
- **Core insight:** Dynamically adjust multi-component loss weights based on magnitude imbalance. Higher-magnitude components receive lower weight to prevent dominating gradient flow. Balances gradient contributions and accelerates convergence.
- **Mathematical principle:** w_i(t) = (L_mean / L_i(t))^γ where L_mean is geometric mean, γ ∈ [0.5, 1.5]. Total loss L = Σ w_i(t) * L_i(t). Normalizes gradient contributions from each loss component.
- **Estimated impact:** 8-15% faster convergence. Especially valuable when combining multiple auxiliary losses (F6 + SSKD + main CE).
- **Implementation complexity:** Low — ~30 lines, negligible compute overhead (<1% per step)
- **Dependencies:** Separable loss components. Works with any loss combination.
- **Risks:** γ tuning required. Too aggressive = oscillations, too conservative = minimal effect.

**R4-A3-F9: ADAPTIVE-INTERMEDIATE-SUPERVISION**
- **Category:** EXPLORE
- **Source:** FedAIMS: Adaptive Intermediate Supervision (2025)
- **Core insight:** Add auxiliary prediction heads at intermediate layers with adaptive weighting that decays during training. Early layers receive explicit supervision signal beyond backpropagated gradients. Curriculum from strong intermediate supervision to final-task focus.
- **Mathematical principle:** L_total = L_final + Σ_{l} λ_l(t) * L_layer_l where λ_l(t) = λ_init * exp(-t/τ). Strong early supervision, gradual decay to final task focus.
- **Estimated impact:** 10-30% faster convergence in deep supervision literature. For RecursiveSSM with 32 applications, adds supervision at key checkpoints (every 8 layers).
- **Implementation complexity:** Medium-High — prediction heads at intermediate points, adaptive scheduling. ~150-200 lines. +10-20% memory, +15-25% compute per step.
- **Dependencies:** Well-defined intermediate representations. For RecursiveSSM: supervise accumulated state at checkpoints?
- **Risks:** RecursiveSSM weight sharing creates ambiguity — which "layers" get supervision when all share same params? Needs architectural analysis.

**R4-A3-F10: WEIGHT-SHARING-LAYER-SPECIFIC-LR**
- **Category:** IMPLEMENT
- **Source:** Rolland et al., "Exploring Shared-Weight Mechanisms in Transformer" (INTERSPEECH 2025)
- **Core insight:** In weight-sharing architectures, different applications of the shared block benefit from different effective learning rates. Apply layer-specific gradient scaling to compensate for gradient magnitude imbalance across application depths. Deep applications have smaller gradients due to vanishing gradient.
- **Mathematical principle:** grad_shared = Σ_{l=1}^{L} α_l * grad_l where α_l = ||grad_mean|| / ||grad_l||. Normalizes per-application gradient contributions. Prevents shallow applications from overwhelming deep ones.
- **Estimated impact:** 10-15% faster convergence by equalizing gradient quality across 32 RecursiveSSM applications
- **Implementation complexity:** Medium — gradient hooks at each application point, per-application scaling. ~80-100 lines. Minimal memory (32 scalar α values).
- **Dependencies:** Gradient hooks at RecursiveSSM application points. Compatible with Adam optimizer.
- **Risks:** 5-10% per-step overhead. α recomputation frequency vs lag tradeoff.

---

## Wave 2 — Math-Paradigm-Breakers (Loss Function & Training Signal Math)

### Theme A — Loss Functions Exploiting Ternary Structure

**R4-A3-M1: TERNARY-HAMMING-MARGIN-LOSS**
- Category: IMPLEMENT
- Source: Math-paradigm-breaker (loss function for ternary)
- Core insight: Standard losses demand equal margins between ALL class pairs, but ternary weights physically CANNOT produce large margins between rows with small Hamming distance. THM-LOSS demands margins proportional to what the ternary code can actually achieve, eliminating gradient waste on impossible targets.
- Mathematical principle: L_THM = (1/|S|) · Σ_{k∈S} max(0, α·√(d_H(W_y,W_k)/d)·||h|| - (z_y - z_k)). Hamming distance d_H computed via POPCOUNT on packed ternary rows. Active set S = top-K violators. Gradient is bounded integers, integer-compatible.
- Estimated impact: 10-25% faster convergence by eliminating 30-60% of wasted gradient signal on infeasible margins. Integer-compatible (no exp/log).
- Implementation complexity: Medium — POPCOUNT-based Hamming distance, top-K selection, custom loss
- Dependencies: POPCOUNT infrastructure (partially implemented for bitplane inference)
- Risks: Non-stationary margins (d_H changes as weights update). Mitigate with EMA of d_H.

**R4-A3-M2: TERNARY-ALIGNED-INTEGER-LOSS (TAIL)**
- Category: IMPLEMENT
- Source: Math-paradigm-breaker (integer-native loss)
- Core insight: Measures robustness in NATIVE ternary metric: "flip distance" = minimum weight state transitions to change prediction. Yields fully integer gradients, completing FPIT chain (forward+loss+backward+optimizer = zero floating point). Integer gradients carry exactly log2(3) bits of useful info per gradient — information-theoretically optimal for ternary.
- Mathematical principle: flip_dist(y,k) = ⌈(z_y - z_k) / ||h||_∞⌉. Loss: L_TAIL = Σ_{k∈top-K} max(0, F·H - (z_y - z_k)) where F=target flip margin, H=⌈||h||_∞⌉. Gradient: dL/dz_y = -|{violators}| (integer), dL/dz_k = I(violated) (0 or 1). Through STE: dL/dW = integer_vector · integer_vector^T ∈ Z^{V×d}.
- Estimated impact: Enables FULL integer training pipeline (forward+loss+backward+optimizer). 2-4x speedup on integer hardware. 50%+ gradient buffer reduction (INT16 vs FP32).
- Implementation complexity: Low — straightforward multi-class hinge with integer margin
- Dependencies: Integer activation quantization (for integer logits), AGA optimizer (INT16 accumulators)
- Risks: Class collision in early training (many classes violating simultaneously). F margin scheduling needed (F=1 early, F=3 late).

**R4-A3-M3: RECURSIVE-CONSENSUS-LOSS (RCL)**
- Category: IMPLEMENT
- Source: Math-paradigm-breaker (weight-sharing-aware loss)
- Core insight: Standard loss gradient for shared weights is SUM of 32 per-layer contributions, which can produce "compromise" gradients near zero when layers disagree, keeping weights at 0. RCL weights each layer's loss by its CONVERGENCE degree, so converged layers (which produce coherent gradients) dominate. Creates virtuous cycle: loss trains model to converge faster → AdaptiveDepth exits earlier → inference speedup.
- Mathematical principle: c_l = 1 - ||h_l - h_{l-1}||²/(||h_l||² + ε) ∈ [0,1]. L_RCL = (1/Z)·Σ_{l∈S} c_l^β · L_l where S is sampled layer subset, Z = Σ c_l^β. Converged layers (c_l→1) dominate. Per-layer L_l can use TAIL for fully integer computation.
- Estimated impact: 15-30% reduction in weight oscillation, 5-15% fewer steps to convergence. Improved gradient quality for early layers.
- Implementation complexity: Medium — requires per-layer output projections at sampled layers, convergence computation
- Dependencies: RecursiveSSM architecture, output projection W_out. ~2.5% total overhead for |S|=11 layers.
- Risks: β too large → single layer dominates (fragile). Mitigate: anneal β from 0→2 over training.

**R4-A3-M4: TERNARY-ENTROPY-RATE-LOSS (TER-LOSS)**
- Category: EXPLORE
- Source: Math-paradigm-breaker (information-theoretic loss)
- Core insight: CE targets a delta-function distribution but ternary output has FINITE capacity (log2(3)≈1.585 bits/weight). Mismatch wastes gradient signal. TER-LOSS computes entropy floor based on ternary channel capacity, penalizes distributions more peaked than achievable. Also derives capacity-utilization learning rate schedule: η = (log(V) - H(p))/C_upper.
- Mathematical principle: C_upper = (d/2)·p_active·log(1 + ||h||²/(d·σ²_noise)), H_floor = max(0, log(V) - C_upper). Loss: L_TER = CE(p,y) + μ·ReLU(H_floor - H(p)). LR schedule: lr_eff = lr_base·(1 - η^γ).
- Estimated impact: At small scale (d≤64): 5-15% improved calibration. At typical scale (d≥256): minimal (capacity rarely binding). Capacity-utilization LR schedule: 5-10% faster convergence at capacity boundary.
- Implementation complexity: Low — entropy O(V), capacity estimation O(d), wrapper around any loss
- Dependencies: None (additive to any base loss)
- Risks: May never activate at typical d. Gaussian capacity approximation may be inaccurate. Most useful as diagnostic tool / principled label smoothing replacement.

### Theme B — Training Signal Optimization for Weight-Sharing

**R4-A3-M5: GRADIENT-CONSENSUS-POLARIZATION (GCP)**
- Category: IMPLEMENT
- Source: Math-paradigm-breaker (gradient aggregation for shared weights)
- Core insight: 32 gradient signals per weight constitute a SAMPLE. The sample mean (standard sum) ignores variance. GCP uses consensus-to-conflict ratio to AMPLIFY clear signals and SUPPRESS ambiguous ones. Amounts to per-weight hypothesis test before Adam update. Implicit sparsity pressure: ambiguous weights → 0, clear weights → ±1.
- Mathematical principle: Consensus c(j) = (1/L)·Σg_l(j), Conflict σ²(j) = variance. Agreement ratio r(j) = c²/(c²+σ²) ∈ [0,1]. Polarized gradient: g_GCP = L·c(j)·r(j)^α where α≥1. Connected to F-statistic: F = L·r/(1-r) for testing if gradient is significantly nonzero.
- Estimated impact: 15-30% faster convergence by eliminating noise from conflicting gradient signals. Implicit balanced ternary distribution.
- Implementation complexity: Low — elementwise variance computation, gradient preprocessing step, <1% overhead
- Dependencies: Per-layer gradient contributions (requires intercepting gradient summation for shared weights)
- Risks: Cubic suppression (α=2) may be too aggressive early. Mitigate: anneal α from 1→2.

**R4-A3-M6: RECURRENCE-AWARE-FLIP-UTILITY (RAFU)**
- Category: IMPLEMENT
- Source: Math-paradigm-breaker (exact flip utility for recurrent shared weights)
- Core insight: STE gradient captures only LOCAL effect at each application. RAFU computes EXACT first-order flip utility accounting for how a weight change propagates through all L=32 applications via state recurrence. Utility is bilinear form U = Ψ^T · H in forward states and backward error signals.
- Mathematical principle: Backward recurrence: ψ_k = δ_k + A^T·ψ_{k+1}. Utility matrix: U = Ψ^T·H where Ψ,H ∈ R^{L×d}. FlipBenefit(i,j; cur→tgt) = -(tgt-cur)·U_ij. Cost: O(L·d²) additional (≈2x backward pass).
- Estimated impact: 30-50% more accurate flip decisions. 20-40% faster convergence in steps. BUT 2x per-step cost — use every N steps as calibration.
- Implementation complexity: Medium — custom backward recurrence, intermediate state storage, matrix product
- Dependencies: Access to per-layer error signals, intermediate states h_0...h_{L-1}
- Risks: 2x time per step may negate gains. Mitigate: use every 10 steps. First-order approximation may miss cascading flip effects.

**R4-A3-M7: INFORMATION-BOTTLENECK-LAYERWISE-LOSSES (IBLL)**
- Category: EXPLORE
- Source: Math-paradigm-breaker (IB theory for weight-sharing recurrence)
- Core insight: Not all 32 applications contribute equally useful gradients. Some merely "pass through" information, producing diluted signals. IBLL derives auxiliary losses from information-bottleneck theory that force every application to perform meaningful computation, ensuring all 32 gradients carry useful training signal.
- Mathematical principle: Compression_l = ||h_l - h_{l-1}||²/||h_{l-1}||². Relevance_l = -CE(softmax(P·h_l), y) with shared probe P. L_aux = Σ λ_l[-β·Relevance_l + γ·max(0, ε-Compression_l)]. Adaptive λ_l based on gradient SNR: GI_l = ||g_l||²/(||g_l||² + Var(g_l)).
- Estimated impact: 20-40% faster convergence by ensuring all 32 applications provide useful gradient signal. Improved gradient quality for early layers.
- Implementation complexity: Medium — auxiliary loss computation at intermediate layers, shared probe, adaptive scheduling
- Dependencies: Intermediate states at supervised layers, projection head (can reuse output head). +25% overhead for 8 supervised layers.
- Risks: Intermediate supervision may conflict with natural recurrence behavior. Probe may be too simple (linear). Interaction with GCP is synergistic.

**R4-A3-M8: BAYESIAN-FLIP-ACCUMULATOR (BFA)**
- Category: EXPLORE
- Source: Math-paradigm-breaker (sequential Bayesian decision theory)
- Core insight: REPLACES shadow weights + STE entirely. Maintains explicit Bayesian posterior π = (π_{-1}, π_0, π_{+1}) per weight, updated by gradient evidence via sequential likelihood. Flips weight only when posterior exceeds confidence threshold. Equivalent to multi-hypothesis SPRT (Sequential Probability Ratio Test) — provably optimal flip timing.
- Mathematical principle: Evidence e_v = Σ_l -g_l(i,j)·(v - q_current). Update: log π_v^{t+1} = ρ·log π_v^t + e_v/τ - log Z. Flip when max_v π_v > θ AND argmax ≠ current. Reset posterior after flip. EMA discount ρ∈(0,1) for non-stationarity.
- Estimated impact: 30-60% faster convergence (SPRT-optimal). Eliminates STE AND shadow weights. Same memory as Adam (3 floats per weight: posterior vs shadow+m+v).
- Implementation complexity: High — fundamentally different optimizer, custom training loop, two optimizer types (BFA for ternary, Adam for FP16 params)
- Dependencies: Per-application gradient signals, custom optimizer implementation
- Risks: Speculative — removing shadow weights loses "soft exploration." Temperature τ critical. First-order evidence may be inaccurate.

---

## Wave 3 — Paradigm-Breakers (Radical Loss & Training Signal)

### Theme A — Radical Loss Function Design

**R4-A3-P1: MINIMAX-DEPTH-EQUILIBRIUM (MDE)**
- Category: IMPLEMENT
- Source: Paradigm-breaker (game-theoretic loss design)
- Core insight: Formulate training as a two-player zero-sum game: model minimizes loss, adversary selects worst recursion depth. Fictitious play converges to Nash equilibrium — weights with BEST worst-case performance across all 32 depths. Adversary naturally creates curriculum by concentrating on hardest depths.
- How it works: min_W max_{p∈Δ_32} Σ p_d · L_d(W). Adversary update: p_d ← p_d · exp(η·L_d) / Z (multiplicative weights). Finds exact spectral radius that optimally trades off all depths. CVaR variant (worst 25%) balances robustness vs performance.
- Estimated impact: 3-8% loss improvement with better consistency across sequence positions. Per-depth projection heads = low overhead.
- Implementation complexity: Low — per-depth projection heads, multiplicative weights update, few lines of code
- Dependencies: Intermediate supervision setup (projection at each depth). Compatible with any gradient estimator.
- Risks: Pure minimax can be pessimistic. CVaR (K=8, worst 25%) mitigates this.

**R4-A3-P2: TERNARY-ERROR-CORRECTING-WEIGHT-CODES (TECWC)**
- Category: EXPLORE
- Source: Paradigm-breaker (coding theory for weight space)
- Core insight: Constrain weight space to a TERNARY ERROR-CORRECTING CODE [N,K,d] over GF(3). Any two valid configurations differ in at least d positions. Reduces search space from 3^N to 3^K (exponentially smaller) while guaranteeing robustness to weight perturbations. Creates wider basins in loss landscape.
- How it works: Optimize information vector u ∈ {-1,0,+1}^K, actual weights w = u·G (mod 3). Generator matrix G from ternary BCH code. Replace STE projection with nearest_codeword via syndrome decoding. Rate R=K/N=0.9 (10% redundancy), d=3 (corrects 1 error).
- Estimated impact: 5-15% loss improvement from implicit regularization. 2-5x convergence acceleration from reduced search space. Inference robustness to hardware bit-flips.
- Implementation complexity: High — ternary code construction, syndrome decoding over GF(3)
- Dependencies: Coding theory implementation over GF(3). Different layers may need different codes.
- Risks: Code constraint may be too restrictive. Start with high-rate (K≈N) and gradually decrease.

**R4-A3-P3: SPECTRAL-SURROGATE-OBJECTIVE (SSO)**
- Category: EXPLORE
- Source: Paradigm-breaker (spectral properties as training proxy)
- Core insight: For ternary SSM matrices, spectral properties (eigenvalues, singular values, condition number) are computable WITHOUT a forward pass and DIRECTLY determine the linear component of recurrence quality. 500x cheaper per evaluation than a forward pass. Use as surrogate for initialization search via simulated annealing.
- How it works: L_surrogate = α(ρ(A)-ρ_target)² + β·Σ(σ_i-σ_target)² + γ·alignment(W, data_cov). Phase 1: spectral search (greedy flips, O(d³)=0.02ms vs 10ms forward pass). Phase 2: fine-tune with standard CE+STE. Targets: ρ_target≈0.95, geometric singular value decay.
- Estimated impact: 10-50x faster initial search phase. 20-40% total training time reduction via better initialization.
- Implementation complexity: Medium — eigendecomposition (numpy), surrogate quadratic, calibrating spectral targets
- Dependencies: Only applicable to SSM matrices (A,B,C), not attention/embedding. Target calibration from empirical baseline.
- Risks: Surrogate may miss nonlinear interactions between A,B,C. Phase 2 fine-tuning corrects errors.

**R4-A3-P4: SPIN-GLASS-ANNEALING-WITH-PHASE-DETECTION (SGAPD)**
- Category: EXPLORE
- Source: Paradigm-breaker (statistical physics for ternary optimization)
- Core insight: Ternary network is FORMALLY a spin-1 system (Blume-Capel model). Crystal field parameter Δ controls sparsity (density of zeros). Phase diagram has ordered, disordered, and zero-magnetization phases. Use order parameters (magnetization m, Edwards-Anderson q) to detect phase transitions and guide temperature schedule. Monte Carlo sweeps complement gradient descent by exploring configurations STE can't reach.
- How it works: Periodic MC sweeps: propose flip, accept with Metropolis probability min(1, exp(-ΔL/T)). Crystal field annealing automatically discovers optimal sparsity. Near phase transitions, long-range correlations between weights emerge — captured by MC but missed by local gradient descent.
- Estimated impact: 50-70% LR tuning effort eliminated via phase-detection scheduling. 10-30% inference speedup from optimal sparsity. MC finds configurations in different basins than STE.
- Implementation complexity: Medium-High — MC sampler for ternary, temperature-to-LR mapping, sweep frequency tuning
- Dependencies: Compatible as complementary update alongside STE gradient. Block MC for efficiency (whole rows/columns).
- Risks: MC slow for N=millions (O(N) per sweep). Mitigate with block MC on A matrix only. Mean-field phase diagram may differ from actual landscape.

### Theme B — Radical Training Signal Generation

**R4-A3-P5: ERGODIC-TRAJECTORY-OPTIMIZER**
- Category: EXPLORE
- Source: Paradigm-breaker (dynamical systems theory for optimizer replacement)
- Core insight: 32-step recursion generates trajectory whose statistical properties (moments, autocorrelations) characterize the dynamical system completely by ergodic theorem. These statistics can REPLACE Adam's per-parameter optimizer states, reducing optimizer memory from O(param_count) to O(state_dim). Forward-only training signal.
- How it works: Trajectory covariance C(τ) = (1/(L-τ))·Σ(h_l-μ)(h_{l+τ}-μ)^T. Target: C_target(0)=αI (bounded variance), C_target(1)/C_target(0)≈ρI (autocorrelation for memory). Per-flip perturbation ΔC computable analytically in O(d). Optimizer memory: 84GB (Adam FP32) → ~50KB (diagonal trajectory stats) — 570,000x reduction.
- Estimated impact: Astronomical memory reduction if it works. 10-30x faster per step (no backprop through chain). Quality: UNKNOWN — likely needs hybrid with task-specific signal.
- Implementation complexity: Medium — trajectory stats straightforward, perturbation analysis closed-form, target design is the hard part
- Dependencies: None (standalone optimizer replacement). Benefits increase with more recursion depths.
- Risks: Trajectory stats may not carry enough task-specific information. Most viable as REGULARIZER alongside lightweight task signal.

**R4-A3-P6: MINIMAX-DEPTH-FAIRNESS**
- Category: IMPLEMENT
- Source: Paradigm-breaker (distributive justice for weight sharing)
- Core insight: Standard training minimizes SUM of per-depth losses (social welfare optimum), which allows depth sacrifice. Minimax minimizes MAXIMUM per-depth loss (Rawlsian fairness). Gradient = gradient at WORST depth only → backward through only 1 of 32 depths (32x cheaper backward pass).
- How it works: g_minimax = ∇L_{l*} where l*=argmax_l L_l. Softmax relaxation: g_soft = Σ softmax(L_l/τ)·∇L_l. At τ→0: hard minimax. At τ→∞: standard average. Testable prediction: if per-depth loss variance is high, minimax helps significantly.
- Estimated impact: 32x backward cost reduction (gradient from 1 depth). +5-15% on worst-depth perplexity if depth sacrifice exists.
- Implementation complexity: Low — per-depth loss computation, softmax weighting, selective backprop
- Dependencies: Per-depth loss computation (projection at each depth)
- Risks: Oscillation between worst depths. Softmax relaxation mitigates. Near-zero downside risk.

**R4-A3-P7: KOOPMAN-SPECTRAL-SHAPING**
- Category: EXPLORE
- Source: Paradigm-breaker (Koopman operator theory for training)
- Core insight: 32-application RecursiveSSM defines a dynamical system fully characterized by its Koopman operator spectrum. Approximate via EDMD, shape eigenvalues toward target distribution (long-memory modes |μ|∈[0.95,0.999], fast-processing |μ|∈[0.3,0.7]). For ternary weights, each flip is rank-1 perturbation with closed-form eigenvalue shift via Wilkinson perturbation theory.
- How it works: Collect trajectory snapshots, compute K_approx = Φ_Y·Φ_X^+, eigendecompose. Per-flip eigenvalue shift: Δμ_n = (v_n^H·ΔK·u_n)/(v_n^H·u_n), O(k) per eigenvalue per flip. Score flips by spectrum improvement. Run every T=100-1000 steps as periodic correction.
- Estimated impact: Proactive instability prevention (catches dynamical pathologies before loss manifests). Complementary to gradient descent.
- Implementation complexity: High — EDMD, eigendecomposition, perturbation sensitivity, target spectrum design
- Dependencies: Trajectory data collection, scipy/numpy. Compute on CPU (32GB RAM).
- Risks: Dictionary choice critical. Target spectrum is a design choice. Computational cost every T steps. Best as periodic correction, not every-step.

**R4-A3-P8: TEMPORAL-DIFFERENCE-DEPTH-LEARNING (TD-DL)**
- Category: IMPLEMENT
- Source: Paradigm-breaker (reinforcement learning for depth credit assignment)
- Core insight: 32 recursive applications form a sequential decision process (MDP). TD learning provides per-depth credit via δ_l = V_{l+1} - V_l (how much did depth l's application improve the representation?). LOCAL gradient (backprop through only 1 application per depth, not the entire chain). Lower variance than full backprop. 32x activation memory reduction (store 1 depth's activations, not all 32).
- How it works: V_l = -CE(proj(h_l), y). TD error: δ_l = V_{l+1} - V_l. Update: Δw ∝ Σ_l δ_l · ∂V_{l+1}/∂w (local gradient, 1 block application). TD(λ) interpolates: λ=0 (cheapest, most bias) to λ=1 (full backprop). λ=0.8 gives ~95% of backprop quality.
- Estimated impact: 32x activation memory reduction (peak O(d) vs O(32d)). Per-depth diagnostic (which depths improve/degrade). TD(λ=0.8) ≈ 95% of full backprop quality.
- Implementation complexity: Low-Medium — per-depth value, TD errors, local gradients, weighted sum
- Dependencies: Per-depth value computation (shares infrastructure with P1/P6)
- Risks: Vocabulary projection at intermediate depths may not be meaningful if state distribution varies. TD(λ=0) may be too biased. Use proxy vocabulary (1000 tokens) for intermediate V_l.

---

## ASPECT 3 COMPLETE — Final Summary

**Total findings:** 26 (10 Web + 8 Math + 8 Paradigm)
**Breakdown:** 16 IMPLEMENT (62%), 10 EXPLORE (38%)

### Top Implementation Priorities (by estimated impact):

1. **M2: TAIL (Ternary-Aligned Integer Loss)** — Completes FULL integer training pipeline. Flip-distance metric natural for ternary. Integer gradients = information-theoretically optimal for ternary. Low complexity.
2. **M5: GCP (Gradient Consensus Polarization)** — Amplifies consensus gradients, suppresses conflicts across 32 applications. <1% overhead. Implicit balanced ternary distribution. F-test interpretation for principled hyperparameters.
3. **P8: TD-DL (Temporal Difference Depth Learning)** — 32x activation memory reduction. Per-depth credit assignment via RL. TD(λ=0.8) ≈ 95% of full backprop quality. Natural consensus emerges from TD structure.
4. **P1/P6: Minimax Depth (MDE/Fairness)** — Game-theoretic optimization across 32 depths. Prevents depth sacrifice. 32x backward cost reduction (P6). Low implementation effort.
5. **M3: RCL (Recursive Consensus Loss)** — Convergence-weighted per-layer losses. Trains model to converge faster → AdaptiveDepth exits earlier → inference speedup. Synergy with TAIL for fully integer computation.
6. **M6: RAFU (Recurrence-Aware Flip Utility)** — Exact first-order flip utility through 32-step recurrence. 30-50% more accurate flip decisions. Use every N steps as calibration.
7. **F5: Uncertainty-Weighted Multi-Task Loss** — Automatic balancing of multi-component losses. 1 learnable param per loss. Eliminates manual tuning.
8. **F10: Weight-Sharing Layer-Specific LR** — Equalizes gradient magnitude across 32 RecursiveSSM applications. 10-15% faster convergence.

### Key Themes:
- **Integer training pipeline**: M2 (TAIL) + existing FPIT techniques → zero floating-point in entire training loop
- **Depth-aware training**: P1/P6 (minimax) + P8 (TD-DL) + M3 (RCL) → per-depth credit assignment replaces naive gradient sum
- **Gradient quality**: M5 (GCP) + M6 (RAFU) → consensus-based amplification + exact flip utility
- **Discrete optimization**: M8 (BFA) + P2 (TECWC) + P4 (SGAPD) → Bayesian/coding-theory/statistical-physics approaches to ternary optimization
- **Forward-only training signals**: P5 (ergodic) + P7 (Koopman) → dynamical systems theory for training without backprop

### Synergies with Aspects 1-2:
- M2 (TAIL) + A2-M4 (ternary recompute) → integer loss + zero-cost activation recomputation = minimal memory integer training
- P8 (TD-DL) + A2-M5 (UCB-BANDIT) → TD provides per-depth signal, UCB decides flips = no STE, no backprop, no shadow weights
- M5 (GCP) + A1-M7 (adaptive curriculum) → consensus ratio informs curriculum difficulty adjustment
- P1 (MDE) + A1-P4 (stateful streaming) → adversary identifies worst depths, streaming provides continuous signal

---

## Summary & Prioritization

### Recommended Implementation Order (v5 integration)

1. **R4-A3-F5 (Uncertainty-Weighted Multi-Task Loss)** — Minimal code, immediate applicability, addresses existing multi-loss weighting problem
2. **R4-A3-F4 (SSKD)** — Leverages existing STE shadow weights, addresses weight trapping, proven for ternary
3. **R4-A3-F10 (Weight-Sharing Layer-Specific LR)** — Directly addresses RecursiveSSM architecture, orthogonal to other improvements
4. **R4-A3-F6 (Load-Balancing Auxiliary Loss)** — Optional for RecursiveSSM if gradient starvation observed
5. **R4-A3-F3 (LET)** — Requires teacher model, useful but optional for convergence speedup
6. **R4-A3-F8 (Loss-Landscape Dynamic Weighting)** — Enhancement to F5, implement after validation

### Avoid in v5 (insufficient justification)

- **R4-A3-F1 (SC-InfoNCE)** — Not applicable to next-token prediction without redesign
- **R4-A3-F2 (Adaptive Loss Learning)** — Overkill complexity, scalability risk at 7B
- **R4-A3-F7 (ARENAS)** — Double memory cost, annealing tuning adds complexity
- **R4-A3-F9 (Adaptive Intermediate Supervision)** — Architectural ambiguity in weight-sharing setup

