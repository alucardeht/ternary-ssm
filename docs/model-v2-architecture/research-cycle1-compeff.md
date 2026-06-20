# Research Cycle 1 — Computational Efficiency Findings

## Step 1: Online Research (2 researchers, 2026-02-26)

### Researcher 1: "Beyond Backpropagation"

**R1-CE-01: Forward-Forward Algorithm (Hinton 2022, updated April 2025)**
- Two forward passes (positive + negative data) instead of forward+backward
- Layer-local losses, no global backprop needed
- Could eliminate STE gradient approximation entirely for ternary
- Challenge: generating "negative" data for language modeling
- Verdict: WORTH EXPLORING at 6M scale, 1000 steps

**R1-CE-02: torch.compile Regional Compilation (2025)**
- SSM scan fails torch.compile (dynamic shapes, recurrent boundaries)
- SOLUTION: compile ONLY non-SSM layers (FFN, projections, output head)
- Expected: +30-50% speedup on non-SSM compute
- Production-ready TODAY, minimal risk
- Verdict: HIGH PRIORITY, immediate action

**R1-CE-03: Perturbation-Based Training (SPSA, ES)**
- Zeroth-order methods scale O(d) with parameters
- For 50M params: ~50M forward passes per update (vs 1 forward + 1 backward)
- Verdict: NOT VIABLE for full training

**R1-CE-04: Direct Feedback Alignment**
- Random feedback weights + STE = double approximation, too lossy
- No evidence of working with ternary
- Verdict: SKIP

**R1-CE-05: Sparse Training from Scratch (RigL, SET)**
- Ternary already ~33% zeros naturally
- Can we force 60% zeros without quality loss?
- Unexplored territory: dynamic sparse + ternary
- Verdict: MEDIUM PRIORITY, test at 6M

**R1-CE-06: Mixed-Precision Gradient Exploration**
- Do STE gradients NEED FP32 shadow weights?
- FP16 gradients + FP32 accumulation = -50% gradient memory
- FP16 SSM states = unknown quality impact
- Verdict: HIGH PRIORITY, low-hanging fruit

**R1-CE-07: Helion (PyTorch, beta Oct 2025)**
- High-level Triton kernel interface (looks like eager PyTorch)
- Could rewrite ternary quantization + STE as Helion kernels
- Automatic fusion with surrounding ops
- Verdict: INVESTIGATE, potentially 2-3x on ternary ops

### Researcher 2: "Hardware-Algorithm Co-Design"

**R1-CE-08: SSM Tensor Parallelism (arXiv 2602.21144, Feb 2026)**
- Communication-efficient TP specifically for SSM
- SSM state sharded by channels (GPU-local, no sync)
- Results: 1.6-2.1x on 2 GPUs, 2.6-4.0x on 4 GPUs
- Quantized AllReduce: +10-18% additional
- Verdict: HIGH for multi-GPU, directly applicable

**R1-CE-09: BitNet b1.58 2B4T Training Recipe (arXiv 2504.12285, May 2025)**
- First open-source native 1-bit LLM at 2B scale
- Two-stage LR schedule (high initial → cooldown)
- Two-stage weight decay (cosine to 0.1 → disabled)
- Loss summation (not averaging) for SFT
- Squared ReLU instead of SwiGLU (better sparsity for 1-bit)
- subln normalization for quantized stability
- 1-bit models MORE STABLE than full-precision (allows higher LR!)
- Verdict: MULTIPLE zero-cost experiments available

**R1-CE-10: Knowledge Distillation During Pre-training (ACL 2025)**
- KD applied during pre-training reduces total steps needed
- Difficulty-aware KD (DA-KD): focus on hard examples
- Requires pre-trained teacher model
- Verdict: MEDIUM, requires teacher

**R1-CE-11: Fused Triton Kernels for Training (Jan 2026)**
- Single-launch kernels merging norm+activation+embedding+communication
- For ternary: fuse quantize+matmul+STE+gradient_update in 1 kernel
- Eliminates launch overhead and memory traffic
- TritonForge: automated profiling-guided optimization
- Verdict: HIGH, the frontier for ternary training speedup

**R1-CE-12: Dynamic Data Pruning (OpenReview 2025)**
- Bernoulli sampling weighted by historical loss/gradient statistics
- Train on SUBSET of most informative data
- If 30% redundant: 1.43x speedup at same quality
- Unexplored for ternary/quantized models
- Verdict: HIGH, underexplored high upside

**R1-CE-13: Curriculum Learning — Sequence Length (2025)**
- Start short (128-256), gradually increase to target (512+)
- Faster SSM scan on shorter sequences
- Risk: SSM HiPPO initialization may need re-init
- Verdict: MEDIUM-HIGH

**R1-CE-14: BitNet Packing Strategy (Ladder Framework)**
- Pack 4 ternary values into 1 int8 for HBM
- 4x memory bandwidth reduction for weight loading
- Already noted for inference, applicable to training forward pass
- Verdict: MEDIUM for training

## Priority Summary (Step 1)

### Tier 1 — Paradigm Potential
- R1-CE-08: SSM Tensor Parallelism (2-4x multi-GPU)
- R1-CE-11: Fused Triton kernels for training
- R1-CE-12: Dynamic data pruning (1.43x)

### Tier 2 — Immediate/Zero-Cost
- R1-CE-02: Regional torch.compile (+30-50%)
- R1-CE-09: BitNet training recipe (multiple zero-cost tests)
- R1-CE-06: Mixed-precision gradients (-50% memory)

### Tier 3 — Worth Exploring
- R1-CE-01: Forward-Forward algorithm
- R1-CE-05: Forced sparsity in ternary
- R1-CE-07: Helion kernels
- R1-CE-13: Curriculum learning

### Tier 4 — Skip/Defer
- R1-CE-03: Perturbation training (NOT VIABLE)
- R1-CE-04: DFA (SKIP)
- R1-CE-10: KD (needs teacher)
- R1-CE-14: BitNet packing (inference focus)

## Step 2: Mathematical Analysis (2 math-paradigm-breakers, 2026-02-26)

### Mathematician 1: Kernel Fusion & Precision Analysis

**M1-CE-01: Fused Quantize-Matmul-STE Kernel — PASS HIGH**
- Q(W)·x = sign(W)·mask(W)·x — never need to materialize Q(W)
- Forward: sparse signed gather-accumulate (additions only, no multiplication)
- Backward (STE): dL/dw_ij = dL/dy_i · x_j — needs only output grad and input, NOT quantized weights
- At ~42% zeros: 30-40% reduction in forward-pass arithmetic
- Memory: eliminates temporary Q(W) tensor (~50 MB at 50M)
- Cross-pollination: combines naturally with shadow compression
- **Implementation: MODERATE (custom CUDA kernel for sparse signed gather)**

**M1-CE-02: Shadow Weight Compression via Boundary Distance — PASS HIGH**
- Boundary distance d_i = |w_i| - threshold determines ternary stability
- Committed weights (d_i >> η): need only 3 bits to maintain ternary assignment
- Boundary weights (d_i ~ 0): need 8+ bits (full FP16)
- Derived: b > log2(R / 2d_i) bits needed per weight
- Average committed: d_i ≈ 0.295, needs 3 bits. Boundary: needs 8 bits.
- Tiered scheme: 60-70% at FP8, 20-30% at FP16, 10-20% at FP8
- **Savings: 67.5% of shadow weight + optimizer state (~405 MB at 50M)**
- Re-evaluation cost: negligible (every 5 steps, element-wise comparison)
- Weight decay does NOT threaten tiering (decay rate << gradient rate)

**M1-CE-03: Gradient Sparsity in Ternary — PASS MEDIUM**
- STE computes nonzero gradients for weights whose TRUE gradients are exactly zero
- Committed weights (35-45%): gradient is noise, not signal
- Derived: committed weights need gradient updates only every ~200 steps
- Theoretical saving: ~40% of backward computation
- BUT: sparse backward GEMM is less hardware-efficient than dense
- Practical approach: gradient freeze for most committed 20%, measure speedup
- **Implementation: HARD (requires custom sparse backward kernels)**

**M1-CE-04: Forward-Forward for Ternary — MEDIUM (speculative)**
- FF goodness g(x) = ||h||^2 with ternary weights: h = signed selection of x components
- Key insight: layer-local losses = no gradient compounding across layers (unlike backprop+STE)
- STE-FREE variant: for each w_ij, try all 3 ternary values, pick best goodness. Cost: O(mn) scalar ops.
- Memory: eliminates ALL shadow weights AND optimizer state = 12x training memory reduction
- Risk: FF unproven for generative language modeling
- Hybrid: FF for ternary structure discovery → STE for fine-tuning
- **Implementation: MODERATE for algorithm, HIGH for matching backprop quality**

**M1-CE-05: SSM State Precision — PASS MEDIUM**
- Error propagation: E[||e||^2] = σ_ε^2 / (1 - α^2), logarithmic in memory length
- Fast decay heads (α<0.95): 8 bits suffice. Medium (0.99): 10-12 bits. Slow (0.999): 13 bits.
- Average: ~10-11 bits vs current 32 bits
- Training saving: ~1 MB (small — state is not bottleneck)
- Inference at batch 256: ~275 MB savings
- Practical: BF16 safe for 75% of heads (α < 0.99)
- **Implementation: MODERATE (mixed-precision state, per-head format selection)**

### Mathematician 2: Algorithmic Reformulations

**M2-CE-01: Cumsum SSM Reformulation — PASS HIGH (MAJOR)**
- Key insight: scalar A per head means ALL state dims share same decay
- Reformulation: h_t = φ_t · cumsum_s[dt_s·B_s·x_s / φ_s] where φ_t = exp(A·S_t)
- This is a CAUSAL CUMULATIVE SUM + pointwise multiply (trivially parallelizable)
- FLOPs: O(T·d_state·n_heads) vs SSD's O(T·C·d_state·n_heads) — 16x fewer FLOPs
- Numerically stable via log-space arithmetic
- **At 50M: 3-6% net training speedup (scan is ~15-25% of total)**
- **At 1B+: more significant as d_state grows**
- CAVEAT: FLA may already achieve similar efficiency through fusion

**M2-CE-02: Sparse Training Matmul — FAIL**
- Dense cuBLAS wins at batch>4 due to tensor core efficiency (16x advantage)
- Crossover requires >62% sparsity, and only for GEMV (not GEMM)
- Our sparsity ~42% is below crossover
- Only viable for inference (already exploited — Triton GEMV kernel)

**M2-CE-03: Optimizer State Compression — PASS MEDIUM-HIGH**
- 85-95% of weights are "committed" (far from boundary)
- STE gradients for committed weights = approximation noise (true gradient is zero!)
- Recommended: INT8 for committed m,v → 3.4x optimizer memory compression
- At 50M: optimizer from 534 MB to 156 MB
- At 1B: from 10.08 GB to ~3.1 GB (enables single-GPU training!)
- Risk: full freeze is dangerous (Finding 85: drift is beneficial)
- **Implementation: MODERATE (custom mixed-precision optimizer groups)**

**M2-CE-04: Information-Theoretic Training Bound — PASS HIGH (FRAMEWORK)**
- Ternary capacity: 1.585 × N bits. Training transfers info at rate R bits/step.
- Regime 1 (Discovery, ~1K steps): ~100K-700K bits/step, fills 92% of capacity
- Regime 2 (Refinement, ~69K steps): ~1K-10K bits/step, fills remaining 8%
- This PREDICTS Finding 88 (92.2% in first 1K steps) from first principles!
- Training is near information-theoretic optimum (within 1-3x)
- Actionable: variable batch sizing (small batch Regime 1, large batch Regime 2)
- **Implementation: TRIVIAL (change batch schedule)**

**M2-CE-05: Curriculum Learning for SSM — PASS MEDIUM**
- Derived: optimal L* = 1.278 × L_eff where L_eff = -1/(A·mean(dt))
- For our SSM: L_eff ≈ 100 tokens → L* ≈ 128 (matches validation scale!)
- Dynamic curriculum: monitor L_eff, adjust sequence length automatically
- Estimated savings: 20-25% total training wall-clock time
- **Implementation: LOW (~10 lines, adjust data loader)**

**M2-CE-06: Shadow Weight Elimination — PARTIAL**
- Full elimination: FAIL (all alternatives use more memory or worse convergence)
- Compression: PASS — hybrid scheme achieves 2.9x total ternary state compression
- Committed (85%): 2-bit ternary + 4-bit offset + INT8 optimizer = 2.75 bytes/param
- Boundary (15%): full FP32 = 12.25 bytes/param
- Average: 4.18 bytes/param vs current 12 = 2.87x compression
- **At 1B: from 10.08 GB to 3.51 GB — fits on RTX 5070!**

**M2-CE-07: Boundary-Focused Optimizer (BFO) — SPECULATIVE MOONSHOT**
- Only update ~12-15% of weights per step (those near boundaries)
- Skip committed weights entirely (not even reduced precision — just skip)
- Concentrate compute on boundary fraction with enhanced LR
- Could use second-order methods on small active set
- Predicted: 50-80% training time reduction
- Combined with info-theoretic framework: huge savings in Regime 2
- **Risk: does skipping committed updates cause regression?**
- **Testable: 6M scale, 1K steps, <5 minutes**

### Priority Ranking (Math-Derived)

| Rank | Finding | Impact | Viability |
|------|---------|--------|-----------|
| 1 | M2-CE-04 Info-theoretic framework | Changes all training decisions | Conceptual |
| 2 | M1-CE-02 + M2-CE-03 Shadow+optimizer compression | 2.9x memory (enables 1B!) | HIGH |
| 3 | M2-CE-05 Curriculum learning | -20-25% training time | HIGH |
| 4 | M2-CE-01 Cumsum SSM | 16x fewer scan FLOPs | MEDIUM-HIGH |
| 5 | M1-CE-01 Fused quantize-matmul | -30-40% forward compute | MEDIUM |
| 6 | M2-CE-07 BFO optimizer | -50-80% training time | SPECULATIVE |
| 7 | M1-CE-04 FF for ternary | 12x memory during training | SPECULATIVE |

## Step 3: Paradigm Breaking (2 paradigm-breakers, 2026-02-26)

### Paradigm-Breaker 1: Structural Training Optimizations

**P1-CE-01: Ternary Backward Kernel — HIGH VIABILITY (IMMEDIATE)**
- Key insight: backward pass through F.linear dispatches to cuBLAS FP16 even though weight is ternary
- Custom torch.autograd.Function using Triton GEMV for grad_input = grad_output @ W_ternary
- Activation gradient is ~50% of backward cost → 40-50% savings on that portion
- Estimated: 1.3-1.5x overall training speedup
- Code path: ternary.py line 455, existing Triton GEMV kernel can be adapted
- Test: 6M scale, 500 steps, compare wall-clock and gradient norms

**P1-CE-02: Frozen Core Training — HIGH VIABILITY (COMBINES WITH P1-CE-01)**
- Treat committed ternary weights (85%) as FROZEN constants during backward
- Only maintain gradient/optimizer for 15% boundary weights
- Boundary distance: d(w) > eta * lr * G_max * K guarantees no flip within K steps
- Reclassify every K=50-200 steps
- Memory: 85% reduction on optimizer states (600MB → 90MB at 50M)
- Compute: ~42.5% savings on weight gradient computation
- Combined with P1-CE-01: backward from 2x FP16 cost → 0.575x FP16 = **3.5x backward speedup**
- Risk: scale factor changes can shift boundaries; reclassification frequency critical
- Test: classify 6M weights, measure flip rate in committed set over 200 steps

**P1-CE-03: Bayesian Ternary Beliefs — MEDIUM VIABILITY (THEORETICAL)**
- Replace shadow weights + Adam with categorical probability p = (p_{-1}, p_0, p_{+1})
- Update: p_v = softmax(log p_v - beta * g * v) — online Bayesian inference
- Memory: 8 bytes/weight (vs 12 for shadow+Adam), or 4 bytes in FP16
- Natural momentum via belief inertia (strong belief ≈ large Adam second moment)
- Natural committed detection: p ≈ (0, 0, 1) = fully committed
- Risk: convergence differs from STE+Adam; team found optimizer sensitivity
- Better as THEORETICAL FRAMEWORK than practical replacement

**P1-CE-04: Top-K Coordinate Descent — MEDIUM VIABILITY (MOONSHOT)**
- Directly flip K most impactful weights per step in discrete ternary space
- No shadow weights, no optimizer states needed
- Memory: from 12 bytes/weight to 0.25 bytes/weight (packed 2-bit) = **50x reduction**
- Uses diagonal Hessian for flip quality: Delta_L ≈ g*(v_new - v_old) + 0.5*H*(delta_v)^2
- Adaptive K: start 1%, decay to 0.01% (simulates LR decay)
- Risk: STE was found essential — this eliminates it entirely
- Test: 6M, 1000 steps, compare loss within 10% = viable

### Paradigm-Breaker 2: Architectural Paradigm Shifts

**P2-CE-01: Input-Conditional Ternary Selection — HIGH VIABILITY (ATTACKS ANGULAR GAP)**
- MAJOR INSIGHT: angular gap exists because ONE ternary config compromises across all inputs
- Generate K=4-16 ternary quantizations (different thresholds or rotations)
- Tiny selector network S(x) picks best quantization per input
- Inference: selector is one small FP16 layer; matmul still ternary
- Memory: K * 2N bits. For K=4, 50M: 4 * 84MB = 336MB (fits easily)
- **ZERO-COST TEST (30 min):** oracle best-of-4 with thresholds 0.6x/0.7x/0.8x/0.9x on trained checkpoint
- If oracle PPL improves >0.5%, concept proven

**P2-CE-02: Codebook-First Training — MEDIUM-HIGH VIABILITY (GAME-CHANGER)**
- Train in codebook space from start (not post-hoc compression)
- K=4096 ternary prototype vectors, each row = one prototype index
- Optimizer state: only K*group_size floats vs N floats = **84x reduction at 50M**
- At 7B: optimizer from 23.5 GB to 2 MB — **7B training on RTX 5070 12GB!**
- Gradient noise reduction: each prototype gets gradients from thousands of groups
- Risk: codebook may be too restrictive; index STE may fail
- Test: 6M, K=256, group_size=64, 1000 steps

**P2-CE-03: Dilated Ternary Convolutions — MEDIUM VIABILITY (BOLD)**
- Replace SSM recurrence entirely with dilated causal convolutions
- Full parallelism (no sequential scan), full GPU utilization
- L=18 layers, K=16, max dilation limited → receptive field 16K tokens
- Training: 10-50x throughput improvement (eliminates 87% sequential bottleneck)
- Risk: quality degradation vs SSM (HiPPO theory gives SSM better compression)
- Possible hybrid: train-as-convolution, infer-as-SSM (S4 conversion)
- Test: 6M, replace SSM with dilated conv, 1000 steps, compare loss

**P2-CE-04: Wolff Cluster Discrete Refinement — MEDIUM VIABILITY (POST-TRAINING)**
- Statistical physics approach: ternary weights as spin-1 Ising variables
- Compute gradient correlations J_ij, identify clusters via Wolff algorithm
- Flip entire clusters simultaneously (escapes local minima single-flip can't)
- Wolff has O(1) autocorrelation time (vs O(L^z) for single-spin Metropolis)
- Applied POST-TRAINING as discrete refinement of STE solution
- Risk: J_ij computation is O(N^2) — must use local neighborhoods only
- Test: one 6M layer, compute row-local correlations, try cluster flips

**P2-CE-05: PBO-Exact Layer Refinement — LOW-MEDIUM VIABILITY (EXACT)**
- Pseudo-Boolean Optimization solver for exact optimal ternary weights per layer
- Fix all layers except one, solve PBO for that layer, iterate
- Modern solvers handle 10^6+ boolean variables
- Risk: scalability to 640x640 layers, linearization accuracy
- Test: one 6M layer (256x256 = 131K booleans), compare to STE solution

### Cross-Paradigm Synthesis

| Rank | Finding | Impact | Viability | Combinable |
|------|---------|--------|-----------|------------|
| 1 | P1-CE-01 + P1-CE-02 Ternary backward + frozen core | 1.7-2.2x training, 6x mem | HIGH | Stack multiplicatively |
| 2 | P2-CE-01 Input-conditional selection | Attacks angular gap directly | HIGH | With codebook |
| 3 | P2-CE-02 Codebook-first training | 84x optimizer mem, enables 7B | MEDIUM-HIGH | With P2-CE-01 |
| 4 | P1-CE-04 Top-K coordinate descent | 50x memory, eliminates STE | MEDIUM | Replaces optimizer |
| 5 | P2-CE-03 Dilated ternary conv | 10-50x training throughput | MEDIUM | Architectural change |
| 6 | P2-CE-04 Wolff clusters | Post-training refinement | MEDIUM | Stacks on any |
| 7 | P1-CE-03 Bayesian beliefs | Theoretical framework | MEDIUM | Alternative to STE |

### KEY PARADIGM INSIGHT (recurring across both breakers)

**The project has been optimizing within the "single ternary configuration" paradigm.** Both breakers independently identified that the angular gap is GEOMETRIC and cannot be closed by better training. The paradigm shift is either:
- (a) Allow MULTIPLE ternary configurations selected per input (P2-CE-01)
- (b) Optimize in STRUCTURED discrete space instead of treating weights as independent (P2-CE-02)
- (c) Eliminate the continuous relaxation entirely and work directly in ternary space (P1-CE-04)

**Hidden assumption exposed:** The backward pass uses cuBLAS FP16 for ternary weights. This is an immediate, concrete optimization (P1-CE-01) that requires only adapting the existing Triton GEMV kernel.
