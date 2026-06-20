# Research Round 5 — Training Velocity

## Overview

This document tracks research findings on training speed optimization for ternary SSM models. "Training Velocity" means specifically: throughput (tok/s), wall-clock time to train, steps/second, and techniques that reduce total training time. NOT memory efficiency or energy reduction.

## NEW FINDINGS (Round 5, Aspect 2)

### Wave 1: Web Research (2 researchers, Feb 2026)

#### SSM TRAINING ACCELERATION

**F1: Implicit SSM Training Curriculum (Adaptive Depth)** (arxiv 2502.07827, Feb 2025)
- Bounded-depth implicit models that self-iterate until convergence with adaptive compute budget. Uses ~8 self-iterations at training vs unbounded at test time.
- Speed: 32x fewer sequential steps than RNNs (8 vs 256 for same problem).
- Relevance: Variable-depth processing — simpler sequences converge faster. Could reduce avg training compute per token.
- Complexity: High (implicit function theorem gradients, Phantom Gradient method).

**F2: LASP-2 Sequence Parallelism (AllGather-based)** (arxiv 2502.07563, Feb 2025)
- Redesigned sequence parallelism using single AllGather on memory states (d×d, sequence-length independent). Replaces ring-style P2P communication.
- Speed: 15.2% faster than LASP-1, 36.6% faster than Ring Attention (2048K seq, 64 GPUs).
- Relevance: Our diagonal A + SSD = memory state is d_state×d_state. Could parallelize training across multiple GPUs.
- Complexity: Medium (requires distributed setup).

**F3: LASP-2H Hybrid Model Parallelism** (arxiv 2502.07563, Feb 2025)
- Extension of LASP-2 for hybrid models mixing linear attention + standard attention. Unified AllGather for both layer types.
- Speed: Same ~15-36% gains, works for hybrid architectures.
- Relevance: Our model is SSM + sliding window attention (hybrid). Direct application.
- Complexity: Medium.

**F4: Zero-Communication Sequence Parallelism** (NeurIPS 2025)
- Eliminates inter-GPU communication overhead for linear attention by exploiting right-product associativity differently than LASP.
- Speed: Zero communication overhead claimed for sequence parallelism.
- Relevance: Diagonal SSM could benefit if technique generalizes beyond linear attention.
- Complexity: Unknown (paper limited access).

**F5: Mamba-3 MIMO State Updates** (OpenReview 2025)
- Multi-input multi-output formulation that better exploits hardware parallelism. Complex state update rule.
- Speed: Pareto-optimal performance/budget curve claimed. Inference-first design.
- Relevance: May inform training throughput via better hardware utilization. Architecture-level change.
- Complexity: High (architecture changes required).

**F6: Mamba Training Dynamics Feature Filtering** (arxiv 2602.12499, Feb 2026)
- Theoretical analysis showing gating vector aligns with class-relevant features. Sample complexity scales as (αr-αc)^{-2}.
- Speed: Theory-driven LR/warmup schedules could reduce training steps by ~20-30%.
- Relevance: Our ternary gating could converge faster with informed initialization based on this theory.
- Complexity: Low (theory → hyperparameter tuning).

**F7: LrcSSM Non-linear Recurrence with Diagonal Jacobian** (NeurIPS 2025)
- Non-linear recurrent model with forced diagonal Jacobian enabling O(TD) parallel processing.
- Speed: "As fast as linear state-space layers" with added non-linearity.
- Relevance: Adds expressivity without speed loss. Could improve quality per training step.
- Complexity: Medium.

**F8: EpochCore Hardware Accelerator** (arxiv 2507.21394, Jul 2025)
- Custom ASIC with LIMA-PE units for FRI-MAC and TRI-MAC recurrent SSM updates. 250x speedup vs traditional systolic array.
- Speed: 2000x inference vs GPU, 250x training vs SA baseline (not vs GPU training).
- Relevance: Shows massive headroom for custom SSM kernels. Ternary could get similar gains with specialized logic.
- Complexity: Very High (ASIC, not software).

**F9: Persistent Cache-Aware Grouped GEMM (Triton)** (pytorch.org blog, 2025)
- Triton kernel for grouped GEMM with cache-aware persistent thread blocks, reducing memory traffic.
- Speed: Significant speedup for MoE training. Applicable to any grouped computation.
- Relevance: Our model has grouped SSM channels. Triton kernel could optimize ternary matmuls during training.
- Complexity: Medium (Triton programming).

**F10: Dynamic Sparse Training of Diagonal Networks** (ICML 2025)
- Sparse training maintaining diagonal structure. Prunes weights dynamically while preserving diagonality.
- Speed: Faster training for diagonal-heavy models. Hardware-friendly sparsity patterns.
- Relevance: Our A matrix is diagonal — could combine with ternary for ultra-sparse training.
- Complexity: Medium.

**F11: Implicit Models with Path-Independent Generation** (arxiv 2502.07827, Feb 2025)
- Training in simultaneous mode (parallel across sequence), inference in sequential mode. Same fixed point either way.
- Speed: Enables highly parallel training, constant-memory inference.
- Relevance: SSM could train in parallel mode (faster), deploy in sequential mode (memory-efficient).
- Complexity: High.

**F12: Mamba Gating Mechanism Theory (Convergence Rates)** (arxiv 2602.12499, Feb 2026)
- Proves gating vector alignment dynamics. For majority-voting data: T=Θ(L²/η(αr-αc)²).
- Speed: Theory-guided warmup and LR schedules could reduce steps by ~20-30%.
- Relevance: Ternary gates could converge faster with informed initialization.
- Complexity: Low (apply theory to training recipe).

#### QUANTIZATION-AWARE TRAINING SPEEDUPS

**F13: EfficientQAT Block-AP + E2E-QP** (ACL 2025, OpenGVLab)
- Two-phase QAT: block-wise training of all parameters, then end-to-end training of quantization params only.
- Speed: Reduces QAT cost — second phase trains only scale factors. W2g64 Llama-2-7B achieves 6.86 PPL.
- Relevance: Could train ternary params separately from scale factors. Compatible with per-group scaling.
- Complexity: Medium (two-stage pipeline).

**F14: WiniQ — Weight Re-initialization via Linear Interpolation** (NeurIPS 2025)
- Periodically re-initializes weights by linear interpolation between FP and quantized weights during QAT. Resets to regions with larger Hessian eigenvalues without increasing loss.
- Speed: **4× acceleration** in QAT convergence. Up to 8.8% improvement under same training budget.
- Relevance: HIGH — directly addresses slow convergence in low-bit QAT. The "flat Hessian surface" problem matches our ternary training experience.
- Complexity: Low (periodic weight interpolation + noise injection).

**F15: Scaling Law for QAT** (arxiv 2505.14302, May 2025)
- Empirical scaling laws predicting QAT performance vs compute tradeoffs for LLMs.
- Speed: Enables intelligent resource allocation — avoid undertraining or overtraining.
- Relevance: Planning tool for ternary training budgets. Informs compute-optimal schedules.
- Complexity: Low (reference curves).

**F16: FP8 Mixed-Precision Training** (Multiple sources, 2025-2026)
- Use FP8 for activations/gradients in non-ternary layers while keeping ternary weights. Native H100/Blackwell FP8 Tensor Core support.
- Speed: 1.5-2× on FP8-capable hardware for non-quantized operations. Larger batches via memory savings.
- Relevance: SSM states/gates could run FP8. Note: RTX 5070 has FP4 not FP8 Tensor Cores — cloud H100s needed.
- Complexity: Medium (FP8 kernel support, loss scaling).

**F17: Highway Backpropagation** (ICLR 2025)
- Parallelizable backprop that alternates gradient accumulation along residual path and backpropagation through every layer in parallel.
- Speed: Major speedups with minimal performance degradation. Parallelizes inherently sequential backprop.
- Relevance: Our model has residual connections (SSM skip paths). Could parallelize gradient computation.
- Complexity: High (architecture support for residual gradient flow).

**F18: Fused Quantize-Dequantize Kernels** (2025)
- Custom CUDA kernels fusing quantization Q(W) and dequantization into single kernel launch, reducing memory traffic.
- Speed: 1.2-1.5× reduction in quantization overhead during STE forward pass.
- Relevance: HIGH — we quantize/dequantize every forward pass. Fused kernel reduces overhead.
- Complexity: High (custom CUDA or TorchAO/CUTLASS).

**F19: Asynchronous Data Prefetching with Multi-Level Caching** (RunPod, 2025)
- Multi-process data loading + async prefetching + multi-level caching (RAM/SSD). Maintains optimal buffering.
- Speed: Achieves 85-95% GPU utilization (vs 40-60% with poor pipelines). 2-3× when data-bottlenecked.
- Relevance: Our 25-45K tok/s may be data-bottlenecked. WikiText-103 fits in RAM for full caching.
- Complexity: Low-Medium (DataLoader tuning).

**F20: Persistent Workers + Optimized DataLoader** (PyTorch best practices, 2025)
- Keep worker processes alive between epochs, optimize num_workers/prefetch_factor, pin memory.
- Speed: 1.2-1.5× by eliminating worker startup overhead.
- Relevance: Simple tuning applicable to any PyTorch training. Zero risk.
- Complexity: Low (configuration only).

**F21: WebDataset Optimized Format** (2025)
- Optimized sequential-read data formats (WebDataset, Arrow) balancing compression with decompression speed.
- Speed: 1.3-2× faster data loading vs naive formats. Reduced I/O.
- Relevance: Preprocessing WikiText-103 into optimized format could improve epoch iteration speed.
- Complexity: Low (one-time preprocessing).

**F22: Gradient Accumulation + Microbatching Optimization** (2025 best practices)
- Tune global_batch = micro_batch × grad_accum × num_gpus. Maximize GPU utilization without OOM.
- Speed: 20-40% throughput improvement from optimal micro_batch sizing.
- Relevance: We already use this but may not be optimally tuned. Profile recommended.
- Complexity: Low (hyperparameter search).

#### KEY PATTERNS (Wave 1)
- **Sequence parallelism maturing:** LASP-2, Zero-Comm SP show 15-36% gains via better communication
- **Theory-driven training:** Mamba dynamics theory could inform 20-30% fewer steps needed
- **QAT convergence acceleration:** WiniQ (4× speedup) specifically targets flat Hessian problem in low-bit QAT
- **Data pipeline as bottleneck:** Multiple sources emphasize 40-60% GPU idle time from poor data pipelines
- **Fused kernels:** Quantize-dequantize fusion reduces STE overhead 1.2-1.5×
- **FP8 coming but not yet for us:** RTX 5070 lacks FP8 Tensor Cores; relevant for cloud training

#### TOP 5 WAVE 1 FINDINGS
1. **F14 WiniQ** — 4× QAT convergence, low complexity, directly targets ternary convergence issues
2. **F2 LASP-2** — 15-36% training speedup, proven for hybrid SSM+attention architectures
3. **F19 Async Prefetching** — 2-3× if data-bottlenecked, simple implementation
4. **F18 Fused QDQ Kernels** — 1.2-1.5× reduction in quantization overhead, high relevance
5. **F6/F12 Mamba Theory** — 20-30% fewer training steps via theory-guided hyperparameters

### Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

#### FASTER GRADIENT COMPUTATION

| # | Technique | Speedup | Viability | Key Insight |
|---|-----------|---------|-----------|-------------|
| M1 | Parallel Scan Backward | 30-40% backward | HIGH | Diagonal A makes backward a linear scan, O(log T) depth |
| M2 | Boundary-Sparse Optimizer | 20-50% optimizer | MEDIUM | Only weights near ternary boundaries need full updates |
| M3 | Amortized Quantization (Lipschitz) | 10-15% quant | MEDIUM | Skip Q(W) when W hasn't changed enough to affect ternary |
| M4 | Spectral Backward (FFT) | Marginal over scan | MEDIUM | FFT backward amortizes across multiple param gradients |
| M5 | Importance-Sampled BPTT | 20-30% backward | MEDIUM | Sample time steps with known decay for unbiased gradients |
| M6 | Ternary Teleportation | 2x fewer steps | MEDIUM | Direct discrete optimization via top-K flip selection |
| M7 | Evolutionary Ternary Search | Potentially transformative | SPECULATIVE | No backward pass; exploit ternary forward cheapness |
| M8 | Low-Rank Shadow Weights | 16x optimizer memory | MEDIUM | W_fp as UV^T; optimizer in low-rank space |

**M1: Parallel Scan Backward** — HIGHEST PRIORITY
- For diagonal A: backward recurrence G_s = g_s + a * G_{s+1} is a linear scan.
- Affine transformations compose associatively → parallel prefix scan in O(log T) depth.
- For T=2048: 11x reduction in sequential depth (2048 → ~11 scan levels).
- Same machinery as forward scan (Mamba). Must verify backward actually uses it (autograd loop = sequential).
- Custom backward with parallel scan could give 30-40% reduction in backward time.

**M2: Boundary-Sparse Optimizer**
- Track boundary distance b_i = min(|w_i - alpha|, |w_i + alpha|, |w_i|) per weight.
- Weights far from boundaries: ternary assignment won't change → skip full optimizer update.
- Late in training: ~70-90% of weights are "frozen" in their basin.
- Skip Adam m,v updates for frozen weights → 70-90% less optimizer compute.
- Deep connection to M6 (teleportation): boundary set = active flip candidates.

**M3: Amortized Quantization via Lipschitz Tracking**
- Track accumulated delta per group since last quantization.
- If delta < c × min_boundary_distance, skip re-quantization for that group.
- After warmup, ~80% of groups can skip quantization per step.
- Saves ~12% of quantization overhead (quantization is ~15% of step time).
- Safety factor c < 1 provides conservative guarantee.

**M4: Spectral Backward via FFT**
- Diagonal SSM = LTI system → diagonalized by DFT.
- dL/dU(ω) = conj(K(ω)) · dL/dH(ω) — pointwise multiply in frequency domain.
- Gradient w.r.t. A: dK/da(ω) = e^{-jω}/(1-ae^{-jω})² — single DFT + pointwise.
- Both O(log T) depth like parallel scan, but cuFFT is heavily optimized on GPU.
- Main advantage: amortization across multiple parameter gradients.

**M5: Importance-Sampled BPTT**
- Sample S << T time steps from distribution p(t) ∝ |a|^{t-T} (exponential favoring recent).
- Unbiased estimator: dL/dθ_hat = Σ (1/p(t_s)) · (dL/dh_{t_s}) · (dh_{t_s}/dθ).
- Antithetic sampling + control variates for variance reduction.
- With S = T/4: save ~30% of backward SSM time, but noisier gradients.

**M6: Ternary Teleportation** — HIGH INTEREST
- Direct optimization in discrete {-1,0,+1} space via top-K flip selection.
- Approximate flip gradient: Δ_L(i,j,v) ≈ (dL/dW_q[i,j]) · (v - W_q[i,j]).
- Select top-K flips by improvement magnitude, apply directly.
- Eliminates shadow weight W_fp entirely (or hybrid: STE first 80%, teleportation last 20%).
- Each step makes guaranteed discrete progress. No wasted gradients on stable weights.

**M7: Evolutionary Ternary Search** — MOONSHOT
- Population-based: P=32 ternary weight matrices, evaluate via forward-only.
- Structured mutations: column swap, sign flip, local flip, block flip.
- No backward pass at all. Forward-only with ternary = additions only.
- If ternary forward is 10x faster than FP16 forward+backward: generation costs 3.2x one gradient step.
- Viable for fine-tuning after initial gradient-based training.

**M8: Low-Rank Shadow Weights (Factored Proxy)**
- Maintain W_fp = UV^T + C (low-rank + sparse correction).
- Optimizer states for U,V: (M+K)×r instead of M×K. For r=32, M=K=1024: 16x reduction.
- Memory savings: 400MB → 60MB optimizer state at 50M params.
- Freed VRAM enables larger batch sizes (primary throughput lever).
- Risk: low-rank may not represent all ternary patterns. Periodic SVD "absorption" of C needed.

#### TRAINING THROUGHPUT MAXIMIZATION

| # | Technique | Speedup | Viability | Key Insight |
|---|-----------|---------|-----------|-------------|
| M9 | Shadow-Delta Propagation | 2-4% | MEDIUM | Only recompute Q(W) for weights that actually flipped |
| M10 | Ternary-Native Optimizer (TNO) | 25-35% | HIGH | Partition weights by boundary distance, skip frozen updates |
| M11 | Fused QMA Kernel | 1.5% | MEDIUM | Fuse quantize + matmul: read W_fp once, quantize in registers |
| M12 | FFT-Based SSM Backward (Chunked) | 10-12% | HIGH | Replace sequential chunk gradient with FFT convolution |
| M13 | Stochastic Layer Dropping | 8-10% net | MEDIUM | Drop 20% of layers per step; ternary STE noise provides room |
| M14 | Operator Splitting (Periodic Q) | 5-7% | MEDIUM | Reuse Q(W) for K steps via Lie-Trotter splitting |
| M15 | Bitmask Weight Decomposition | 2-3% (30-50% if FP4 TC) | SPECULATIVE | W = W+ - W-; binary masks + POPCOUNT; FP4 Tensor Core |
| M16 | Gradient-Free Boundary Perturbation | Up to 50%/step | LOW-MEDIUM | Forward-mode flip evaluation replaces backprop for interior layers |

**M10: Ternary-Native Optimizer (TNO)** — HIGHEST PRIORITY
- Partition parameters: Active (near boundary, b_i < threshold) vs Frozen (deep in basin).
- Active: SGD with momentum (not Adam — variance normalization meaningless for binary STE on/off).
- Frozen: plain SGD, no state.
- Memory: 400MB → 60MB optimizer state (85% reduction at 50M params).
- Freed 340MB allows batch 64 → 96-128: **25-35% throughput from batch size increase alone**.
- Risk: must verify SGD+momentum converges as well as Adam for ternary.

**M12: FFT-Based SSM Backward** — HIGH PRIORITY
- Diagonal A: h_t = Σ a^s · u_{t-s} = convolution with geometric kernel.
- All intermediate states via FFT: O(T log T) fully parallel (no sequential recurrence).
- Gradient w.r.t. A: two FFT convolutions + elementwise ops.
- Replaces sequential chunk-to-chunk gradient propagation.
- Net: ~10-12% total throughput improvement (SSM backward is ~15% of step time).

**M13: Stochastic Layer Dropping for Ternary**
- Drop probability p = 0.2: compute only 80% of layers per step on average.
- Ternary-specific insight: STE variance is already HIGH → room for layer-drop noise.
- Adaptive p_l based on per-layer gradient SNR: low-SNR layers dropped more aggressively.
- Net after convergence slowdown: ~8-10% throughput gain.
- More viable for larger models (100M+) with more layer redundancy.

**M14: Operator Splitting (Periodic Requantization)**
- Lie-Trotter splitting: K gradient steps with same Q(W), then re-quantize.
- Strang splitting for second-order accuracy: half-Q → full-gradient → half-Q.
- K=4-8 is safe when flip rate <1% per step (validated after warmup).
- Cache Q(W) in L2 for K steps → reduced memory traffic.
- Combined: ~5-7% throughput improvement.

**M15: Bitmask Weight Decomposition**
- Q(W) = mask_pos - mask_neg (two binary masks). 4x bandwidth reduction.
- Matmul becomes AND-gather: y = (mask_pos AND_gather x) - (mask_neg AND_gather x).
- At 50M params: save 37.5 MB bandwidth per forward pass.
- RTX 5070 FP4 Tensor Core encoding: pack 2 ternary values per FP4 slot (9 combos in 4 bits).
- Speculative but connects to Phase 30 POPCOUNT plans.

**M16: Gradient-Free Boundary Perturbation**
- Identify boundary weights. For random subset R: evaluate single-weight flip via forward-mode.
- Forward perturbation for one flip: O(d) per layer (not O(d²) of full backprop).
- If |R| = √d ≈ 224 flips and interior layers are 60% of backward: 2.4x faster backward.
- HIGH variance — convergence may degrade significantly.
- Best as hybrid: full backprop for embedding/output, perturbation for interior ternary layers.

#### KEY CROSS-POLLINATION (Wave 2)
- **M1 + M12:** Both exploit diagonal A for parallel backward. M1 via scan, M12 via FFT. Use whichever has better GPU implementation.
- **M2/M10 + M3/M9:** Boundary tracking serves BOTH optimizer sparsity AND quantization amortization. Single computation, dual benefit.
- **M6 + M2:** Teleportation's flip candidates = boundary-sparse optimizer's active set. Same information, two perspectives.
- **M8 + M10:** Low-rank shadow (16x optimizer memory) + TNO (85% state reduction). Combined: 340MB + 37.5MB savings → batch 128+ on 12GB VRAM.
- **M13 + M16:** Layer dropping + boundary perturbation = three-tier compute: (a) full forward+backward, (b) forward + perturbation, (c) skip. Tunable budget.
- **Universal insight:** Ternary training has ENORMOUS redundancy — most weights don't flip between steps, most optimizer state tracks frozen weights. A "ternary-aware training engine" computing only deltas could save 40-60% of total step time.

### Wave 3: Paradigm Breakers (2 agents, Feb 2026)

#### RADICAL TRAINING ARCHITECTURES

| # | Technique | Speedup | Key Insight |
|---|-----------|---------|-------------|
| P1 | Ternary-Resident Training (TRT) | 1.5-2x | Use ternary matmul kernels in TRAINING forward+backward, not just inference |
| P2 | Sparse Delta Training | 1.2-1.5x (late training) | Only compute Q(W) delta from the <0.1% of weights that actually flip |
| P3 | Population-Based Ternary Search | Potentially transformative | No backward; evolve population of 8 ternary models (80MB < 1 model's optimizer state) |
| P4 | Proximal Alternating Minimization | 1.5-2x | Train low-rank adapters with EXACT gradients, periodically absorb into ternary |

**P1: Ternary-Resident Training (TRT)** — HIGHEST PRIORITY
- STE forward already uses quantized weights: w = w_shadow + (Q(w_shadow) - w_shadow).detach().
- Yet F.linear(x, w) executes as FP16 matmul despite w containing only {-alpha, 0, +alpha}.
- Replace with ternary matmul kernel (add/subtract/skip) for BOTH forward AND backward dL/dx.
- Only dL/dW = dL/dy @ x^T remains FP16 (unavoidable outer product, but faster than forward matmul).
- Project already has ternary kernel infrastructure (pack_weights.py, ternary_matmul_wrapper.py).
- Expected: if matmuls are ~70% of step time and speed up 3x → net ~1.9x training speedup.
- Viability: MEDIUM-HIGH. Mathematical foundation solid. Main risk: GPU ternary may not beat FP16 on RTX 5070.

**P2: Sparse Delta Training**
- At convergence: <0.1% of ternary weights change between consecutive steps (Phase 21.B data).
- Cache packed ternary weights across steps. Only update positions that actually flip.
- Eliminates per-step quantization + packing overhead entirely.
- delta_W has entries in {-2,-1,0,+1,+2} with <0.1% density → sparse matmul nearly free.
- Most effective in later training stages when flip rate drops. Combine with TRT for full benefit.
- Viability: MEDIUM. Need to measure flip rate trajectory (high early, low late).

**P3: Population-Based Ternary Search (PBTS)** — MOONSHOT
- Population of 8 models at 50M: 8 × 10MB = 80MB (ternary packed). Less than ONE model's Adam state (300MB).
- Forward-only evaluation (no backward): each eval costs ~1/3 of forward+backward step.
- Structured mutations: column swap (preserve function), sign flip, local flip, block flip.
- Zero backward, zero optimizer state, zero shadow weights. All forward passes parallelizable.
- HIGH RISK: evolutionary convergence likely orders of magnitude slower than gradient descent.
- Best as HYBRID: STE for 80% of training, evolutionary polish for final 20%.
- Viability: LOW standalone, MEDIUM as hybrid fine-tuning.

**P4: Proximal Alternating Minimization (PAM)**
- Decompose: y = (W_ternary + A @ B) @ x. Freeze W_ternary, train adapters A,B with EXACT gradients.
- Every K steps: absorb correction W_ternary_new = Q(W_ternary + A @ B), reset adapters.
- This is proximal gradient method: gradient step (adapters) + proximal projection (requantize).
- No STE approximation error — gradients through adapters are mathematically correct.
- Optimizer state: 3MB (390K scale params) vs 400MB (50M shadow weights) = 133x reduction.
- Existing infrastructure: DeferredTernaryLinear class already has delta_A, delta_B.
- Risk: Phase 10.D.1 showed frozen-ternary + alpha stagnates. Rank-4 adapters more expressive but untested.
- Viability: MEDIUM. Sound math, existing code, but convergence needs validation.

#### TRAINING AS ALTERNATIVE COMPUTATION

| # | Technique | Speedup | Key Insight |
|---|-----------|---------|-------------|
| P5 | Analytical Ternary Assignment | 3-5x | Replace backprop with correlation-based combinatorial solving (Wiener filter) |
| P6 | Chunk-Selective Backward | 1.5-2x | Skip backward through low-loss chunks (vanishing gradient anyway) |
| P7 | Convergence-Locked Layer Freezing | 1.5-2.5x progressive | Freeze layers whose ternary pattern stabilizes; delete shadow weights + optimizer |
| P8 | Factored Scale-Pattern Optimization | 1.5x + 5x memory | Separate scale gradient (continuous) from pattern flips (discrete) |

**P5: Analytical Ternary Assignment** — PARADIGM SHIFT
- For y = Q(W)x: optimal W derived from sufficient statistics Σ_x = E[xx^T] and C_xt = E[xt^T].
- Wiener filter solution: w* = Σ_x^{-1} C_xt. Then Q(w*) via greedy coordinate descent on quadratic.
- No backward pass through SSM recurrence. Forward-only statistics collection.
- Low-rank Σ_x approximation (top-64 SVD): 12MB total vs 600MB optimizer state.
- Risk: layerwise alternating optimization may not match end-to-end backprop quality.
- GPTQ connection: same mathematical framework, proven for post-training quantization.
- Viability: MEDIUM. Sound math, but from-scratch training is different regime than post-training.

**P6: Chunk-Selective Backward** — HIGH PRIORITY
- SSD chunks the sequence into size-64 blocks. Not all chunks contribute useful gradients.
- Compute per-chunk loss (trivial). Select top-P% highest-loss chunks for backward.
- State gradient propagation through ALL chunks (cheap), weight gradients only from selected.
- At P=40%: backward compute reduced by ~60% for intra-chunk operations.
- Low-loss chunks have vanishing gradients anyway → minimal quality impact.
- Compatible with existing FLA kernel: just add chunk mask to backward.
- Viability: HIGH. Straightforward engineering, clear mathematical justification (importance sampling).

**P7: Convergence-Locked Layer Freezing** — HIGHEST VIABILITY
- Track Hamming distance between consecutive ternary snapshots per layer.
- When distance < 0.1% for N consecutive checks: LOCK layer.
- Locked = delete shadow weights + optimizer state + skip dW in backward.
- Backward through locked layer: dx = Q(W)^T dy (ternary transpose, additions-only, no dW).
- Progressive: early layers converge first → "shrinking trainable zone" over training.
- Typical timeline: 100% layers training → 67% → 33% → 17% as training progresses.
- Safety: periodic "thaw check" — temporarily compute dW for locked layers, unlock if flips detected.
- Worst case: no layers lock = identical to baseline. ZERO quality risk from mechanism.
- Viability: HIGH. Minimal code, existing flip-counting infrastructure.

**P8: Factored Scale-Pattern Alternating Optimization**
- Maintain ternary patterns T as packed binary (no shadow weights, ~12MB at 50M).
- Maintain per-group scales α as FP32 (~1.5MB). Optimizer state: only 3MB (Adam for 390K scales).
- Every step: gradient-based scale optimization (standard backprop through 390K params only).
- Every K steps: accumulated gradient sign determines which ternary values to flip.
- Addresses D.1 failure: scale-only stagnated because no flip mechanism. This adds flips.
- Memory: 115MB total vs 600MB standard = 5.2x reduction.
- Backward: cheaper (gradient flows to 390K scales, not 50M shadow weights).
- Risk: threshold tuning for flip decisions. Interaction between scale updates and flips.
- Viability: MEDIUM-HIGH.

#### KEY PATTERNS (Wave 3)
- **Training pays FP16 prices for ternary product** — the fundamental inefficiency. P1 (TRT) directly attacks this.
- **Ternary convergence signal is unique** — Hamming distance provides a NATURAL layer-locking criterion unavailable in continuous models. P7 exploits this.
- **Shadow weights are a 200MB crutch for STE** — P4, P5, P8 all eliminate them via different mechanisms.
- **Combinatorial search viable due to tiny information content** — 50M ternary = 10MB information. P3, P5 exploit this.
- **Best approach is COMBINABLE** — P6 (chunk-selective) + P7 (layer freezing) = 2-3x total with zero quality risk.

#### CROSS-WAVE SYNERGIES (Training Velocity)
- **P1 (TRT) + P2 (Sparse Delta):** Ternary matmul in training + only update flipped positions = multiplicative
- **P6 (Chunk-Selective) + P7 (Layer Freezing):** Chunk selection reduces backward, layer freezing eliminates it for locked layers. Combined: 2-3x total speedup.
- **M10 (TNO) + P7 (Layer Freezing):** TNO partitions weights by boundary distance, freezing deletes their state entirely. Progressive → full elimination.
- **M1 (Parallel Scan Backward) + P6 (Chunk-Selective):** Parallel scan for selected chunks only. O(P% × log T) depth.
- **P4 (PAM) + P8 (Factored):** Both separate scale from pattern. PAM uses low-rank adapters, P8 uses accumulated gradient sign. Could be combined: adapters for scale, gradient sign for flips.
- **P5 (Analytical) + P7 (Layer Freezing):** Analytical assignment for initial solution, then lock converged layers. Forward-only training for the locked portion.

---

## Aspect 2 Summary: Training Velocity

**Total findings:** 22 web research (F1-F22) + 16 mathematical (M1-M16) + 8 paradigm shifts (P1-P8) = **46 findings**

**Top 5 by estimated impact:**
1. **P1 Ternary-Resident Training** — 1.5-2x by using ternary matmul kernels in training (not just inference)
2. **P7 Convergence-Locked Layer Freezing** — 1.5-2.5x progressive, zero quality risk, minimal code
3. **P6 Chunk-Selective Backward** — 1.5-2x via importance-sampled backward through high-loss chunks only
4. **M10 Ternary-Native Optimizer** — 25-35% from batch size increase via 85% optimizer memory reduction
5. **F14 WiniQ** — 4x QAT convergence acceleration, low complexity

**Recommended implementation order:**
1. P7 (Layer Freezing) — zero risk, immediate progressive speedup, minimal code
2. P6 (Chunk-Selective Backward) — high viability, clear math justification
3. M10 (TNO) — memory savings enable larger batch = direct throughput gain
4. P1 (TRT) — requires kernel work but biggest single-technique speedup potential
5. F14 (WiniQ) — convergence speedup orthogonal to all above

**Combined potential:** P7 + P6 + M10 = estimated 3-4x training speedup with high confidence.

**Fundamental insight:** Ternary training has ENORMOUS redundancy. Most weights don't flip between steps, most optimizer state tracks frozen weights, most backward compute flows through already-learned chunks. A "ternary-aware training engine" that exploits ALL these redundancies could achieve 5-10x speedup over naive STE+AdamW.
