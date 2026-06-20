# R8-A3: Compile-time Optimization Findings

**Synthesis Date:** 2026-03-06
**Research Waves:** 2 (W1: Manual + torch.compile defaults | W2: Math analysis of overhead and state-visible optimizations)
**Total Findings:** 15 unique (5 W1-AgA, 5 W1-AgB, 5 W2-AgA, 5 W2-AgB, 3 deduplicated W1 overlaps)

## Summary: 12 Distinct Findings — 7 IMPLEMENT, 3 EXPLORE, 2 SKIP

### Context: Critical Bottleneck Analysis

**Step time baseline (v5 FP16, 6M params):** 45.3 ms/step
**Kernel-only time (theoretical FLOP-limited):** 0.06 ms
**Overhead fraction:** 75-85% of wall time is Python/kernel-launch overhead, not compute
**Key insight:** Memory-bandwidth and launch-overhead bound, NOT compute-bound

This changes prioritization: **FLOP-reduction techniques have low leverage. Launch-amortization and state-visibility have 10-100× leverage.**

---

## Tier 1 — IMPLEMENT (7 findings)

### R8-A3-F1: ALLOW-IN-GRAPH-ALL-STE-FUNCTIONS
**Tier:** IMPLEMENT
**Principle:** `torch.compile` (TorchDynamo) breaks the graph on every untagged `autograd.Function` subclass, losing fusion opportunities. Marking with `torch.compiler.allow_in_graph` treats the function as an opaque black-box that can live inside compiled graphs, enabling Inductor to fuse surrounding ops across quantization boundaries.

**Application:** `src/model_v7/components/ternary.py` — mark STE classes at lines ~181, ~251, ~321, ~589
**Pseudocode:**
```python
torch.compiler.allow_in_graph(TernaryQuantize)
torch.compiler.allow_in_graph(BoundaryAwareTernaryQuantize)
torch.compiler.allow_in_graph(CTCETernaryQuantize)
torch.compiler.allow_in_graph(BiSTETernaryQuantize)
# (LSQTernaryQuantize, DualRateTernaryQuantize, GASTETernaryQuantize already done)
```

**Expected gain:** 5-15% e2e throughput from eliminating graph breaks and enabling vertical fusion
**Dependency:** None
**Synergies:** Prerequisite for W1-B-F5 (fullgraph-disable-guard-pattern)
**Confidence:** VERY HIGH — 4 untagged classes currently cause guaranteed graph breaks

---

### R8-A3-F2: COMPILE-MODE-REDUCE-OVERHEAD
**Tier:** IMPLEMENT
**Principle:** Current mode `"default"` applies Inductor kernel fusion. Mode `"reduce-overhead"` additionally uses CUDAGraph Trees — captures entire forward+backward as a replayable CUDA graph, eliminating per-step CPU kernel launch overhead via single `graph.replay()` call instead of hundreds of kernel dispatches. On 6M models where kernel launch is proportionally large, measured gain: 2-10× reduction in CPU overhead.

**Application:** `src/model_v7/train.py:2448` — change compile mode + add `mark_step_begin` calls
**Pseudocode:**
```python
model = torch.compile(model, mode="reduce-overhead", dynamic=False)
# In training loop, before each forward pass:
torch.compiler.cudagraph_mark_step_begin()
```

**Expected gain:** 10-25% wall-clock reduction for 6M model
**Dependency:** None (standalone)
**Synergies:** Enables W2-A-F1 (registered buffer EMA), W2-A-F2 (C2 static marking)
**Caveat:** Requires no data-dependent control flow inside compiled model. Test for breakage; fallback to `"default"` if needed
**Confidence:** HIGH — PyTorch official feature with documented gains

---

### R8-A3-F3: COMPILE-PRINT-SUPPRESSION
**Tier:** IMPLEMENT
**Principle:** Every `print()` inside a compiled function causes a graph break. Current model has debug `print()` at train.py:3720-3723. Registering `print` as reorderable logging function eliminates break while preserving output.

**Application:** `src/model_v7/train.py` — before training loop starts
**Pseudocode:**
```python
import torch._dynamo.config as dynamo_cfg
dynamo_cfg.reorderable_logging_functions.add(print)
```

**Expected gain:** 2-8% e2e — eliminates 1-3 graph breaks in forward pass
**Dependency:** None
**Synergies:** None
**Confidence:** HIGH — documented PyTorch feature

---

### R8-A3-F4: COMPILE-DYNAMIC-SHAPES-FOR-SEQ-WARMUP
**Tier:** IMPLEMENT
**Principle:** Model uses `seq_warmup` which changes `input_ids` length during warmup. With `dynamic=False`, each new shape triggers full recompilation (30-120s each). With `dynamic=True`, single graph handles variable shapes symbolically, paying ~2% per-step overhead but eliminating catastrophic recompilation.

**Application:** `src/model_v7/train.py:2443-2448` — conditional dynamic flag
**Pseudocode:**
```python
if seq_warmup:
    model = torch.compile(model, mode="default", dynamic=True)
else:
    model = torch.compile(model, mode="default", dynamic=False)
```

**Expected gain:** Eliminates O(warmup_steps) × 30-120s recompilations = potentially 6000-24000s (1.7-6.7 hours) saved during warmup
**Dependency:** Requires `seq_warmup` to be active (check train.py:3680)
**Synergies:** After warmup, revert to `dynamic=False`
**Confidence:** HIGH — documented PyTorch optimization

---

### R8-A3-F5: STATEFUL-UPDATE-INPLACE-REGISTERED-BUFFER (EMA)
**Tier:** IMPLEMENT
**Principle:** EMA, ADMM, C2 updates currently execute as Python attribute mutations opaque to the graph. Converting to in-place ops on registered buffers makes them compiler-visible. Enables Inductor to fuse EMA update kernel with optimizer weight-write — both read/write same weight, memory accesses coalesce into single kernel pass.

**Application:** `src/model_v7/training/ema.py` — ExponentialMovingAverage.update()
**Pseudocode:**
```python
for buf, param in zip(self.shadow_buffers, model.parameters()):
    torch.lerp_(buf, param.detach(), 1.0 - self.decay)
# Buffers pre-registered via model.register_buffer("ema_w0", ...)
```

**Expected gain:** 3-8% throughput from fusing EMA update with optimizer step into one kernel
**Dependency:** EMA must use registered buffers (convert if not already)
**Synergies:** Prerequisite for W1-B-F1 (CUDA graph training replay) — EMA must be in-graph
**Confidence:** HIGH — straightforward in-place conversion

---

### R8-A3-F6: C2-STATE-AS-CONSTANT-WITHIN-STEP-GRAPH
**Tier:** IMPLEMENT
**Principle:** C2 state (committed_mask, sigma_g) is read during forward but NOT updated — updated AFTER optimizer step. Therefore within a single step, C2 state is mathematically constant. Mark with `torch.compiler.mark_static_address` so compiler hoists reads outside inner loop, eliminates redundant re-reads, treats as compile-time constant.

**Application:** `src/model_v7/components/commitment.py` — CommitmentTracker.__init__ and forward
**Pseudocode:**
```python
torch.compiler.mark_static_address(self.committed_mask)
torch.compiler.mark_static_address(self.sigma_g)
# Update AFTER optimizer step, with torch.compiler.disable() decorator
```

**Expected gain:** 2-5% throughput from eliminating memory barriers, enabling fusion of mask operations with gradient ops
**Dependency:** Requires C2 state to exist (already in v7 per architecture-brief)
**Synergies:** Prerequisite for F2 (reduce-overhead mode) — CUDAGraph requires static-address pointers
**Confidence:** HIGH — pure API usage, no structural changes

---

### R8-A3-F7: LAYER-LOCAL-LION-UPDATE-VIA-POST-ACCUMULATE-GRAD-HOOK
**Tier:** IMPLEMENT
**Principle:** Lion's update for layer L depends only on (m_L, g_L), NOT on gradients of layers L+1...N. Using `register_post_accumulate_grad_hook` fires optimizer step for each parameter immediately after its gradient is ready, eliminating intermediate gradient buffer materialization. At 6M/12 layers: saves ~3.5MB VRAM peak.

**Application:** `src/model_v7/train.py` — register hooks at model init, remove explicit `optimizer.step()`
**Pseudocode:**
```python
for name, param in model.named_parameters():
    param.register_post_accumulate_grad_hook(
        lambda p: lion_update_single(p, optimizer.state[p], lr, beta1, beta2)
    )
# Training loop: no explicit optimizer.step() — hooks fire automatically
```

**Expected gain:** 3-8% wall-clock + 3.5MB VRAM reduction (enables batch=8 if VRAM-bound → 2× convergence speedup)
**Dependency:** Lion optimizer required
**Synergies:** W1-B-F1 (CUDA graph) — optimizer interleaving reduces peak gradient storage
**Confidence:** MEDIUM-HIGH — requires careful hook design to avoid double-updates

---

## Tier 2 — EXPLORE (3 findings)

### R8-A3-F8: PYTHON-OVERHEAD-AMORTIZATION-VIA-N-STEP-GRAPH-UNROLLING
**Tier:** EXPLORE
**Principle:** At 6M scale, ~75-85% of 45.3ms step time is Python/kernel-launch overhead. CUDAGraph captures ONE step. Unrolling N steps into single compiled function makes inter-step state transitions (EMA, C2, ADMM) intra-graph operations, allowing CUDAGraph to capture all N steps as single replayable graph. For N=2: effective overhead per step drops to 19ms → 1.7× speedup.

**Application:** `src/model_v7/train.py` — replace inner training loop with N-step unrolled compiled function
**Pseudocode:**
```python
@torch.compile(mode="reduce-overhead", dynamic=False)
def two_step(model, W, opt_state, ema, x0, x1):
    loss0, W1, opt1 = model.step(W, opt_state, x0)
    ema1 = torch.lerp(ema, W1.detach(), 1 - ema_decay)  # now in-graph
    loss1, W2, opt2 = model.step(W1, opt1, x1)
    return loss0 + loss1, W2, opt2, torch.lerp(ema1, W2.detach(), 1 - ema_decay)
# Training loop calls two_step every 2 batches
```

**Expected gain:** 1.5-2.5× throughput at 6M scale by amortizing overhead across N steps
**Constraint:** Activation checkpointing for N steps (2× VRAM for activations at N=2) — tractable at 886MB baseline
**Dependency:** Markov property of SSM — training steps must be structurable as pure function
**Synergies:** Amplifies benefit of F2 (reduce-overhead) and F5 (EMA in-graph) — larger graphs = more replay benefit
**Validation needed:** Measure actual Python overhead fraction (decompose step time: kernels vs Python)
**Confidence:** MEDIUM — highest-potential but largest scope, requires architecture restructuring

---

### R8-A3-F9: LOSS-PROPORTIONAL-TOKEN-IMPORTANCE-SAMPLING
**Tier:** EXPLORE
**Principle:** Standard training uses uniform token sampling. Optimal importance sampling weights gradients by token importance, optimally estimated by loss. High-loss tokens have provably larger gradients (CE loss surrogate theorem). Sampling k=T//2 tokens with IS weights from previous step gives unbiased estimator with 50% variance reduction → statistically equivalent to full T tokens at half compute cost.

**Application:** `src/model_v7/train.py` — modify loss computation to subsample with IS weights
**Pseudocode:**
```python
prev_token_losses = loss_unreduced.detach()  # from previous step
p = (prev_token_losses ** 0.5 + 1e-6); p /= p.sum(-1, keepdim=True)
idx = torch.multinomial(p.view(B, T), num_samples=T//2)
w = 1.0 / (p.gather(1, idx) * (T//2))
loss = (w * loss_unreduced.gather(1, idx)).mean()
```

**Expected gain:** 10-20% reduction in steps-to-target-loss (only finding that attacks convergence, not just throughput) + 20-40% speedup per step from fewer tokens = 1.5-2.5× total training time reduction
**Safeguard:** Clip IS weights at `w_max=5.0` to prevent rare extreme-loss tokens from dominating variance
**Dependency:** Previous-step losses must be computed and retained (already done for logging)
**Synergies:** Works synergistically with gradient-based importance sampling but can be applied standalone
**Validation needed:** Verify loss-gradient-norm correlation empirically (literature: 0.6-0.9 for LMs); measure variance reduction on actual training curve
**Confidence:** MEDIUM — strong theoretical foundation but requires empirical validation on v7 architecture

---

### R8-A3-F10: SSM-COMMITTED-STATE-REPLAY-VIA-CONVOLUTION-KERNEL
**Tier:** EXPLORE
**Principle:** Committed SSM layers have frozen ternary weights A, B, C. The recurrence h_t = A·h_{t-1} + B·x_t has impulse response c_k = C·A^k·B that decays geometrically. For ternary matrices, A^k precomputable via integer matrix exponentiation. Precompute truncated kernel {c_k}, replace recurrent scan with parallel convolution F.conv1d. Eliminates sequential dependency.

**Application:** `src/model_v7/components/ssm.py` + `src/model_v7/training/autofreeze.py`
**Pseudocode:**
```python
# On commit (autofreeze.py):
A_q, B_q, C_q = quantize_ternary(A), quantize_ternary(B), quantize_ternary(C)
kernel = compute_ssm_convolution_kernel(A_q, B_q, C_q, T_trunc=128)
layer.register_buffer('conv_kernel', kernel)

# In ssm.py forward — committed fast path:
if layer.committed and hasattr(layer, 'conv_kernel'):
    y = F.conv1d(x, layer.conv_kernel, padding=T_trunc-1)[..., :T]
```

**Expected gain:** 5-15% e2e throughput, growing with commitment fraction (3-8× speedup on committed heads themselves)
**Mathematical guarantee:** Precomputed kernel is exact (not approximate) for sequences shorter than T_trunc
**Approximation:** Truncation at T_trunc introduces error ~||A^{T_trunc}|| · ... (typically <1% for T_trunc=128)
**Dependency:** Commitment tracker must exist and report stable masks; FLOPS of A^k precomputation amortized over many forward passes
**Synergies:** Pairs with F1 (gradient-chain-truncation) — F1 skips committed backward, F10 parallelizes committed forward
**Validation needed:** (a) Precompute timing vs. forward kernel, (b) Truncation error empirically <1%, (c) Conv kernel cache correctness
**Confidence:** MEDIUM-HIGH — mathematically sound but requires new kernel implementation

---

## Tier 3 — SKIP (2 findings)

### R8-A3-F11: ADMM-STATE-ELIMINATION-VIA-CLOSED-FORM-ABSORPTION
**Status:** CONDITIONAL SKIP
**Reason:** Gate G3 FAIL — requires ADMM to be active in v7. Architecture-brief lists "ADMM dual variable accumulation" in Already Done (v5→v7 carry-over), but this must be verified in `src/model_v7/` code. If ADMM is not present, this finding does not apply.

**Verification step:** Search `src/model_v7/` for `admm`, `dual_var`, `u_accumulate` — if not found, SKIP. If found, this becomes a Tier 2 EXPLORE.

**Principle (if applicable):** Instead of maintaining ADMM dual variable u as separate state and updating it via `u += ρ·(W_q - W_shadow)` (Python mutation), absorb regularization into gradient: `grad_effective = grad + ρ·(W_shadow - W_q)`. Moves ADMM computation inside compiler-visible graph, enabling fusion with grad_weight.

---

### R8-A3-F12: COMPILER-OPAQUE-STATE-AS-SYMBOLIC-CONSTANT-VIA-SHAPE-SPECIALIZATION
**Status:** CONDITIONAL SKIP (high complexity, marginal gain with F6 present)
**Reason:** Gate G1 BORDERLINE — gain comes from content-specialization of committed_mask, but F6 (mark_static_address) already provides address-stability. Additional content-specialization via `mark_static` + dynamo cache invalidation on mask update is high complexity for incremental gain (5-10% beyond F6).

**Recommendation:** Implement F6 first. If profiling shows gradient operations on committed layers still bottleneck, revisit F12 as advanced optimization.

---

## Implementation Order

**Phase 1 (Week 1) — Compilation Foundation (Low Risk, High Reward):**
1. **F1: ALLOW-IN-GRAPH-ALL-STE-FUNCTIONS** — 5 lines, zero risk, 5-15% gain
2. **F3: COMPILE-PRINT-SUPPRESSION** — 2 lines, zero risk, 2-8% gain
3. **F4: COMPILE-DYNAMIC-SHAPES-FOR-SEQ-WARMUP** — 5 lines, conditional, saves 1.7+ hours per warmup

**Phase 2 (Week 1-2) — State Visibility & CUDAGraph (Medium Risk, High Reward):**
5. **F2: COMPILE-MODE-REDUCE-OVERHEAD** — 2 lines + loop change, medium risk (test for breakage), 10-25% gain
6. **F5: STATEFUL-UPDATE-INPLACE-REGISTERED-BUFFER (EMA)** — convert EMA to registered buffer ops, enables CUDAGraph payoff
7. **F6: C2-STATE-AS-CONSTANT-WITHIN-STEP-GRAPH** — 2 lines, prerequisite for reduce-overhead, 2-5% gain

**Phase 3 (Week 2-3) — Optimizer Interleaving & Advanced (Medium Complexity, Targeted Gains):**
8. **F7: LAYER-LOCAL-LION-UPDATE-VIA-HOOKS** — medium complexity, 3-8% + VRAM savings
9. **F10: SSM-COMMITTED-STATE-REPLAY** — high complexity, requires new kernel, 5-15% but growing with commitment

**Phase 4 (Week 3+) — Exploration (Large Scope):**
10. **F8: PYTHON-OVERHEAD-AMORTIZATION-VIA-N-STEP-UNROLLING** — EXPLORE-tier (requires measurement, potential 1.5-2.5×)
11. **F9: LOSS-PROPORTIONAL-TOKEN-IMPORTANCE-SAMPLING** — EXPLORE-tier (requires validation, potential 1.5-2.5×)

---

## Synergy Matrix

| Finding | Synergizes With | Effect |
|---------|-----------------|--------|
| F1 (STE in-graph) | F5 (reduce-overhead) | Removes graph breaks → larger CUDA graph |
| F2 (reduce-overhead) | F5 (EMA in-graph), F6 (C2 static) | Requires in-graph state for correctness |
| F5 (EMA registered buffer) | F2 (CUDAGraph), F7 (hooks) | EMA update now fuse-able with optimizer |
| F6 (C2 static address) | F2 (reduce-overhead) | Prerequisite for CUDAGraph pointer stability |
| F7 (Lion hooks) | F2 (CUDAGraph), F5 (EMA) | Interleaved optimizer reduces peak gradient storage |
| F8 (N-step unrolling) | F2 (reduce-overhead), all state-visibility finds | Amplifies overhead amortization benefit |
| F9 (token importance) | None (orthogonal) | Standalone, attacks convergence not just throughput |
| F10 (SSM committed conv) | F8 (N-step unrolling) | Committed-fast-path enabler for batched conv |

---

## Risk Mitigation

**High-risk findings (require testing before production):**
- **F2 (reduce-overhead):** Test with `torch.compile(..., fullgraph=False)` first; enables on opt-in basis until validated
- **F8 (N-step unrolling):** Requires Markov property validation; measure step-time decomposition to confirm overhead fraction
- **F10 (SSM convolution):** Implement with A_log-based decay check; ensure truncation error <1% on empirical A values

**Medium-complexity findings (require new code paths):**
- **F5 (registered buffer EMA):** Refactor ExponentialMovingAverage.update() with buffers instead of attributes
- **F7 (Lion hooks):** Careful hook design to avoid double-updates; test against baseline optimizer.step()
- **F10 (convolution kernel):** Implement integer-matrix-exponentiation for A^k; cache and versioning on mask change

---

## Measurement Plan

**For each IMPLEMENT finding:**
1. **Baseline:** Run smoke test (50 steps, batch=4) before change. Record: tok/s, VRAM peak, loss@50.
2. **After change:** Run same smoke test. Record same metrics.
3. **Delta:** Compute Δtok/s, ΔVRAM, Δloss. Gate: Δtok/s > 0 (no regression), ΔVRAM not worse.
4. **Log:** Document activation (debug log) to confirm technique is active.

**For each EXPLORE finding:**
1. Implement on separate feature branch.
2. Run 200-500 step training convergence test (not just smoke).
3. Measure: steps-to-target-loss (F9), e2e throughput (F8, F10), profile overhead breakdown (F8).
4. Decide IMPLEMENT vs SKIP based on empirical results.

---

## References

- PyTorch torch.compile documentation: https://pytorch.org/docs/stable/user_guide/torch_compiler/
- CUDAGraph Trees: https://pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html
- FX graph pass patterns: https://pytorch.org/docs/stable/fx.html
- SSM impulse response & convolution theory: Referenced from W2-AgB mathematical derivations
