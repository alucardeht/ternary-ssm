# 06 — Validation: Training Stability (Ternary + SSM + Routing)

## VERDICT: HIGH RISK BUT FEASIBLE — Progressive Training Essential

No one has combined all three (ternary + SSM + routing). Each works individually. The combination introduces triple gradient approximation error. Manageable with progressive training and careful monitoring.

---

## 1. The Triple-STE Problem

Our architecture has cascaded discrete operations:
1. **Ternary weight quantization** → STE for weight gradients
2. **Selective scan** → input-dependent state transitions
3. **Discrete routing decisions** → STE for router top-k

**No direct research on double/triple-STE found.** This is novel territory.

### Gradient Quality Degradation Chain

```
Ternary Weights ──STE──> Noisy Gradients
        ↓
    SSM State ──Recurrence──> Error Accumulation O(T·λ_max)
        ↓
Router Decisions ──Double STE──> Compound Gradient Approximation
        ↓
   Load Imbalance ──Expert Collapse──> Training Divergence
```

### Why SSMs Help (Partially)

SSMs naturally avoid vanishing gradients (linear state updates, no nonlinear activation in recurrence). But ternary STE reintroduces gradient approximation error at every projection, partially breaking this advantage.

---

## 2. Compound Error in Recurrent Paths

### Error Bound

```
Error grows as O(T · λ_max)
T = sequence length, λ_max = largest eigenvalue of A
```

Quantization noise at timestep t propagates to t+1, t+2... with multiplicative amplification.

### Critical: A Matrix Cannot Be Ternary

Diagonal A with ternary values → eigenvalues can only be {-1, 0, +1}. This is far too restrictive for stable state evolution. **Confirmed again: A stays FP16 or INT8 minimum.**

### Selective Scan Amplifies Errors

Unlike time-invariant SSMs, Mamba's input-dependent B, C, Δ mean quantization errors vary per token. Error propagation becomes **non-stationary** — harder to bound theoretically.

### Sequence Length Is Critical

Longer sequences = more error accumulation. Must test: 512 → 2K → 8K → 32K to find the practical limit.

---

## 3. Router Stability Concerns

### Router Collapse Risk

Early/late layers may funnel >70% tokens to 1-2 paths. Known MoE problem.

### Mitigation (Day 1 — Non-Negotiable)

- **Load balancing loss** — encourage uniform utilization
- **Z-loss** — bound router logits (prevent softmax instability)
- **Entropy regularization** — prevent overconfident routing
- **Separate LR for router** — often needs higher LR than model weights
- **Monitor every 10 steps** — early detection of collapse

### Double-STE Interaction (Untested)

Router computes scores from ternary-layer features (first STE), then makes discrete top-k decision (second STE). The router's gradient signal is already weak (only k/N tokens contribute). Noisy features from ternary layers may make it worse.

---

## 4. Training Recipe (Consensus from Literature)

### Progressive Approach (Recommended)

```
Phase 1 (0-10%):   FP16 SSM + routing baseline
                    → Validate recurrence and routing work
                    → Establish performance ceiling

Phase 2 (10-30%):  QAT for projections → ternary (STE)
                    → Keep A at FP16
                    → Monitor gradient norms carefully
                    → Expect larger LR than FP (2-5×)

Phase 3 (30-50%):  Enable dynamic routing
                    → Freeze/slow ternary weight LR
                    → Activate router with load balancing + z-loss
                    → Higher LR for router parameters

Phase 4 (50-100%): Full training, all components active
                    → Fine-tune, monitor, stabilize
```

### Alternative: Joint Training From Scratch

BitNet shows ternary from-scratch works. But adding SSM + routing simultaneously is riskier. Progressive is safer for a novel combination.

### Key Insight from QAT Research

"The best accuracy is achieved when a QAT phase follows a full-precision training phase." — NVIDIA QAT Best Practices

---

## 5. Normalization: RMSNorm Is Mandatory

### Where to Apply

- **Before each ternary SSM block** (Pre-LN style)
- **After SSM output** (before routing decision)
- **Extra RMSNorm positions** for ternary models (arXiv 2505.08823)
- Inside SSM block after state update (if instability observed)

### Why RMSNorm Over LayerNorm

- Preferred for quantized models (empirical consensus 2024-2025)
- Jamba uses RMSNorm specifically for Mamba layer stability at scale
- Pre-LN (norm before sublayer) more stable than Post-LN

---

## 6. Gradient Management

### Adaptive Per-Layer Clipping

- Global clipping is suboptimal — different layers have different gradient distributions
- **AdaGC**: Adjusts per-layer based on gradient statistics
- Ternary layers: expect higher baseline norms → separate thresholds
- Router: separate clipping threshold (different gradient scale)

### Practical Settings

```
Ternary projection layers: clip at 95th percentile of recent norms
A matrix (FP16): standard clipping
Router: separate, typically less aggressive clipping
Global fallback: clip if any layer exceeds 10× recent average
```

---

## 7. Monitoring: Early Warning System

### Every 10 Steps
- Gradient norms per layer
- Router entropy (target: close to log(num_choices))
- State magnitude ||h_t|| (should be bounded, not growing with T)

### Every 100 Steps
- Weight distribution histograms {-1, 0, +1} ratios
- Eigenvalue tracking for A matrices (max|λ| should stay <1)
- Δ parameter statistics (discretization step)

### Red Flags (Stop and Investigate)

| Signal | Meaning |
|--------|---------|
| Router entropy < 0.5 × log(K) for >100 steps | Router collapse |
| State magnitude grows with sequence length | Quantization error accumulation |
| Gradient norm > 10× recent average | Approaching divergence |
| Weight distribution >80% zeros | Quantization collapse |
| Loss spikes every few hundred steps | Persistent instability |

### Automated Response

```
On gradient norm spike:   → Reduce LR by 50%, increase clipping
On router collapse:       → Increase load balancing λ, add entropy bonus
On state magnitude growth: → Check A eigenvalues, increase A precision
On loss spike:            → Immediate checkpoint + diagnostic dump
```

---

## 8. Fallback Strategies

If full ternary + SSM + routing is unstable:

1. **Hybrid precision:** Ternary projections but INT8 for A (instead of ternary everywhere)
2. **Simplify routing:** Static MoE instead of dynamic MoD
3. **Increase precision:** 2-bit instead of ternary (adds zero point, more expressiveness)
4. **Stability anchors:** Keep every Nth layer in FP16

---

## 9. Risk Assessment Summary

| Component | Risk | Mitigation |
|-----------|------|------------|
| Ternary projections (90% params) | LOW — proven by BitNet/Bi-Mamba | QAT from scratch |
| A matrix FP16 | LOW — standard approach | Keep high precision |
| B, C, Δ ternary | HIGH — untested | Start FP16, incrementally quantize |
| Dynamic routing | MEDIUM — MoD proven but not with ternary | Load balancing + z-loss |
| Double-STE gradient | HIGH — unexplored | Progressive training, careful monitoring |
| Long sequence stability | MEDIUM — error accumulates | Test incrementally (512→32K) |

### Overall: Proceed With Caution

- **Each component works individually** — proven in literature
- **The combination is novel** — no guarantees
- **Progressive training is the key** — don't enable everything at once
- **Budget 2-3× more experimentation time** than standard model development

---

*Sources: BitNet (JMLR 2024), Mamba (arXiv 2312.00752), MoD (arXiv 2404.02258), Jamba (arXiv 2403.19887), Q-S5 (arXiv 2406.09477), Extra RMSNorm (arXiv 2505.08823), AdaGC (arXiv 2502.11034), Router Z-Loss, MoE Load Balancing (HuggingFace)*

*Last updated: February 2026*
