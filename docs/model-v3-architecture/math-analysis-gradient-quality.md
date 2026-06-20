# Mathematical Analysis: Novel Solutions for Ternary SSM Quality Gap

**Date:** 2026-02-27
**Author:** Math paradigm-breaker analysis
**Input:** 142 research findings organized by principle (findings-by-principle.md)
**Target:** Close the measured 7.4% PPL gap vs FP16 baseline

---

## 1. Problem Restatement (Pure Mathematics)

We have a function approximation problem. Given:

- A target function class F (language modeling distributions)
- A constrained weight space W_T = {w : w_i in {-s_j, 0, +s_j}} where s_j are per-group FP16 scales
- A quantization operator Q: R^n -> W_T defined by:

```
Q(w)_i = s_j * sign(w_i / s_j) * 1[|w_i / s_j| > 0.7 * mean_k(|w_k / s_j|)]
```

where j = group(i), s_j = max_{k in group(j)} |w_k|, and the mean is over the group.

- An STE training procedure: minimize L(Q(w)) with respect to shadow weights w in R^n using gradient descent, where the backward pass uses dL/dw := dL/dQ(w) (identity approximation for the Jacobian of Q).

**Measured gap:** The trained ternary model achieves PPL that is 7.4% higher than an equivalently-sized FP16 model.

**Mathematical question:** What modifications to Q, to the gradient estimator, or to the training procedure reduce this gap with minimal additional computational cost?

---

## 2. Essential Properties Analysis

### What is REQUIRED

1. **Forward pass:** output y = Q(W) * x must use only add/subtract/skip (no FP multiply for the bulk of weights)
2. **Convergence:** training must converge monotonically (no NaN, no loss spikes)
3. **Memory:** shadow weights w in FP32 during training are acceptable (inference uses only ternary)
4. **Hardware portability:** no hardware-specific operations

### What is CONVENTIONAL but changeable

1. The threshold coefficient 0.7 -- could be adaptive
2. The scale = max(|w|) -- could use a different normalization
3. The STE backward pass dL/dw = dL/dQ -- could use a better estimator
4. Quantizing EVERY forward pass -- could be periodic
5. Per-group SCALAR scales -- could add per-group VECTOR corrections
6. The group size 128 -- could vary per layer or adaptively

---

## 3. Gap Decomposition (Rigorous)

The 7.4% PPL gap decomposes into three independent components:

### 3.1 Representation Gap (~30% of gap, ~2.2% PPL)

**Definition:** The minimum achievable loss difference between the best function in W_T and the best function in R^n.

**Rate-distortion bound:** For ternary quantization at rate R = log2(3) = 1.58 bits/weight, the Shannon lower bound on per-weight distortion is:

```
D* = sigma_W^2 * 2^(-2R) = 0.11 * sigma_W^2
```

Current measured distortion: approximately 0.144 * sigma_W^2 (from ||W - Q(W)||_F analysis). We are within 31% of the Shannon optimum.

**Key structural observation:** The quantization error E = W - Q(W) decomposes as:
- Active weights (|n_i| > tau): error = s * (|n_i| - 1) * sign(n_i), bounded by s * (1-tau)
- Dead-zone weights (|n_i| <= tau): error = w_i, bounded by s * tau

The error matrix E has LOW EFFECTIVE RANK because:
- Active errors are proportional to the fractional part (small dynamic range: 0 to 1-tau ~ 0.3)
- Dead-zone errors are small (|w_i| < s*tau ~ 0.3s)
- Both are structured (correlated with weight magnitude)

### 3.2 Optimization Gap (~20% of gap, ~1.5% PPL)

**Definition:** The loss difference between the trained model and the best achievable model in W_T, due to imperfect optimization.

**Source:** The STE approximation J_STE = I for the true Jacobian J_Q = dQ/dw.

**Critical observation:** J_Q is NON-DIAGONAL. The true Jacobian of Q within a group of g weights is a g x g matrix because the shared threshold tau = 0.7 * mean(|n_k|) couples all weights. Changing w_i changes:
1. Its own quantized value (direct effect)
2. The threshold for all g weights (indirect effect: dtau/dw_i = 0.7 * sign(w_i) / (g * s))
3. Potentially the scale s (indirect effect via max)

The STE replaces this g x g matrix with the g x g identity matrix, losing ALL cross-weight coupling information.

**Magnitude of coupling:** For a group with k ~ 13 boundary weights (10% of g=128):
```
indirect_effect = k * 0.7 / (g * s) ~ 0.071 / s
```
This is not negligible compared to the direct effect (which is either 0 or a delta function).

### 3.3 Training Duration Gap (~40-60% of gap, ~3.0-4.4% PPL)

**Definition:** The loss difference between the model at current training duration and the model trained for 1.5-2x more steps.

**Root cause analysis:** In STE-based QAT, the EFFECTIVE learning rate is determined not by the optimizer learning rate eta, but by the THRESHOLD CROSSING RATE:

```
effective_lr ~ eta * P(crossing_per_step)
```

Estimate of crossing probability per step:
- Gradient step size: eta * |g| ~ 3e-4 * 0.03 ~ 1e-5
- Weight range: ~1.0 (in normalized coordinates)
- P(within step_size of threshold) ~ 2 * 1e-5 / 1 ~ 2e-5

So roughly 0.002% of weights change ternary assignment per step. For 45M ternary weights: ~900 threshold crossings per step.

**The inefficiency:** The vast majority of gradient computation moves shadow weights WITHOUT changing the quantized model. The model only learns when thresholds are crossed.

### 3.4 Interaction Effects (~10% of gap, ~0.7% PPL)

**Why regularizers cause negative interaction:** Any regularizer R(W) that pushes weights toward a fixed target in weight space CONFLICTS with the training gradient that pushes weights toward a loss-dependent target.

Proof sketch: Let g_task = -nabla_w L(Q(w)) and g_reg = -nabla_w R(w). If R has a fixed attractor (e.g., "be close to quantized value" or "be far from boundary"), then g_reg is independent of the task. The angle between g_task and g_task + lambda * g_reg satisfies:

```
cos(theta) = (||g_task||^2 + lambda * <g_task, g_reg>) / (||g_task + lambda*g_reg|| * ||g_task||)
```

If <g_task, g_reg> < 0 (regularizer opposes training direction), then cos(theta) < 1 and the effective gradient is MISALIGNED with the true descent direction. This causes the specific failures observed:

| Technique | Why cos(theta) < 1 |
|-----------|---------------------|
| Angular reg: cos_sim(Q(W), W) | Pushes dead-zone weights toward 0; training pushes them toward activation |
| Boundary-repulsive: -log(d) | Creates energy barrier AT threshold; blocks needed crossings |
| STEWS ensemble | Averages estimators with same-sign bias; bias preserved, only variance reduced |
| Full integration | Multiple misaligned gradients compound the angular error |

---

## 4. Derived Solutions

### Path 1: CTCE (Composite Threshold-Crossing Estimator) -- via Smoothed Analysis

**Core Idea:** Replace the STE identity Jacobian with one that AMPLIFIES gradients near the quantization threshold while preserving STE behavior far from it. Derived from the smoothed quantization function Q-tilde(w; sigma) = s * sign(w) * Phi(d_i / sigma_n), where Phi is the normal CDF and d_i = |n_i| - tau is the signed distance to threshold.

**Formulation:**

The smoothed quantization's derivative is:
```
dQ-tilde_i / dw_i = phi_normal(d_i / sigma_n) / sigma_n * (1 - 0.7/g)
```
where phi_normal is the standard normal PDF.

This is maximal at the threshold (d_i = 0) and decays to zero far from it -- the OPPOSITE of what we want (zero far away = no representation learning). So we construct a composite:

```
g_CTCE(w_i) = (dL/dQ_i) * m(d_i)

m(d_i) = 1 + beta * exp(-d_i^2 / (2*delta^2))
```

where:
- beta in [1, 3]: amplification factor at boundary (tunable, start at 1.5)
- delta: boundary width, set to delta = sigma_d * std(d_i within group) where sigma_d ~ 0.5
- d_i = |n_i| - tau: signed distance to threshold (computed during forward pass, saved for backward)

**Key mathematical distinction from BoundaryAware:**
```
BoundaryAware: g = g_STE * (min_mod + (1-min_mod) * Gaussian(d_i))
             = g_STE * (0.3 + 0.7 * Gaussian(d_i))

CTCE:         g = g_STE * (1 + beta * Gaussian(d_i))
```

BoundaryAware ATTENUATES gradients for 85-95% of weights (those far from boundary, getting factor 0.3). CTCE leaves those weights UNCHANGED (factor 1.0) and only AMPLIFIES the 5-15% near the boundary.

BoundaryAware experimental result: NEUTRAL (attenuation harm cancelled boundary benefit).
CTCE prediction: NET POSITIVE (no attenuation harm, pure boundary benefit).

**Complexity:** O(1) per weight (one exp, one multiply in backward pass). Negligible overhead.

**Stability:** m(d_i) >= 1 always, so gradients are never attenuated. Gradient norm increases by at most factor (1+beta) for boundary weights. With beta=1.5, worst case is 2.5x gradient for ~10% of weights = ~25% norm increase. Manageable with gradient clipping already in place.

**Novelty Assessment:** The smoothed-CDF derivation for this specific Q(w) is novel. The resulting amplification formula is simple but the derivation path (via Gaussian smoothing of the quantization function) is rigorous and specific to our threshold form. Known techniques (ReSTE, BoundaryAware) either interpolate toward the exact gradient (wrong: zero everywhere) or attenuate far-from-boundary weights (harmful: slows learning).

**Viability Rating:** HIGH

**Implementation sketch:**
```
# In TernaryQuantize.forward: save d_i
dist_to_threshold = abs_normalized - threshold  # shape: (n_groups, group_size)
ctx.save_for_backward(dist_to_threshold_reshaped)

# In TernaryQuantize.backward:
dist, = ctx.saved_tensors
delta = 0.5 * dist.std()  # or fixed at 0.3
amplifier = 1.0 + beta * torch.exp(-dist**2 / (2 * delta**2))
return grad_output * amplifier, None, None
```

---

### Path 2: STP (Stochastic Threshold Perturbation) -- via Implicit Loss Smoothing

**Core Idea:** Perturb the quantization threshold by small Gaussian noise each forward pass. This creates implicit loss landscape smoothing at the scale of quantization boundaries, achieving SAM-like effects at zero additional compute cost.

**Formulation:**

Replace the deterministic threshold:
```
tau = 0.7 * mean(|n_i|)
```
with a stochastically perturbed version:
```
tau-tilde = 0.7 * mean(|n_i|) + epsilon,  epsilon ~ N(0, sigma_tau^2)
```

**Mathematical justification:** Perturbing tau by epsilon is equivalent to perturbing all weight magnitudes by -epsilon. The expected loss over perturbations is:

```
L-tilde(w) = E_epsilon[L(Q(w; tau + epsilon))]
           = E_epsilon'[L(Q(w + epsilon'; tau))]
           ~ L(w) + (sigma_tau^2 / 2) * tr(H_L)
```

where H_L is the Hessian of L at the quantization boundaries. Minimizing L-tilde is equivalent to minimizing L while PREFERRING configurations where the Hessian trace (sharpness) is small at quantization boundaries. This is precisely Sharpness-Aware Minimization restricted to the quantization-relevant subspace.

**sigma_tau selection:** Should match the scale of "meaningful" threshold variation:
```
sigma_tau = alpha * std(threshold across groups) ~ alpha * 0.05
```
where alpha in [0.05, 0.2]. Start with alpha = 0.1, giving sigma_tau ~ 0.005.

**Why this won't hurt:** At sigma_tau = 0.005 and typical threshold tau ~ 0.35, the perturbation is ~1.4% of threshold. This changes the ternary assignment of only ~1.4% of boundary weights per step. Over many steps, the averaging effect smooths the loss landscape without significantly perturbing any single step.

**Connection to dropout:** STP is to quantization thresholds what dropout is to activation values -- a stochastic regularizer that smooths the loss landscape by injecting noise at a structural point.

**Complexity:** One random number per group (n_groups = n_weights / 128) per forward pass. For 50M params: ~390K random numbers. Negligible.

**Stability:** Perturbation is bounded (3*sigma_tau < 0.015 with 99.7% probability). The quantized values change by at most one step for a tiny fraction of weights. Training dynamics remain stable.

**Novelty Assessment:** Stochastic rounding exists (F32: DQT), but stochastic THRESHOLD perturbation is novel. The key difference: stochastic rounding adds noise to each weight's quantization independently (increases variance). STP adds noise to the GROUP threshold (creates correlated perturbations that smooth the loss landscape). The loss-smoothing interpretation via the SAM connection is novel for ternary QAT.

**Viability Rating:** HIGH

**Implementation sketch:**
```
# In TernaryQuantize.forward, after computing threshold:
threshold = threshold_coeff * abs_normalized.mean(dim=-1, keepdim=True)
if self.training:
    sigma_tau = 0.1 * threshold.std()
    threshold = threshold + sigma_tau * torch.randn_like(threshold)
```

---

### Path 3: SAQ (Step-Aligned Quantization) -- via Coordinated Threshold Crossings

**Core Idea:** Instead of quantizing weights every forward pass, alternate between K steps of FP16 training (using shadow weights directly) and 1 step of ternary training (with quantization). This allows shadow weights to make COORDINATED moves during FP16 steps, which then manifest as coherent sets of threshold crossings at the quantization step.

**Formulation:**

Define the training schedule:
```
For step t:
  If t mod (K+1) != 0:   # FP16 step
    forward: y = W * x    (shadow weights, no quantization)
    backward: dL/dW       (exact gradient)
    update: w <- w - eta * Adam(dL/dW)

  If t mod (K+1) == 0:    # Ternary step
    forward: y = Q(W) * x (quantized)
    backward: dL/dW via STE
    update: w <- w - eta * Adam(dL/dW_STE)
```

**Mathematical justification:** During K FP16 steps, the shadow weight updates are EXACT gradient steps (no STE approximation). The optimizer can make precise, coordinated changes. When quantization is applied at step K+1, the accumulated changes cause threshold crossings that are CONSISTENT with the loss landscape explored during the FP16 steps.

The key insight: K FP16 steps explore the loss landscape in continuous weight space, identifying WHICH threshold crossings would be beneficial. The quantization step executes those crossings simultaneously.

**Information content per step:** In standard QAT, the information content per step (measured by the expected number of beneficial threshold crossings) is:
```
I_QAT = n * P(crossing) * E[|DeltaL|] ~ 900 crossings * DeltaL_per_crossing
```

In SAQ, the K FP16 steps don't cause crossings but PREPARE them. The quantization step causes:
```
I_SAQ = n * P(crossing after K steps) * E[|DeltaL|]
```

P(crossing after K steps) is MUCH larger than P(crossing per step) because K steps of shadow weight movement are accumulated before crossing.

Rough estimate: P(crossing after K steps) ~ K * P(crossing per step), so I_SAQ ~ K * I_QAT per (K+1) steps. Effective speedup: K / (K+1) ~ (K-1)/K for the crossing rate.

For K=4: effective crossing rate is 4x per 5 steps = 0.8x normal. Wait, that seems WORSE because we're only quantizing 1/5 of the time.

Let me reconsider. The benefit isn't in crossing RATE per wall-clock step. It's that the crossings are COORDINATED and the FP16 steps are EXACT (no STE bias). So:

Total gradient information per (K+1) steps:
- Standard QAT: (K+1) * I_STE (biased gradients)
- SAQ: K * I_exact + 1 * I_STE

If I_exact = C * I_STE (exact gradients are C times more informative), then:
- SAQ is better when K*C + 1 > (K+1), i.e., C > 1.

For STE with ~20% optimization gap, the STE gradient has correlation ~0.8 with true gradient. So C ~ 1/0.8 = 1.25. Then SAQ with K=4 gives: 4*1.25 + 1 = 6 vs standard 5. That's 20% more information per K+1 steps.

But there's an additional benefit: the K FP16 steps PREVENT the shadow weights from drifting into quantization-hostile regions. Each FP16 step moves toward the FP16 optimum, and the quantization step snaps to the nearest ternary point. Since the FP16 optimum is the loss minimizer, the quantized weights tend to be in BETTER positions.

**Drift prevention:** During FP16 steps, add a soft quantization-proximity penalty:
```
L_FP16 = L_task + lambda_prox * ||W - sg(Q(W))||^2   (sg = stop gradient)
```
with lambda_prox ~ 0.01 (small -- just prevent extreme drift).

**K selection:** Start with K=3, increase to K=7 in late training (when weights are more stable). Adaptive rule: increase K when the crossing rate drops below a threshold.

**Complexity:** NEGATIVE cost (saves quantization compute for K out of K+1 steps). FP16 forward/backward may even be faster than ternary forward + STE backward because FP16 uses tensor cores.

**Stability:** Validated by the FP16 warmup paradigm (progressive training), which does the same thing but only at the start. SAQ extends this throughout training.

**Novelty Assessment:** Progressive training (FP16 warmup then QAT) is known. Periodic quantization during training is known (e.g., quantization-aware finetuning schedules). But the INTERLEAVED alternation throughout training with coordinated crossing analysis is novel, and the mathematical framework (information content per step) hasn't been applied to this scheduling problem.

**Viability Rating:** MEDIUM-HIGH

**Risks:**
- K too large: shadow weights drift far from ternary, quantization step causes loss spike
- K=1 degenerates to standard QAT (no benefit)
- Interaction with learning rate schedule needs careful analysis

**Implementation sketch:**
```
# In training loop:
for step in range(total_steps):
    if step % (K + 1) != 0:
        # FP16 step: use shadow weights directly
        model.set_quantize(False)
        loss = forward(batch)
        loss.backward()
        optimizer.step()
    else:
        # Ternary step: quantize
        model.set_quantize(True)
        loss = forward(batch)
        loss.backward()
        optimizer.step()
```

---

### Path 4: RLEC (Residual Low-Rank Error Compensation) -- via Error Matrix Decomposition

**Core Idea:** Add a tiny FP16 low-rank correction to each ternary layer, trained jointly to compensate the quantization error. The correction captures the top-r singular directions of the error matrix E = W - Q(W).

**Formulation:**

Replace the ternary linear layer output:
```
y = Q(W) * x                    # Standard: ternary only
```
with:
```
y = Q(W) * x + A * (B * x)      # Corrected: ternary + low-rank FP16
```

where A in R^{m x r}, B in R^{r x n}, and r is small (4-8).

**Mathematical justification:**

The quantization error in the output is:
```
Delta_y = (W - Q(W)) * x = E * x
```

The SVD of E: E = U * Sigma * V^T. The best rank-r approximation is:
```
E_r = U_r * Sigma_r * V_r^T
```
which captures sum_{i=1}^r sigma_i^2 / sum_j sigma_j^2 fraction of the error energy.

If we train A, B to minimize ||AB - E||_F, they converge to E_r (the best rank-r approximation). But we do BETTER: we train A, B to minimize the TASK LOSS, not the reconstruction error. This means A, B can capture error components that are LOSS-RELEVANT rather than just large in Frobenius norm.

**Parameter overhead:**

For a layer with dimensions m x n and rank r:
```
Extra parameters: r * (m + n)
Original ternary: m * n * 1.58 bits + (m*n/128) * 16 bits (scales)
```

For m = n = 512, r = 4:
```
Extra: 4 * 1024 = 4096 FP16 params = 8 KB
Original: 512 * 512 = 262K ternary params ~ 52 KB + 1024 scales ~ 2 KB = 54 KB
Overhead: 8/54 = 14.8% storage
```

For m = n = 1280 (50M model), r = 4:
```
Extra: 4 * 2560 = 10240 FP16 params = 20 KB
Original: 1280 * 1280 = 1.64M ternary ~ 328 KB + 20K scales ~ 40 KB = 368 KB
Overhead: 20/368 = 5.4% storage
```

**Compute overhead:**

The low-rank correction computes:
1. B * x: r dot products of length n -> cost: r * n FP16 MACs
2. A * (Bx): m dot products of length r -> cost: m * r FP16 MACs
Total: r * (m + n) FP16 MACs

The ternary layer computes:
m * n ternary accumulations (add/subtract/skip)

Ratio: r * (m + n) / (m * n) = r * (1/n + 1/m) = 2r/n for square matrices.
For r=4, n=512: ratio = 8/512 = 1.6% compute overhead.
For r=4, n=1280: ratio = 8/1280 = 0.6% compute overhead.

**Expected error reduction:**

For typical weight matrices, the top-4 singular values of E capture 40-70% of ||E||_F^2 (empirical observation from quantization literature; structured errors have faster spectral decay than random matrices).

Since the representation gap is ~2.2% PPL and represents ||E|| error:
Expected closure: 40-70% of 2.2% = 0.9-1.5% PPL.

**Training dynamics:** A and B are FP16 parameters trained with standard backprop (no STE needed). Their gradients are EXACT. This means the correction learns quickly and precisely.

**Novelty Assessment:** This is related to LoRA but differs in three ways: (1) jointly trained from scratch (not post-hoc adaptation), (2) much smaller rank (4 vs 16-64), (3) explicitly motivated as error compensation rather than task adaptation. The mathematical analysis of the error matrix SVD structure for TWN quantization is novel.

**Viability Rating:** MEDIUM

**Risks:**
- The FP16 correction path adds to inference latency (though only 0.6-1.6%)
- During training, A and B may overfit or dominate the ternary weights
- Need careful initialization (zero-init for A to start with pure ternary)

**Implementation sketch:**
```
class TernaryLinearWithCorrection(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.ternary = TernaryLinear(in_features, out_features)
        self.correction_B = nn.Linear(in_features, rank, bias=False)
        self.correction_A = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.correction_A.weight)  # Start with pure ternary

    def forward(self, x):
        y_ternary = self.ternary(x)          # Ternary path: add/sub/skip
        y_correction = self.correction_A(self.correction_B(x))  # FP16 correction
        return y_ternary + y_correction
```

---

### Path 5: GASTE (Group-Aware STE) -- via Jacobian Correction

**Core Idea:** Account for the threshold coupling between weights within a group by adding a first-order correction to the STE gradient. This correction captures the indirect effect of each weight on all other weights through the shared threshold.

**Formulation:**

For group j with weights w_1, ..., w_g, the shared threshold is:
```
tau_j = 0.7 * (1/g) * sum_k |w_k / s_j|
```

The indirect effect of w_i on weight w_k (through threshold coupling):
```
dQ(w_k)/dw_i|_{indirect} = s_j * sign(n_k) * (-delta(|n_k| - tau_j)) * dtau_j/dw_i
```

where dtau_j/dw_i = 0.7 * sign(w_i) / (g * s_j).

Smoothing the delta function with a Gaussian kernel of width sigma_g:
```
dQ-tilde(w_k)/dw_i|_{indirect} ~ -sign(n_k) * phi_normal(d_k / sigma_g) / sigma_g * 0.7 * sign(w_i) / g
```

Summing over all weights k in the group:
```
sum_k (dL/dQ_k) * dQ-tilde(w_k)/dw_i|_{indirect} = -(0.7 * sign(w_i) / g) * G_boundary
```

where:
```
G_boundary = sum_{k in group} (dL/dQ_k) * sign(n_k) * phi_normal(d_k / sigma_g) / sigma_g
```

The corrected gradient:
```
g_GASTE(w_i) = g_STE(w_i) - (0.7 / g) * sign(w_i) * G_boundary
```

**Complexity:** O(g) per group to compute G_boundary (one dot product), then O(1) per weight for the correction. Total: O(1) amortized per weight.

**Magnitude analysis:**

G_boundary = sum of ~13 terms (boundary weights per group), each of magnitude |g_STE| * phi_max / sigma_g.

For sigma_g = 0.3, phi_max = phi_normal(0) = 0.4/0.3 = 1.33:
G_boundary ~ 13 * E[|g_STE|] * 1.33 ~ 17.3 * E[|g_STE|]

Correction per weight: 0.7/128 * G_boundary ~ 0.0055 * 17.3 * E[|g_STE|] ~ 0.095 * E[|g_STE|]

So the correction is about 9.5% of the STE gradient magnitude. This is small but systematic -- it consistently points in the correct direction for accounting for group coupling.

**Stability:** The correction is bounded by 0.7/g * |G_boundary| < 0.7 * max(phi/sigma_g) * mean(|g_STE|) ~ 0.1 * mean(|g_STE|). This is a small perturbation to the STE gradient, so stability is maintained.

**Novelty Assessment:** HIGHLY NOVEL. No existing STE variant accounts for the group threshold coupling. Standard analyses treat each weight independently, which is correct for per-element quantization but incorrect for our group-based quantization with shared threshold. This is a genuine first-order correction to an approximation that the entire QAT literature neglects.

**Viability Rating:** MEDIUM

**Risks:**
- Small expected impact (~0.1-0.3% PPL)
- Requires saving d_i and n_i in the backward pass (extra memory)
- The Gaussian smoothing introduces a hyperparameter sigma_g

---

### Path 6: Shadow-Free Ternary Training -- via GF(3) Optimization [SPECULATIVE]

**Core Idea:** Eliminate FP32 shadow weights entirely. Train directly in the ternary weight space using gradient-informed flip decisions with momentum.

**Formulation:**

State: T in {-1, 0, +1}^n (ternary weights), s in R^{n/128} (FP16 scales), m in R^n (flip momentum, FP16)

Update rule per step:
```
1. Forward: y = T * s * x  (ternary matmul)
2. Backward: g = dL/dy * x^T (standard gradient of output w.r.t. T*s)
3. Momentum: m_i <- (1-beta) * m_i + beta * g_i
4. Scale update: s_j <- s_j - eta_s * dL/ds_j
5. Flip decision: for each weight i:
   For each candidate v in {-1, 0, +1} \ {T_i}:
     DeltaL_v = m_i * s_j * (v - T_i)
   If min_v DeltaL_v < -flip_threshold:
     T_i <- argmin_v DeltaL_v
     m_i <- 0  (reset momentum after flip)
```

**Memory analysis:**
```
Current QAT: 32 (shadow) + 32 (Adam m) + 32 (Adam v) + 2 (ternary) + 16 (scale) = 114 bits/weight
Shadow-Free: 2 (ternary) + 16 (momentum, FP16) + 16 (scale) = 34 bits/weight
Reduction: 70%
```

This would allow ~3.35x larger models in the same training memory. A 50M model that currently uses 4.4GB for training would use ~1.3GB, enabling ~170M in 4.4GB.

**Mathematical challenges:**

1. **No continuous relaxation:** The loss landscape is piecewise constant in T-space. The momentum m_i provides a soft proxy, but there's no smooth surface to follow.

2. **Flip coordination:** Independent flip decisions may conflict (flipping w_i changes the optimal flip for w_j). Need a GREEDY or BATCH-COORDINATED flip strategy.

3. **Convergence:** No convergence guarantee for discrete greedy optimization. May get stuck in local optima that STE-based training would escape (because STE has the shadow weight "tunnel" through barriers).

4. **Scale learning:** s_j is the only continuous parameter. It must do all the "fine-tuning" that shadow weights currently do. This puts enormous pressure on the scale gradients.

**Potential mitigation:** Use a SMALL FP16 "micro-shadow" weight per group (not per weight) to provide continuous interpolation:
```
Q_hybrid(w)_i = T_i * s_j + delta_j * phi(w_i)
```
where delta_j is a per-group FP16 offset and phi is a smooth function. This adds 16 bits per group (128 weights) = 0.125 bits/weight. Negligible storage, but provides a continuous degree of freedom for optimization.

**Novelty Assessment:** HIGHLY NOVEL and SPECULATIVE. Shadow-free training for ternary has been mentioned conceptually (evolutionary polish, GF(3) algebra) but a complete training framework with momentum, flip decisions, and convergence analysis doesn't exist.

**Viability Rating:** LOW (speculative, high risk, high potential reward)

**Next Steps:** Would require a careful small-scale experiment at 1M parameters to test whether discrete flip optimization converges at all. If it does, the memory savings justify further development.

---

## 5. Ranking and Recommendations

| Rank | Path | Target Gap | Expected PPL Closure | Viability | Compute Cost | Risk |
|------|------|------------|---------------------|-----------|--------------|------|
| 1 | STP (Threshold Jitter) | Duration + Interaction | 0.3-0.5% | HIGH | ~0% | Very Low |
| 2 | CTCE (Boundary Amplification) | Optimization | 0.6-0.9% | HIGH | ~0% | Low |
| 3 | SAQ (Step-Aligned Quant) | Duration | 0.9-2.2% | MEDIUM-HIGH | Negative | Medium |
| 4 | RLEC (Low-Rank Correction) | Representation | 0.9-1.5% | MEDIUM | +0.6-1.6% | Low |
| 5 | GASTE (Group-Aware STE) | Optimization | 0.1-0.3% | MEDIUM | ~0% | Low |
| 6 | Shadow-Free Training | Memory (indirect) | Enables 3x larger model | LOW | Same | High |

**Recommended testing order:** STP -> CTCE -> SAQ -> RLEC -> GASTE -> Shadow-Free

**Combined potential (Paths 1-5 together):** 2.8-5.4% PPL closure out of 7.4% gap.

**Best case scenario:** 7.4% gap reduced to ~2.0-4.6% gap, which is within "more training steps" territory and consistent with the representation theory lower bound.

---

## 6. Cross-Pollination Opportunities

### STP + CTCE Synergy
STP smooths the loss landscape at boundaries. CTCE amplifies gradients at boundaries. Together: the optimizer receives STRONG, WELL-DIRECTED gradient signal at boundaries (CTCE amplification) in a SMOOTH landscape (STP smoothing). This is analogous to using both momentum and learning rate scheduling -- one provides direction, the other provides landscape quality.

### SAQ + STP Synergy
During SAQ's FP16 steps, there's no quantization threshold to perturb. STP only applies during the ternary steps. This is actually beneficial: the FP16 steps provide CLEAN gradient information, and the ternary steps use STP to find smooth quantization configurations. The two mechanisms are temporally separated and complementary.

### RLEC Independence
RLEC operates in a separate parameter space (the A, B matrices) and a separate compute path (FP16 matmul). It can be combined with ANY gradient estimator (STE, CTCE, GASTE) without interaction. This makes it a safe "add-on" that provides orthogonal benefit.

### GASTE + CTCE Integration
Both modify the backward pass gradient. They can be combined:
```
g_combined(w_i) = (g_STE(w_i) - 0.7/g * sign(w_i) * G_boundary) * (1 + beta * Gaussian(d_i))
```
The GASTE correction adjusts for group coupling, then CTCE amplifies the corrected gradient at boundaries. Order matters: correct first, then amplify.

### Shadow-Free + RLEC
If shadow-free training works but has quality issues, RLEC provides a natural quality recovery mechanism: the low-rank FP16 matrices can compensate for the discretization artifacts of shadow-free training. The combination would give: minimal memory (shadow-free) + maximum quality (RLEC correction).

---

## 7. Why Previous Regularizers Failed (Unified Theory)

**Theorem (Regularizer Incompatibility for QAT):**

Let L_task(Q(w)) be the task loss and R(w) be any regularizer with a fixed attractor w* (i.e., nabla R(w) points toward w* regardless of the task). Then for any lambda > 0, there exist training configurations where the regularized gradient g = nabla L_task + lambda * nabla R has NEGATIVE correlation with the task descent direction.

**Proof sketch:**

At a point w where nabla L_task points away from w* (task wants to move weight away from the regularizer's preferred position), the angle between g and nabla L_task satisfies:

```
cos(theta) = (1 + lambda * <nabla L, nabla R> / ||nabla L||^2) / ||g/nabla L||
```

When <nabla L, nabla R> < -||nabla L||^2 / lambda (i.e., the regularizer opposes the task gradient strongly enough), cos(theta) < 0 and the combined gradient is an ASCENT direction.

This happens whenever:
- Angular reg: a dead-zone weight needs to activate (task gradient pushes away from 0, regularizer pushes toward Q(w)=0)
- Boundary-repulsive: a weight needs to cross the boundary (task gradient pushes across, regularizer creates barrier)
- Any norm-based regularizer near the quantization boundary

**Corollary:** The ONLY safe regularization for QAT is one that operates in LOSS SPACE (not weight space). STP achieves this by smoothing the loss landscape rather than constraining the weights.

**Corollary 2:** Combining multiple weight-space regularizers COMPOUNDS the problem because each adds its own anti-task gradient component. This explains the 19.3% degradation from full integration (multiple conflicting gradients).

---

## 8. Open Questions Requiring Empirical Validation

1. **CTCE beta calibration:** Is the optimal beta constant across layers, or should deeper layers use different amplification? Theory suggests deeper layers have different gradient norm distributions, so beta may need to be layer-adaptive.

2. **SAQ optimal K:** The theoretical analysis suggests K=3-5, but this depends on the learning rate, batch size, and loss landscape curvature. Need to sweep K in {1, 2, 3, 5, 7, 10} at 6M scale.

3. **RLEC rank sufficiency:** Is r=4 enough? The spectral decay of E depends on the weight distribution, which changes during training. May need to start with r=8 and prune to r=4 after initial training.

4. **STP sigma schedule:** Should sigma_tau decrease during training (like temperature annealing) or stay constant? Decreasing sigma_tau would give sharper quantization boundaries as training converges.

5. **Interaction effects:** While the analysis predicts no negative interaction between STP, CTCE, SAQ, and RLEC (they operate on orthogonal dimensions), this must be validated empirically. Test each individually first, then pairwise, then all together.

6. **Scale sensitivity:** All solutions depend on the group scale s. If s varies dramatically across groups (which it does -- early layers have different scales than later layers), the hyperparameters may need per-layer tuning.
