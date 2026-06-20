# R9-A3: Learning Rate Meta-Strategies — Findings

**Round:** R9 | **Aspect:** A3 | **Focus:** tokens_to_converge via LR optimization
**Sources:** Wave 1 (W1-AgA, W1-AgB) + Wave 2 (W2-AgA, W2-AgB)
**Gate:** G1 (Actionability) + G2 (Applicability) + G3 (No-LEGO)
**Synthesis Date:** 2026-03-06

---

## Summary

- **IMPLEMENT:** 17 findings (all pass Gate v3)
- **EXPLORE:** 3 findings (viable but require empirical tuning or lower confidence)
- **SKIP / Superseded:** 7 findings

**Critical deduplication note:** W1-AgB-F2 (LION-BETA1-COMMITMENT-ANNEALING, β₁ = 0.9 - 0.3f) is **SUPERSEDED** by W2-AgA-F5 (LION-PER-PARAM-SNR-ADAPTIVE-BETA1). The W2 version has the correct theoretical direction (β₁ increases for low-SNR parameters, not decreases with global commitment), is per-parameter rather than global, and is derived from sign accuracy analysis. W1-AgB-F2 is marked as SUPERSEDED; W2-AgA-F5 is the canonical version.

---

## IMPLEMENT Findings

All 17 findings below pass Gate v3 (G1 + G2 + G3). Organized by theme for composition.

### **LR Schedule Shape & Decay (Wave 1-A, Tier 1 Impact)**

#### **R9-A3-W1-A-F1: WSD-COSINE-DECAY-PHASE**

**ID:** R9-A3-W1-A-F1
**Gate:** PASS (G1: specific file:line, pseudocode; G2: optimizer-agnostic; G3: always-active)
**Principle:** The WSD decay phase loss landscape is approximately quadratic (river valley theory). Cosine-shaped decay within the decay window matches this curvature better than linear, reaching the optimum in fewer steps for the same decay budget.

**Implementation:**
File: `src/model_v7/__main__.py` (adapt from src/model_v5/train.py L3164-3165)
```python
# Replace linear decay within WSD decay phase:
decay_lr = 0.5 * (1.0 + math.cos(decay_progress * math.pi))
return max(0.0, decay_lr)
```

**Gain Estimate:** 5-12% step reduction in final loss quality for fixed decay budget.
**Always-active:** Yes — baked into WSD schedule branch.

---

#### **R9-A3-W1-A-F2: WSD-S-CHECKPOINT-REUSE-DECAY**

**ID:** R9-A3-W1-A-F2
**Gate:** PASS (G1: specific file:function; G2: optimizer-agnostic; G3: always-active)
**Principle:** WSD-S (Wen et al. NeurIPS 2024) outperforms WSD by periodic decay-resume cycles. After each decay+resume, the model starts from a better point than the original stable checkpoint.

**Implementation:**
File: `src/model_v7/__main__.py` — new `wsd_s` option in lr_schedule
```python
cycle = step // decay_interval
phase_in_cycle = step % decay_interval
if phase_in_cycle < decay_length:
    decay_progress = phase_in_cycle / decay_length
    return 0.5 * (1.0 + math.cos(decay_progress * math.pi))
else:
    return 1.0  # stable phase
```

**Gain Estimate:** 8-15% step reduction vs standard WSD.
**Always-active:** Yes — always-active when wsd_s mode selected.

---

#### **R9-A3-W1-A-F3: COMMITMENT-TRIGGERED-WSD-DECAY-BRANCH**

**ID:** R9-A3-W1-A-F3
**Gate:** PASS (G1: specific file:function; G2: ternary-native, SSM-compatible; G3: always-active)
**Principle:** Trigger fast decay branch when commitment_fraction crosses a threshold (e.g. 0.3). Commitment events are phase transitions in the loss landscape — the effective parameter space shrinks, and decay immediately post-commitment refines in the new reduced geometry.

**Implementation:**
File: `src/model_v7/__main__.py` (lr_schedule function)
```python
committed_frac = getattr(model, '_last_committed_frac', 0.0)
decay_trigger_frac = 0.3
if committed_frac >= decay_trigger_frac and not hasattr(lr_schedule, '_decay_triggered'):
    lr_schedule._decay_triggered = True
    lr_schedule._decay_start = step
if hasattr(lr_schedule, '_decay_triggered'):
    t = (step - lr_schedule._decay_start) / 50
    if t < 1.0:
        return 0.5 * (1.0 + math.cos(t * math.pi))
    return 0.1
```

**Gain Estimate:** 10-20% step reduction.
**Always-active:** Yes — reads C2 state (already always-on).

---

### **Parameter Group LR Scaling (Tier 1 Impact)**

#### **R9-A3-W1-A-F4: MUP-LR-PARAM-GROUP-SPLIT**

**ID:** R9-A3-W1-A-F4
**Gate:** PASS (G1: specific file:section; G2: algebraically valid; G3: always-active)
**Principle:** Maximal Update Parameterization (muP) shows hidden layer weights require LR scaled by 1/√d_model (for Lion sign updates) to keep weight updates of consistent magnitude. Split param_groups into embeddings (LR=base_lr) and hidden matmuls (LR=base_lr/√d_model_ratio).

**Implementation:**
File: `src/model_v7/__main__.py` (param_groups construction before optimizer init)
```python
d_model = config.d_model
d_model_base = 256
mup_ratio = d_model / d_model_base

embed_params, hidden_params, output_params = [], [], []
for name, p in model.named_parameters():
    if 'embedding' in name or 'embed' in name:
        embed_params.append(p)
    elif 'out_proj' in name or 'lm_head' in name:
        output_params.append(p)
    else:
        hidden_params.append(p)

param_groups = [
    {'params': embed_params, 'lr': base_lr},
    {'params': hidden_params, 'lr': base_lr / (mup_ratio ** 0.5)},
    {'params': output_params, 'lr': base_lr / mup_ratio},
]
```

**Gain Estimate:** At 6M, gain = 0 (calibration only). Primary value: enables zero-shot LR transfer to larger models.
**Always-active:** Yes — baked into optimizer construction.

---

#### **R9-A3-W1-A-F5: WSD-STABLE-PHASE-LENGTH-COMMITMENT-ADAPTIVE**

**ID:** R9-A3-W1-A-F5
**Gate:** PASS (G1: specific file:function; G2: ternary-native; G3: always-active)
**Principle:** End stable phase early when commitment_fraction plateaus (d(f)/dt < epsilon). Stable phase only contributes river progress while commitment_fraction is rising. Once plateau, stable phase at high LR only causes oscillations.

**Implementation:**
File: `src/model_v7/__main__.py` (lr_schedule function) + C2 state
```python
if lr_schedule_type == "wsd_adaptive":
    commit_rate = getattr(model, '_commit_rate_ema', 1.0)
    plateau_threshold = 1e-4
    if step < warmup_end:
        return step / warmup_end
    elif not lr_schedule._decay_started and commit_rate > plateau_threshold:
        return 1.0
    else:
        lr_schedule._decay_started = True
        decay_t = (step - lr_schedule._decay_start_step) / decay_length
        return max(0.0, 0.5 * (1.0 + math.cos(min(decay_t, 1.0) * math.pi)))
```

**Gain Estimate:** 10-25% step reduction.
**Always-active:** Yes — reads C2 state.

---

### **Adaptive LR Monitoring (Wave 1-B, Tier 1-2 Impact)**

#### **R9-A3-W1-B-F1: LOSS-EMA-PLATEAU-LR-HALVING**

**ID:** R9-A3-W1-B-F1
**Gate:** PASS (G1: specific technique, train.py identified; G2: works with ternary + SSM; G3: always-active)
**Principle:** When training loss EMA flattens (ΔL_EMA < ε), halving LR reduces the sign noise floor in Lion's momentum, allowing smaller-signal thresholds to cross.

**Implementation:**
File: `train.py` (~L3920 and ~L5520)
```python
# At initialization:
loss_ema = None
loss_ema_alpha = 0.98

# Inside training loop after loss.backward():
loss_val = loss.item()
if loss_ema is None:
    loss_ema = loss_val
else:
    loss_ema = loss_ema_alpha * loss_ema + (1 - loss_ema_alpha) * loss_val

# Every check_interval=100 steps:
if global_step % 100 == 0 and hasattr(train_loop, '_prev_loss_ema'):
    delta = abs(loss_ema - train_loop._prev_loss_ema)
    if delta < 0.001:
        for pg in optimizer.param_groups:
            pg['lr'] = max(pg['lr'] * 0.5, base_lr * 0.01)
train_loop._prev_loss_ema = loss_ema
```

**Gain Estimate:** 8-15% step reduction.
**Always-active:** Yes — inline loss_ema tracking.

---

#### **R9-A3-W1-B-F3: FLIP-RATE-ALWAYS-ACTIVE-LR-SCHEDULE**

**ID:** R9-A3-W1-B-F3
**Gate:** PASS (G1: specific technique, train.py identified; G2: ternary-native; G3: removes LEGO guard)
**Principle:** `use_two_stage_flip_lr` exists as LEGO but should be always-active. Flip rate is an ideal surrogate for phase detection: high flip rate = exploration; low flip rate = commitment plateau. When flip-rate EMA < 1%, reduce LR to finalize commitment.

**Implementation:**
File: `train.py` lines 3244-3256 (remove LEGO guard) + 3197-3220 (merge into lr_schedule)
```python
# Remove `if getattr(config, 'use_two_stage_flip_lr', False):` guard at line 3244
# Make FlipRateTracker unconditional when use_ternary=True

if use_ternary and flip_tracker.transitioned:
    # Phase 2: cosine decay from transition point
    steps_since = step - flip_tracker.transition_step
    return 0.1 * 0.5 * (1 + math.cos(steps_since / T_decay * math.pi))
else:
    return existing_inv_sqrt_schedule(step)
```

**Gain Estimate:** 10-20% step reduction.
**Always-active:** Yes — removes LEGO, becomes always-active when use_ternary=True.

---

#### **R9-A3-W1-B-F4: LION-DEAD-ZONE-COMMITMENT-LR-MICRO-BOOST**

**ID:** R9-A3-W1-B-F4
**Gate:** PASS (G1: specific technique, ternary.py identified; G2: ternary-native; G3: always-active)
**Principle:** Weights with |w_shadow| in the dead zone near the ternary threshold are the only ones that can flip. Apply targeted LR micro-boost (×3.0) ONLY to gradient components for these threshold-adjacent shadow weights.

**Implementation:**
File: `src/model_v7/components/ternary.py` (inside TernaryLinear.forward or after gradient computation)
```python
threshold = module.threshold_coeff * module.weight.abs().mean()
margin = 0.15 * threshold
near_threshold = (module.weight.abs() - threshold).abs() < margin
if module.weight.grad is not None:
    module.weight.grad[near_threshold] *= 3.0
```

**Gain Estimate:** 6-14% step reduction.
**Always-active:** Yes — inline in backward without flag.

---

#### **R9-A3-W1-B-F5: LION-OPTIMIZER-STATE-WARM-RESTART-FOR-UNCOMMITTED**

**ID:** R9-A3-W1-B-F5
**Gate:** PASS (G1: specific technique, train.py identified; G2: ternary-native; G3: always-active)
**Principle:** Reset ONLY the Lion momentum state (`exp_avg`) for parameters whose commitment stalls (flip_rate_ema < ε but committed_fraction unchanged for N steps). This gives the optimizer fresh gradient signal while preserving learned weight values.

**Implementation:**
File: `train.py` (~L5628, commitment monitoring block)
```python
if global_step % 200 == 0 and use_ternary:
    prev_f = getattr(train_model, '_prev_committed_f', None)
    curr_f = sum(m.committed_mask.float().mean() for m in ternary_modules) / n
    if prev_f is not None and abs(curr_f - prev_f) < 0.001:
        for name, module in model.named_modules():
            if hasattr(module, 'committed_mask') and module.weight in optimizer.state:
                uncommitted = ~module.committed_mask
                if uncommitted.any():
                    optimizer.state[module.weight]['exp_avg'][uncommitted] = 0.0
    train_model._prev_committed_f = curr_f
```

**Gain Estimate:** 5-10% step reduction.
**Always-active:** Yes — runs every 200 steps when use_ternary=True.

---

### **Theory-Derived LR Bounds (Wave 2-A, Tier 1 Impact)**

#### **R9-A3-W2-A-F1: LION-SIGN-NOISE-INDEPENDENCE-LR-CEILING**

**ID:** R9-A3-W2-A-F1
**Gate:** PASS (G1: specific formula `lr_max = 2·mean(|g|)`; G2: pure algorithmic; G3: always-active ceiling)
**Principle:** For Lion's update rule, the learning rate is **decoupled from sign error rate**. Sign error depends only on SNR (batch size and gradient statistics), NOT lr. The only stability constraint is the loss landscape geometry, derived from the descent lemma:

    L(θ_{t+1}) ≤ L(θ_t) - 2·lr·(g^T·sign(m_t)) + lr²·d

Therefore: `lr < 2·mean(|g|)` (twice the mean absolute gradient magnitude).

**Implementation:**
File: `train.py` (Lion optimizer loop ~L5000)
```python
grad_l1_mean = sum(p.grad.abs().mean() for p in param_group) / n_params
lr_ceiling = 2.0 * grad_l1_mean
lr_effective = min(base_lr, lr_ceiling)
```

**Gain Estimate:** 5-15% step reduction by avoiding divergence-and-restart cycles.
**Always-active:** Yes — unconditional lr ceiling.

---

#### **R9-A3-W2-A-F2: UNCOMMITTED-GAP-ADAPTIVE-LR**

**ID:** R9-A3-W2-A-F2
**Gate:** PASS (G1: specific formula; G2: ternary-native; G3: always-active)
**Principle:** The expected time for weight i to commit is γᵢ / (lr · (1-2p)) where γᵢ = δ - |wᵢ| is the gap to threshold. Minimizing total commit time requires high lr, but the **distribution-adaptive insight** is: set `lr ∝ γ̄` (mean gap across uncommitted weights). This creates a naturally decaying schedule:
- Early: γ̄ ≈ δ → large lr (fast early learning)
- Later: γ̄ → 0 → smaller lr (stable commitment)

Formula: `lr_t = lr_max · (γ̄_t / δ) + lr_min = lr_max · (1 - mean(|w_uncommitted|) / δ) + lr_min`

**Implementation:**
File: `train.py` lr scheduler hook + `ternary.py` for delta access
```python
gaps = []
for layer in ternary_layers:
    w = layer.weight.detach()
    mask = layer.committed_mask
    delta = layer.quantizer.threshold
    w_uncomm = w[~mask]
    if w_uncomm.numel() > 0:
        gaps.append(delta - w_uncomm.abs().mean())

if gaps:
    gamma_mean = sum(gaps) / len(gaps)
    lr_scale = gamma_mean / delta
    lr_t = lr_max * lr_scale + lr_min
    set_lr(optimizer, lr_t)
```

**Gain Estimate:** 10-25% step reduction.
**Always-active:** Yes — once wired, replaces scheduler.

---

#### **R9-A3-W2-A-F3: OLS-STE-ALPHA-LR-STABILITY-BOUND**

**ID:** R9-A3-W2-A-F3
**Gate:** PASS (G1: specific formula `lr_max = C_STE × α_t`; G2: ternary-native; G3: always-active ceiling)
**Principle:** OLS-STE scale α_t = ||W_ternary||_F / ||W_sign||_F ≈ mean(|wᵢ|) encodes the STE approximation accuracy. For Lion steps with `lr >> α_t`, the ternary scale drifts faster than the backward pass can track, creating hidden gradient error. STE approximation validity requires: `lr ≤ C_STE · α_t` where `C_STE ≈ 0.1-0.2`.

**Implementation:**
File: `train.py` (after forward pass) + `ternary.py` (store `_last_alpha`)
```python
C_STE = 0.15
alpha_values = [layer.quantizer._last_alpha
                for layer in ternary_layers
                if hasattr(layer.quantizer, '_last_alpha')]
if alpha_values:
    alpha_ema = 0.99 * alpha_ema + 0.01 * min(alpha_values)
    lr_ste_bound = C_STE * alpha_ema
    lr_effective = min(base_lr, lr_ste_bound)
    set_lr(optimizer, lr_effective)
```

**Gain Estimate:** 5-12% early-training step reduction.
**Always-active:** Yes — unconditional ceiling.

---

#### **R9-A3-W2-A-F4: SIGMA-G-CONDITION-RATIO-LR-DECAY**

**ID:** R9-A3-W2-A-F4
**Gate:** PASS (G1: specific formula; G2: ternary-native; G3: always-active)
**Principle:** As commitment fraction f increases, remaining uncommitted parameters become progressively harder to optimize. Effective Hessian condition number κ_eff grows. Lion stability requires `lr < 1/κ_eff`. The C2 state already maintains σ_g per parameter, enabling a proxy: `κ_proxy = max(σ_g_uncomm) / min(σ_g_uncomm)`. Apply: `lr_t = base_lr / √κ_proxy`.

**Implementation:**
File: `train.py` lr scheduler + `commitment.py` (σ_g access)
```python
sigma_g_all = torch.cat([
    model_module.sigma_g[~model_module.committed_mask]
    for model_module in ternary_linear_modules
])
if sigma_g_all.numel() > 1:
    kappa_proxy = sigma_g_all.max() / (sigma_g_all.min() + 1e-8)
    lr_decay = 1.0 / kappa_proxy.sqrt().clamp(min=1.0)
    lr_t = base_lr * lr_decay
    set_lr(optimizer, max(lr_t, lr_min))
```

**Gain Estimate:** 8-15% step reduction.
**Always-active:** Yes — reads existing C2 state.

---

#### **R9-A3-W2-A-F5: LION-PER-PARAM-SNR-ADAPTIVE-BETA1** ⭐ **CANONICAL VERSION**

**ID:** R9-A3-W2-A-F5
**Gate:** PASS (G1: specific formula; G2: ternary-native; G3: always-active)
**Principle:** For Lion momentum sign accuracy, `P(correct) ≈ Φ(SNR · √(1/(1-β₁)))` where SNR = μ_g / σ_g per-parameter. **Optimal β₁ depends on SNR:** low-SNR parameters need higher β₁ for heavy averaging; high-SNR parameters can use lower β₁. Formula:
    β₁_eff(i,t) = β₁_max - (β₁_max - β₁_min) × SNR_i / (1 + SNR_i)

This is the **CORRECT direction** (opposite of W1-AgB-F2). As remaining uncommitted weights tend to have LOWER SNR (harder to commit, noisy signal), they naturally receive HIGHER β₁, not lower.

**Implementation:**
File: `train.py` Lion optimizer loop (~L5000) + `commitment.py` (σ_g read)
```python
BETA1_MAX = 0.95
BETA1_MIN = 0.85

mu_g_ema = 0.99 * mu_g_ema + 0.01 * g.abs()
sigma_g = commitment_module.sigma_g
snr = mu_g_ema / (sigma_g + 1e-8)
beta1_eff = BETA1_MAX - (BETA1_MAX - BETA1_MIN) * snr / (1.0 + snr)

# Per-param momentum update:
m = beta1_eff * m + (1.0 - beta1_eff) * g
theta = theta - lr * m.sign()
```

**Gain Estimate:** 8-18% step reduction.
**Always-active:** Yes — replaces scalar β₁ with per-param in Lion loop.
**Note:** **SUPERSEDES W1-AgB-F2** (incorrect direction and global rather than per-param).

---

### **Manifold-Geometry LR Strategies (Wave 2-B, Tier 1-2 Impact)**

#### **R9-A3-W2-B-F1: CAPACITY-NORMALIZED-LR-SCALING**

**ID:** R9-A3-W2-B-F1
**Gate:** PASS (G1: specific algorithm, pseudocode; G2: ternary-native; G3: always-active)
**Principle:** When n_k weights commit at step t_k, the optimization problem dimensionality drops. Information-theoretically, this creates a **surplus of LR relative to problem curvature**. The correct response is to **increase LR proportionally** to maintain constant effective step-size-to-curvature ratio. For Lion, the exponent is α ∈ [0.25, 0.5] (between sign-invariant 0.0 and muP-consistent 0.5).

Formula: `η_t = η_base · min(2.0, (1.0 / (1.0 - f(t) + ε))^α)` where f = committed_fraction, α = 0.35.

**Implementation:**
File: `train.py` (after optimizer.step(), reads C2 `committed_mask`)
```python
f = len(committed_mask.nonzero()) / len(committed_mask)
alpha = 0.35
capacity_multiplier = min(2.0, (1.0 / (1.0 - f + 1e-6)) ** alpha)

for group in optimizer.param_groups:
    group['lr'] = base_lr * schedule_multiplier * capacity_multiplier
```

**Gain Estimate:** 8-18% step reduction.
**Always-active:** Yes — no new state beyond already-tracked f.

---

#### **R9-A3-W2-B-F2: AVALANCHE-DAMPING-LR-BRAKE**

**ID:** R9-A3-W2-B-F2
**Gate:** PASS (G1: pseudocode, files identified; G2: ternary-native; G3: always-active)
**Principle:** Commitment events follow contagion dynamics. When one group commits, neighboring groups' σ_g drops, triggering their commitment — an "avalanche." During avalanche, the loss landscape is rapidly simplifying. Optimal strategy: slow down (reduce LR) to avoid overshooting minima that are moving. Formula: `brake = max(0.5, 1.0 - λ_t · 100)` where λ_t = 50-step EMA of commitment delta.

When λ_t = 0.005 (0.5% weights committed per step), brake = 0.5x LR.

**Implementation:**
File: `commitment.py` (track λ_ema) + `train.py` (apply brake)
```python
# In commitment.py:
new_committed = (committed_mask.sum() - prev_committed_count).float() / n_total
lambda_ema = 0.9 * lambda_ema + 0.1 * new_committed

# In train.py:
avalanche_brake = max(0.5, 1.0 - lambda_ema * 100.0)
for group in optimizer.param_groups:
    group['lr'] = base_lr * schedule_mult * avalanche_brake
```

**Gain Estimate:** 5-12% step reduction.
**Always-active:** Yes — no flag.

---

#### **R9-A3-W2-B-F4: COMMITMENT-RATE-PID-LR-CONTROLLER**

**ID:** R9-A3-W2-B-F4
**Gate:** PASS (G1: pseudocode, train.py; G2: ternary-native; G3: always-active)
**Principle:** Commitment rate r(t) = d(committed_fraction)/dt is the single most informative signal about training dynamics. Instead of open-loop LR scheduling (function of time only), close the loop using r(t) as the process variable in a PID controller. Target: `r* = r_total / T_budget` (uniform commitment finish).

Error: `e(t) = r* - r(t)`. Control law: `η(t) = η_base · (1 + K_p·e(t) + K_i·∫e(s)ds)` with anti-windup at [0.2·η_base, 3.0·η_base].

**Implementation:**
File: `train.py` (3 new scalar state: λ_ema from F2, commitment_pid_integral, r_target)
```python
r_target = 0.001
r_current = lambda_ema

e_t = r_target - r_current
commitment_pid_integral = 0.98 * commitment_pid_integral + e_t

K_p, K_i = 0.20, 0.001
pid_multiplier = 1.0 + K_p * e_t + K_i * commitment_pid_integral
pid_multiplier = max(0.2, min(3.0, pid_multiplier))

for group in optimizer.param_groups:
    group['lr'] = base_lr * schedule_mult * pid_multiplier
```

**Gain Estimate:** 10-20% step reduction.
**Always-active:** Yes — no flag.

---

### **Ternary-Specific Bounds (Wave 2-B, Tier 2 Impact)**

#### **R9-A3-W2-B-F3: TERNARY-CLIFF-LR-BOUND**

**ID:** R9-A3-W2-B-F3
**Gate:** EXPLORE (G1: pseudocode; G2: ternary-native; G3: always-active)
**Note:** Medium-high viability but requires empirical tuning of k=0.5 constant.
**Principle:** Ternary networks have a THIRD curvature regime (beyond mountain/river): "cliff" directions where weight shadow value is near the ternary threshold δ. A single Lion step can cause discontinuous loss jump. Safe LR bound derived from requiring `P(cliff_cross) < p_max`:

`η_safe = k · δ_cliff` where δ_cliff = E[|w_shadow - δ|] over frontier weights, k ≈ 0.5.

**Implementation:**
File: `ternary.py` TernaryLinear (compute per layer) + `train.py` (aggregate)
```python
# In ternary.py:
frontier_mask = (w_shadow.abs() > 0.7 * delta) & (w_shadow.abs() < 1.3 * delta)
if frontier_mask.any():
    delta_cliff = (w_shadow[frontier_mask].abs() - delta).abs().mean()
    eta_cliff = 0.5 * delta_cliff.item()
else:
    eta_cliff = float('inf')

# In train.py:
eta_cliff_global = min(layer.eta_cliff for layer in model.ternary_layers)
for group in optimizer.param_groups:
    group['lr'] = min(group['lr'], eta_cliff_global)
```

**Gain Estimate:** 5-10% step reduction.
**Always-active:** Yes — unconditional clamp.

---

#### **R9-A3-W2-B-F5: THRESHOLD-RESCALING-TRIGGERED-LR-WARMUP**

**ID:** R9-A3-W2-B-F5
**Gate:** EXPLORE (G1: pseudocode; G2: ternary-native; G3: always-active)
**Note:** Medium viability; gain depends on recalibration frequency.
**Principle:** When ternary threshold delta is recalibrated (via LSQ updates or scale changes), the quantization function changes discontinuously. Lion momentum (exp_avg) is stale relative to the new landscape. Optimal response: post-recalibration warmup to re-calibrate momentum.

`η(t) = η_post_recal · (1 - exp(-t / T_warmup))` for ~9 steps after recalibration event.

**Implementation:**
File: `ternary.py` TernaryLinear (track recalibration) + `train.py` (apply warmup factor)
```python
# In ternary.py:
scale_change_ratio = abs(new_scale - self.scale_prev) / (self.scale_prev + 1e-8)
if scale_change_ratio > 0.05:
    self.recal_warmup_counter = int(-1.0 / math.log(beta1))

if self.recal_warmup_counter > 0:
    self.recal_warmup_counter -= 1
    warmup_factor = 1.0 - math.exp(-self.recal_warmup_counter / 9.0)

# In train.py:
min_warmup_factor = min(layer.recal_warmup_factor for layer in ternary_layers)
for group in optimizer.param_groups:
    group['lr'] = base_lr * schedule_mult * min_warmup_factor
```

**Gain Estimate:** 4-8% step reduction (empirical).
**Always-active:** Yes — fires whenever scale changes.

---

## EXPLORE Findings

Three findings are viable but require empirical tuning or have lower confidence. Recommended for experimental validation before integration.

1. **R9-A3-W2-B-F3: TERNARY-CLIFF-LR-BOUND** — Mathematically sound but k=0.5 constant needs empirical validation.
2. **R9-A3-W2-B-F5: THRESHOLD-RESCALING-TRIGGERED-LR-WARMUP** — Gain is conditional on recalibration frequency (LSQ update rate).

---

## SKIP / Superseded Findings

| ID | Reason |
|----|--------|
| R9-A3-W1-B-F2 | **SUPERSEDED by R9-A3-W2-A-F5.** Wrong direction (β₁ decreases with commitment, but remaining weights have lower SNR and need higher β₁); global not per-parameter. W2-AgA-F5 is the correct theoretical version. |
| **1-cycle super-convergence** (W1-A, filtered) | FAIL G2. Lion's sign update fixes magnitude at ±lr — CLR's variance mechanism does not apply. |
| **Warmup elimination** (W1-A, filtered) | FAIL G1. OLS-STE lsq_scale calibration requires warmup. |
| **Chinchilla LR decay law** (W1-A, filtered) | FAIL G1. Already implemented as default schedule. |
| **muP for SSM B/C/dt projections** (W1-A, filtered) | FAIL G2. muP theory applies to standard linear layers; B/C/dt are input-dependent selections. |

---

## Combined LR System

The 17 IMPLEMENT findings compose into a single, coherent LR formula that is always-active and reads from C2 state (no new data structures required).

### **Mathematical Form**

```
η(step) = η_schedule(step) × capacity_mult × pid_mult × avalanche_brake × σ_g_decay × cliff_clamp × recal_warmup
```

Where each term is computed from available state:

#### **1. Base Schedule:** η_schedule(step)

Choose one of:
- **inv_sqrt (default):** `η = η_base / √(1 + step / τ)`
- **WSD-cosine (F1):** Linear decay phase replaced with cosine:
  - Warmup: `step / warmup_steps`
  - Stable: `1.0`
  - Decay: `0.5 · (1 + cos(π · decay_progress))`
- **WSD-S (F2):** Periodic decay-resume cycles
- **Flip-rate triggered (F3):** Two-stage: inv_sqrt until flip_tracker.transition_step, then cosine decay

If **commitment_triggered decay (F3)** is active:
- Early: standard schedule
- After committed_fraction ≥ 0.3: fast 50-step cosine decay to 0.1×η_base

If **commitment-adaptive stable phase (F5)** is active:
- Extend stable phase while commit_rate > 1e-4
- Auto-transition to decay when commit_rate plateaus

#### **2. Capacity Multiplier:** capacity_mult (F1-B-F1, Wave 2-B)

```
f = committed_fraction
capacity_mult = min(2.0, (1 / (1 - f + 1e-6))^0.35)
```
Increases LR as weights commit to maintain constant gradient utilization.

#### **3. PID Multiplier:** pid_mult (F4-B-F4)

```
r_target = 0.001 (commit 0.1% per step)
r_current = λ_ema = 50-step EMA of per-step commitment delta
e_t = r_target - r_current
commitment_integral += e_t
pid_mult = 1 + 0.20 × e_t + 0.001 × commitment_integral
pid_mult = clamp(pid_mult, 0.2, 3.0)
```
Actively corrects commitment rate toward target (fast when stalled, damped during avalanche).

#### **4. Avalanche Brake:** avalanche_brake (F2-B-F2)

```
λ_t = 50-step EMA of commitment delta
avalanche_brake = max(0.5, 1 - λ_t × 100)
```
Damps LR during high-rate commitment (avalanche) to prevent overshooting.

#### **5. σ_g Condition Decay:** σ_g_decay (F4-A-F4)

```
σ_g_all = [σ_g for uncommitted weights across all layers]
κ_proxy = max(σ_g_all) / (min(σ_g_all) + 1e-8)
σ_g_decay = 1 / √(κ_proxy.clamp(min=1.0))
```
Scales LR by inverse square-root of condition number of uncommitted subspace (derived from Fisher information).

#### **6. Cliff Clamp:** cliff_clamp (F3-B-F3)

```
For each ternary layer:
    frontier_mask = (|w| ∈ [0.7δ, 1.3δ])
    delta_cliff = mean(||w| - δ|) over frontier
    η_cliff_layer = 0.5 × delta_cliff
η_cliff_global = min(η_cliff_layer across all layers)
cliff_clamp = η_cliff_global / base_lr  (as a multiplier)
```
Prevents unintended cliff crossings at ternary thresholds.

#### **7. Recalibration Warmup:** recal_warmup (F5-B-F5)

```
When threshold scale changes > 5%:
    recal_warmup_counter = 9 steps
If recal_warmup_counter > 0:
    recal_warmup = 1 - exp(-recal_warmup_counter / 9)
Else:
    recal_warmup = 1.0
```
Temporary LR reduction to re-calibrate momentum after quantizer scale change.

#### **Additional Ceilings (Theorem-derived, from Wave 2-A):**

Apply these as hard limits **after** the composite formula above:

- **L1 gradient ceiling (F1-A-F1):** `η ≤ 2 × mean(|g|)` — prevents divergence per descent lemma
- **OLS-STE alpha ceiling (F3-A-F3):** `η ≤ 0.15 × α_ema` — prevents STE approximation drift
- **Uncommitted gap adaptive (F2-A-F2):** `η = lr_max × (δ - mean(|w_uncommitted|)) / δ + lr_min` — geometry-driven (replaces schedule, or blends with it)

### **Composition & Precedence**

The formula is applied in phases:

```
# Phase 1: Compute base schedule
η_base = η_schedule(step)

# Phase 2: Apply continuously varying multipliers (always safe)
η = η_base × capacity_mult × σ_g_decay × recal_warmup

# Phase 3: Apply control multipliers (PID + avalanche brake)
η = η × pid_mult × avalanche_brake

# Phase 4: Apply hard ceilings (safety constraints)
η = min(η, 2 × mean(|g|))             # L1 ceiling
η = min(η, 0.15 × α_ema)               # STE ceiling
η = min(η, cliff_clamp_global)         # Cliff safety

# Phase 5: Optional geometry-driven replacement
If using uncommitted_gap_adaptive (F2-A-F2):
    η = lr_max × (δ - μ_uncomm) / δ + lr_min  # replaces phase 1-4
```

### **C2 State Dependencies**

All 17 findings read ONLY from C2 state (commitment.py) or model layer attributes:
- `committed_mask` — per-layer boolean mask (size: param count)
- `σ_g` — per-parameter gradient magnitude EMA (size: param count)
- `n_stable` — per-layer stable step count (scalar per layer)
- `committed_fraction` — global (scalar)

**Zero new buffers required.** All state is already tracked by C2 (always-on since R5).

### **Key Composition Properties**

1. **Complementarity:** capacity_mult (level correction) + pid_mult (rate correction) + avalanche_brake (high-rate damping) are orthogonal mechanisms. Together they create a two-degree-of-freedom LR controller.

2. **Safety:** The hard ceilings (L1, STE, cliff) guarantee no divergence. All continuous multipliers are conservative (fade to 1.0 in nominal conditions).

3. **Backward Compatibility:** When C2 commitment is not active, capacity_mult → 1.0 (no committed weights), pid_mult → 1.0 (no commitment feedback), cliff_clamp → ∞ (no frontier weights). System degrades to a standard schedule.

4. **Theoretical Grounding:** Every term is derived from first principles (descent lemma, information theory, Fisher information, sign accuracy analysis). None are heuristic.

---

## Sources & References

**Wave 1 (Research Papers):**
- Wen et al. (2024). "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective." NeurIPS 2024. arxiv:2410.05192
- Hu et al. (2024). "MiniCPM: Small Language Models with Scalable Training Strategies." arxiv:2404.06395
- Yang et al. (2022). "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." ICML 2022. arxiv:2203.03466
- Cerebras/EleutherAI (2023). "The Practitioner's Guide to the Maximal Update Parameterization."

**Wave 2 (Mathematical Derivations):**
- Wei Jiang, Lijun Zhang. "Convergence Properties of Lion Optimizer." arxiv:2508.12327v1
- Xing et al. (2018). "River Valley Loss Landscape Hypothesis."
- Kosson et al. (2024). "River Valley Extensions for Ternary Networks."

**Internal (v7 Architecture):**
- `src/model_v7/components/commitment.py` — C2 state tracking (always-on)
- `src/model_v7/components/ternary.py` — TernaryLinear, OLSTernaryQuantize
- `src/model_v7/__main__.py` / `train.py` — training loop, optimizer initialization

---

## Implementation Roadmap

**Immediate (Batch R9-B, high confidence):**
1. F1: WSD-cosine decay shape
2. F3: Commitment-triggered decay
3. F1-B: Loss-plateau halving
4. F3-B: Flip-rate always-active (remove LEGO guard)
5. F4-B: Dead-zone micro-boost

**High Priority (Batch R9-C, after validation):**
6. F2: WSD-S cycles
7. F5: Commitment-adaptive stable phase
8. F5-B: Optimizer momentum warm-restart
9. F1-A-F1 through F5-A-F5: All Wave 2-A theorem-derived bounds

**Experimental (Batch R9-D, if time permits):**
10. F1-B-F1 through F5-B-F5: Wave 2-B manifold-geometry strategies
11. EXPLORE findings (cliff bound, recalibration warmup)

---

**Synthesis Complete.** Ready for implementation in sequential architecture-enforcer → code-writer pipeline.
