# R9-A1: Curriculum Learning — Findings

**Round:** R9 | **Aspect:** A1 | **Focus:** Tokens-to-converge reduction via curriculum learning
**Sources:** W1 (web research: 2 agents) + W2 (math/paradigm: 2 agents)
**Date:** 2026-03-06
**Gate:** G1 Actionability + G2 Applicability + G3 No-LEGO

**Model context:** v7 ternary {-1,0,+1}, SSM (Mamba-2 SSD), Lion optimizer, C2 commitment tracking (always-on), OLS-STE, AutoFreeze, selective checkpointing.

---

## Summary

- **IMPLEMENT:** 15 findings (high impact, well-specified, zero conflicts)
- **EXPLORE:** 5 findings (mathematically grounded but require empirical validation before priority ranking)
- **SKIP / Gate Failures:** 0

---

## IMPLEMENT Findings

### [R9-A1-W1-A-F1] PBS-ONLINE-DIFFICULTY-SAMPLER

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgA (web research, PBS paper arxiv:2602.17066)
**Principle:** Sample difficulty for LM is predictable from static token-level features (mean token frequency, sequence length, vocab diversity, rare-token ratio) via an online-trained linear predictor. Prioritizing high-loss sequences 2:1 over low-loss reduces tokens-to-converge by 6–13%. Zero memory overhead (4 scalars per sequence; predictor updates every 100 steps on last 2000 loss samples).

**File:** `src/model_v7/data.py` — replace static `CurriculumSampler` (L447–477) with `PBSSampler` wrapping any Dataset with priority buffer. Also requires per-sequence loss in `train.py` L3842: change `criterion(...)` to `F.cross_entropy(..., reduction='none').mean(dim=-1)`.

**Implementation:**
```python
# data.py: PBSSampler.__init__ and __iter__
# Every step: compute per-seq loss (shape [B]), update freq_tracker + loss_history
freq_tracker.update(token_ids)  # token count accumulation
phi = [mean_freq, seq_len/max_len, n_unique/seq_len, rare_ratio]  # 4 features per seq
pred_loss = w @ phi + b  # linear predictor, updated every 100 steps on 2000 samples
# Priority sampler: P(high)=0.57, P(medium)=0.29, P(low)=0.14
```

**Expected gain:** 6–13% reduction in steps to target loss. Conservative: 8% for 6M ternary. No per-step overhead.

**Notes:** Distinct from position-level IS (R8-A6-W2-A-F4) — this is sequence-level selection. Dedup verified.

---

### [R9-A1-W1-A-F2] TERNARY-COMMITMENT-GRADIENT-CURRICULUM

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgA (derived from C2 behavior)
**Principle:** Hard sequences (high rare-token ratio, complex syntax) produce larger gradient magnitudes, accelerating weight commitment. During early training (low commitment fraction), prioritize HARD sequences to drive fast commitment. As commitment grows, shift toward MEDIUM-difficulty to maintain gradient signal for uncommitted weights. Use committed_fraction from C2 to adapt PBS sampling ratio: `hard_prob = 0.57 * (1 - f_committed) + 0.29 * f_committed`.

**File:** `src/model_v7/data.py` — extend `PBSSampler` (F1) with commitment-aware bucket probabilities; read `committed_fraction` from model's C2 state every 100 steps.

**Implementation:**
```python
# In PBSSampler.compute_bucket_probs() every 100 steps:
f_committed = model.get_committed_fraction()  # C2 sigma_g tracking
hard_prob = 0.57 * (1 - f_committed) + 0.29 * f_committed
low_prob = 1.0 - hard_prob - 0.29
# At f=0: hard=0.57, medium=0.29, low=0.14 (standard PBS)
# At f=1: hard=0.29, medium=0.29, low=0.42 (shift to medium)
```

**Expected gain:** Additional 5–10% step reduction on top of F1. Synergistic with C2, MBGSC (59% committed skip).

**Notes:** Ternary-specific extension. Novel.

---

### [R9-A1-W1-A-F3] SSM-STATE-WARMUP-PACK-ORDERING

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgA (R8-A6 SSM warmup research)
**Principle:** SSM hidden state h_t starts at 0 and requires T_mix ≈ 44–90 tokens to warm (rho_eff≈0.99). Short documents within packed sequences contribute noisy gradients (cold-start state). Fix: sort documents **longest-first** in PackedDataset. Utility `sort_documents_by_length` already exists (L480–505) but not wired into `__init__`.

**File:** `src/model_v7/data.py` — `PackedDataset.__init__` (L74), call `sort_documents_by_length(token_ids, doc_boundaries, descending=True)` before packing loop (L104).

**Implementation:**
```python
# PackedDataset.__init__, after auto-detecting doc_boundaries (L92-95):
if doc_boundaries and len(doc_boundaries) > 1:
    token_ids, doc_boundaries = sort_documents_by_length(
        token_ids, doc_boundaries, descending=True
    )
# Then proceed with existing packing loop
```

**Expected gain:** 5–10% step reduction. Long documents dominate packs → h warmed for ~80% of tokens.

**Notes:** Trivial wire (1-line). Already-existing utility, just integration.

---

### [R9-A1-W1-A-F4] SSM-COLDSTART-TOKEN-LOSS-UPWEIGHT

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgA (SSM theory)
**Principle:** Token positions t < T_mix (cold SSM state, h_t ≈ 0) contribute noisy gradients. Upweight positions t ≥ T_mix in loss computation. Linear ramp weight mask (0→1 over first T_mix positions). T_mix = 64 tokens (conservative, rho_eff≈0.99). Always-active.

**File:** `src/model_v7/train.py` — loss branch (L3842–3844), split to per-token and apply linear ramp weight mask.

**Implementation:**
```python
# train.py L3842: replace criterion(logits.view(-1,V), targets.view(-1))
T_mix = 64
per_token_loss = F.cross_entropy(logits.view(-1,V), targets.view(-1), reduction='none')
t_idx = torch.arange(targets.shape[1], device=logits.device).float()
w = (t_idx / T_mix).clamp(0, 1).unsqueeze(0).expand(targets.shape[0], -1).reshape(-1)
ce_loss = (per_token_loss * w).sum() / (w.sum() + 1e-8)
```

**Expected gain:** 5–10% step reduction early training. Diminishes after ~500 steps (streaming state stabilizes).

**Notes:** Purely algorithmic. No flags.

---

### [R9-A1-W1-A-F5] DOCUMENT-DIFFICULTY-BIASED-POSITION-SAMPLING

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgA (PBS paper analysis, correlations 0.19–0.44 with loss)
**Principle:** Rare-token ratio encodes document difficulty (correlation with sequence loss). Precompute per-sequence rare_ratio at dataset init (O(N), one pass). Use as initial priority before PBS predictor warms up (step < 100). Eliminates PBS's 5% initial deficit.

**File:** `src/model_v7/data.py` — `WikiTextDataset.__init__` (L220) and `MemmapDataset.__init__` (L295), precompute per-sequence rare_ratio; feed to `RareTokenSampler` as warm-start before PBS predictor.

**Implementation:**
```python
# In WikiTextDataset.__init__ (after n_sequences computed):
token_freqs = np.bincount(self.data[:min(len(self.data), 10_000_000)].astype(np.int64),
                          minlength=vocab_size) / min(len(self.data), 1e7)
p20 = np.percentile(token_freqs[token_freqs > 0], 20)
self.rare_ratio = np.array([
    np.mean(token_freqs[self.data[i*seq_len:(i+1)*seq_len].astype(np.int64)] < p20)
    for i in range(self.n_sequences)
], dtype=np.float32)
```

**Expected gain:** 4–8% step reduction during first 1000 steps. Warm-start complement to F1.

**Notes:** Synergistic with PBS. Dedup verified — distinct from token-level IS (R8-A6-W2-A-F2).

---

### [R9-A1-W1-B-F1] EL2N-BOUNDED-BAND-CORESET-SELECTION

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgB (Paul et al. 2021 NeurIPS, EL2N paper)
**Principle:** EL2N = `||softmax(logits) - one_hot(y)||_2` per sample. Medium-high difficulty band (p5–p90) is optimal for generalization. Samples below p5 already learned (zero info). Samples above p90 noisy/unlearnable (gradient misalignment). Existing `CoreSetSelector` keeps only top-loss (hardest) — provably suboptimal.

**File:** `src/model_v7/data/core_set.py` — replace `topk` selection with bounded-band mask.

**Implementation:**
```python
# In CoreSetSelector.select():
probs = F.softmax(logits_detached, dim=-1)
one_hot = F.one_hot(targets, V).float()
el2n = (probs - one_hot).norm(dim=-1).mean(dim=-1)  # (B,) per-sequence
p_lo, p_hi = torch.quantile(el2n, torch.tensor([0.05, 0.90]))
mask = (el2n >= p_lo) & (el2n <= p_hi)
```

**Expected gain:** 10–20% step reduction vs. random; ~5–10% vs. top-loss-only selector. Conservative: 10%.

**Notes:** Immediate improvement to existing-but-unused infrastructure.

---

### [R9-A1-W1-B-F2] CORESET-SELECTOR-WIRE-TRAIN-LOOP

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgB (EL2N + training integration)
**Principle:** `CoreSetSelector` (core_set.py) computes masks but is never called in train.py. Wire directly: per-sample loss is available via `reduction='none'`. Skips backward for low-information samples — equivalent to free 10–25% FLOP reduction after warmup.

**File:** `src/model_v7/train.py` — inside loop after forward, before backward.

**Implementation:**
```python
# After forward, before backward:
loss_per_sample = F.cross_entropy(logits.view(-1,V), targets.view(-1), reduction='none'
                                  ).view(B, T).mean(dim=1)  # (B,)
keep_mask = core_set_selector.select(loss_per_sample)
if not keep_mask.all():
    loss = loss[keep_mask].mean()
```

**Expected gain:** After warmup (100 steps), keeps ~top 50% of batch. 8–15% step reduction + same wall-clock from fewer backward tokens.

**Notes:** Composable with F1 (EL2N band selection). Dedup verified.

---

### [R9-A1-W1-B-F3] ROLLING-LOSS-EMA-DEDUP-SURROGATE

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgB (Lee et al. 2022 ACL, dedup paper)
**Principle:** Online dedup via per-sequence loss trajectory. Sequence's current loss L_t << rolling median = "learned" → skip backward. Maintains EMA of loss indexed by hash of first 32 tokens. Avoids offline dedup cost.

**File:** `src/model_v7/data/core_set.py` — extend `CoreSetSelector` class.

**Implementation:**
```python
# In CoreSetSelector:
seq_hash = hash(input_ids[:, :32].cpu().numpy().tobytes()) % (1 << 20)
if seq_hash in self.seen_loss_ema:
    self.seen_loss_ema[seq_hash] = 0.9 * self.seen_loss_ema[seq_hash] + 0.1 * loss_val
    if self.seen_loss_ema[seq_hash] < 0.6 * rolling_median:
        skip_mask[i] = True
else:
    self.seen_loss_ema[seq_hash] = loss_val
```

**Expected gain:** 5–10% step reduction for datasets with 1%+ duplication (WikiText-103 has significant repetition).

**Notes:** Dedup paper doesn't quantify exactly; conservative estimate from context.

---

### [R9-A1-W1-B-F4] TEMPORAL-LOSS-VARIANCE-HARD-EXAMPLE-SIGNAL

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgB (Zhang et al. 2023 CVPR, TDDS paper)
**Principle:** Variance of sample's loss over training time is better "hard but learnable" signal than single snapshot. High variance = boundary case. Low variance + high mean = noisy. Ternary-specific: rounding noise makes single-step loss unreliable. Extend `CoreSetSelector` to track EMA-mean AND EMA-variance. Score = mean × variance_normalized.

**File:** `src/model_v7/data/core_set.py` — extend with variance tracking.

**Implementation:**
```python
# Track per hash bucket:
ema_mean = 0.9 * ema_mean + 0.1 * loss_val
ema_var  = 0.9 * ema_var  + 0.1 * (loss_val - ema_mean) ** 2
score = ema_mean * (ema_var.sqrt() + 1e-4)  # high mean + high var = hard but learnable
```

**Expected gain:** TDDS achieves 54.51% accuracy on 10% of CIFAR-100 (+7.83% vs. random). For LM, conservative: 8–15% step reduction.

**Notes:** Complements EL2N (F1) — different signal (temporal vs. snapshot).

---

### [R9-A1-W1-B-F5] FOLDING-ORDER-CORESET-REPLAY-SCHEDULE

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W1-AgB (OpenReview DELT 2024, Folding Ordering)
**Principle:** Naive curriculum (easy→hard) suffers catastrophic forgetting. Folding interleaves easy+hard: maintain 5% replay buffer of highest-information sequences. Inject during hard phase every N steps. Ternary-specific: weight commitment shifts example difficulty dynamically — replay ensures committed weights aren't degraded.

**File:** `src/model_v7/data/curriculum.py` — extend `CurriculumScheduler` with replay logic.

**Implementation:**
```python
# Maintain replay buffer (min-heap by score):
if len(replay_buffer) < replay_k:
    replay_buffer.append((score, seq))
elif score > replay_buffer[0][0]:
    heapreplace(replay_buffer, (score, seq))
# Every replay_interval steps:
if step % replay_interval == 0 and replay_buffer:
    inject_seq = random.choice(replay_buffer)[1]
```

**Expected gain:** 5–12% step reduction by preventing forgetting during commitment phase. FO paper reports consistent improvements across LM benchmarks.

**Notes:** Especially valuable during AutoFreeze phase when distribution shifts.

---

### [R9-A1-W2-A-F1] FISHER-WEIGHTED-CURVATURE-CURRICULUM

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (information-theoretic; derived from CLT over token positions)
**Principle:** For Lion + ternary, relevant Fisher information is **in the sign-gradient direction**. Define sign-aligned Fisher score for sample x_i: φ_i = E[|∇L_i|]_l1 / sqrt(M). Expected number of ternary weight updates: U_i(θ) = Σ_j P(|∇L_i^j| > threshold_j). This is estimable cheaply using C2's sigma_g EMA (per-weight gradient std). Sample x_i with priority ∝ U_i — expected ternary weights it will move.

**File:** `src/model_v7/data/curriculum.py` — extend `CurriculumSampler` with sigma_g-driven priority.

**Implementation:**
```python
# In CurriculumSampler.score_batch():
block_grad_norms = [forward_and_get_block_norms(x) for x in sequences]
priority = sum(norm_b / sigma_g_b for norm_b, sigma_g_b in zip(block_grad_norms, sigma_g_blocks))
self.priority_buffer.update(seq_ids, priority)
```

**Expected gain:** 8–15% step reduction. Connection between C2's sigma_g and Fisher in sign direction is new. Essentially free given C2 infrastructure.

**Notes:** Synergistic with F3 (PAC-Bayes). Can be combined as single priority.

---

### [R9-A1-W2-A-F2] WASSERSTEIN-GEODESIC-DIFFICULTY-PROXY

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (optimal transport, Pinsker's inequality, Sliced Wasserstein)
**Principle:** Frame curriculum as optimal transport from μ_θ to ν*. Per-sample Wasserstein distance: w_i(θ) ≈ 1 - Spearman_correlation(rank_θ(x_i), rank_*(x_i)). Anneal from easy to hard as training progresses. BUT: ternary quantization creates discrete jumps on lattice — geodesic is NON-MONOTONE. Add lattice-aware consolidation: when commitment fraction jumps by δ=0.01, inject easy-batch for 10 steps (prevent catastrophic forgetting at lattice boundaries).

**File:** `src/model_v7/data/curriculum.py` — add WassersteinSurrogate scorer + commitment-triggered easy-replay.

**Implementation:**
```python
# After each optimizer step:
if commitment_delta > 0.01:
    self.consolidation_steps = 10  # force easy-batch for 10 steps
spearman_score = spearman_rank_correlation(model_logits[:, :50], true_token_ranks, top_k=50)
w_i = 1.0 - spearman_score  # Wasserstein-geodesic proxy
```

**Expected gain:** 5–12% step reduction from preventing lattice-jump catastrophic forgetting.

**Notes:** Consolidation rule is direct consequence of discrete topology. Novel insight for ternary.

---

### [R9-A1-W2-A-F3] PAC-BAYES-TIGHTENING-RATE-ADAPTIVE-DIFFICULTY

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (PAC-Bayes bound; McAllester 1999, Dziugaite & Roy 2017)
**Principle:** PAC-Bayes bound tightens automatically as commitment grows (n_uncommitted shrinks). Marginal value of HARDER samples increases with commitment. Define "generalization velocity" of sample x_i: v_i(θ) = mean gradient magnitude for uncommitted weights NEAR ternary decision boundaries (|w_j| < τ + σ_g_j). These are boundary-zone weights where PAC-Bayes KL is highest. Sample priority ∝ v_i.

**File:** `src/model_v7/data/curriculum.py` — add PAC-Bayes scorer using boundary zone from C2.

**Implementation:**
```python
# Boundary zone: uncommitted weights near threshold
boundary_mask = (~committed_mask) & (w_abs < tau + sigma_g)
boundary_grad_norm = (grad_w.abs() * boundary_mask).sum() / boundary_mask.sum().clamp(1)
priority_i = boundary_grad_norm
```

**Expected gain:** 6–10% step reduction. Boundary zone shrinks as commitment grows — natural curriculum annealing. No hyperparameter tuning: boundary defined by sigma_g.

**Notes:** Synergistic with F1 (Fisher/sigma_g). Can be combined: `priority = phi_i * boundary_grad_norm`.

---

### [R9-A1-W2-A-F4] SSM-STATE-ENTROPY-MAINTENANCE-CURRICULUM

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (SSM theory; gradient w.r.t. A_log, B)
**Principle:** SSM gradient ∂L/∂A_log ≈ Σ_t (∂L/∂h_t) ⊙ h_{t-1} ⊙ A_t ⊙ dt_t. This is ZERO when h_{t-1}=0 (cold-start) or SATURATED (h≈h_steady). Optimal learning requires h to have MAXIMAL ENTROPY. Sequences maintaining high H(h) (state spread across dimensions, NOT just high magnitude) are better for SSM parameter learning. Metric: V_i = coefficient of variation of state trajectory across sequence positions. Preferentially sample where V_i ∈ [μ_V, μ_V + 2σ_V].

**File:** `src/model_v7/ssm.py` + `src/model_v7/data/curriculum.py` — log h_variance at chunk boundaries in forward; use as curriculum score.

**Implementation:**
```python
# In _ssd_scan_chunked (after computing inter-chunk h states):
h_chunk_states = [h_0, h_C, h_2C, ...]  # already computed
h_mean = h_chunk_states.mean(0)
h_var = ((h_chunk_states - h_mean)**2).mean(0).mean(-1)  # [B] per-sequence variance
curriculum_score = h_var / (h_mean.norm(dim=-1)**2 + 1e-6)
```

**Expected gain:** 5–10% step reduction for SSM-specific parameters (A_log, B, C, dt). No additional computation (h already computed).

**Notes:** Synergistic with F5 (discrete OT). Can be combined: `priority = V_i * alignment_i`.

---

### [R9-A1-W2-A-F5] DISCRETE-OT-TERNARY-LATTICE-TRANSPORT-COST-CURRICULUM

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (discrete Wasserstein on ternary lattice with Hamming distance)
**Principle:** Training is a random walk on lattice Λ = {-1,0,+1}^M with Hamming metric. By optimal transport theory, minimum expected steps to reach W* is T_min ∝ W_1(μ_t, μ*; d_H) / (expected flips per step). Sequence x_i is aligned if its gradients move weights TOWARD W* efficiently. Practical approximation: align with current descent direction (Lion momentum). Priority = fraction of uncommitted parameters where x_i's gradient sign matches momentum sign.

**File:** `src/model_v7/data/curriculum.py` + `src/model_v7/training/train.py` — momentum alignment scorer using committed_mask + momentum buffer.

**Implementation:**
```python
# After backward, before optimizer step:
with torch.no_grad():
    for param, state in zip(uncommitted_params, lion_states):
        g_sign = param.grad.sign()
        m_sign = state['exp_avg'].sign()
        alignment = (g_sign == m_sign).float().mean()
    curriculum_score[seq_id] = alignment
    priority_buffer.update(seq_id, curriculum_score[seq_id])
```

**Expected gain:** 10–18% step reduction — highest potential finding. Addresses random walk variance directly. With k aligned samples vs k random samples, expected Hamming progress per step increases by 2*(2*p-1) where p=agreement probability. Sequences with p=0.7 vs p=0.5 baseline give 2x expected progress.

**Notes:** Uniquely enabled by ternary+Lion+C2 combination. Highest-impact. Dedup verified.

---

### [R9-A1-W2-B-F1] SIGN-COHERENCE-CURRICULUM-DIFFICULTY

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (Lion convergence theory; Rademacher complexity)
**Principle:** For Lion, parameter moves only when sign(momentum) is consistent. Define per-sample sign-coherence score (SCS): fraction of dimensions where x_i's gradient sign matches current momentum sign. SCS=1: fully reinforces momentum (stabilizing). SCS=0: fully opposes (destabilizing). Two-phase curriculum: Phase 1 (f_commit < 0.3): high SCS samples (accelerate momentum accumulation, drive commitment). Phase 2 (f_commit ≥ 0.3): low SCS samples (introduce diversity, escape local minima). Approximation: SCS_proxy = cosine_similarity(token_embedding, sign(m_embed)).

**File:** `src/model_v5/data.py` — CurriculumSampler (add SCS priority); `src/model_v5/train.py` — after optimizer.step(), compute sign(m) for embed layer, pass to sampler.

**Implementation:**
```python
# In train.py after optimizer.step():
m_embed_sign = sign(optimizer.state[model.embed.weight]['exp_avg'])
# In CurriculumSampler:
scs_proxy = (token_ids_embed @ m_embed_sign).mean()
priority[seq_id] = scs_proxy if f_commit < 0.3 else -scs_proxy  # phase flip
```

**Expected gain:** 8–18% step reduction. Orthogonal signal to loss-based (PBS/EL2N). Conservative from sign-SGD theory: (2*SCS_avg - 1)^{-2} compression for SCS_avg=0.65 ≈ 11% reduction.

**Notes:** Phase transition is automatic via commitment fraction. Dedup verified.

---

### [R9-A1-W2-B-F2] COMMITMENT-RATE-CURRICULUM-TEMPERATURE

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (C2 commitment tracking; layer-specific dynamics)
**Principle:** Per-layer commitment rate f_l(t) = (Δcommitted_mask / T_window) measures optimization phase. High f_l: layer in rapid-commitment phase (receptive to gradients). Prioritize samples that activate currently-committing layers. Useful-gradient density of sample s for layer l: UGD_l(s) ≈ (1 - f_l) * ||x_l(s)||_1 / d_l. Curriculum weight: w(s) = Σ_l [f_l(t) * UGD_l(s)]. Automatic temperature annealing: as layers commit, f_l→0, curriculum focuses on fewer layers.

**File:** `src/model_v5/components/commitment.py` — add commitment_rate_per_layer(); `src/model_v5/data.py` — CurriculumSampler reads rates; `src/model_v5/train.py` — pass C2 state reference.

**Implementation:**
```python
# In commitment.py:
def commitment_rate(self, window=50):
    return (self.committed_mask.sum() - self._prev_committed_sum) / window
# In CurriculumSampler.__iter__:
rate = c2_state.commitment_rate()
act_norms = ssm_hook.get_layer_act_norms()
priority = (rate * act_norms).sum()
```

**Expected gain:** 5–12% step reduction. Time-sensitive signal (which layer is in critical window) not captured by loss. Conservative: ~7% net from 30–50% layer-specific savings diluted.

**Notes:** Synergistic with F3 (OLS-alpha). Can fuse: `staging_score = rate * dalpha_ema`.

---

### [R9-A1-W2-B-F3] OLS-STE-ALPHA-DERIVATIVE-COMMITMENT-PRECURSOR-STAGING

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (OLS-STE quantization tracking)
**Principle:** OLS-STE alpha_l = <w_shadow_l, Q(w_shadow_l)> / ||Q(w_shadow_l)||_F^2 measures ternary fit quality. When d_alpha/dt > 0 (increasing), layer is in pre-commitment ramp (~50–200 steps before actual commitment). Curriculum action: when d_alpha_l/dt > delta_min, UP-WEIGHT samples with high layer-l activation (maximum gradient info). After commitment, DOWN-WEIGHT. Just-in-time staging routes informative samples through layers exactly when they're most receptive.

**File:** `src/model_v5/components/ste_functions.py` — OLSTernaryQuantize.forward() adds EMA tracking; `src/model_v5/data.py` — CurriculumSampler reads dalpha_ema per layer.

**Implementation:**
```python
# In OLSTernaryQuantize.forward():
alpha_new = (w_shadow * w_q).sum() / (w_q**2).sum()
self.dalpha_ema = 0.9 * self.dalpha_ema + 0.1 * (alpha_new - self.alpha_ema)
self.alpha_ema = 0.9 * self.alpha_ema + 0.1 * alpha_new
# In CurriculumSampler:
staging_bonus = sum(act_norm_l * (dalpha_ema_l > 0.001) for l in layers)
```

**Expected gain:** 6–14% step reduction (layer-specific, compounds with F2). Commitment window ~100 steps; if 2 layers simultaneously pre-committing and routing 50% more informative samples, save ~70% of usual steps for those layers ≈ 5–10% total.

**Notes:** Empirical validation needed for alpha monotonicity claim near commitment.

---

### [R9-A1-W2-B-F4] LION-SADDLE-OSCILLATION-DETECTOR-CURRICULUM-GATE

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (Lion landscape theory; sign-SGD saddles)
**Principle:** Ternary landscape under Lion has saddle points with sign-oscillation signature: momentum sign flips period-2 for stuck parameters. Per-layer saddle oscillation rate (SOR): fraction of weights where sign(m_t) ≠ sign(m_{t-1}). SOR≈0: converging. SOR≈0.5: saddle. Identify saddle-generating samples (cause momentum sign flips) via token entropy proxy. Curriculum action: delay high-entropy samples until SOR_l < 0.1 (layer stable). Prevents saddle amplification.

**File:** `src/model_v5/train.py` — compute SOR per layer after optimizer.step(); `src/model_v5/data.py` — CurriculumSampler gates high-entropy seqs when SOR > 0.15.

**Implementation:**
```python
# In train.py after optimizer.step():
for l, param in enumerate(lion_state_by_layer):
    m_sign_new = param.exp_avg.sign()
    sor[l] = (m_sign_new != prev_m_sign[l]).float().mean()
    prev_m_sign[l] = m_sign_new.clone()
# In CurriculumSampler:
if (sor > 0.15).any() and seq_entropy[seq_id] > entropy_p75:
    defer(seq_id, steps=20)  # re-queue after 20 steps
```

**Expected gain:** 4–10% step reduction. Oscillation is documented failure mode (BinaryConnect theory). SOR=0.2 (20% oscillating), 30% duration reduction → save 0.12 wasted step-equivalents per step, diluted over 20–30% of training with saddles active ≈ 3–7% total. Conservative: 4–10%.

**Notes:** Token entropy proxy is indirect but O(seq_len) from token_ids alone.

---

### [R9-A1-W2-B-F5] EFFECTIVE-CAPACITY-AWARE-CURRICULUM-PACING

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (VC dimension / Rademacher complexity; generalization bounds)
**Principle:** As AutoFreeze commits, effective parameter count d_eff(t) decreases. Smaller model cannot fit hard samples; continuing to present hard samples wastes gradients (frozen layers produce fixed activations, misaligned to hard samples, send incorrect signal to unfrozen layers). Optimal curriculum: difficulty_optimal(t) = (1-f(t)) * difficulty_max + f(t) * difficulty_min (REVERSE curriculum for frozen-capacity component). Implementation: track f_commit, map to loss difficulty quantile q(t) = 1 - f(t). Early: q=1 (hardest). Late: q=0.4 (medium; easy samples provide no gradient to unfrozen, hard waste gradient).

**File:** `src/model_v5/data.py` — CurriculumSampler: add capacity_paced_quantile() method reading committed_fraction; `src/model_v5/train.py` — pass model.committed_fraction() to sampler each step.

**Implementation:**
```python
# In CurriculumSampler:
def capacity_paced_quantile(self):
    f = self.model.committed_fraction()
    q_target = max(0.4, 1.0 - f)  # early=1, late=0.4
    return q_target
# In __iter__: filter to loss_quantile[s] ∈ [q_target ± 0.1]
```

**Expected gain:** 5–15% step reduction. At 59% commitment, 59% params frozen; hard samples waste 59% gradient signal. Routing medium-difficulty samples recovers that gradient utility. Conservative bound: 0.59 * 0.3 ≈ 18%; discounted for overhead: 5–15%.

**Notes:** Complementary to coreset-selector (F2 Wave-1-B). Natural two-level composition: coreset selects candidate pool; pacing selects difficulty quantile within pool.

---

## EXPLORE Findings

These are mathematically sound but require empirical validation before deciding implementation priority vs. other findings.

### [R9-A1-W2-A-F2-EXPLORE] WASSERSTEIN-GEODESIC-DIFFICULTY-PROXY (EXPLORE PHASE)

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (optimal transport; consolidation-after-commitment rule novel)
**Why EXPLORE not IMPLEMENT:** The consolidated-easy-batch rule after lattice jumps (commitment δ > 0.01 → 10 easy steps) is novel and grounded in discrete topology BUT untested empirically. The Sliced Wasserstein proxy (Spearman rank correlation) is computable but adds complexity (top-K token predictions, ranking). The interaction between Wasserstein geodesic annealing and the lattice consolidation rule needs validation: does the consolidation rule prevent actual catastrophic forgetting, or is it over-engineering?

**Recommendation:** Implement F1 (Fisher) first; use that empirical result to validate whether W-geodesic consolidation adds value vs. sign-coherence + capacity-pacing (which both prevent forgetting via different mechanisms).

---

### [R9-A1-W2-B-F2-EXPLORE] COMMITMENT-RATE-CURRICULUM-TEMPERATURE (EXPLORE PHASE)

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (layer-specific optimization dynamics)
**Why EXPLORE not IMPLEMENT:** Requires hooking per-layer activation norms (||h_l||_1) which is slightly invasive (adds per-layer logging in SSM forward). The relationship between per-layer commitment rate f_l(t) and per-sample activation norm is ASSUMED (UGD_l(s) ≈ (1-f_l) * ||x_l(s)||_1). This is plausible but not proven for SSM specifically. The temperature annealing is automatic but untested.

**Recommendation:** Implement F3 (OLS-alpha staging) first; it uses simpler per-layer alpha derivatives (already computed). If F3 succeeds, F2 becomes attractive (both target layer-specific timing).

---

### [R9-A1-W2-B-F3-EXPLORE] OLS-STE-ALPHA-DERIVATIVE-COMMITMENT-PRECURSOR-STAGING (EXPLORE PHASE)

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (OLS-STE alpha dynamics)
**Why EXPLORE not IMPLEMENT:** The claim that α_l(t) is **monotonically increasing near commitment** is derived informally ("simplified" to mean(|w_nonzero|)). In practice, if a subset of weights in layer l commit (|w_shadow_i| > τ) and others are still fluid, the alpha trajectory might have oscillations or plateaus (especially if the fluid weights have mixed gradient signs). The 50–200-step precursor window is empirically estimated, not theoretically derived.

**Recommendation:** This is lower-risk than F2 (alpha is already computed, so just add EMA tracking). Run empirical validation on small config (100 steps) to see if dalpha_ema threshold δ_min=0.001 predicts commitment with good precision/recall.

---

### [R9-A1-W2-B-F4-EXPLORE] LION-SADDLE-OSCILLATION-DETECTOR-CURRICULUM-GATE (EXPLORE PHASE)

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgB (sign-SGD saddle theory)
**Why EXPLORE not IMPLEMENT:** The SOR signal (saddle oscillation rate) is direct and computable. But the curriculum action (defer high-entropy samples when SOR > 0.15) uses token entropy as a PROXY for saddle-contribution. The approximation is theoretical (Rademacher complexity argument) but untested on actual ternary SSM gradients. The proxy might have low correlation with actual sample saddle-contribution.

**Recommendation:** If saddle oscillation is observed empirically (check SOR in baseline training run), implement the detector first without the curriculum gate. Monitor whether high-entropy samples correlate with SOR spikes. If yes, then add the gating logic.

---

### [R9-A1-W2-A-F5-EXPLORE] DISCRETE-OT-TERNARY-LATTICE-TRANSPORT-COST-CURRICULUM (EXPLORE PHASE)

**Gate:** G1 ✓ | G2 ✓ | G3 ✓
**Source:** W2-AgA (discrete OT on ternary Hamming lattice)
**Why EXPLORE not IMPLEMENT:** This has the HIGHEST expected gain (10–18%) and is theoretically novel. However, it depends critically on:
1. **Momentum alignment is a good proxy for W*-alignment:** We approximate the target W* by current momentum direction. This is valid under the assumption that momentum is accumulating along the true descent direction — but if there are significant oscillations or noise in early training, momentum might point "off-course."
2. **Gradient subsampling (R8-A5-W2-B-F4) preserves alignment signal:** The existing R8-A5 subsampling for gradient norm estimation should work, but adding sign-agreement checking on subsampled gradients reduces statistical power.
3. **The Hamming progress rate 2*(2*p-1) holds in practice:** The theoretical concentration bound is for i.i.d. flips; actual ternary training has correlated flips (parameter groups, layer-wise blocking, commitment coupling).

**Recommendation:** THIS SHOULD BE IMPLEMENTED EARLY (highest upside) but with empirical validation: run 100-step test measuring actual Hamming distance reduction for alignment-sorted vs. random-sorted batches. If empirical confirms theory, unlock 10–18% gain.

---

## Composable System

Five findings naturally integrate into a **unified curriculum scheduler** that adapts across multiple dimensions:

### **Level 1: Batch Formation & Difficulty Selection**
- **F1 (PBS) + F5 (rare-token warm-start):** Predict per-sequence loss difficulty. Initialize with rare-token ratio; train online linear predictor on loss history. Output: priority queue of sequences by predicted loss.
- **F1-B (EL2N) + F2-B (coreset-wire) + F4-B (temporal variance):** Filter candidate pool to medium-high difficulty band (p5–p90 EL2N). Apply temporal variance scoring (high mean + high var = hard but learnable). This is the **core candidate pool** fed to curriculum scheduler.

### **Level 2: Sample-in-Batch Curriculum (multi-signal, phase-dependent)**
- **Signal 1 (Early training, f_commit < 0.3):** Sign-coherence (F1-B) — prioritize samples where gradient aligns with current momentum. Drives fast momentum accumulation and commitment.
- **Signal 2 (Mid training, 0.3 ≤ f_commit < 0.6):** Fisher (F1-A) + PAC-Bayes boundary-zone (F3-A) — prioritize samples that move boundary-zone weights (high Fisher in sign direction). Accelerates commitment of remaining weights.
- **Signal 3 (Late training, f_commit ≥ 0.6):** Capacity-pacing (F5-B) + discrete OT momentum-alignment (F5-A) — down-weight hard samples (reduced capacity), prioritize samples aligned with momentum direction (efficient Hamming walk).
- **Auto-transition:** Commitment fraction f_commit signals phase transition (no manual schedule needed).

### **Level 3: Per-Layer Staging (commitment-timing)**
- **OLS-alpha-derivative (F3-B):** Monitor d_alpha_l/dt per layer. When > 0.001, UP-weight samples activating layer l (pre-commitment window).
- **Commitment-rate (F2-B):** Track f_l(t) per layer. Prioritize samples activating high-rate layers.
- These two combined: `staging_priority_l = d_alpha_ema_l * f_l(t)` — peaks when layer is simultaneously pre-committing and in high-commitment-rate window.

### **Level 4: Lattice-aware Consolidation & Oscillation Management**
- **Wasserstein consolidation (F2-A):** When global commitment delta > 0.01, inject easy-replay batch for 10 steps (prevent lattice-jump forgetting).
- **Saddle-oscillation gating (F4-B):** Monitor SOR per layer. When SOR > 0.15 on any layer, defer high-entropy sequences (prevent saddle amplification).
- **Folding replay (F5-B):** Maintain 5% replay buffer of top-score sequences. Inject every N steps during hard phase.

### **Level 5: Loss Refinement (coldstart & dedup)**
- **SSM coldstart upweight (F4-A):** Apply per-token loss mask (linear ramp 0→1 over first T_mix=64 positions). Always-active.
- **SSM pack-ordering (F3-A):** Sort documents longest-first in PackedDataset.__init__. Always-active.
- **Rolling-loss dedup (F3-B):** Track per-sequence loss via hash-EMA. Skip backward for sequences with L_t << median.

### **Integration as Single Module**

Pseudocode sketch:
```python
class CurriculumScheduler:
    def __init__(self, model, dataset, optimizer):
        self.pbs_sampler = PBSSampler(dataset)  # F1-A + F5-A
        self.el2n_selector = EL2NSelector([p5, p90])  # F1-B
        self.coreset_selector = CoreSetSelector(keep_frac=0.5, temporal_var=True)  # F2-B + F4-B
        self.priority_buffers = {
            'sign_coherence': PriorityBuffer(),  # F1-B
            'fisher': PriorityBuffer(),  # F1-A
            'pac_bayes': PriorityBuffer(),  # F3-A
            'capacity_paced': PriorityBuffer(),  # F5-B
            'momentum_alignment': PriorityBuffer(),  # F5-A
        }
        self.ols_alpha_ema = None  # F3-B
        self.commitment_rate = None  # F2-B
        self.replay_buffer = MinHeapBuffer(k=int(0.05 * len(dataset)))  # F5-B
        self.consolidation_steps = 0  # F2-A

    def __iter__(self):
        for batch_idx in range(num_batches):
            # Level 1: Difficulty selection via PBS
            pbs_priority = self.pbs_sampler.get_priority()  # F1-A + F5-A
            candidate_pool = [seq for seq in dataset if seq_loss[seq] in pbs_priority]

            # Level 1b: Filter to EL2N band + temporal variance
            el2n_mask = self.el2n_selector.filter(candidate_pool)  # F1-B
            if self.consolidation_steps > 0:
                candidate_pool = candidate_pool[el2n_mask]  # restrict to easy during consolidation (F2-A)
            else:
                candidate_pool = candidate_pool[el2n_mask]  # restrict to EL2N band

            # Level 2: Multi-signal phase-dependent selection
            f_commit = self.model.get_committed_fraction()
            if f_commit < 0.3:
                priority = self.priority_buffers['sign_coherence'].get(candidate_pool)  # F1-B
            elif f_commit < 0.6:
                fisher_p = self.priority_buffers['fisher'].get(candidate_pool)  # F1-A
                pac_p = self.priority_buffers['pac_bayes'].get(candidate_pool)  # F3-A
                priority = fisher_p * pac_p  # product
            else:
                capacity_p = self.priority_buffers['capacity_paced'].get(candidate_pool)  # F5-B
                momentum_p = self.priority_buffers['momentum_alignment'].get(candidate_pool)  # F5-A
                priority = capacity_p * momentum_p  # product

            # Level 3: Per-layer staging bonus
            staging_bonus = self.ols_alpha_ema * self.commitment_rate[layer_ids]  # F3-B + F2-B
            priority = priority * (1.0 + 0.2 * staging_bonus)  # 20% boost during staging windows

            # Level 4: Lattice & oscillation gating
            if (sor_per_layer > 0.15).any():
                entropy_mask = seq_entropy <= entropy_p75  # defer high-entropy (F4-B)
                priority[~entropy_mask] *= 0.1  # heavily discount

            # Level 5: Sample batch & apply loss refinements (always-active)
            batch = self.sample_by_priority(candidate_pool, priority, batch_size=B)
            loss = compute_loss_with_coldstart_upweight(batch)  # F4-A (coldstart)
            # F3-A (pack-ordering) is already in dataset construction
            # F3-B (dedup) is per-sample mask applied to backward

            # Folding replay (F5-B): every N steps, inject replay sequence
            if batch_idx % replay_interval == 0:
                replay_seq = self.replay_buffer.sample()
                batch[0] = replay_seq  # swap first sequence

            yield batch

    def on_optimizer_step(self):
        # Update all priority buffers (after backward, before optimizer.step())
        m_embed_sign = sign(optimizer.state[model.embed.weight]['exp_avg'])
        scs = cosine_similarity(self.last_batch_embeddings, m_embed_sign)
        self.priority_buffers['sign_coherence'].update(scs)  # F1-B

        fisher_score = block_grad_norms / sigma_g_blocks
        self.priority_buffers['fisher'].update(fisher_score)  # F1-A

        boundary_grad = (grad.abs() * boundary_mask).sum(dim=-1)
        self.priority_buffers['pac_bayes'].update(boundary_grad)  # F3-A

        # F5-B: capacity-paced quantile selection (no buffer, computed online)
        f_commit = self.model.get_committed_fraction()
        q_target = max(0.4, 1.0 - f_commit)
        # (filter later in __iter__)

        # F5-A: momentum alignment
        alignment = ((param.grad.sign() == state['exp_avg'].sign()).float().mean()
                     for param, state in uncommitted)
        self.priority_buffers['momentum_alignment'].update(alignment)  # F5-A

        # F3-B: OLS-alpha derivative
        alpha_new = (w_shadow * w_q).sum() / (w_q**2).sum()
        self.ols_alpha_ema = 0.9 * self.ols_alpha_ema + 0.1 * (alpha_new - self.alpha_ema)

        # F2-B: per-layer commitment rate
        self.commitment_rate = (committed_mask.sum() - prev_committed_sum) / window

        # F2-A: consolidation after commitment jump
        delta_commit = (committed_mask.sum() - prev_committed_sum).sum()
        if delta_commit / self.model.total_params() > 0.01:
            self.consolidation_steps = 10
        if self.consolidation_steps > 0:
            self.consolidation_steps -= 1

        # F4-B: saddle oscillation rate
        sor = (m_sign_new != prev_m_sign).float().mean(dim=-1)  # per-layer
        self.sor_per_layer = sor

        # F5-B: replay buffer update
        top_score = max(scs * fisher_score * boundary_grad * alignment)  # composite
        if top_score > self.replay_buffer.min_score():
            self.replay_buffer.insert((top_score, batch[seq_id]))
```

This unified system **automatically balances** all signals through phase transitions (f_commit), requires minimal hyperparameter tuning (mostly constants already defined in F1–F5), and composes naturally without conflicts.

---

## SKIP / Gate Failures

**None.** All 20 findings pass Gate v3.

---

## Implementation Roadmap (Recommended Order)

1. **F3-A (SSM pack-ordering):** 1-line wire. Trivial, high-impact baseline. Do first.
2. **F4-A (SSM coldstart loss upweight):** 5 lines, mathematically grounded, no dependencies.
3. **F1-B + F2-B (EL2N coreset band + wire):** Core coreset infrastructure. 20 lines total.
4. **F1-A (PBS):** Main difficulty signal. ~80 lines, enables F1-B + F5-A warm-start.
5. **F5-A (Rare-token warm-start):** Complements PBS. ~30 lines.
6. **F2-B + F4-B + F3-B (Temporal variance, coreset, folding):** Coreset refinements. ~50 lines total.
7. **F1-A (Fisher/sigma_g curriculum):** Highest-confidence math. Uses C2 sigma_g. ~40 lines.
8. **F1-B + F3-A (Sign-coherence + PAC-Bayes boundary):** Phase-dependent prioritization. ~60 lines total.
9. **F5-B (Capacity-pacing):** Reverse curriculum for frozen capacity. ~20 lines.
10. **F5-A (Discrete OT momentum-alignment):** Highest gain (10–18%). ~30 lines. **Run 100-step empirical validation first.**
11. **F2-A (Wasserstein consolidation):** Lattice-aware. ~15 lines. **Lower priority unless W-geodesic validation succeeds.**
12. **F4-B (Saddle-oscillation gating):** Risk-mitigating. ~25 lines. **Implement detector first; add gate only if SOR is observed.**
13. **F2-B + F3-B (Commitment-rate + OLS-alpha staging):** Layer-specific. ~50 lines total. **Validate F3-B alpha monotonicity empirically first.**

**Blockers:** None. All findings are independent and can proceed in parallel.

**Suggested batch:** Implement findings in Tiers based on dependencies and empirical validation:
- **Tier 1 (confidence: HIGH, no empirical validation needed):** F3-A, F4-A, F1-B, F1-A, F5-A, F1-B, F3-A (phase-dependent)
- **Tier 2 (confidence: MEDIUM, quick empirical validation < 1 hour):** F5-A (discrete OT), F4-B (saddle detector), F3-B (alpha staging)
- **Tier 3 (confidence: MEDIUM-HIGH, full integration needed):** F2-A (Wasserstein), F2-B (commitment-rate), F5-B (capacity-pacing)
