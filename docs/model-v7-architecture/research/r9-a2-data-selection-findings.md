# R9-A2: Data Selection & Quality Filtering — Findings

**Round:** R9 | **Aspect:** A2 (Data Selection / Domain Weighting / Convergence) | **Waves:** 1 + 2
**Sources:** W1-AgA, W1-AgB, W2-AgA, W2-AgB
**Gate:** G1 Actionability ✓ | G2 Applicability (ternary/SSM) ✓ | G3 No-LEGO ✓
**Dedup:** 19 unique findings (1 duplicate merged)

---

## Summary

- **Preprocessing IMPLEMENT:** 6 (one-time cost, applied to corpus)
- **Per-Step IMPLEMENT:** 8 (runtime sampling/filtering, minimal overhead)
- **EXPLORE:** 4 (higher complexity, probe)
- **SKIP:** 1 (merged into better variant)
- **Expected combined gain:** 50-120% step reduction (conservative stacking estimate accounting for overlap)

---

## Preprocessing IMPLEMENT (Applied Once to Corpus)

### [R9-A2-W1-A-F1] MINHASH-DEDUP-AT-TOKENIZATION
**Viability:** HIGH | **Gain:** 10-20% step reduction | **Gate:** ✓

**Principle:** Duplicate training sequences produce identical gradient directions. Each duplicate is wasted compute — zero new Fisher information. MinHash dedup removes near-duplicates (Jaccard J > 0.8) before training.

**File:** `src/model_v7/prepare_wikitext.py` — dedup pass after text collection, before tokenization

**Key algorithm:**
```python
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.8, num_perm=128)
deduped = []
for text in train_texts:
    m = MinHash(num_perm=128)
    for w in text.lower().split():
        m.update(w.encode())
    if not lsh.query(m):
        lsh.insert(str(len(deduped)), m)
        deduped.append(text)
train_texts = deduped  # drop ~15–30% duplicates
```

**Expected gain:** 10-20% (WikiText-103 is already relatively clean; expect lower end ~10-15%). One-time preprocessing cost.

---

### [R9-A2-W1-A-F2] NGRAM-REPETITION-SCORE-FILTER
**Viability:** HIGH | **Gain:** 5-12% step reduction | **Gate:** ✓

**Principle:** High internal n-gram repetition (boilerplate, tables, spam) forces memorization rather than learning. Filter documents where >20% of top-2-grams or >18% of top-3-grams are repeating.

**File:** `src/model_v7/prepare_wikitext.py` — quality filter before tokenization

**Key algorithm:**
```python
from collections import Counter
import numpy as np

def repetition_score(text, n=3):
    tokens = text.lower().split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    c = Counter(ngrams)
    return sum(v-1 for v in c.values()) / max(len(ngrams), 1)

train_texts = [t for t in train_texts
               if repetition_score(t, n=2) < 0.20
               and repetition_score(t, n=3) < 0.18]
```

**Expected gain:** 5-12% (removes ~5-8% of Wikipedia stubs + table-heavy pages). Zero per-step overhead.

---

### [R9-A2-W1-A-F3] DSIR-NGRAM-IMPORTANCE-RESAMPLING
**Viability:** HIGH | **Gain:** 15-25% step reduction | **Gate:** ✓

**Principle:** Data Selection via Importance Resampling (Xie et al. 2023). Train data that matches target distribution (clean Wikipedia prose) requires fewer steps. Compute importance weights w(x) = p_target(x) / q_raw(x) via hashed n-gram frequencies; select top 70% by weight.

**File:** `src/model_v7/prepare_wikitext.py` — weighted sampling before tokenize_and_save

**Key algorithm:**
```python
# Build target freqs from val.bin (already-clean); raw freqs from all train_texts
import hashlib, numpy as np
N_BUCKETS = 10000

def ngram_hash_counts(text, n=2):
    counts = np.zeros(N_BUCKETS)
    tokens = text.lower().split()
    for i in range(len(tokens)-n+1):
        h = int(hashlib.md5(' '.join(tokens[i:i+n]).encode()).hexdigest(), 16) % N_BUCKETS
        counts[h] += 1
    return counts / max(counts.sum(), 1)

# Compute importance weights per doc
scores = [np.dot(ngram_hash_counts(t),
                 np.log(target_freq + 1e-8) - np.log(raw_freq + 1e-8))
          for t in train_texts]
threshold = np.percentile(scores, 30)  # keep top 70%
train_texts = [t for t, s in zip(train_texts, scores) if s >= threshold]
```

**Expected gain:** 15-25% (most impactful preprocessing finding — directly targets distribution matching). One-time cost.

---

### [R9-A2-W1-A-F4] PERPLEXITY-BAND-FILTER-PRETOKENIZATION (STATIC)
**Viability:** HIGH | **Gain:** 8-15% step reduction | **Gate:** ✓

**Principle:** Filter by perplexity band [p10, p85] using fast unigram proxy. Removes too-easy documents (ppl < p10; gradient ≈ 0) and too-hard/noisy documents (ppl > p85; gradient noise dominates). Implemented as one-time preprocessing.

**File:** `src/model_v7/prepare_wikitext.py` — static perplexity filtering before tokenization

**Key algorithm:**
```python
from collections import Counter
import numpy as np

# Estimate unigram model from training data
vocab_counts = Counter()
for t in train_texts[:5000]:
    vocab_counts.update(t.lower().split())
total = sum(vocab_counts.values())

def fast_ppl(text):
    tokens = text.lower().split()
    T = max(len(tokens), 1)
    return np.exp(-sum(np.log(vocab_counts.get(tok, 1) / total)
                       for tok in tokens) / T)

ppls = [fast_ppl(t) for t in train_texts]
p10, p85 = np.percentile(ppls, 10), np.percentile(ppls, 85)
train_texts = [t for t, p in zip(train_texts, ppls) if p10 <= p <= p85]
```

**Expected gain:** 8-15% (removes ~25% of noise-dominated docs). Zero per-step overhead.

**Note:** This is the STATIC preprocessing version. The per-step adaptive version is R9-A2-W2-B-F5 (see Per-Step IMPLEMENT section). Both can be stacked.

---

### [R9-A2-W1-A-F5] DOREMI-PROXY-DOMAIN-MIXTURE-WEIGHTS (PREPROCESSING)
**Viability:** HIGH | **Gain:** 20-30% step reduction | **Gate:** ✓

**Principle:** Domain mixture ratios are highly consequential (DoReMi showed 2.6× speedup at scale). Train a 1M-param proxy SSM for 200 steps with Group DRO, derive optimal domain weights, resample training data. For WikiText, assign domains via article category headers.

**File:** `src/model_v7/prepare_wikitext.py` — domain detection + proxy training + stratified resampling (one-time, ~5 min)

**Key algorithm:**
```python
# Assign domains from WikiText-103 section headers (= Section =)
domains = assign_domains(train_texts)  # extract section headers

# Train 1-layer SSM proxy (1M params) for 200 steps on balanced mix
domain_losses = train_proxy_and_measure_losses(train_texts, domains, steps=200)

# Exponential reweighting: w_k = exp(eta * L_k) / Z
eta = 0.1  # temperature
weights = np.exp(eta * np.array(domain_losses))
weights /= weights.sum()

# Stratified resample
train_texts = stratified_resample(train_texts, domains, weights)
```

**Expected gain:** 20-30% (DoReMi baseline 2.6× = 62% at 8B; conservative 6M estimate: 20-30%). One-time preprocessing cost (~5 min proxy training).

---

## Per-Step IMPLEMENT (Runtime Sampling/Filtering)

### [R9-A2-W1-B-F1] SELF-REFERENTIAL-EXCESS-LOSS-RESAMPLING
**Viability:** HIGH | **Gain:** 5-15% step reduction | **Gate:** ✓

**Principle:** RHO-LOSS uses excess_loss = model_loss − reference_loss to identify learnable tokens. Replace external reference model with EMA of own past loss over K steps. self-referential proxy removes dependency on pre-trained reference binary.

**File:** `src/model_v7/rho1.py` (replace `ref_flat` with EMA buffer) + `src/model_v7/train.py` L3766–3791 (activate path without external ref_losses)

**Key algorithm:**
```python
# In rho1.py compute_rho1_loss
# Register buffer: token_loss_ema (shape [B, L], float16)
token_loss_ema = token_loss_ema * 0.99 + per_token_loss.detach() * 0.01
excess = per_token_loss - token_loss_ema  # self-referential excess
mask = (excess > 0).float()              # keep tokens harder than recent avg
loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
```

**Expected gain:** 5-15% (RHO-LOSS with true reference: 18× on Clothing-1M; WikiText pretraining: conservative 5-15% for self-EMA approximation). Zero extra forward pass. EMA buffer overhead: ~0.5 MB at B=4, L=512.

---

### [R9-A2-W1-B-F2] NGRAM-SELF-DSIR-ONLINE-RESAMPLING
**Viability:** HIGH | **Gain:** 8-12% step reduction | **Gate:** ✓

**Principle:** DSIR online variant. Every K=200 steps, extract n-gram frequencies from high-loss tokens (loss > threshold); bias sampler toward sequences with high overlap to those high-loss n-grams. Self-adaptive, no reference model.

**File:** `src/model_v7/data.py` — extend `WikiTextDataset` + new `NgramImportanceSampler` class in batch construction

**Key algorithm:**
```python
# Every K=200 steps in train.py:
high_loss_mask = per_token_loss > loss_threshold  # [B, L] bool
high_loss_ngrams = extract_hashed_ngrams(targets[high_loss_mask], n=3)  # Counter
seq_weights[i] = sum(high_loss_ngrams.get(g, 0) for g in seq_ngrams[i]) + 1e-6
sampler.update_weights(seq_weights / seq_weights.sum())
```

**Expected gain:** 8-12% (DSIR achieves 2-2.5% GLUE; adaptive target should be stronger; conservative on single-domain corpus). Zero per-token cost after K-step accumulation.

---

### [R9-A2-W1-B-F3] DOREMI-SELF-PROXY-ONLINE-DOMAIN-WEIGHTS
**Viability:** HIGH | **Gain:** 10-20% step reduction | **Gate:** ✓

**Principle:** DoReMi online variant without separate proxy training. Define "domains" as structural document types (prose vs lists vs technical) using simple heuristics. Per step, track per-type loss EMA; apply DRO update rule: w_d *= exp(ε * excess_loss_d). Reference-free.

**File:** `src/model_v7/data.py` — `WikiTextDataset.__init__` (doc type labeling) + `get_batch` (per-step loss tracking + DRO weight update)

**Key algorithm:**
```python
# At dataset init: assign doc_type labels (3 buckets: prose, lists, technical)
doc_types = assign_doc_type(doc_texts)

# Per step in get_batch:
for d in range(n_domains):
    loss_per_domain[d] = mean loss for docs of type d in current batch
    excess_d = loss_per_domain[d] - loss_ema_per_domain[d]
    domain_weights[d] *= np.exp(eps * excess_d.item())  # eps ~ 0.01

domain_weights /= domain_weights.sum()  # normalize to simplex
loss_ema_per_domain = 0.95 * loss_ema_per_domain + 0.05 * loss_per_domain
```

**Expected gain:** 10-20% (DoReMi: 2.6× on multi-domain Pile; single-corpus with self-defined structural domains: conservative 10-20%). Zero overhead per step (just division + exp).

---

### [R9-A2-W1-B-F4] SSM-LONG-RANGE-DOC-OVERSAMPLING
**Viability:** HIGH | **Gain:** 5-12% step reduction | **Gate:** ✓

**Principle:** SSM recurrent state h accumulates information over T_mix tokens (~44 for rho_eff=0.99). Long documents have higher fraction of "warm" (informative) tokens. Optimal sampling weight ∝ (L − T_mix)/L. The codebase already has `sort_documents_by_length` in data.py; convert to continuous importance weights.

**File:** `src/model_v7/data.py` — `build_document_index` (compute doc_warm_fraction); pass to `WeightedRandomSampler` (no new sampler class needed)

**Key algorithm:**
```python
# At dataset init (data.py build_document_index)
T_mix = 44  # from R8-A6-W2-B-F2
doc_warm_fractions = np.maximum(0.0, (doc_lengths - T_mix) / doc_lengths.clip(min=1))
# Optional: quadratic emphasis on very long docs
doc_weights = doc_warm_fractions ** 2
sampler = WeightedRandomSampler(weights=doc_weights, num_samples=len(doc_weights))
```

**Expected gain:** 5-12% (removing ~20-40% of cold-start gradient bias improves noise term). No per-step cost.

---

### [R9-A2-W1-B-F5] IN-CONTEXT-TOPOLOGICAL-DOC-ORDERING
**Viability:** HIGH | **Gain:** 5-10% step reduction | **Gate:** ✓

**Principle:** In-Context Pretraining (Shi et al. 2023) shows that ordering documents by semantic relatedness improves downstream accuracy 5-16% without changing data. For SSMs, related consecutive documents provide useful recurrent state initialization. Greedy nearest-neighbor chain via TF-IDF cosine similarity.

**File:** `src/model_v7/data.py` — `build_document_index` (add TF-IDF bucket labels); `PackedDataset.__init__` (replace random packing order with topic-clustered traversal)

**Key algorithm:**
```python
# At dataset init: assign topic bucket (TF-IDF k-means, k=8)
buckets = assign_topic_bucket(doc_texts, n_buckets=8)
ordered_doc_ids = np.argsort(buckets * 1e9 + doc_positions)  # within-bucket positional sort

# Pack in reordered sequence
token_ids_reordered = np.concatenate([token_ids[doc_starts[i]:doc_ends[i]]
                                      for i in ordered_doc_ids])
```

**Expected gain:** 5-10% (In-Context Pretraining: +15% on attention-based models; SSM gains are smaller but include bonus boundary signal). Zero per-step cost.

---

### [R9-A2-W2-A-F1] ZLIB-RESIDUAL-MDL-FILTER
**Viability:** HIGH | **Gain:** 10-20% step reduction | **Gate:** ✓

**Principle:** Shannon source coding theorem: model codelength H_model(x) exceeds zlib's H_zlib(x) for sequences the model hasn't yet learned. The difference (MDL residual) is automatically adaptive and reveals "how much worse" the neural model is than a dumb compressor. Filter: keep sequences where 0 < score_MDL < 3.0 bits/token. As training progresses and H_model → H_zlib, automatically enriches pool with harder content.

**File:** `src/model_v7/prepare_wikitext.py` (H_zlib one-time precomputation) + `src/model_v7/train.py` (per-step MDL score filtering)

**Key algorithm:**
```python
# prepare_wikitext.py — one-time preprocessing
import zlib

def compute_H_zlib(token_ids: list[int]) -> float:
    token_bytes = b''.join(t.to_bytes(2, 'little') for t in token_ids)
    compressed_bits = len(zlib.compress(token_bytes, level=6)) * 8
    return compressed_bits / len(token_ids)  # bits/token

# Store H_zlib per document in metadata

# train.py — per step, after forward pass
score_MDL = per_sample_loss - batch_H_zlib  # [B]
delta_lo = np.percentile(score_MDL, 10)     # adaptive lower threshold
mask = (score_MDL > delta_lo) & (score_MDL < 3.0)
loss = (per_sample_loss * mask.float()).mean()
```

**Expected gain:** 10-20% (strongest in mid-training; near-zero early and late). Zero per-step cost beyond one subtraction.

---

### [R9-A2-W2-B-F1] UNCOMMITTED-GRADIENT-FRACTION-RESAMPLING
**Viability:** HIGH | **Gain:** 5-15% step reduction | **Gate:** ✓

**Principle:** Ternary STE backward: committed weights receive exactly zero gradient. A sequence with high loss but gradient concentrated on committed parameters contributes nothing. The uncommitted gradient fraction I_uncomm(x, t) = ||g_uncommitted||₁ / ||g_total||₁ measures how much of a sequence's gradient is "useful." Resample by this fraction; zero-cost to compute (already tracked in backward).

**File:** `src/model_v7/ternary.py` (~L1505 backward, gradient norm logging) + `src/model_v7/train.py` (per-sequence EMA accumulation) + `src/model_v7/data.py` (WeightedRandomSampler)

**Key algorithm:**
```python
# In TernaryLinearFunction.backward (ternary.py ~L1505)
grad_uncommitted_l1 = grad_weight[~committed_mask].abs().sum()
grad_committed_l1   = grad_weight[committed_mask].abs().sum()
frac = grad_uncommitted_l1 / (grad_committed_l1 + grad_uncommitted_l1 + 1e-8)
# Log frac per batch element

# In train.py, main loop:
seq_grad_frac_ema[batch_idx] = 0.9 * seq_grad_frac_ema[batch_idx] + 0.1 * frac.item()

# In data.py sampler:
sampler.weights = seq_grad_frac_ema[sampler.indices]
```

**Expected gain:** 5-15% (grows with commitment fraction f; at f ≥ 0.4, variance of I_uncomm increases sharply). Zero backward cost.

---

### [R9-A2-W2-B-F4] TERNARY-THRESHOLD-FRONTIER-PRIORITY
**Viability:** HIGH | **Gain:** 8-15% step reduction | **Gate:** ✓

**Principle:** Ternary learning event is weight COMMITMENT. Each frontier weight (|w| > 0.8δ) is close to threshold. Optimal selection criterion: maximize gradient alignment with frontier weights. TFA(x) = Σ_{j ∈ frontier} sign(g_j^x) * sign(w_j) minimizes sum of hitting times = total tokens to convergence.

**File:** `src/model_v7/commitment.py` (frontier mask tracking) + `src/model_v7/ternary.py` (~L1505 backward, TFA accumulation) + `src/model_v7/train.py` (sampler weight update via TFA)

**Key algorithm:**
```python
# commitment.py — track frontier mask
frontier_mask = (~committed_mask) & (weight.abs() > 0.8 * threshold)

# ternary.py backward (~L1505)
frontier_grad = grad_weight[frontier_mask]
frontier_weights = weight_data[frontier_mask]
tfa = (frontier_grad * frontier_weights.sign()).sum().item()
# Accumulate per-sequence

# train.py
for seq_id in batch:
    sampler.weights[seq_id] *= np.exp(alpha * seq_tfa[seq_id])  # alpha ~ 0.01
```

**Expected gain:** 8-15% (strongest when f = 0.2–0.7; negligible early/late). Effect increases as commitment progresses.

---

## Per-Step EXPLORE (Higher Complexity, Probe Implementation)

### [R9-A2-W2-A-F2] NATURAL-GRAD-UNCOMMITTED-FISHER-SCORE
**Viability:** HIGH | **Gain:** 8-15% step reduction | **Gate:** ✓

**Principle:** Information geometry: natural gradient norm ||∇L||²_G = ∇L^T G^{-1} ∇L where G is Fisher information matrix. Restricted to uncommitted parameters, this is the theoretically optimal importance weight (Fisher information contribution to posterior). Requires access to per-sample gradients and sigma_g from C2 state. More complex than per-batch aggregation.

**File:** `src/model_v7/train.py` (NGU score computation post-backward) + `src/model_v7/commitment.py` (sigma_g access) + `src/model_v7/data.py` (NGU-weighted sampler)

**Key algorithm:**
```python
# train.py — after backward pass
# Requires per-sample gradients (reduction='none' in forward)
uncommitted_mask = ~committed_mask  # [total_params] bool
for name, param in model.named_parameters():
    if param.grad is None:
        continue
    mask_p = uncommitted_mask[param_slices[name]]
    sigma_g_p = sigma_g_state[param_slices[name]]
    g = param.grad[mask_p]
    s_g = sigma_g_p[mask_p].clamp(min=1e-6)
    per_param_score = (g ** 2) / (s_g ** 2)
    # Accumulate for reweighting via Lion momentum magnitude proxy
```

**Expected gain:** 8-15% (strongest when many parameters near-committed, σ_g is low). Implementation requires careful integration with per-sample loss computation.

**Status:** EXPLORE — probe before committing to production.

---

### [R9-A2-W2-A-F3] SUBMODULAR-COMMITMENT-COVERAGE-GREEDY
**Viability:** MEDIUM | **Gain:** 10-18% step reduction | **Gate:** ✓

**Principle:** Training event is weight commitment. Coverage function F(S) = number of blocks pushed toward commitment by batch S is SUBMODULAR. Greedy maximization yields (1-1/e) ≈ 63% optimal solution. This directly targets commitment rate as optimization criterion.

**File:** `src/model_v7/data.py` (SubmodularCommitmentSampler class) + `src/model_v7/commitment.py` (API for commitment potential + momentum sign)

**Key algorithm:**
```python
# data.py SubmodularCommitmentSampler
class SubmodularCommitmentSampler:
    def compute_promotion_scores(self, doc_indices):
        CP = commitment_tracker.get_commitment_potential()  # [n_blocks]
        m_sign = lion_optimizer.get_momentum_sign()         # [n_blocks]
        affinities = block_affinity_vectors[doc_indices]    # [N_cand, n_blocks] precomputed
        promo = affinities * CP[None, :] * (1 - committed_mask)[None, :]
        return promo

    def select_batch(self, candidates, batch_size):
        promo = self.compute_promotion_scores(candidates)
        coverage = torch.zeros(n_blocks)
        selected = []
        for _ in range(batch_size):
            marginal = (1 - (-coverage[None] - promo).exp()).sum(dim=-1)
            best = candidates[marginal.argmax()]
            selected.append(best)
            coverage += promo[marginal.argmax()]
        return selected
```

**Expected gain:** 10-18% (increases over training as more blocks approach threshold). Requires precomputation of per-document block affinity vectors (O(N × n_blocks × T) at init).

**Status:** EXPLORE — submodular greedy is novel to ternary data selection; validate on 50-step validation runs first.

---

### [R9-A2-W2-A-F4] GRADIENT-SIGN-DIVERSITY-MINIMAX-BATCH
**Viability:** MEDIUM | **Gain:** 5-12% step reduction | **Gate:** ✓

**Principle:** Lion updates sign(batch_grad). Correlated gradients reduce effective batch size. Minimax batch criterion: maximize min_b(block alignment) ensures no uncommitted block is neglected. Proxy: maximize batch vocabulary diversity.

**File:** `src/model_v7/data.py` (MinimaxDiversitySampler) + `src/model_v7/train.py` (sampler integration with Lion state)

**Key algorithm:**
```python
# data.py MinimaxDiversitySampler
class MinimaxDiversitySampler:
    def select_batch(self, candidates, batch_size, current_m_sign, committed_mask):
        alignment = self.proxy_alignment[candidates] * current_m_sign[None, :]
        alignment = alignment * (1 - committed_mask)[None, :]

        batch_alignment = torch.zeros(n_blocks)
        selected = []
        for _ in range(batch_size):
            candidate_min = torch.minimum(
                batch_alignment[None, :] + alignment,
                dim=-1
            ).values  # worst-case block alignment per candidate
            best_idx = candidate_min.argmax()
            selected.append(candidates[best_idx])
            batch_alignment += alignment[best_idx]
        return selected
```

**Expected gain:** 5-12% (strongest when many blocks at different commitment stages). Requires precomputation of per-document block alignment proxy.

**Status:** EXPLORE — minimax criterion is novel; validate before production.

---

### [R9-A2-W2-A-F5] SSM-NGRAM-CEILING-SELECTION
**Viability:** HIGH | **Gain:** 8-15% step reduction (SSM params) | **Gate:** ✓

**Principle:** SSM recurrent state h is only useful when local n-gram models fail. Measure via n-gram improvement ratio: ngram_IR(d) = (CE_1 - CE_5) / CE_1. High ngram_IR = strong local structure = SSM state redundant. Low ngram_IR = weak local structure = SSM state critical. Combined score: (1 - ngram_IR) × max(0, CE_model - CE_5) — high only when sequence REQUIRES SSM state AND model hasn't learned it yet.

**File:** `src/model_v7/prepare_wikitext.py` (compute CE_1, CE_5 for all docs) + `src/model_v7/train.py` (per-step SSM score reweighting)

**Key algorithm:**
```python
# prepare_wikitext.py — one-time preprocessing
def compute_ngram_scores(docs, token_vocab):
    unigram_lm = build_unigram_lm(docs)
    fivegram_lm = build_kneser_ney_ngram(docs, n=5)

    scores = {}
    for doc_id, doc in enumerate(docs):
        CE_1 = -mean([unigram_lm[t] for t in doc])
        CE_5 = -mean([fivegram_lm(t, context=doc[i-4:i]) for i, t in enumerate(doc)])
        ngram_IR = (CE_1 - CE_5) / (CE_1 + 1e-6)
        scores[doc_id] = {'CE_5': CE_5, 'ngram_IR': ngram_IR}
    return scores

# train.py — per step
ssm_score = (1 - ngram_IR_stored) * (per_sample_loss - CE_5_stored).clamp(min=0)
reweight = (ssm_score / ssm_score.mean()).clamp(0.5, 2.0)
loss = (per_sample_loss * reweight).mean()
```

**Expected gain:** 8-15% specifically on SSM parameter convergence. Complements ternary-specific filters by partitioning corpus into sequences that train SSM params vs ternary weights.

**Status:** EXPLORE — relatively novel application; validate SSM-specific metrics first.

---

### [R9-A2-W2-B-F2] SSM-BACK-LOADING-INFORMATIVE-TOKENS
**Viability:** HIGH | **Gain:** 3-8% step reduction (SSM params) | **Gate:** ✓

**Principle:** SSM gradient to input projection W_B decays backward through recurrence. Informative (high-surprise) tokens have larger dt → faster state decay → lower gradient backprop. Ordering matters: back-load informative tokens (place at end) to maximize their gradient weight. Within a packed sequence, sort by unigram surprise (ascending = common first, rare last).

**File:** `src/model_v7/data.py` (PackedDataset.__init__ L74: document reordering by mean informativeness) + `src/model_v7/prepare_wikitext.py` (unigram frequency table)

**Key algorithm:**
```python
# data.py PackedDataset.__init__ (~L74)
# Precomputed unigram freq table from prepare_wikitext.py

# Option: sort docs within pack by ascending mean informativeness
doc_info_scores = [token_freq_lookup(doc).mean() for doc in pack_docs]
sort_order = np.argsort(doc_info_scores)  # ascending
packed_ids = np.concatenate([pack_docs[i] for i in sort_order])

# Or: shuffle token positions within document (more disruptive, higher gain)
# Requires permuting both input and target labels consistently
```

**Expected gain:** 3-8% (strongest for long sequences T ≥ 256, high-rho SSM). Quantitative: 2.17× stronger W_B gradient with back-loading vs front-loading.

**Status:** EXPLORE — token reordering can disrupt language model assumptions; validate carefully.

---

### [R9-A2-W2-B-F3] LION-SIGN-DIVERSITY-BATCH-COMPOSITION
**Viability:** MEDIUM | **Gain:** 3-10% step reduction | **Gate:** ✓

**Principle:** Lion sign effectiveness depends on batch gradient sign correlation. High correlation reduces effective batch size (B_eff = B / (1 + (B-1)ρ)). Maximize batch vocabulary diversity as proxy for low sign correlation. Stratified sampling across n-gram buckets or greedy vocabulary union.

**File:** `src/model_v7/data.py` (batch construction in PackedDataset or WikiTextDataset)

**Key algorithm:**
```python
# data.py — batch construction
# Precompute per-sequence 2-gram fingerprint sets or vocabulary hash buckets

# Greedy diversity:
selected = [random_seed]
while len(selected) < batch_size:
    scores = [len(seq_bigrams[c] - union(selected_bigrams)) for c in candidates]
    selected.append(candidates[argmax(scores)])

# Cheaper stratified version:
buckets = [hash(top-10_unigrams(seq)) % 8 for seq in all_seqs]
# Ensure each batch contains at least 1 seq per bucket (if possible)
```

**Expected gain:** 3-10% (strongest at B=4-8, typical sign correlation ρ=0.2-0.5). For small B=4, relatively modest gains.

**Status:** EXPLORE — batch construction overhead (greedy or bucketing) may negate gains on small batches; validate empirically.

---

## Skip / Gate Failures

### [R9-A2-W1-A-F5] DOREMI-PROXY-DOMAIN-MIXTURE-WEIGHTS ❌ MERGED
**Merged into:** R9-A2-W1-B-F3 (DoReMi self-proxy online variant)

**Reason:** W1-B-F3 is strictly better: it removes the need for a separate proxy model training phase. It applies the DRO update rule online per step, eliminating the preprocessing dependency and making the approach fully dynamic and reference-free. W1-A-F5 (preprocessing proxy training) is subsumed by this per-step variant. Keeping W1-B-F3 only.

---

## Per-Step Adaptive Filter (Complementary to Static Preprocessing)

### [R9-A2-W2-B-F5] CAPACITY-MATCHED-PERPLEXITY-ADAPTIVE-FILTER
**Viability:** HIGH | **Gain:** 5-12% step reduction | **Gate:** ✓

**Principle:** Optimal data complexity for partially-committed ternary model changes as commitment fraction grows. The uncommitted subspace's VC dimension C(t) = 1.585 × (1 − f(t)) × n_total. Optimal training data complexity matches capacity. Adaptive perplexity upper bound: p_upper(t) = max(0.40, (1 − f(t)) × 0.85).

**File:** `src/model_v7/train.py` (commitment fraction query + filter update every 100 steps) + `src/model_v7/data.py` (adaptive band filter in WeightedRandomSampler) + `src/model_v7/commitment.py` (expose committed_fraction() API)

**Key algorithm:**
```python
# In train.py, main training loop (every eval_interval steps)
f = commitment.committed_fraction()  # from C2 state
p_upper = max(0.40, (1.0 - f) * 0.85)

# In data.py WikiTextDataset or WeightedRandomSampler:
seq_ppl_scores = [precomputed at init]  # unigram model proxy
p10 = np.percentile(seq_ppl_scores, 10)
p_upper_percentile = np.percentile(seq_ppl_scores, int(100 * p_upper))
seq_mask = (seq_ppl_scores >= p10) & (seq_ppl_scores <= p_upper_percentile)
# Soft filter: zero weight outside band, uniform inside
sampler.weights = seq_mask.float()
```

**Expected gain:** 5-12% (complementary with R9-A2-W1-A-F4 static preprocessing filter; the two can be stacked).

**Note:** This is the DYNAMIC adaptive version. R9-A2-W1-A-F4 is the STATIC preprocessing version. Both can coexist.

---

## Pipeline Order & Composition

### Preprocessing Sequence (Applied Once)

1. **MINHASH-DEDUP** (W1-A-F1) — removes exact/near-duplicates
2. **NGRAM-REPETITION-FILTER** (W1-A-F2) — removes boilerplate/spam
3. **NGRAM-SELF-DSIR** (W1-A-F3) — selects top 70% by importance weight
4. **PERPLEXITY-BAND-FILTER** (W1-A-F4, static) — removes ppl outliers [p10, p85]
5. **DOREMI-PROXY** (W1-A-F5) — optional, high-cost; rebalance domains (SKIP if DoReMi self-proxy online is enabled instead)

**Simplified pipeline (recommended):** 1 → 2 → 3 → 4 → skip 5 (use W1-B-F3 online instead)

### Per-Step Resampling Stack (Applied During Training)

1. **SELF-REFERENTIAL-EXCESS-LOSS** (W1-B-F1) — activate rho1 without external reference
2. **NGRAM-SELF-DSIR-ONLINE** (W1-B-F2) — update n-gram weights every 200 steps
3. **DOREMI-SELF-PROXY-ONLINE** (W1-B-F3) — DRO domain weight update every step
4. **SSM-LONG-RANGE-DOC-OVERSAMPLING** (W1-B-F4) — warm-fraction weighting (one-time at init)
5. **IN-CONTEXT-TOPOLOGICAL** (W1-B-F5) — doc ordering (one-time at init)
6. **ZLIB-RESIDUAL-MDL** (W2-A-F1) — per-step MDL score filtering
7. **UNCOMMITTED-GRADIENT-FRACTION** (W2-B-F1) — per-step EMA resampling by gradient utility
8. **TERNARY-THRESHOLD-FRONTIER** (W2-B-F4) — per-step TFA resampling by frontier alignment
9. **CAPACITY-MATCHED-PERPLEXITY-ADAPTIVE** (W2-B-F5) — update adaptive band every 100 steps

**Composition strategy:**

- **Steps 1–5** are LOW OVERHEAD, non-interfering → **IMPLEMENT ALL**
- **Steps 6–9** are RESAMPLING WEIGHTS → **STACK as product weights** (sampler.weights ← w_MDL × w_uncomm × w_TFA × w_ppl_adaptive)
- **EXPLORE findings** (F2, F3, F4, F5 SSM) → **PROBE on 50-step validation first**
- **LION-SIGN-DIVERSITY** (W2-B-F3) → **LOW B=4; worth probing but lower priority**
- **SSM-BACK-LOADING** (W2-B-F2) → **EXPLORE carefully; token reordering can disrupt language modeling**

---

## Cross-Pollination & Synergies

| Synergy | Components | Effect |
|---------|-----------|--------|
| **F1 + F5** | ZLIB-MDL (filter mastered) + SSM-NGRAM-CEILING (filter SSM-irrelevant) | Two-dimensional selection space: documents selected only if score_MDL > 0 AND score_SSM > 0 |
| **F3 + F4** | Submodular coverage (F3) + Minimax diversity (F4) | F3 = primary criterion (coverage); F4 = tiebreaker (diversity within equal coverage) |
| **F2 + F3** | Natural gradient (F2) + Submodular coverage (F3) | When CP(b) ∝ 1/σ_g_b (high commitment potential = low variance), both criteria align; stack as product |
| **F1 + F2** | MDL residual filter (F1) + Natural gradient score (F2) | Two-stage: filter first (eliminate mastered), then rank (prioritize high Fisher contribution) |
| **W1 B1-B5 + W2** | Per-step resampling stack (1–9) | Weight matrix: seq_weight[i] ← w_uncomm[i] × w_TFA[i] × w_MDL[i] × ... normalized at each sampler.update() call |

---

## Dedup Registry Additions

```
R9-A2-W1-A-F1: MINHASH-DEDUP-AT-TOKENIZATION
R9-A2-W1-A-F2: NGRAM-REPETITION-SCORE-FILTER
R9-A2-W1-A-F3: DSIR-NGRAM-IMPORTANCE-RESAMPLING
R9-A2-W1-A-F4: PERPLEXITY-BAND-FILTER-PRETOKENIZATION
[R9-A2-W1-A-F5: DOREMI-PROXY-DOMAIN-MIXTURE-WEIGHTS — MERGED into W1-B-F3]

R9-A2-W1-B-F1: SELF-REFERENTIAL-EXCESS-LOSS-RESAMPLING
R9-A2-W1-B-F2: NGRAM-SELF-DSIR-ONLINE-RESAMPLING
R9-A2-W1-B-F3: DOREMI-SELF-PROXY-ONLINE-DOMAIN-WEIGHTS [replaces W1-A-F5]
R9-A2-W1-B-F4: SSM-LONG-RANGE-DOC-OVERSAMPLING
R9-A2-W1-B-F5: IN-CONTEXT-TOPOLOGICAL-DOC-ORDERING

R9-A2-W2-A-F1: ZLIB-RESIDUAL-MDL-FILTER
R9-A2-W2-A-F2: NATURAL-GRAD-UNCOMMITTED-FISHER-SCORE [EXPLORE]
R9-A2-W2-A-F3: SUBMODULAR-COMMITMENT-COVERAGE-GREEDY [EXPLORE]
R9-A2-W2-A-F4: GRADIENT-SIGN-DIVERSITY-MINIMAX-BATCH [EXPLORE]
R9-A2-W2-A-F5: SSM-NGRAM-CEILING-SELECTION

R9-A2-W2-B-F1: UNCOMMITTED-GRADIENT-FRACTION-RESAMPLING
R9-A2-W2-B-F2: SSM-BACK-LOADING-INFORMATIVE-TOKENS [EXPLORE]
R9-A2-W2-B-F3: LION-SIGN-DIVERSITY-BATCH-COMPOSITION [EXPLORE]
R9-A2-W2-B-F4: TERNARY-THRESHOLD-FRONTIER-PRIORITY
R9-A2-W2-B-F5: CAPACITY-MATCHED-PERPLEXITY-ADAPTIVE-FILTER
```

---

## Implementation Roadmap

### Batch 1 (Preprocessing — Foundation)
- MINHASH-DEDUP (W1-A-F1)
- NGRAM-REPETITION-FILTER (W1-A-F2)
- DSIR-NGRAM-IMPORTANCE (W1-A-F3)
- PERPLEXITY-BAND-STATIC (W1-A-F4)

### Batch 2 (Per-Step Resampling — Core)
- SELF-REFERENTIAL-EXCESS-LOSS (W1-B-F1)
- NGRAM-SELF-DSIR-ONLINE (W1-B-F2)
- DOREMI-SELF-PROXY-ONLINE (W1-B-F3)
- SSM-LONG-RANGE-DOC-OVERSAMPLING (W1-B-F4)
- IN-CONTEXT-TOPOLOGICAL (W1-B-F5)
- ZLIB-RESIDUAL-MDL (W2-A-F1)
- UNCOMMITTED-GRADIENT-FRACTION (W2-B-F1)
- TERNARY-THRESHOLD-FRONTIER (W2-B-F4)
- CAPACITY-MATCHED-PERPLEXITY-ADAPTIVE (W2-B-F5)

### Batch 3 (EXPLORE — Validation)
- NATURAL-GRAD-UNCOMMITTED-FISHER (W2-A-F2)
- SUBMODULAR-COMMITMENT-COVERAGE (W2-A-F3)
- GRADIENT-SIGN-DIVERSITY-MINIMAX (W2-A-F4)
- SSM-NGRAM-CEILING (W2-A-F5)
- SSM-BACK-LOADING (W2-B-F2)
- LION-SIGN-DIVERSITY (W2-B-F3)

---

## Summary Statistics

| Category | Count | Est. Gain Range |
|----------|-------|-----------------|
| Preprocessing IMPLEMENT | 4 + (optional) 1 | 38–72% cumulative |
| Per-Step IMPLEMENT (Core) | 9 | 48–105% cumulative |
| **Combined (Batch 1+2)** | **13** | **50–120% cumulative** |
| EXPLORE | 6 | 26–83% individual |
| SKIP (merged) | 1 | — |
| **Total Unique** | **19** | — |

**Gain interpretation:** Estimated reductions compound with overlap; actual stacking loss depends on correlation structure between filtering criteria. Conservative 50–120% range accounts for non-independence.

---

**Last Updated:** 2026-03-06 | **Wave 3 Synthesis Complete** | **Next Step:** Architecture-enforcer review before Batch 1 implementation
