# Research Round 5 — Data Engineering & Tokenization

## Overview

Research findings on tokenization strategies, data quality/filtering, and curriculum learning for ternary SSM models. Focus: how data preparation impacts training quality and efficiency.

## NEW FINDINGS (Round 5, Aspect 3)

### Wave 1: Web Research (2 researchers, Feb 2026)

#### TOKENIZATION STRATEGIES

**F1: Byte-Level Models (BLMs) — Bolmo & BLT** (AI2/Meta, Dec 2024 - Feb 2026)
- Tokenizer-free models operating on UTF-8 bytes via hierarchical patch encoding. Bolmo uses learnable cycle modules + dynamic patching. BLT uses entropy-driven boundary detection.
- Bolmo-7B: 78.6% on CUTE (char understanding) vs 56.9% for subword teacher. ~125 bytes/s inference.
- Relevance: HIGH — eliminates vocab_size entirely → no LM head matmul. Your 32K vocab → 256 byte vocab (125x reduction in output layer params).
- Complexity: Medium (requires local encoder, boundary predictor, hierarchical patching).

**F2: Larger Vocabulary = Better Quality** (ACL 2025)
- Empirical: larger subword vocabs (64K-128K vs 32K) improve LLM performance across tasks.
- Tension with ternary: larger vocab = larger LM head = more ternary params. Quality vs efficiency trade-off.
- Resolution: combine with F3 (VocabTailor) or F4 (CCE) to decouple training quality from deployment cost.

**F3: VocabTailor — Dynamic Vocabulary Selection** (ICLR 2025)
- Decoupled vocab: full tokenizer preserved, LM head uses hybrid static+dynamic selection per task.
- 99% reduction in vocab-related memory. Code completion: 11% vocab retained, Pass@1 = 53.87% (vs 54.10%).
- Relevance: CRITICAL — ternary LM head (d_model→32K) is 16.4M params. VocabTailor reduces to ~500K-5M dynamically.
- Complexity: Medium-High.

**F4: Cut Cross-Entropy (CCE)** (ICLR 2025)
- Computes cross-entropy without materializing full logit matrix. Flash memory kernel: on-the-fly logit computation.
- Gemma 2B: loss memory 24GB → 1MB. Classifier head: 28GB → 1GB. Zero speed/convergence penalty.
- Relevance: HIGH — enables large vocab training on consumer GPU. Orthogonal to vocab reduction.
- Complexity: Low-Medium (reference implementations exist).

**F5: Hierarchical BPE Patching** (Oct 2025)
- Two-stage BPE: standard + second stage merging frequent n-grams within patch boundaries.
- Lowest bits-per-byte compression vs char/space/entropy methods.
- Relevance: LOW-MEDIUM. Improves compression but doesn't address LM head size.

**F6: Multi-Resolution Tokenization** (2025-2026 trend)
- Process sequences at multiple granularities simultaneously. HPMixer, PyraTok examples.
- Coarse patches (global context) + fine patches (local detail) → SOTA in time series/vision.
- Relevance: MEDIUM — SSM already has multi-scale via state evolution. Adds complexity.

**F7: Token Merging/Pruning for Inference** (LightVLM, PTP, 2025)
- Hierarchical token reduction: merge/prune low-importance tokens at each layer.
- LightVLM: 100% performance at 35% token retention, 2x speedup.
- Relevance: HIGH for inference — reduce tokens flowing through sequential SSM scan.

**F8: Lookup-Free VQ for Byte Models** (PyraTok, Jan 2026)
- VQ-based byte tokenizer with hierarchical consistency. Lookup-free quantization.
- Relevance: MEDIUM — VQ codebooks are naturally ternary-friendly (discrete, compact).

**F9: Distillation-Based Byteification** (Bolmo, Dec 2024)
- Convert existing subword model to byte-level via 2-stage distillation (125B bytes).
- Retains >90% teacher performance with 20-point gain on character tasks.
- Relevance: CRITICAL adoption path — train ternary SSM with subword first, then byteify.

**F10: CCE + VocabTailor Synthesis** (2025 combined insight)
- Train with large vocab (CCE handles memory), deploy with dynamic small vocab (VocabTailor).
- Decouples training quality from inference cost. Best of both worlds.
- Relevance: HIGHEST strategic value — solves core ternary vocab tension.

#### DATA QUALITY & CURRICULUM

**F11: Curriculum Learning with Compression/Lexical Metrics** (arxiv 2506.11300, Jun 2025)
- Easy→hard ordering using compression ratio, lexical diversity (MTLD), readability (Flesch).
- 18-45% reduction in steps to reach baseline. 3.5% sustained improvement as warmup.
- Relevance: Directly applicable to WikiText-103, no model changes needed.
- Complexity: Low.

**F12: Perplexity-Correlation Data Selection** (ICLR 2025)
- Select data by computing correlation between sample perplexity and downstream benchmark performance.
- Outperforms DSIR on all benchmarks. pip install perplexity-correlations.
- Relevance: Can filter/weight WikiText-103 samples using GPT-2 as reference model.
- Complexity: Medium.

**F13: FineWeb Deduplication Pipeline** (arxiv 2406.17557, Jun 2024)
- MinHash LSH dedup + quality filtering. 15T token dataset from CommonCrawl.
- FineWeb-Edu (educational filter) shows dramatic gains on MMLU/ARC.
- Relevance: Open-source pipeline applicable to any corpus. Code publicly released.
- Complexity: Medium.

**F14: SemHash Semantic Deduplication** (MinishLab, 2025)
- Fast semantic dedup using Model2Vec embeddings + LSH. pip install semhash.
- 1.8M records deduplicated in ~83s on CPU. Supports outlier filtering.
- Relevance: Lightweight, can dedupe WikiText-103 semantically (not just exact match).
- Complexity: Low.

**F15: AskLLM + Density Sampling** (ICLR 2025)
- LLM scores sample quality via zero-shot reasoning + density sampling for diversity.
- Training on 10% of data (selected) outperforms full-data training. 70% faster convergence.
- Relevance: Expensive but powerful. Density sampling component is lightweight.
- Complexity: High (AskLLM), Medium (Density).

**F16: Scaling Laws for Optimal Data Mixtures** (NeurIPS 2025)
- Predictive framework for domain mixture ratios without trial-and-error.
- Eliminates expensive grid search. Applicable when scaling to multi-domain data.
- Complexity: Medium.

**F17: Data Mixing Laws** (ICLR 2025)
- Predict LLM performance from data mixtures at low cost.
- Complements F16 for pretraining and post-training mixture optimization.
- Complexity: Medium.

**F18: MGA Reformulation for Data Augmentation** (arxiv 2502.04235, Feb 2025)
- Rephrase text into diverse genre/audience variations via LLM prompting.
- Prevents repetition degradation when training multiple epochs. 770B token MGACorpus released.
- Relevance: HIGH for ternary — convergence takes many epochs, reformulation prevents degradation.
- Complexity: High (requires LLM for generation).

**F19: RedPajama-v2 with Quality Annotations** (Together AI, 2024-2025)
- 30T token open dataset with 40+ precomputed quality annotations.
- Ready-to-use quality signals for large-scale training.
- Relevance: For scaling beyond WikiText-103 to billion-token pretraining.
- Complexity: Low (data is preprocessed).

**F20: DCLM Baseline Filtering** (NeurIPS 2024, DataComp-LM)
- Benchmark-driven data curation via systematic ablations. Open-source tooling.
- Relevance: Reference pipeline for corpus construction at scale.
- Complexity: High.

**F21: Linguistically-Informed Curriculum** (Jan 2026)
- Curriculum based on dependency tree depth, clause count, syntactic complexity.
- Complements F11 with structural linguistic metrics beyond surface readability.
- Complexity: Low.

#### KEY PATTERNS (Wave 1)
- **Byte-level convergence:** 2024-2026 research overwhelmingly favors byte-level over subword
- **Dynamic vocabulary as decoupling:** VocabTailor + CCE decouple quality (train large) from efficiency (deploy small)
- **Three-tier data pipeline:** Dedup (F13,F14) → Quality filter (F12,F15) → Curriculum order (F11,F21)
- **Multi-epoch augmentation critical for ternary:** Ternary convergence is slow → MGA reformulation prevents repetition degradation
- **Lexical locality:** Each inference uses tiny vocab subset (0.08-28%) → ternary LM head vastly oversized

#### TOP 5 WAVE 1 FINDINGS
1. **F10 CCE+VocabTailor** — solves the ternary vocab tension completely (train large, deploy small)
2. **F1 Byte-Level Models** — eliminates LM head entirely, paradigm shift for ternary efficiency
3. **F11 Curriculum Learning** — 18-45% fewer steps, zero complexity, immediate
4. **F14 SemHash Dedup** — fast semantic dedup, pip install, applies to WikiText-103
5. **F18 MGA Augmentation** — prevents multi-epoch degradation critical for slow ternary convergence

### Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

#### TOKENIZATION OPTIMIZED FOR TERNARY

| # | Technique | Speedup (LM head) | Viability | Key Insight |
|---|-----------|-------------------|-----------|-------------|
| M1 | LSH-Head (Ternary Hash Output) | 780x | HIGH | Ternary projections ARE locality-sensitive hash functions; hash to bucket then score |
| M2 | Binary Decision Forest | 6500x average | HIGH | Predict token via O(log V) ternary binary classifiers (tree depth) |
| M3 | Error-Correcting Code Head (ECC) | 1000x | HIGH | Predict L=31 code bits instead of V=32K logits; ternary = optimal 1-bit measurement |
| M4 | Compositional Token Algebra | 90-330x | HIGH | Factor V = V1×V2×V3; predict each factor independently |
| M5 | Ternary Superposition Codes | 3-10x | MEDIUM | B basis ternary vectors + small code matrix A; factored prediction |
| M6 | Hyperdimensional Computing (HDC) | 16-128x | SPECULATIVE | All-integer output via POPCOUNT nearest-neighbor in Hamming space |

**M1: LSH-Head** — NATURAL FIT
- K=10 ternary hash functions → 1024 buckets of ~32 tokens each.
- Compute K hash bits: b_k = sign(g_k^T h). Costs O(10d) ternary ops.
- Score only bucket contents: O(32d) ternary ops. Total: O(42d) vs O(32768d).
- Ternary projections are literally Charikar/Indyk LSH. Zero entries = feature subsampling (exactly what makes LSH work).
- Multi-probe LSH (check neighboring buckets) mitigates quality loss.

**M2: Binary Decision Forest** — EXTRAORDINARY SPEEDUP
- Huffman-like tree: each internal node has one ternary classifier w_n ∈ {-1,0,+1}^d.
- Average depth = H(p) ≈ 4-6 bits for English text. Cost: O(5d) vs O(32768d).
- Ternary binary classifiers are OPTIMAL building blocks (1-bit compressed sensing theory).
- Zero entries provide natural feature selection per node.
- Training via hierarchical softmax (standard, Morin & Bengio 2005).
- Risk: error accumulation at early tree levels. Mitigate with beam search.

**M3: ECC-Head** — TOP RECOMMENDATION
- Assign each token a BCH(31,16,7) codeword. LM head predicts 31 bits.
- Head shrinks from d×32K to d×31. **1000x parameter reduction in LM head.**
- Deep connection: ternary projection computing sign(w^T h) IS a 1-bit measurement.
- 1-bit compressed sensing theory proves L=O(log V) ternary projections sufficient to distinguish V tokens.
- Training: independent binary cross-entropy per bit (simpler than softmax).
- Syndrome decoding for error correction: O(L²) = O(961) — negligible.
- Learned codeword assignment co-optimized with ternary projectors.

**M4: Compositional Token Algebra**
- Factor V=32768 as V1×V2 (e.g., 181×181) or V1×V2×V3 (33³).
- Two-factor: O(362d) vs O(32768d) — 90x reduction.
- Three-factor: O(99d) vs O(32768d) — 330x reduction.
- Chinese Remainder Theorem: represent token ID as tuple of residues mod coprime numbers.
- Semantic factorization: group tokens by type → predict type first (coarser, easier for ternary).
- Well-understood math (adaptive softmax, class decomposition).

**M5: Ternary Superposition Codes**
- B=256 ternary basis vectors. Each token = sparse combination of basis vectors.
- Compute B inner products O(Bd), then small matrix A*s O(VB) or O(Vk) if sparse.
- With sparse A (k=8): O(256d + 262K) — significant savings.
- Based on frame theory: ternary frames with Welch bound analysis.

**M6: Hyperdimensional Computing (HDC)** — MOONSHOT
- Each token gets random binary hypervector c_t ∈ {-1,+1}^D (D=4096). Fixed, no learning.
- Prediction: sign(P*h) then nearest-neighbor in Hamming space via XOR+POPCOUNT.
- With AVX-512: 262K SIMD ops. All-integer, zero floating-point in entire output.
- Risk: binarization bottleneck, training difficulty through sign().

#### OPTIMAL DATA REPRESENTATION FOR SSMs

| # | Technique | Impact | Viability | Key Insight |
|---|-----------|--------|-----------|-------------|
| M7 | Hadamard-Rotated Embeddings | 1-5% ppl + multiply-free | HIGH | WHT decorrelates inputs → optimal for ternary projection information throughput |
| M8 | Eigenvalue-Matched Curriculum | 10-30% fewer steps | MEDIUM | Present dependencies at timescale matching SSM memory horizons |
| M9 | Recurrence-Harmonic Augmentation | 2-8% ppl | HIGH | Multi-rate subsampling = 2-3x effective data for free |
| M10 | MI-Optimal Chunk Boundaries | 1-5% long-range | MEDIUM | Place SSD chunk boundaries at natural information minima |
| M11 | Spectral-Aligned Tokenization | 5-15% if mismatch | LOW | Match tokenization power spectrum to SSM filter bank response |
| M12 | State-Predictive Tokenization | 0-20% unknown | SPECULATIVE | Invert pipeline: desired states → tokens that produce those states |

**M7: Hadamard-Rotated Embeddings** — IMPLEMENT IMMEDIATELY
- WHT (Walsh-Hadamard Transform) decorrelates embeddings: x_t = H_d · E'[token_t].
- Info theory: I(x; Bx) maximized when Cov(x) = σ²I. WHT achieves this computationally.
- WHT requires ONLY additions/subtractions, O(d log d) via butterfly algorithm.
- Combined with ternary B projection: ENTIRE embedding-to-state pipeline is multiply-free.
- Orthogonal transform → preserves norms, cannot hurt stability. Zero downside risk.
- Implementation: literally one line of code using fast WHT.

**M8: Eigenvalue-Matched Curriculum**
- Each SSM dimension has memory horizon τ_i = -1/ln|a_i|.
- Gradient from dependency at distance k peaks at k ≈ τ_i (calculus: max of k·r^{k-1}).
- Curriculum: present sequences whose dependency timescales match current eigenvalue distribution.
- Proxy: estimate D(τ,s) from document structure (sentences ~20-50, paragraphs ~100-500).
- 10-30% faster convergence. Does not change final quality, just navigates loss landscape faster.

**M9: Recurrence-Harmonic Augmentation** — IMPLEMENT IMMEDIATELY
- Multi-rate subsampling: take every r-th token. Effective eigenvalue becomes a_i^r.
- Forces scale-invariant features across timescales.
- Effectively 2-3x data augmentation for free (train on rate-1, rate-2, rate-3 versions).
- Block shuffle augmentation: permute within window w. Tests robustness at timescale w.
- Selective token dropout at timescale τ: p(drop_t) = 0.5·(1+cos(2πt/τ)).

**M10: MI-Optimal Chunk Boundaries for SSD**
- At chunk boundary, state decays by a^L. Information loss = (1-|a_i|^{2L})·Var(h_b^(i)).
- Place boundaries where Var(h) is small (natural topic breaks, sentence boundaries).
- Surprise proxy: S(t) = -log p(x_{t+1}|x_{1:t}). High surprise = natural boundary.
- Dynamic programming: O(T·W) for optimal segmentation.

**M11: Spectral-Aligned Tokenization**
- SSM filter bank response F(ω) = Σ 1/|1-a_i·e^{-jω}|².
- Token sequence has power spectrum S_V(ω). Objective: maximize min(F, S_V) at all frequencies.
- Water-filling analogy: every frequency band of the filter bank needs sufficient input signal.
- Practical: add multi-scale position information (char count, sentence position).

**M12: State-Predictive Tokenization** — MOONSHOT
- Compute optimal input x_t = B⁺·(h_t* - diag(a)·h_{t-1}) from desired state trajectory.
- Assign tokens whose embeddings are closest to optimal inputs.
- Inverts usual pipeline: SSM dynamics DICTATE tokenization.
- Partial version: vocabulary reordering for smooth state trajectories (Hungarian algorithm).

#### KEY CROSS-POLLINATION (Wave 2)
- **M3 (ECC) + M4 (Compositional):** ECC on factored token space. Factor V=V1×V2, ECC per factor: L1+L2 << L_flat.
- **M1 (LSH) + M2 (Tree):** LSH to identify subtree, then fine-grained tree decisions within. Cost: O(Kd + H(p|bucket)·d).
- **M7 (Hadamard) + M9 (Multi-Rate):** WHT decorrelates at ALL rates. Multi-rate forces scale invariance. Orthogonal improvements, compose multiplicatively.
- **M8 (Eigenvalue Curriculum) + M9 (Augmentation):** Curriculum pushes frontier forward; augmentation consolidates what's behind.
- **M3 (ECC) + M6 (HDC):** ECC codeword IS a hyperdimensional representation. Increase L to 4096 and decode via POPCOUNT.
- **Universal insight:** Diagonal SSM is a FILTER BANK. Data optimization = designing optimal input signal for exponentially-decaying bandpass filters with ternary coefficients.

### Wave 3: Paradigm Breakers (2 agents, Feb 2026)

#### RADICAL DATA ENGINEERING APPROACHES

| # | Technique | Impact | Viability | Key Insight |
|---|-----------|--------|-----------|-------------|
| P1 | Z3-Pipeline (Ternary Ring Pipeline) | 3,300x embedding reduction | MEDIUM-HIGH | Entire data path in Z_3 — ternary codes for tokens, ternary embedding MLP, ternary output head |
| P2 | Ternary Bitstream LM (TBLM) | 10,800x I/O reduction | MEDIUM-HIGH | Eliminate tokens — predict next TRIT (3-way), ternary Huffman coding, info-theoretic symmetry |
| P3 | Character-As-State-Transition (CAST) | 20-40% param reduction | MEDIUM | Characters ARE ternary transition matrices — no embedding, no in_proj, no dt_proj |
| P4 | Entropic State Segmentation (ESS) | Tokenizer-free | MEDIUM-LOW | SSM state entropy defines segment boundaries — emergent context-sensitive tokenization |
| P5 | Ternary Native Tokenization (TNT) | 80x embedding, L1-cache fit | MEDIUM | Tokens as ternary codes, first layer becomes POPCOUNT (ternary×ternary) |
| P6 | Residual-Aligned Data Selection (RADS) | 2-5x convergence | MEDIUM-HIGH | Quantization residual R = W_shadow - Q(W) guides data selection for threshold crossings |
| P7 | Eigenvalue-Resonant Multi-Rate (ERMT) | 2.35x compute reduction | LOW-MEDIUM | Multi-rate SSM: bytes→fast eigenvalues, subwords→medium, phrases→slow |
| P8 | Threshold Geography Cartography | 1.5-3x training efficiency | HIGH | Shadow weight boundary distance maps reveal WHICH linguistic patterns model struggles with |

**P1: Z3-Pipeline** — HIGHEST PRIORITY (TERNARY PURITY)
- Assign each token a ternary codeword c(t) ∈ Z_3^10 (10 trits = 59,049 > 32K vocab).
- Ternary embedding MLP: e(t) = α₂·W₂·σ(α₁·W₁·c(t)), W₁ ∈ {-1,0,+1}^{128×10}, W₂ ∈ {-1,0,+1}^{768×128}.
- Embedding shrinks from 49 MB FP16 → 24 KB ternary (2,040x reduction).
- Output head: predict 10 trits independently → W_out ∈ {-1,0,+1}^{30×768} = 5.6 KB (vs 49 MB = 8,750x).
- Decode via nearest valid codeword in Z_3^10 trie (O(10) lookup).
- Error correction: ternary linear codes over GF(3), [13,10,3] code corrects 1 trit error.
- Gray-code-like assignment: semantically similar tokens get low Hamming distance.
- Combined with ternary model: ENTIRE pipeline is add/subtract/skip. >99% ternary (vs 84% current).
- **Risk:** Independent trit prediction assumes separability. May need autoregressive trit factorization.

**P2: Ternary Bitstream Language Model (TBLM)** — DEEPEST SYNERGY
- Build ternary Huffman code over vocab: expected length L = H(V)/log₂(3) ≈ 4.73 trits/token.
- Model processes ONE TRIT per SSM step, predicts next trit (3-way classification).
- Embedding: 3 learned vectors (4.5 KB vs 49 MB = 10,800x). Output: 3 logits (4.5 KB vs 49 MB).
- Information-theoretic symmetry: ternary weights (1.58 bits) process ternary data (1.585 bits). NO mismatch.
- Automatic variable compute: common tokens = short codes = few steps. Rare tokens = long codes = more compute.
- LM head eliminated entirely. At 7B: replaces 204M multiplies with 58K ternary ops (3,500x reduction).
- **Risk:** 4.73x more SSM steps per token. Beam search operates at trit level.

**P3: Character-As-State-Transition (CAST)** — MOST RADICAL
- Each byte c defines a ternary transition matrix T_c ∈ {-1,0,+1}^{16×16} and input b_c ∈ {-1,0,+1}^{16}.
- SSM step becomes: h_{t+1} = σ(s_A · T_{c_t} · h_t + s_B · b_{c_t}). Pure add/subtract/skip.
- Eliminates: embedding, in_proj, x_proj, dt_proj, continuous A matrix, discretization.
- Multi-head (48 heads, d_h=16): 256 characters × 48 × (256+16) ternary params = 835 KB total.
- K=4 mode routing for selectivity: mode_index = argmax_k(r_k^T · h_{t-1}), ~3.3 MB total.
- 3^256 possible transitions per head per character — enormous expressivity.
- **Risk:** Without selectivity, essentially a weighted finite automaton. Byte-level = 5x more steps.

**P4: Entropic State Segmentation (ESS)** — TOKENIZER-FREE
- Two-tier SSM: small byte-level segmenter (d=64) + main segment processor (d=768).
- Segmenter monitors its own state entropy H_t = -Σ p_i·log(p_i) where p_i = |h[i]|²/||h||².
- When entropy drops (state "saturates"), emit segment representation to main model.
- Differentiable via soft gates: gate_t = σ(w^T · h_t^{seg}). Gumbel-softmax for hard boundaries.
- Context-sensitive: same text segments differently based on preceding context (unlike BPE).
- Ternary benefit: quantized outputs have CRISPER entropy signal → sharper segmentation boundaries.
- Language-agnostic: no tokenizer training, no unknown tokens, works for any script/code.
- **Risk:** Degenerate segmentation (every byte or never). Joint two-tier training is unstable.

**P5: Ternary Native Tokenization (TNT)** — HARDWARE EFFICIENCY
- Each token gets ternary code c ∈ {-1,0,+1}^{d_code} with d_code=64 (3^64 ≈ 10^30 capacity).
- Learned code assignment maximizes: separation (Hamming distance) + semantic similarity + filter match.
- First layer becomes ternary×ternary = POPCOUNT: y_j = popcount(w⁺∧c⁺) + popcount(w⁻∧c⁻) - ...
- Embedding storage: 32K × 64 × 1.58 bits ≈ 400 KB (vs 32 MB FP16 = 80x reduction).
- Entire embedding table fits in CPU L1 cache.
- **Risk:** Can 64 ternary dimensions match 512 FP16 dimensions? Geometric structure differs.

**P6: Residual-Aligned Data Selection (RADS)** — TERNARY-SPECIFIC ACTIVE LEARNING
- Quantization residual R = W_shadow - Q(W_shadow) points where optimizer WANTS to go.
- Gradient-Residual Alignment Score: GRAS(x) = Σ_l ⟨g_l(x), R_l⟩ / (||g_l(x)||·||R_l||).
- Select top-K examples by GRAS from candidate batch → maximizes threshold crossing probability.
- ~90% of current gradient updates do NOT cross a ternary threshold → 90% compute wasted.
- GRAS-selected data maximizes E[crossings] = Σ_{w near boundary} P(|W + lr·g| crosses threshold).
- **MEANINGLESS for FP16 models** — exists ONLY because ternary has discrete thresholds.
- **Risk:** Biased selection could overfit to threshold-crossing patterns. ~50% overhead per step.

**P7: Eigenvalue-Resonant Multi-Rate Tokenization (ERMT)** — NATURAL TIMESCALES
- Partition SSM state dims into bands by eigenvalue magnitude: fast (|λ|<0.3), medium, slow (|λ|>0.8).
- Each band processes data at its natural rate: bytes (every step), subwords (every 4), phrases (every 16).
- Compute: C_total = L·(C_0 + C_1/4 + C_2/16) ≈ 0.425·L·C_1 → 2.35x reduction.
- Each level has different ternary sparsity: byte-level = high sparsity (simple patterns), phrase-level = dense.
- **Risk:** Multi-rate synchronization is complex. Needs custom CUDA kernels.

**P8: Threshold Geography as Data Cartography** — IMMEDIATE USE
- d_boundary(w) = min(|w - threshold_+|, |w - threshold_-|, |w - threshold_0|).
- Uncertainty map: U_l = {(i,j) : d_boundary(w_l[i,j]) < ε}. Fraction f_uncertain per layer.
- Non-uniform uncertainty profile reveals WHICH linguistic abstraction levels need more training.
- Early layers uncertain → train on unusual character sequences, rare words, morphology.
- Late layers uncertain → train on complex reasoning, long-range dependencies.
- Match score: match(x) = Σ_l f_uncertain_l · a_l(x) / Σ_l a_l(x).
- ZERO extra compute — reads existing shadow weights. Implementable in hours.
- Three boundary types (+1/0, -1/0, +1/-1) each indicate different learning challenges.
- **Risk:** Layer-abstraction correlation is approximate. Small models may lack differentiation.

#### KEY CROSS-POLLINATION (Wave 3)
- **P1 (Z3) + P2 (TBLM):** Ternary codes for input (Z3 embedding) + trit prediction for output (TBLM). Best of both: compact input AND compact output without extending input sequence.
- **P6 (RADS) + P8 (Threshold Geography):** P8 identifies WHICH layers need help. P6 selects data that helps those layers cross thresholds. Complete ternary-aware active learning system.
- **P3 (CAST) + P4 (ESS):** CAST processes bytes as state transitions; ESS identifies when to emit. Natural pairing for tokenizer-free byte-level processing with dynamic segmentation.
- **P5 (TNT) + M3 (ECC):** Ternary codes for input (TNT) + error-correcting codes for output (ECC). Both ends of the pipeline become ternary-native with error correction.
- **P7 (ERMT) + M9 (Multi-Rate Augmentation):** ERMT provides multi-rate processing; M9 provides multi-rate training data. Matched pair for scale-invariant learning.

#### META-INSIGHT FROM PARADIGM BREAKERS
> **"The FP16 embedding table is the last vestige of continuous-precision thinking in a ternary model.**
> Five independent approaches (P1, P2, P3, P5, TNT) converge on the same conclusion:
> tokens should be represented AS ternary vectors, not embedded INTO FP16 space.
> This eliminates the information explosion (15-bit token ID → 4096-bit FP16 vector)
> followed by immediate compression (ternary projection discards most precision).
> The model should speak ternary natively, from input to output."
>
> The second convergence: **data selection should exploit the UNIQUE structure of ternary training.**
> Shadow weight residuals (P6) and boundary distances (P8) are signals that FP16 models
> cannot produce. These ternary-specific signals enable active learning that is
> provably impossible with continuous weights.

---

## Aspect 3 Summary: Data Engineering & Tokenization

**Total findings:** 21 web research (F1-F21) + 12 mathematical (M1-M12) + 8 paradigm shifts (P1-P8) = **41 findings**

**Top 5 by estimated impact:**
1. **P2 TBLM (Ternary Bitstream)** — eliminates tokens entirely, 10,800x I/O reduction, info-theoretic symmetry
2. **P1 Z3-Pipeline** — entire data path in ternary ring, >99% ternary purity, 3,300x embedding reduction
3. **M3 ECC-Head** — predict 31 BCH code bits instead of 32K logits, 1000x LM head reduction
4. **P6 RADS + P8 Threshold Geography** — ternary-specific active learning, 2-5x convergence
5. **F10 CCE+VocabTailor** — pragmatic solution: train large vocab, deploy small dynamically

**Immediate actions (can implement now):**
- P8 (Threshold Geography): Zero-code observational analysis of shadow weight boundary distances
- M7 (Hadamard Embeddings): One-line WHT decorrelation, zero downside risk
- M9 (Multi-Rate Augmentation): Data augmentation via subsampling, 2-3x effective data
- F11 (Curriculum Learning): Easy→hard ordering, 18-45% fewer steps
- F14 (SemHash Dedup): pip install semhash, fast semantic dedup for WikiText-103

**Recommended implementation order:**
1. F14 + F11 (dedup + curriculum) — pure data pipeline, no model changes
2. P8 (threshold geography analysis) — observational, validates signals
3. M7 + M9 (Hadamard + multi-rate augmentation) — embedding improvement + data aug
4. P6 (RADS) — ternary-aware data selection using validated P8 signals
5. P1/P5 (ternary embedding) — replace FP16 embedding with ternary codes
6. M3 (ECC-Head) — replace LM head with error-correcting code prediction
7. P2 (TBLM) — full trit-level prediction (most radical, test last)
