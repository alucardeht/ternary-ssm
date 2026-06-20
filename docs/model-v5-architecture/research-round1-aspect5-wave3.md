# Round 1, Aspect 5: Memory-Efficient Training & Inference — Wave 3 (Paradigm-Breakers)

## PB1: Radical Paradigms for Near-Zero Training Memory

### P1: Potts Model Parallel Tempering
- **Paradigm shift:** Training is NOT optimization of a differentiable function — it's finding the ground state of a 3-state Potts model Hamiltonian
- **How:** Each ternary weight is a spin σ_i ∈ {-1,0,+1}. Metropolis-Hastings single weight flips with O(1) energy evaluation. Δy_i = (b-a) × x_j where (b-a) ∈ {-2,-1,+1,+2}. Parallel tempering with R=8 replicas at different temperatures. Layer-local Hamiltonian via Forward-Forward goodness.
- **Memory:** 1.5 GB for 7B (55× reduction vs 84 GB standard STE)
  - Packed ternary weights: 1.4 GB
  - 8 replica diffs (sparse mutations): 80 MB
  - Current layer activations: 50 MB
- **Ternary-specific:** Only 2 alternative values per weight → perturbation fully enumerable. O(1) energy evaluation possible because weight change is small integer. 3-state Potts model has 50+ years of theory.
- **Risk:** Impact HIGH, Difficulty HIGH. Layer-local Hamiltonians (Forward-Forward) may not produce competitive LM quality.

### P2: Hypernetwork Weight Compilation
- **Paradigm shift:** Don't STORE 7B weights + optimizer state — train a TINY generator (4M params) that OUTPUTS ternary weights on-the-fly
- **How:** Small GRU (h_dim=1024, 2 layers) takes layer spec → produces ternary weight rows via argmax(logits). Generator trained via STE through argmax. Weights generated per-layer, used, discarded.
- **Memory:** 500 MB for 7B (168× reduction)
  - Generator params (FP32): 16 MB
  - Generator optimizer: 32 MB
  - Generated layer (temporary): 9 MB
  - Activations (checkpointed): 200 MB
- **Ternary-specific:** Ternary matrices have lower Kolmogorov complexity → small generator can capture structure. argmax naturally produces discrete values. Generator can learn global patterns (sparsity ratios, spectral properties).
- **Risk:** Impact MEDIUM-HIGH, Difficulty MEDIUM. 1750× compression ratio (4M→7B) may be too aggressive.

### P3: Block-SPSA Zeroth-Order Training
- **Paradigm shift:** Eliminate backpropagation entirely — estimate gradient DIRECTION by comparing loss at two random perturbations
- **How:** Select random block of K=10K params, flip randomly, evaluate L+ and L−, gradient estimate ĝ_i = (L+ − L−) / 2δ_i. Accumulate and flip when |A_i| > threshold. Only needs SIGN of gradient (flip left or right).
- **Memory:** 2.4 GB for 7B (35× reduction)
  - Packed weights: 1.4 GB
  - Active block accumulator (INT16): 20 KB
  - Forward pass activations: 1 GB
- **Ternary-specific:** Forward pass 3× cheaper → 2× forward SPSA cost is relatively cheap. Only need sign of gradient, not magnitude. Discrete search makes "try and evaluate" natural. Embarrassingly parallelizable.
- **Risk:** Impact MEDIUM, Difficulty MEDIUM. May need 5-20× more data than STE backprop.

### P4: Forward-Forward Layer-Streaming Training
- **Paradigm shift:** Train ONE LAYER at a time with Hebbian local objective — peak memory = single layer, not full model. No global backprop ever.
- **How:** Forward-Forward goodness G_l(x) = ||W_l·x||². Per-weight signal: dG/dw_{ij} = 2·y_i·x_j (Hebbian product). Accumulate-and-flip with INT8 accumulators. Self-contrastive negative generation.
- **Memory:** 132 MB peak for 7B (636× reduction!!)
  - Current training layer: 9 MB
  - Accumulators (INT8): 45 MB
  - Frozen inference layers (streaming): 18 MB
  - Activation buffers: 50 MB
- **Ternary-specific:** Hebbian learning is ADD/SUBTRACT when computed from ternary weights. Entire training loop (forward, signal, update) is INTEGER arithmetic. Ternary self-regularization prevents per-layer overfitting. SSM state carries rich signal between independently-trained layers.
- **Risk:** Impact HIGH, Difficulty HIGH. Forward-Forward has NOT achieved competitive results on large-scale LM. Key bet: 636× scale headroom compensates inferior optimizer.

### P5: Discrete Score-Matching Weight Diffusion
- **Paradigm shift:** Training is DENOISING — start from random ternary config, iteratively denoise toward optimal config via learned score function over discrete space
- **How:** Forward corruption: flip weights randomly with probability β_t. Reverse denoising: score network s_φ predicts which weights are "wrong" given current noisy config. Score network: ~10M params, trained on small model weights (6M, 50M). Cross-scale transfer: local statistical patterns of good ternary matrices generalize across sizes.
- **Memory:** 1.5 GB for 7B (56× reduction)
  - Packed weights: 1.4 GB
  - Score network (FP16): 20 MB
  - Active block scores: 10 KB
  - Forward activations: 100 MB
- **Ternary-specific:** D3PM discrete diffusion designed for categorical variables — ternary is exactly 3-category. Local structure of ternary matrices simpler than FP16 → small score network suffices. Cross-scale generalization hypothesis.
- **Risk:** Impact HIGH, Difficulty HIGH. Cross-scale transfer is key gamble. If works: meta-learning across training runs.

### Combined Pipeline (P4 → P5 → P3)
- Small model training via FF Layer-Streaming (P4): 132 MB
- Score network training on small models (P5): 200 MB
- Large model weight GENERATION via diffusion (P5): 1.5 GB
- Optional fine-tuning via Block-SPSA (P3): 2.4 GB
- **Peak at any point: 2.4 GB for 7B. Backpropagation: NEVER used. FP multiplication: NEVER in forward pass.**

---

## PB2: Radical Paradigms for Ultra-Minimal Inference Memory

### P6: Micro-Stripe Pipelined Inference
- **Paradigm shift:** A weight matrix doesn't need to exist in RAM as a whole — process in vertical stripes of S columns, streaming each from storage
- **How:** Partition d_model into B = d_model/S blocks. y = Σ W[:,bS:(b+1)S] · x[bS:(b+1)S]. Each stripe: m×S ternary weights (256 KB) + S FP16 input + m FP16 accumulator. Input also produced stripe-by-stripe by previous layer.
- **Working memory:** ~852 KB for 7B
  - Weight stripe buffer: 256 KB
  - Input activation stripe: 256 B
  - Output accumulator: 16 KB
  - SSM state (one layer): 256 B
  - KV-cache (8 SWA layers, 3-bit, window=128): 576 KB
- **Throughput:** I/O bound — NVMe 2.8 tok/s, DDR5 20 tok/s, GDDR7 270 tok/s
- **Ternary-specific:** Weight stripes are 2-bit per element → 8× denser than FP16 → 8× smaller loads. Computation is add/subtract → completes faster than I/O (purely I/O bound). FP16 would need 8× larger stripes.
- **Risk:** Impact HIGH, Difficulty LOW. Mathematically IDENTICAL to standard matvec. Zero quality impact.

### P7: Ternary Sign Sufficiency (TSS)
- **Paradigm shift:** In ternary networks, activation MAGNITUDE is largely irrelevant — downstream ternary weights only route add/subtract based on sign
- **How:** Replace intermediate activations with sign(y) × scale (single FP16 per dimension). Error analysis: relative error O(1/√(d/3)) — for d=4096 this is ~2.8% per layer. SSM state stored as (ternary_sign, single_scale_factor).
- **Working memory with P6:** ~865 KB (activation stripe drops from FP16 to ternary)
- **Ternary-specific:** Continuous weights amplify magnitude differences → sign loses critical info. Ternary weights treat all non-zero inputs equally → magnitude is second-order. Residual connections preserve magnitude in skip path.
- **Risk:** Impact MEDIUM, Difficulty MEDIUM. 2.8% per-layer error compounds across 64 layers.

### P8: Incremental Logit Refinement (ILR)
- **Paradigm shift:** Never compute the V×d_model output head matmul — instead each layer incrementally updates running top-K candidate scores
- **How:** Each layer has topic detector t_l ∈ {-1,0,+1}^d (1 KB). Score s_l = t_l · h_l. Precomputed table maps (layer, score_bin) → sparse update over K=256 candidates. Running scores maintained as K INT32 accumulators.
- **Working memory:** ~9.0 MB for 7B (50% reduction vs M14's 18.4 MB)
  - Peak SSM layer weights (streamed): 8.4 MB
  - SSM state (T3S): 3.6 KB
  - KV-cache: 576 KB
  - ILR candidate scores: 1.5 KB
  - Eliminates 25 MB output head entirely
- **Ternary-specific:** Ternary output head produces INTEGER dot products with bounded range → binning and table lookup are EXACT. Top-256 captures >99.9% probability mass.
- **Risk:** Impact HIGH, Difficulty HIGH. Candidate selection correctness is critical.

### P9: Hash-Indexed Inference (HII)
- **Paradigm shift:** Inference is NOT computation but RETRIEVAL — precompute (state→prediction) table offline, convert inference to O(1) hash lookup
- **How:** Offline: cluster SSM states into K centroids, store mean next-token distribution per centroid. Online: hash(quantized_state) → lookup prediction. SSM update requires ONLY A (diagonal) and B (ternary projection), not full network.
- **Working memory:** ~165 KB for 7B (!!)
  - SSM state (all 56 layers, FP16): 14.3 KB
  - Current layer B matrix (streamed): 128 KB
  - A matrix (all layers, diagonal): 14.3 KB
  - Hash table entry: 400 B
- **Disk storage:** 400 MB - 4 GB (K=1M to 10M centroids)
- **Ternary-specific:** Ternary state hashes are EXACT → no rounding in trie traversal. Hamming distance via POPCOUNT for nearest centroid. SSM update is add/subtract only.
- **Risk:** Impact EXTREMELY HIGH, Difficulty HIGH. Centroid coverage of state space is the key risk.

### P10: Trie-Compressed Suffix Memory
- **Paradigm shift:** Model doesn't COMPUTE next token — it RETRIEVES it from compressed trie indexed by ternary state hash. Intelligence is in organization, not computation.
- **How:** Offline: build ternary-branching trie from (SSM_state_hash, next_token) pairs. Online: quantize SSM state → traverse trie (ternary decisions at each node) → read stored distribution at leaf. D_eff ~ 20-40 dimensions suffice (language occupies tiny manifold).
- **Working memory:** ~143 KB for 7B
  - SSM state: 14.3 KB
  - Current trie node: 11 B
  - Node stack (depth 30): 360 B
  - Retrieved distribution: 400 B
  - SSM update buffer: 128 KB
- **Disk storage:** ~165 MB (15M nodes × 11 bytes)
- **Ternary-specific:** Ternary state → exact trie paths (no rounding). Ternary branching factor=3 creates compact tries. SSM hash computation is add/subtract only. FP16 states would require KD-tree (degrades in high dims).
- **Risk:** Impact EXTREMELY HIGH, Difficulty VERY HIGH. Generalization limited to seen contexts. Best as HYBRID: trie fast path + neural fallback.

---

## Aspect 5 Complete Summary

### Findings Count
- Wave 1 (Web): 14 findings (F1-F14)
- Wave 2 (Math): 16 formulations (M1-M16)
- Wave 3 (Paradigm): 10 proposals (P1-P10)
- **Total: 40 findings**

### Key Convergences

**Training Memory (from impossible to practical):**
| Approach | Memory (7B) | Reduction | Uses Backprop? |
|----------|------------|-----------|----------------|
| Standard STE | 84 GB | 1× | Yes |
| TDO + Plateau (M1+M4) | 15 GB | 5.6× | Yes |
| Block-SPSA (P3) | 2.4 GB | 35× | No |
| Potts Tempering (P1) | 1.5 GB | 55× | No |
| Weight Diffusion (P5) | 1.5 GB | 56× | No |
| HyperNet (P2) | 500 MB | 168× | Generator only |
| FF Streaming (P4) | 132 MB | 636× | No |

**Inference Memory (from megabytes to kilobytes):**
| Approach | Working Memory | Throughput |
|----------|---------------|------------|
| Standard (full model in RAM) | ~1.4 GB | 1000+ tok/s |
| Streaming Certificate (M14) | 18.4 MB | bandwidth-limited |
| ILR + Streaming (P8) | 9.0 MB | bandwidth-limited |
| Micro-Stripe (P6) | 852 KB | 270 tok/s (GDDR7) |
| Stripe + TSS (P6+P7) | 865 KB | 270 tok/s |
| Hash-Indexed (P9) | 165 KB | lookup speed |
| Trie Memory (P10) | 143 KB | lookup speed |

### Top 5 Findings for v5

1. **P4 (FF Layer-Streaming)** — 636× training memory reduction. If quality scales, this is paradigm-defining.
2. **P6 (Micro-Stripe Pipeline)** — Sub-1 MB inference with ZERO quality loss. Implementable immediately.
3. **M1 (TDO)** — Conservative 5× optimizer savings with mathematical proof. Safe to implement.
4. **P9 (Hash-Indexed Inference)** — 165 KB working memory. Moonshot but mathematically grounded.
5. **M6 (Binary Gradient Pipeline)** — Full integer backward pass. Eliminates FP from training entirely.

### Overarching Insight
Ternary models uniquely enable gradient-free and local training methods because: (1) search space is discrete with only 3 states per param, (2) decision per weight is binary (flip or not), (3) forward pass is cheap (add/subtract). These properties make methods impractical for FP16 suddenly viable for ternary. Similarly for inference: ternary enables sub-MB working memory via stripe pipelining (8× denser than FP16) and exact hash-based retrieval (discrete state → exact indexing).
