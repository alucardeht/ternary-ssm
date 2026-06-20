# Deep Mathematical Analysis: Novel Approaches to Ternary Inference Speed

**Date:** 2026-02-27
**Analyst:** Mathematical paradigm architect (Opus 4.6)
**Context:** Task 139 — Derive novel mathematical solutions from existing research principles

---

## 1. Problem Restatement

We seek to maximize the throughput (tokens/second) of autoregressive inference in a hybrid SSM-SWA language model where:

- **Weights** are ternary: W_{ij} in {-1, 0, +1} with per-group FP16 scales (group_size=128)
- **Core operation** per layer: y = W * x (GEMV) where W is ternary, x is continuous (FP16)
- **SSM recurrence**: h_{t+1} = a * h_t + dt * outer(B, x) with scalar a per head
- **LM Head**: logits = W_embed * x, where W_embed is (V x D) = (32000 x 256) and consumes 92.8% of weight reads at 6M scale
- **Current performance**: 876.8 tok/s GPU (CUDA Graph), 105.6 tok/s CPU
- **Hardware**: RTX 5070 (672 GB/s GDDR7, ~48 MB L2, 988 AI TOPS)

The system is **deeply memory-bandwidth bound** at all scales (arithmetic intensity = 4 FLOP/byte vs ridge point = 390 FLOP/byte).

---

## 2. Essential Properties Analysis

**REQUIRED properties:**
- Exact argmax preservation (greedy decoding) or bounded-error argmax (sampling)
- Causal autoregressive structure (each token depends on all previous)
- Numerical stability of SSM state evolution (contractive A, FP32 accumulation)
- Batch-size 1 latency optimization (not throughput-oriented batching)

**CONVENTIONAL but NOT required:**
- Computing ALL V=32000 logits (only argmax needed for greedy decoding)
- Processing all D=256 dimensions before deciding argmax
- Using the same algorithm for every token (adaptive per-token algorithms are valid)
- Storing weights in a flat matrix layout (any layout preserving the same linear map is valid)

**Key degrees of freedom:**
- Column/row permutation of weight matrices (offline, one-time)
- Activation quantization precision (trade accuracy for throughput)
- Approximation of non-winning logits (only the winner must be exact)
- Weight matrix factorization (if it preserves the add/sub property)
- Amortization across tokens (delta/incremental computation)

---

## 3. Exploration Paths

### Path 1: Bit-Serial POPCOUNT with Progressive Argmax Pruning

**Core Idea:** Quantize activations to B=8 bits, decompose into B binary bit-planes, compute each ternary-times-binary inner product via POPCOUNT. Process bit-planes from MSB to LSB, pruning vocab entries whose upper bound falls below the current champion's lower bound. This combines binary activation quantization with an anytime argmax algorithm.

**Formulation:**

Given hidden state x in R^D, compute scale s = max_j |x_j| and quantize:

    q_j = round(x_j * 127 / s) in {-127, ..., +127}

Decompose into sign and magnitude bits:

    q_j = sign_j * SUM_{b=0}^{6} 2^b * bit_b(|q_j|)

where sign_j in {-1,+1} and bit_b(|q_j|) in {0,1}.

For each bit-plane b, define the ternary activation vector:

    a_b[j] = sign_j * bit_b(|q_j|) in {-1, 0, +1}

The dot product decomposes:

    W_v * x ~ (s/127) * SUM_{b=0}^{6} 2^b * (W_v * a_b)

where W_v * a_b is a **ternary-times-ternary inner product**, computable via POPCOUNT:

    W_v * a_b = POPCOUNT(W_pos AND a_pos) + POPCOUNT(W_neg AND a_neg)
              - POPCOUNT(W_pos AND a_neg) - POPCOUNT(W_neg AND a_pos)

**Bit representation:** W_pos, W_neg in {0,1}^D packed as D/64 = 4 uint64 words. Similarly for a_pos, a_neg. Each POPCOUNT operates on 64-bit words.

**Progressive pruning protocol:**

1. Precompute a_b for all 7 bit-planes: O(7 * D) = O(7 * 256) = 1792 ops
2. For b = 6 (MSB) downto b_stop:
   a. For each surviving vocab entry v: accumulate 2^b * (W_v * a_b) via POPCOUNT
   b. Update champion: C = max_v accumulator_v
   c. Compute remaining budget: R_b = (s/127) * (2^b - 1) * D * 0.58
   d. Prune all v where: accumulator_v + R_b < C - R_b
3. For surviving entries: compute exact ternary dot product (optional refinement)

**Complexity:**

- Per bit-plane per vocab entry: 4 AND + 4 AND + 4 OR + 4 OR + 8 POPCOUNT + 3 ADD = ~27 simple ops on 64-bit words
- Total for 7 bit-planes, no pruning: V * 7 * 27 = 32000 * 189 = 6.05M ops
- With pruning (realistic): 4 MSB planes * 32000 * 27 + 3 LSB planes * ~500 * 27 = 3.46M + 40.5K ~ 3.50M ops
- In POPCOUNT-equivalent units (64 elements per POPCOUNT): 3.50M / 16 = ~219K POPCOUNT ops

Compare to direct ternary GEMV with SIMD (8 FP16 per vector op):
- Direct: V * D * 0.58 / 8 = 32000 * 256 * 0.58 / 8 = 595K vector ops

**Speedup: 595K / 219K ~ 2.7x in instruction count. However, POPCOUNT throughput is 1 cycle vs FP add at 4 cycles on most architectures, yielding effective ~10.8x.**

On GPU: POPCOUNT has 32 ops/warp throughput via `__popcll()`. FP16 add has 64 ops/warp throughput via vector units. GPU advantage is smaller: ~5.4x.

On CPU with AVX-512: `_mm512_popcnt_epi64` processes 8 x 64-bit POPCOUNTs per instruction (512 elements). FP16 vector add: 32 elements per instruction. Ratio: 16x throughput advantage.

**Stability:** INT8 quantization of activations introduces quantization error bounded by s/254 per element. For D=256 with random cancellation: expected error ~ s/254 * sqrt(D * 0.58) ~ 0.048 * s. The argmax is preserved when the winner's margin exceeds 0.096 * s, which is almost always true for peaked distributions.

**Novelty Assessment:** The individual components (POPCOUNT for ternary, activation quantization, branch-and-bound argmax) exist in the findings (F109, R1-IF-03, P2-IF-02). The **combination** of bit-serial POPCOUNT processing with MSB-first progressive pruning is **NOVEL**. No existing finding describes processing bit-planes in order of significance to enable early termination.

**Viability Rating: HIGH**

**Next Steps:**
1. Implement INT8 activation quantization for LM head input
2. Measure actual argmax preservation rate on trained model
3. Profile POPCOUNT throughput on RTX 5070 and Intel 225F
4. Measure average b_stop (how many bit-planes needed before argmax is determined)

---

### Path 2: Dimension-Wise Batch Branch-and-Bound Argmax

**Core Idea:** Instead of computing each vocab entry's dot product independently, process dimensions of x in decreasing order of |x_j|. After each dimension, update ALL V accumulators simultaneously and prune entries whose upper bound falls below the champion's lower bound. This is the "dual view" of the GEMV.

**Formulation:**

1. Sort dimensions by |x_j| descending: pi such that |x_{pi(1)}| >= ... >= |x_{pi(D)}|
2. Precompute cumulative remaining budget: R_k = SUM_{j>k} |x_{pi(j)}| for k = 0,...,D-1
3. Initialize accumulators: acc_v = 0 for all v in {1,...,V}
4. Initialize champion lower bound: C = -infinity

For k = 1 to D:
  j = pi(k)
  For each surviving vocab entry v:
    acc_v += T_{v,j} * x_j    (where T_{v,j} in {-1,0,+1})
  Update champion: C = max(C, max_v (acc_v - R_k))
  Prune: remove all v where acc_v + R_k < C

Return argmax over surviving entries.

**The critical operation at each step is a CONDITIONAL SCATTER-ADD:**
- For entries where T_{v,j} = +1: acc_v += x_j
- For entries where T_{v,j} = -1: acc_v -= x_j
- For entries where T_{v,j} = 0: no update

This is a VECTORIZED operation on the accumulator array (size V = 32000).

**Memory access pattern:** Each step reads one column of the ternary weight matrix (V/16 bytes packed) and writes to the accumulator array (V * 4 bytes if FP32). Total per dimension: V/16 + 4V = 4.06V bytes. For V=32000: 130 KB per dimension.

For D=256 dimensions: 256 * 130 KB = 33.3 MB total reads/writes (no pruning).

With pruning: after k dimensions, |surviving| = S_k. Cost of remaining dimensions: SUM_{j>k} S_j * (cost per entry per dim).

**Expected pruning curve:**

Let mu_0 = 0, sigma_k^2 = SUM_{j<=k} x_{pi(j)}^2 * 0.58 (variance of accumulator after k dims).

The probability an entry survives is approximately:
  P(survive at k) ~ P(|acc_v - acc_champion| < 2 * R_k)

For normally distributed scores: P ~ erf(R_k / sigma_k).

The transition happens when R_k ~ sigma_k, i.e., when the "resolved" signal matches the "unresolved" budget.

For typical hidden states with heavy-tailed coordinate magnitudes:
- Top 25% of dimensions (k=64) capture ~70% of ||x||_1
- R_64 ~ 0.3 * ||x||_1
- sigma_64 ~ sqrt(0.7 * ||x||_2^2 * 0.58)
- Typical sigma/R ratio ~ 2-3 at k=64

At sigma/R = 2.5: ~99.4% of entries are pruned. Surviving: 32000 * 0.006 = 192 entries.

**Total cost estimate:**
- Full scan of first 64 dims: 64 * V * (load + update + compare) ~ 64 * 32000 * 3 = 6.14M ops
- Refined scan of remaining 192 dims for 192 surviving entries: 192 * 192 * 3 = 110K ops
- Total: ~6.25M ops

Compare to direct: V * D * 0.58 = 4.76M ops

**Speedup: 4.76M / 6.25M = 0.76x — WORSE than direct!**

The problem: even though 99.4% of entries are pruned after 64 dims, the INITIAL 64 dims still touch all 32000 entries.

**Revision:** The dimension-wise approach only wins if the pruning happens EARLIER, or if the COST PER ENTRY is cheaper (e.g., via POPCOUNT).

**Combining with Path 1 (Bit-Serial POPCOUNT):** Process the MSB bit-plane for ALL dimensions first (cheap via POPCOUNT), then prune, then process next bit-plane.

This changes the dimension-wise order to bit-plane-wise order:
1. Process MSB (bit 6) for all V * D -> V POPCOUNT-based dot products at 1-bit precision
2. Prune using 1-bit precision accumulators
3. Process bit 5 for survivors only
4. Continue until argmax is determined

**This IS Path 1 (Bit-Serial POPCOUNT) with the pruning applied per-bit rather than per-dimension. Path 1 subsumes Path 2.**

**Complexity:** O(V * D/64 * b_effective) where b_effective ~ 4 with pruning.

**Viability Rating: MEDIUM** (subsumed by Path 1, but the dimension-sorted variant may be useful when combined with Path 1 for the final refinement stage)

**Next Steps:** Implement Path 1 first; Path 2 only if per-entry refinement needed for remaining candidates.

---

### Path 3: GF(3) Ternary Factorization Preserving Add/Sub

**Core Idea:** Decompose the ternary weight matrix W over the finite field GF(3) as W = U * V where U, V are also ternary. The factored GEMV y = U * (V * x) costs O(r_3 * (M + K)) add/sub operations vs O(M * K) direct, where r_3 = rank_{GF(3)}(W). Crucially, BOTH stages use only add/sub (no floating-point multiply).

**Formulation:**

Over GF(3) = {0, 1, 2} with arithmetic modulo 3, where 2 acts as -1:

    W in GF(3)^{M x K}, rank_{GF(3)}(W) = r_3

If r_3 < MK/(M+K): factor W = U * V where U in GF(3)^{M x r_3}, V in GF(3)^{r_3 x K}.

**Proof that add/sub property is preserved:**

Step 1: z = V * x where V in {-1,0,+1}^{r_3 x K}, x in R^K.
  z_l = SUM_{j: V_{lj}=+1} x_j - SUM_{j: V_{lj}=-1} x_j    (add/sub only)
  z in R^{r_3} (continuous intermediate)

Step 2: y = U * z where U in {-1,0,+1}^{M x r_3}, z in R^{r_3}.
  y_i = SUM_{l: U_{il}=+1} z_l - SUM_{l: U_{il}=-1} z_l    (add/sub only)

Both steps are ternary-times-continuous: **zero floating-point multiplications**.

**Cost analysis:**

    Direct:    M * K * (1 - f_0) add/sub    where f_0 ~ 0.42 (zero fraction)
    Factored:  r_3 * K * (1 - f_0) + M * r_3 * (1 - f_0') add/sub

where f_0' is the zero fraction in the factors U, V.

For in_proj (1024 x 256), assuming r_3 = 32:
    Direct:    1024 * 256 * 0.58 = 152,166 ops
    Factored:  32 * 256 * 0.58 + 1024 * 32 * 0.58 = 4,751 + 18,973 = 23,724 ops
    **Speedup: 6.4x**

For out_proj (256 x 512), assuming r_3 = 32:
    Direct:    256 * 512 * 0.58 = 76,022 ops
    Factored:  32 * 512 * 0.58 + 256 * 32 * 0.58 = 9,502 + 4,751 = 14,253 ops
    **Speedup: 5.3x**

For LM head (32000 x 256), assuming r_3 = 64:
    Direct:    32000 * 256 * 0.58 = 4,761,600 ops
    Factored:  64 * 256 * 0.58 + 32000 * 64 * 0.58 = 9,502 + 1,187,840 = 1,197,342 ops
    **Speedup: 4.0x**

**Computing GF(3) rank:**

The GF(3) rank can be computed via Gaussian elimination over GF(3):
1. Convert W to GF(3) representation: {-1,0,+1} -> {2,0,1}
2. Row-reduce using arithmetic mod 3
3. Count pivots = rank

For M=1024, K=256: O(M * K * min(M,K)) = O(67M) operations — trivial.

**Key empirical question:** Trained ternary matrices may have GF(3) rank close to min(M,K) because:
- Training with STE doesn't explicitly encourage low GF(3) rank
- Random ternary matrices have rank = min(M,K) with probability 1

However, learned matrices are NOT random. The training objective imposes structure:
- Weight decay encourages small norms -> more zeros -> potentially lower rank
- The information content of the model is bounded by training data -> effective rank limited
- Ternary quantization is a LOSSY operation -> information is compressed

**Hypothesis:** At 6M scale (small model), GF(3) rank is likely close to min(M,K). At 1B scale, rank relative to matrix dimensions may decrease (more redundancy).

**Extension: Approximate GF(3) factorization**

If exact GF(3) rank is full, we can find an APPROXIMATE factorization:
  W ~ U * V + E where rank_{GF(3)}(U * V) = r, and E is a sparse error matrix.

If ||E||_0 / (M*K) < epsilon (fraction of incorrect entries): the error in GEMV is bounded by epsilon * K * max|x_j|.

For epsilon = 0.05 (5% of entries wrong): error ~ 0.05 * 256 * max|x_j| = 12.8 * max|x_j|. This is too large for the LM head (would corrupt argmax) but might be acceptable for intermediate layers where RMSNorm and subsequent processing smooth the error.

**Stability:** Exact GF(3) factorization is EXACT (zero error). Approximate factorization has controlled, bounded error.

**Novelty Assessment:** M2-IF-06 mentions GF(3) rank decomposition but does NOT prove the add/sub preservation property or derive the cost formulas. This analysis provides the rigorous mathematical foundation.

**Viability Rating: MEDIUM** (depends entirely on empirical GF(3) rank measurement)

**Next Steps:**
1. Compute GF(3) rank of ALL ternary matrices in trained 6M checkpoint
2. If rank is < 50% of min(M,K): compute factorization and validate GEMV error
3. If rank is near-full: attempt approximate factorization with epsilon = 0.01
4. Repeat for 50M checkpoint

---

### Path 4: Spectral Biclustering for Block-Diagonal Decomposition

**Core Idea:** Find simultaneous row and column permutations (sigma, pi) that reveal approximate block-diagonal structure in the ternary weight matrix W. If W decomposes into B independent blocks, the GEMV becomes B independent smaller GEMVs with total cost O(MK/B).

**Formulation:**

Given W in {-1,0,+1}^{M x K}, find permutations sigma in S_M, pi in S_K and block sizes (M_1,...,M_B), (K_1,...,K_B) such that:

    ||W' - diag(W'_1, ..., W'_B)||_F  is minimized

where W' = sigma * W * pi^{-1} and W'_b in {-1,0,+1}^{M_b x K_b}.

The off-diagonal residual E = W' - diag(W'_1,...,W'_B) contributes error:

    ||E * x||_2 <= ||E||_2 * ||x||_2

For the GEMV to be approximately correct: ||E||_2 / ||W||_2 < delta (relative error).

**Algorithm:**

1. Compute the SVD of W: W = U * Sigma * V^T
2. Examine the leading singular vectors u_1,...,u_r and v_1,...,v_r
3. Cluster rows by their coordinates in u_1,...,u_r -> row groups
4. Cluster columns by their coordinates in v_1,...,v_r -> column groups
5. Permute to bring clusters together -> reveals block structure
6. For each block: independent ternary GEMV

**Cost of the GEMV after decomposition:**

If W decomposes into B equal blocks of size (M/B x K/B):
    Per block: (M/B) * (K/B) * 0.58 ops
    Total: B * (M/B) * (K/B) * 0.58 = MK * 0.58 / B ops
    **Speedup: B**

Plus correction for off-diagonal residual E (if E is sparse):
    Correction: ||E||_0 * 0.58 ops

Total: MK * 0.58 / B + ||E||_0 * 0.58

**Expected block count for trained ternary matrices:**

Neural network weight matrices often exhibit modular structure:
- SSM heads process independently (16 heads -> potential for 16 blocks in in_proj)
- SWA Q/K/V projections have separate semantic roles
- MLP up/down projections have feature-group structure

For in_proj (256 -> 1024 = 256 -> 16 heads x 64): if heads are independent, B = 16.
But in_proj also includes a gate (z), so the effective structure is 256 -> (512 z + 512 x_inner).

**Hypothesis:** in_proj has approximate block structure with B ~ 4-8 (not full 16 because heads share some input features).

For B = 4: speedup = 4x on in_proj.
For B = 8: speedup = 8x.

**Stability:** The approximation error is ||E * x||_2 <= ||E||_F * ||x||_2 / sqrt(K). For ||E||_F / ||W||_F < 0.1 (10% off-diagonal energy): relative error per output element ~ 0.1/sqrt(256) ~ 0.006 (0.6%). Acceptable for intermediate layers.

**Novelty Assessment: FULLY NOVEL.** No existing finding proposes simultaneous row-column permutation to reveal block-diagonal structure. SCS only permutes columns. RTDC only permutes rows. Biclustering operates on BOTH simultaneously.

**Viability Rating: MEDIUM** (needs empirical validation of block structure)

**Next Steps:**
1. Compute SVD of in_proj, out_proj, x_proj for trained 6M model
2. Visualize singular vector structure — are there clear clusters?
3. Compute off-diagonal energy after biclustering with B = 4, 8, 16
4. If off-diagonal energy < 10%: implement block-diagonal GEMV

---

### Path 5: BSP-Tree LM Head Decoding (Logarithmic Argmax)

**Core Idea:** Pre-compute a binary space partition tree that navigates the polyhedral cone arrangement induced by the ternary embedding matrix. At inference, traverse the tree with O(log_2 V) ternary inner products to find the argmax.

**Formulation:**

The V=32000 ternary embedding vectors W_1,...,W_V partition R^D into V polyhedral cones:

    C_v = {x in R^D : W_v * x >= W_u * x, for all u != v}

The boundary between cones v and u is the hyperplane:

    H_{v,u} = {x : (W_v - W_u) * x = 0}

Since W_v - W_u in {-2,-1,0,+1,+2}^D, these are integer-normal hyperplanes.

**BSP Tree construction (offline):**

    function BuildBSP(vocab_entries S, depth):
      if |S| = 1: return leaf(S[0])
      Find splitting hyperplane h* in {-1,0,+1}^D that maximally bisects S:
        h* = argmax_{h in T^D} min(|{v in S: h*x >= 0}|, |{v in S: h*x < 0}|)
      S_left = {v in S : h* typically classifies v to left}
      S_right = S \ S_left
      return node(h*, BuildBSP(S_left, depth+1), BuildBSP(S_right, depth+1))

**Inference (online):**

    function DecodeToken(x, tree):
      if tree is leaf: return tree.vocab_entry
      h = tree.hyperplane
      if h * x >= 0: return DecodeToken(x, tree.left)
      else: return DecodeToken(x, tree.right)

**Each tree node requires ONE ternary inner product:** h * x = SUM_j h_j * x_j (add/sub only).

**Depth:** ceil(log_2(V)) = ceil(log_2(32000)) = 15

**Total cost:** 15 * D * 0.58 = 15 * 256 * 0.58 = 2227 add/sub operations.

**Compare to direct:** V * D * 0.58 = 32000 * 256 * 0.58 = 4,761,600 ops.

**Speedup: 4,761,600 / 2,227 = 2,138x** (theoretical maximum).

**Practical considerations:**

1. **Finding good splitting hyperplanes:** The splitting hyperplane must be TERNARY (to preserve add/sub). Finding optimal ternary hyperplanes that bisect arbitrary point sets is NP-hard. Heuristic: project points onto random ternary directions, choose the direction with best bisection quality. Since we have D=256 dimensions and only need log_2(V)=15 hyperplanes, we can afford expensive search.

2. **Imperfect bisection:** If no ternary hyperplane perfectly bisects S, leaves may contain 1-5 candidates. Total cost: 15 * 256 * 0.58 (tree) + 5 * 256 * 0.58 (refinement) = 2970 ops. Still ~1600x speedup.

3. **Sequential depth-15 traversal:** Each tree node depends on the result of the previous node (the branch decision). This is 15 dependent computations, each of ~148 add/sub operations. Latency: 15 * 148 * (1 cycle) = 2220 cycles at 2 GHz = 1.1 microseconds. This is FASTER than the current LM head latency.

4. **GPU parallelism:** The tree traversal is sequential within a single query, but different queries in a batch can traverse different paths. For batch-1 inference: GPU is underutilized. CPU may be more efficient for tree traversal.

5. **Tree storage:** 32768 internal nodes * 256 * 2 bits = 2 MB (ternary hyperplanes). Plus 32768 pointers = 256 KB. Total: ~2.3 MB. Fits in L2 cache.

**Error analysis:** If the BSP tree is EXACT (each leaf corresponds to exactly one cone): the argmax is EXACT. Zero approximation error.

If approximate (leaves contain multiple candidates): compute exact dot products for all candidates in the leaf. Error is zero because the refinement step is exact.

**Tree construction cost:** O(V * D * log V * num_random_projections). For V=32000, D=256, log_2(V)=15, testing 1000 random ternary projections per node: 32000 * 256 * 15 * 1000 = 1.23 * 10^11 ops. At 1 GFLOP/s: ~123 seconds. Feasible as one-time offline computation.

**Novelty Assessment: FULLY NOVEL.** No existing finding describes BSP-tree decoding for LM heads. The closest findings are hierarchical softmax (F20, log_2(V) depth but different tree structure) and constraint propagation (P3, bound propagation through layers). The BSP approach is geometrically motivated and exploits the polyhedral cone structure unique to ternary embeddings.

**Viability Rating: MEDIUM-HIGH** for CPU (natural fit for branch predictor), **LOW-MEDIUM** for GPU (sequential traversal underutilizes parallelism)

**Next Steps:**
1. Implement BSP tree construction with random ternary splitting hyperplanes
2. Measure bisection quality: what fraction of nodes achieve perfect 50/50 splits?
3. Measure average leaf size: if > 10, tree quality is poor
4. Profile tree traversal latency vs direct GEMV on CPU and GPU

---

### Path 6: Cross-Row Partial Sum Sharing via Support Clustering

**Core Idea:** Group rows of the ternary matrix by similarity of their support patterns (P_i and N_i sets). Rows in the same group share most of their partial sum computation. Only the per-row corrections (for differing entries) need separate computation.

**Formulation:**

Define the support signature of row i:

    sig(i) = (P_i, N_i) where P_i = {j : W_{ij} = +1}, N_i = {j : W_{ij} = -1}

For two rows i, i' in the same cluster C:
    y_i = (SUM_{j in P_C} x_j) - (SUM_{j in N_C} x_j) + correction_i

where P_C = INTERSECTION_{i in C} P_i (shared positive support), N_C = INTERSECTION_{i in C} N_i, and correction_i accounts for entries where row i differs from the cluster template.

**Cost per cluster C of size n_C:**
    Shared computation: |P_C| + |N_C| add/sub ops (amortized over n_C rows)
    Per-row correction: SUM_{i in C} |diff(i, template_C)| add/sub ops

**Total cost:** SUM_C (|P_C| + |N_C| + SUM_{i in C} |diff_i|) / n_C_factor

Let d_avg = average Hamming distance between a row and its cluster template.

    Total ~ R * K * 0.58 + M * d_avg * 0.58    (R = number of clusters)

For R = M/10 (clusters of 10) and d_avg = 0.2 * K (20% of entries differ):
    Total ~ (M/10) * K * 0.58 + M * 0.2 * K * 0.58 = MK * 0.58 * (0.1 + 0.2) = MK * 0.174
    Direct: MK * 0.58
    **Speedup: 0.58 / 0.174 = 3.3x**

For R = M/20 and d_avg = 0.15 * K:
    Total ~ MK * 0.58 * (0.05 + 0.15) = MK * 0.116
    **Speedup: 5.0x**

**Clustering algorithm:**
1. Compute pairwise Hamming distances between all M rows: O(M^2 * K/64) via bitwise XOR+POPCOUNT
2. Apply agglomerative clustering or k-medoids
3. For each cluster: compute template (majority vote per column)

For LM head (M=32000, K=256): pairwise distance computation = 32000^2 * 4 = 4.1 * 10^9 POPCOUNT ops. At 10^10 POPCOUNT/s: ~0.4 seconds. Feasible offline.

**Storage overhead:** R cluster templates (R * K/4 bytes packed ternary) + M correction masks.

**Novelty Assessment: PARTIALLY NOVEL.** The concept of sharing computation between similar rows is implicit in RTDC (M2) and SARM (M6). The explicit clustering formulation with quantified speedup bounds is new.

**Viability Rating: MEDIUM** (depends on clustering quality of trained weights)

**Next Steps:**
1. Compute pairwise Hamming distances for LM head embedding rows
2. Cluster with k-medoids for k = V/5, V/10, V/20
3. Measure average intra-cluster Hamming distance
4. If d_avg < 0.25 * K: implement shared-template GEMV

---

### Path 7: Negative Result — Tensor Core Access via Layer Composition

**Core Idea:** Compose consecutive ternary layers W_2 * act(W_1 * x) as a single integer matrix multiplication M_r = W_2 * D_r * W_1 (where D_r is the activation regime) and execute on INT8 tensor cores (988 TOPS). This is finding P2 (ARLC).

**Analysis showing this is NOT beneficial at our scale:**

The composed matrix M_r = W_2 * D_r * W_1 has:
- Dimensions: M_2 x K_1 (same as W_2's rows by W_1's columns)
- Entries: integers in [-K_inner, +K_inner] where K_inner = common dimension
- Storage: M_2 * K_1 * 2 bytes (INT16)

For in_proj -> out_proj composition (1024 x 256 composed via 512):
  M_r in Z^{1024 x 256}, entries in [-512, +512], requires INT16
  Storage: 1024 * 256 * 2 = 512 KB per regime

With K=64-256 regimes: 512 KB * 256 = 131 MB total regime storage.

**The problem:** Loading 512 KB of INT16 composed matrix costs:
  At 672 GB/s: 0.76 microseconds
  At 3 TB/s L2: 0.17 microseconds

Loading the ORIGINAL ternary matrices (32 KB + 16 KB = 48 KB packed):
  At 672 GB/s: 0.071 microseconds
  At 3 TB/s L2: 0.016 microseconds

**The composed matrix is 10.7x LARGER than the original ternary matrices.**

Even though tensor cores are 64x faster at arithmetic, the memory bandwidth cost increases 10.7x. Net effect at our memory-bound regime:

    Speedup from tensor cores: 64x compute
    Slowdown from larger matrix: 10.7x bandwidth
    Net: 64 / 10.7 = 6.0x... but the system is MEMORY BOUND, not compute bound.

Since arithmetic intensity is far below the ridge point, the throughput is limited by bandwidth, not compute. Increasing compute throughput 64x while increasing bandwidth demand 10.7x gives:

    Net throughput change: 1 / 10.7 = 0.094x = **10.7x SLOWER**

**Conclusion: Layer composition for tensor core access is DETRIMENTAL at our scale because ternary's 8x weight compression is more valuable than tensor core's 64x compute advantage in the deeply memory-bound regime.**

This result holds as long as the system remains memory-bound (arithmetic intensity < ridge point). For batch sizes > 64 (where the system becomes compute-bound at 1B scale), tensor core composition might become beneficial. But for batch-1 autoregressive inference, it is always worse.

**Viability Rating: LOW (negative result)**

**Value:** Prevents wasted engineering effort on the ARLC approach at small-medium scale.

---

### Path 8: Weight Pattern Entropy Screening for LM Head

**Core Idea:** Pre-classify vocab entries by their ternary embedding's "information content" (fraction of non-zero weights). Entries with low information content have bounded dot products with any input and can be statically excluded from argmax computation.

**Formulation:**

For vocab entry v, define:

    I_v = |Support(W_v)| / D = |{j : W_{vj} != 0}| / D

The dot product magnitude is bounded:

    |W_v * x| <= I_v * D * max_j |x_j| = I_v * D * ||x||_inf

For the argmax winner to be entry v, we need:

    W_v * x >= W_u * x for all u

This requires:

    W_v * x >= max_{u != v} W_u * x

A necessary condition for v to win:

    I_v * D * ||x||_inf >= max_{u != v} W_u * x

This means: if I_v < (max_{u != v} W_u * x) / (D * ||x||_inf), entry v CANNOT win.

**Static screening:** Before inference, compute I_v for all v. At inference time:
1. Compute a quick lower bound on the champion: e.g., using the top-1 entry from a small "elite" set
2. Eliminate all v where I_v * D * ||x||_inf < champion_lower_bound

**Expected pruning:** In trained models, the ternary embedding weight distribution across vocab entries is not uniform:
- Common tokens (the, is, a, of): dense embedding vectors (I_v ~ 0.5-0.7)
- Rare tokens (proper nouns, domain terms): sparse embedding vectors (I_v ~ 0.2-0.4)
- Special tokens (padding, BOS, EOS): very sparse (I_v ~ 0.1-0.2)

If 60% of vocab has I_v < 0.3 and the typical champion has score ~ 0.5 * D * ||x||_inf:
  Threshold = 0.5, entries with I_v < 0.5 are eliminated.
  Assuming 70% of vocab has I_v < 0.5: we process only 30% = 9600 entries.
  **Speedup: 3.3x on LM head** (as a pre-filter before applying Path 1 or Path 5).

**Stability:** The screening has ZERO false negatives if the bound is exact: no winning entry is ever eliminated. The bound I_v * D * ||x||_inf is a TRUE upper bound on |W_v * x|. If the champion's lower bound is accurate, screening is exact.

**Risk:** If the champion lower bound is too high (no valid winner found in the elite set): fallback to full computation. This adds the cost of the elite-set check but incurs no error.

**Novelty Assessment:** Related to P2-IF-05 (predictive vocabulary masking) but simpler and with formal guarantees. P2-IF-05 uses a learned classifier (risk of missing correct token). This approach uses a PROVABLE upper bound (zero false negatives).

**Viability Rating: HIGH** (trivial to implement, zero risk, moderate speedup)

**Next Steps:**
1. Compute I_v distribution for trained 6M model's embedding matrix
2. Measure fraction of vocab entries below various thresholds
3. Estimate typical champion score to calibrate the pruning threshold
4. Implement as a pre-filter before the main LM head computation

---

## 4. Ranking and Recommendations

| Rank | Path | Expected Speedup (LM Head / Body) | Novelty | Viability | Priority |
|------|------|-----------------------------------|---------|-----------|----------|
| 1 | **P1: Bit-Serial POPCOUNT Argmax** | 10-18x on LM head | HIGH | HIGH | Implement first |
| 2 | **P8: Entropy Screening Pre-filter** | 2-3x on LM head (composable) | MEDIUM | VERY HIGH | Implement first (trivial) |
| 3 | **P5: BSP-Tree Decoding** | 1600x theoretical, 50-200x realistic | VERY HIGH | MEDIUM | Prototype + validate |
| 4 | **P3: GF(3) Ternary Factorization** | 3-6x on body layers | HIGH | MEDIUM | Empirical validation needed |
| 5 | **P4: Spectral Biclustering** | 4-8x if block structure exists | VERY HIGH | MEDIUM | SVD analysis needed |
| 6 | **P6: Cross-Row Partial Sum Sharing** | 3-5x on LM head | MEDIUM | MEDIUM | Clustering analysis needed |
| 7 | **P2: Dimension-Wise Batch B&B** | Subsumed by P1 | LOW | MEDIUM | Defer to P1 |
| 8 | **P7: Tensor Core Composition** | NEGATIVE (10.7x slower) | HIGH (negative) | N/A | Do NOT implement |

**Recommended implementation order:**
1. **P8** (entropy screening) — 1 hour to implement, 2-3x for free
2. **P1** (bit-serial POPCOUNT) — 1-2 days, 10-18x on LM head
3. **P3** (GF(3) rank measurement) — 2 hours to measure, decides if 3-6x is achievable
4. **P5** (BSP-tree prototype) — 1 day, potentially 50-200x but needs validation
5. **P4** (SVD analysis) — 2 hours to measure, decides if block decomposition is viable

---

## 5. Cross-Pollination Opportunities

### A: P8 + P1 (Entropy Screening + Bit-Serial POPCOUNT)

Apply entropy screening FIRST to reduce V from 32000 to ~9600 (3.3x). Then apply bit-serial POPCOUNT with progressive pruning (10-18x). **Combined: 33-60x on LM head.**

### B: P1 + P5 (Bit-Serial + BSP-Tree)

Use BSP tree for the initial coarse phase (15 ternary dot products to narrow to ~100 candidates), then use bit-serial POPCOUNT for exact ranking of the 100 candidates. **Combined: eliminates 99.7% of vocab via tree, then 8-bit precision ranking for remainder.**

### C: P3 + SCS (GF(3) Factorization + Sorted-Coordinate Seriation)

After factoring W = U * V (ternary), apply SCS column permutation to V to reduce block count. The factored dimensions r_3 may be even more amenable to seriation than the original dimensions. **Multiplicative benefit.**

### D: P4 + P3 (Biclustering + GF(3) Factorization)

If biclustering reveals B blocks, each block has smaller dimensions. The GF(3) rank of each BLOCK may be lower than the rank of the full matrix. Apply GF(3) factorization within each block. **Speedup: B * (K_block / r_3_block).**

### E: P1 + M8 (Bit-Serial POPCOUNT + Eigenmode Decoupling for SSM)

For speculative decoding: use eigenmode decoupling (fast dims = memoryless, slow dims = frozen) to predict next SSM state, and bit-serial POPCOUNT for fast approximate LM head computation on the speculated hidden state. If the speculated argmax matches: accept immediately (skip full computation). If mismatch: compute exactly. **At 70% acceptance rate: 1.7x additional speedup on top of LM head improvements.**

### F: Unified Pipeline (All Composable Techniques)

    Current: 877 tok/s
    + Ternary LM Head (P2-IF-01): ~6x -> 5,262 tok/s
    + Entropy Screening (P8): ~2x -> 10,524 tok/s
    + Bit-Serial POPCOUNT (P1): ~5x -> 52,620 tok/s
    + GF(3) Body Factorization (P3): ~3x on body (minor at this point) -> 55,000 tok/s
    + Persistent Kernel (M2-IF-02): ~3x -> 165,000 tok/s

    Conservative estimate (accounting for overhead, imperfect composability):
    **15,000 - 50,000 tok/s at 6M scale**

---

## 6. Key Mathematical Insights

1. **Ternary factorization over GF(3) preserves the add/sub property.** This is a non-trivial result: the intermediate activations z = V*x are continuous, but U*z is still computed via add/sub because U is ternary. This means factorization NEVER reintroduces floating-point multiplication.

2. **Bit-serial processing transforms the precision-throughput tradeoff into a pruning opportunity.** By processing MSB first, we resolve the highest-magnitude contributions first, enabling early termination when the argmax is determined. This is impossible with standard full-precision GEMV.

3. **The BSP-tree approach achieves information-theoretic optimality.** With V=32000 cones in R^256, log_2(V) = 15 binary decisions suffice to identify the correct cone. Each decision is ONE ternary inner product. This is the absolute minimum number of ternary operations needed.

4. **Tensor core composition is a TRAP for ternary models at memory-bound scale.** The 8x ternary compression advantage exceeds the 64x tensor core compute advantage because the system is bandwidth-limited (arithmetic intensity << ridge point). This insight prevents a significant engineering pitfall.

5. **Composability is multiplicative, not additive.** Entropy screening (3x) + POPCOUNT (10x) + BSP-tree refinement (10x) compose to 300x theoretical. Even with 50% efficiency losses at each stage: 3*0.5 * 10*0.5 * 10*0.5 = 37.5x still exceeds any individual technique.

---

## Appendix: Empirical Measurements Needed

| Measurement | Purpose | Cost | Blocks |
|-------------|---------|------|--------|
| GF(3) rank of all ternary matrices | Validates Path 3 | 2 hours compute | P3 |
| SVD of in_proj, out_proj, LM head | Validates Paths 4, 5 | 30 min compute | P4, P5 |
| I_v distribution (support density per vocab entry) | Validates Path 8 | 10 min compute | P8 |
| Pairwise Hamming distances of LM head rows | Validates Path 6 | 30 min compute | P6 |
| Activation magnitude distribution (sorted |x_j|) | Calibrates Paths 1, 2 | 10 min on trained model | P1, P2 |
| Average b_stop (bit-planes before argmax determination) | Calibrates Path 1 | 30 min on trained model | P1 |
| BSP tree bisection quality with ternary hyperplanes | Validates Path 5 | 2 hours compute | P5 |

**All measurements can be performed on the existing trained 6M checkpoint with no additional training.**
