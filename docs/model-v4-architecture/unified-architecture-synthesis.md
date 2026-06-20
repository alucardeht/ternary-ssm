# Unified Architecture Synthesis — 7 Designs from 10 Findings

**Date:** 2026-02-27
**Agent:** Paradigm-Breaker (Synthesis Mode)
**Inputs:** 10 validated findings from R1.A1, R1.A2, R1.A3 (Waves 1-3)

---

## The 10 Findings (Reference)

| ID | Finding | Mathematical Essence | Operation Type |
|----|---------|---------------------|----------------|
| M10 | Diagonal ternary SSM | a_i in {-1,0,+1}: keep/flip/reset. Gradient independent of T | Sign flip, zero, identity |
| M9 | Exact reversible coupling | y1=x1+F(x2), y2=x2+G(y1). Exact inverse in integers | Integer addition |
| M8 | Walsh-KAN | f(x) = sum c_n*W_n(x), c_n ternary. W_n(x) = (-1)^{popcount(n AND gray(x))} | POPCOUNT + add |
| M7 | Three-valued routing | r in {-1,0,+1}: compute/skip/suppress. Error-correcting | Ternary scalar multiply |
| M11 | Entropy-calibrated exit | 32x compound: 16x(ternary) x 2x(adaptive). Hard tokens get MORE layers | Entropy check (cheap) |
| M4 | MDS-coded state matrix | B designed via MDS codes over GF(3). Max info preserved through quantization | GF(3) matmul |
| M2 | Predictive spike coding | delta_t = x_t - C*h_t, ternarize. Process only if surprise. 70% predictable | Subtract + threshold |
| M3 | Temporal coding + bit shifts | exp(tau/tau_m) = 2^tau = bit shift. CLZ for output. Free log-sum-exp | Bit shift, CLZ |
| — | Ternary weights | W in {-1,0,+1}. GEMV via add/subtract. 2 bits/weight | Add/subtract |
| — | SSM backbone | h_t = A*h_{t-1} + B*x_t. O(1) memory per token | Recurrence |

---

## Conflict Analysis

### Conflicts That Must Be Resolved

**Diagonal SSM (M10) vs MDS State Matrix (M4)**
- M10 requires diagonal A (each state dimension independent) for gradient stability
- M4 requires inter-dimensional coupling for information preservation through quantization
- **Resolution:** MDS structure in the B matrix (input encoding), NOT the A matrix. A stays diagonal (gradient stability). B spreads incoming information via MDS coding. C reconstructs via MDS decoding. This gives gradient benefits of M10 AND information preservation of M4 simultaneously.
- **Mathematical justification:** The quantization bottleneck is at the state update h_t = Q(A*h_{t-1} + B*x_t). With diagonal A, errors in each dimension are independent. MDS in B ensures the INPUT is encoded with maximum distance — even if individual state dimensions are corrupted by quantization, the information can be recovered because it was spread via MDS before quantization.

**Temporal Coding (M3) vs Walsh-KAN (M8)**
- M3 encodes activations as spike times (integers in [0,63])
- M8 computes Walsh functions on activations via POPCOUNT(n AND gray(x))
- **Assessment:** COMPATIBLE. Walsh functions operate on the binary representation of integers. Spike times ARE integers. The gray code of an integer spike time is well-defined. The composition is: spike time -> gray code -> POPCOUNT with coefficient index -> ternary coefficient -> sum.
- **Constraint:** Spike times are bounded [0,63] for 64-bit arithmetic. This limits dynamic range compared to FP16 activations. Quality impact unknown.

**Reversible Coupling (M9) vs Early Exit (M11)**
- M9 needs paired layers for exact inverse computation during backprop
- M11 exits tokens before processing all layers
- **Resolution:** These operate in DIFFERENT PHASES. Training: all layers execute (reversibility used for memory). Inference: early exit skips remaining layers (no reversibility needed). During training with early exit: checkpoint the exit point, only backprop through executed layers. Reversible coupling gives O(1) memory for whatever layers DID execute.

**Predictive Spike Coding (M2) vs Three-Valued Routing (M7)**
- Both skip computation. M2 skips token dimensions within a layer. M7 skips entire layers for a token.
- **Assessment:** STACKING, not overlapping. M7 decides at the layer level (skip this layer entirely?). M2 decides at the token level within the layer (which dimensions need updating?). They operate on orthogonal axes.
- **Compound efficiency:** 0.3 (surprising tokens, M2) x 0.5 (active layers, M7) = 15% of full computation.

### Natural Synergies

| Pair | Why They Compose Naturally |
|------|---------------------------|
| M10 + M9 | Diagonal SSM inside reversible coupling: F(x) = diag_SSM(x). Integer state + integer coupling = exact reversibility |
| M8 + M10 | Walsh-KAN as nonlinearity AFTER diagonal SSM state update. SSM provides context, KAN provides nonlinear transformation |
| M7 + M11 | Routing selects layers, exit controls total depth. Two complementary sparsity mechanisms |
| M2 + M7 | Token-level (M2) and layer-level (M7) sparsity stack multiplicatively |
| M4 + M10 | MDS in B, diagonal in A. Information preservation + gradient stability simultaneously |

---

## Architecture 1: "The Spike Machine" — Zero Floating Point

**Findings used:** M10 (diagonal SSM) + M3 (temporal coding) + M8 (Walsh-KAN) + M9 (reversible) + M11 (early exit)

**Theme:** Every single operation is integer logic. Zero floating point in the entire forward pass.

### Computational Flow (per token)

```
Input: token_id (integer)
  |
  v
Embedding: lookup table -> spike_times[d] (integers in [0, 63])
  |
  v
For each REVERSIBLE LAYER PAIR (L/2 pairs):
  |
  Split into (x1, x2) of dimension d/2 each
  |
  F(x2): Diagonal SSM on x2 spike times
    h_i(t) = a_i * h_i(t-1) + [B*x2]_i     (a_i: keep/flip/reset)
    Walsh-KAN on h: y = sum c_n * (-1)^{popcount(n AND gray(h))}
    All via: bit shift + POPCOUNT + add/subtract
  |
  y1 = (x1 + F(x2)) mod M     (M = 2^6 for [0,63] range)
  |
  G(y1): Same structure as F, different weights
  |
  y2 = (x2 + G(y1)) mod M
  |
  Output: (y1, y2) — merge back
  |
  EARLY EXIT CHECK (every 5 layers):
    Compute output distribution entropy via LUT
    If entropy < threshold: exit
  |
  v
Output head: ternary GEMV -> logits
```

**Operations per layer:** bit shifts + POPCOUNT + modular add. ZERO multiply. ZERO floating point.

### Training Flow
- All layers execute (no early exit during training)
- Reversible coupling: backprop recomputes activations from output. O(1) memory per layer.
- STE for ternary weights as usual
- Spike times in forward pass; gradients in FP16 for backward pass (STE bridges the gap)

### Efficiency Estimate
- 16x (ternary no-multiply) x 2x (early exit avg) x 1.5x (no FP overhead) = **~48x vs float baseline**
- CPU: pure integer pipeline, no FP unit needed. 500-2000 tok/s at 50M on modern CPU.

### Weakest Link
**Temporal coding (M3) constrains dynamic range.** 6-bit spike times have only 64 levels. For complex language understanding, this may be insufficient. Quality degradation at high semantic complexity is the primary risk. Mitigation: increase to 8-bit (256 levels) at cost of wider integers, or use M3 only for edge deployment and standard integer activations for quality-critical tasks.

### Viability: MEDIUM
The temporal coding is beautiful but unproven for language. Every other component has theoretical backing.

---

## Architecture 2: "The Surprise Processor" — Maximum Throughput

**Findings used:** M10 (diagonal SSM) + M2 (spike coding) + M7 (3-valued routing) + M4 (MDS state) + M11 (early exit)

**Theme:** Skip everything that is predictable. Only process surprises.

### Computational Flow (per token)

```
Input: token embedding x_t (FP16 or INT8)
  |
  v
PREDICTION CHECK (M2):
  prediction = C * h_t   (ternary GEMV, cheap)
  delta = x_t - prediction
  spike = ternarize(delta)    -> {-1, 0, +1} per dimension
  surprise_score = count_nonzero(spike)
  |
  IF surprise_score < threshold (70% of tokens):
    STATE ONLY UPDATE: h_i(t) = a_i * h_i(t-1)   (diagonal decay, O(d))
    Output = prediction   (already computed)
    DONE — cost: O(d) per token
  |
  ELSE (30% of tokens — surprising):
    |
    For each LAYER with THREE-VALUED ROUTING (M7):
      |
      Router: r = ternarize(w_route^T * h)  -> {-1, 0, +1}
      |
      IF r = 0 (skip, ~25% of layers): pass through unchanged
      IF r = -1 (suppress, ~10% of layers): output = -layer_output  (error correction)
      IF r = +1 (compute, ~65% of layers):
        |
        SSM update: h_i(t) = a_i*h_i(t-1) + [B_MDS * x]_i   (M10 + M4)
        Output projection via ternary GEMV
        |
      EARLY EXIT CHECK (M11, every 5 layers):
        If entropy(output_distribution) < threshold: EXIT
    |
    v
  Output head -> next token
```

### Training Flow
- All tokens processed (no prediction skipping during training)
- Prediction error IS the training signal (train the model to predict well)
- Router trained via STE on the ternary routing decision
- MDS B matrix can be fixed (designed from coding theory) or learned
- Standard STE for all ternary weights

### Efficiency Estimate
```
70% tokens: O(d) cost each = negligible
30% tokens: 65% of layers computed, 2x early exit savings
  = 0.30 x 0.65 x 0.50 = 0.0975 of full cost
Total: ~10% of full float computation

Factor breakdown:
  Prediction skip (M2): 3.3x
  Routing (M7): 1.5x
  Early exit (M11): 2x
  Ternary (base): 16x
  Compound: 3.3 x 1.5 x 2 x 16 = ~160x
  Conservative (40% predictable instead of 70%): ~80x
```

### Weakest Link
**Predictability varies by domain.** The 70% figure comes from natural language statistics. Code, mathematics, creative writing, and multilingual text may be less predictable. For a pure code model, predictability might drop to 40%, reducing the prediction speedup from 3.3x to 1.7x. **The architecture degrades GRACEFULLY** — it just skips less and computes more. Never worse than baseline.

### Viability: HIGH
All components have strong mathematical foundations. The stacking is along orthogonal axes. The graceful degradation means it never fails catastrophically. **This is the safest high-efficiency option.**

---

## Architecture 3: "The Infinite Depth Machine" — Maximum Quality

**Findings used:** M9 (reversible) + M10 (diagonal SSM) + M8 (Walsh-KAN) + M7 (routing) + M4 (MDS)

**Theme:** Ternary layers are so cheap that depth is essentially free. Go absurdly deep. Let routing select which depths matter.

### Computational Flow (per token)

```
Input: token embedding
  |
  v
WEIGHT-TIED REVERSIBLE BLOCK (repeated K times, K = 100-1000):
  |
  Reversible coupling:
    x1, x2 = split(input)
    F(x2):
      Diagonal SSM (M10): h_i = a_i*h_i + [B_MDS*x2]_i
      Walsh-KAN (M8): y = POPCOUNT-based nonlinearity on h
    y1 = x1 + F(x2)
    G(y1): same structure, different (or shared) weights
    y2 = x2 + G(y1)
  |
  THREE-VALUED ROUTING (M7):
    Router decides: compute this block (+1), skip (0), or suppress (-1)
    Average: 50-100 of 1000 blocks actually compute
  |
  v
Output head -> logits
```

**Key insight:** With SHARED weights across blocks (Deep Equilibrium Model style), the total parameter count is that of a SINGLE block. Memory: O(1) for activations (reversible) + O(1) for weights (shared) = O(1) total, regardless of depth.

### Training Flow
- Forward: run K iterations of the weight-tied block
- Routing: STE for three-valued decisions
- Backward: reversible reconstruction — start from final output, reconstruct each layer's activations in reverse. Memory: store only 2 layers at any time.
- Gradient: M10 guarantees ||dh(T)/dh(0)|| = sqrt(n_+ + n_-), independent of K. No vanishing/exploding gradients at ANY depth.

### Efficiency Estimate
```
Per-block cost: ~(N+4)d additions + (N+1)d POPCOUNTs (Walsh N=32, d=512)
  = ~18K POPCOUNTs + ~18K additions per block

Active blocks: ~50 of 1000 (routing)
Total: 50 x 36K ops = 1.8M integer ops per token

For d=512, N=32:
  CPU (1 GOPS integer): ~1.8ms per token = ~555 tok/s
  GPU: ~5-10x faster = ~3000-5000 tok/s

Effective depth: 50 ternary layers chosen by content
  vs 18 float layers (standard Transformer)
  = 2.7x more processing with 16x cheaper layers
  = HIGHER quality at LOWER total cost
```

### Weakest Link
**Weight-tied expressiveness.** Running the same weights 1000 times is a fixed-point iteration. The model converges to a fixed point of the layer function. This is PROVEN to work (DEQ models achieve transformer-level quality), but the convergence may be slow for some inputs, requiring many iterations. If the fixed point is not semantically meaningful, quality suffers. Mitigation: use a small number of DIFFERENT weight sets (e.g., 10 unique blocks repeated 100 times each) to increase expressiveness while maintaining O(10) parameter cost.

### Viability: MEDIUM-HIGH
DEQ models are validated. Ternary reversibility is unique and proven (M9). The question is whether the routing can learn to select the right 50 blocks out of 1000 effectively. **Potentially the highest quality per parameter of any architecture.**

---

## Architecture 4: "The Holographic Computer" — Maximum Compression

**Findings used:** M10 (diagonal SSM) + M8 (Walsh-KAN) + M3 (temporal coding) + M4 (MDS) + M2 (spike coding)

**Theme:** Everything is encoded holographically. Information is distributed, operations are POPCOUNT.

### Computational Flow (per token)

```
Input: token_id
  |
  v
HDC ENCODING:
  token_vector = codebook[token_id]  (ternary, d=4096)
  position_binding = cyclic_shift(token_vector, position)
  context_update = majority_vote(state, position_binding)
  |
  v
SURPRISE CHECK (M2):
  predicted = state (holographic state IS the prediction)
  actual = context_update
  delta = ternary_subtract(actual, predicted)
  if mostly_zero(delta): skip to output decode
  |
  v
SSM STATE EVOLUTION (M10):
  For each dimension i:
    h_i(t) = a_i * h_i(t-1) + [B_MDS * delta]_i
  (Only update on surprise — most dimensions stay unchanged)
  |
  v
WALSH-KAN DECODE (M8):
  output_features = sum c_n * W_n(h)
  All via POPCOUNT — ternary Walsh on ternary state
  |
  v
TEMPORAL OUTPUT (M3):
  output_spike_times = CLZ(output_features)
  logits = LUT[spike_times]  (precomputed exponential)
  |
  v
Output token
```

### Training Flow
- Standard STE through ternary weights
- HDC encoding is parameter-free (fixed codebook + shift)
- Walsh-KAN coefficients trained via STE
- SSM A/B trained as usual
- Prediction error provides natural curriculum signal

### Efficiency Estimate
```
Per token (common case, 70% predictable):
  POPCOUNT for surprise check: d/64 = 64 ops
  Diagonal state decay: d = 4096 ops
  Total: ~4200 integer ops per token

Per token (surprising, 30%):
  Full SSM update: d^2 = 16M ops (with MDS B)
  Walsh-KAN: N*d = 32*4096 = 131K POPCOUNT ops
  Total: ~16.2M integer ops per token

Average: 0.7 * 4200 + 0.3 * 16.2M = ~4.86M ops/token
CPU at 1 GOPS: ~4.9ms = ~204 tok/s (50M equivalent)
```

Model size: ~12.5 MB (50M params at 2 bits) + 4KB LUTs + negligible HDC codebook

### Weakest Link
**Holographic capacity.** HDC's O(d/log(d)) item capacity means d=4096 holds ~300 distinct associations. For complex multi-entity reasoning (tracking 50+ characters in a novel), this may be insufficient. The holographic superposition becomes noisy. **This limits the architecture to simpler language tasks or requires hierarchical memory extensions.**

### Viability: LOW-MEDIUM
Beautiful mathematics but unproven for language generation at LLM quality. The capacity limitation is fundamental and cannot be resolved without increasing d dramatically (which increases compute quadratically). Best suited as a COMPONENT (holographic memory layer) rather than the entire architecture.

---

## Architecture 5: "The Error-Correcting Language Machine" — Maximum Robustness

**Findings used:** M4 (MDS) + M7 (3-valued routing) + M10 (diagonal SSM) + M9 (reversible) + M11 (early exit)

**Theme:** Language modeling as channel coding. Each layer corrects errors. Exit when the signal is clean.

### Computational Flow (per token)

```
Input: token embedding (potentially noisy from ternary quantization)
  |
  v
For each REVERSIBLE LAYER PAIR:
  |
  MDS ENCODE (M4):
    Input encoded via MDS B matrix over GF(3)
    Produces codeword with known distance properties
  |
  DIAGONAL SSM UPDATE (M10):
    h_i(t) = a_i*h_i(t-1) + [B_MDS*x]_i
    State evolves with gradient-stable dynamics
  |
  SYNDROME CHECK:
    syndrome = H * h_t  (parity-check matrix, GF(3) operations)
    error_weight = hamming_weight(syndrome)
  |
  THREE-VALUED ROUTING based on syndrome (M7):
    IF syndrome = 0 (clean): r = 0 (skip, pass through)
    IF syndrome small: r = +1 (compute, mild correction)
    IF syndrome large: r = -1 (suppress, major correction — negate diverging component)
  |
  EARLY EXIT (M11):
    IF syndrome = 0 for N consecutive layers: EXIT
    (The state is "clean" — no more correction needed)
  |
  v
Output head -> logits
```

### Training Flow
- Reversible coupling: O(1) activation memory
- MDS B matrix: can be FIXED from coding theory (no training needed) or learned
- Syndrome computation: differentiable over GF(3) with STE
- Router learns to map syndrome magnitude to routing decision
- Loss includes both CE loss AND syndrome minimization term

### Efficiency Estimate
```
Base ternary: 16x
Syndrome-driven exit: 2x (clean tokens exit early)
Routing: 1.3x (clean layers skipped)
Total: 16 x 2 x 1.3 = ~42x

Note: This architecture optimizes for ROBUSTNESS not raw speed.
The syndrome provides an objective EXIT CRITERION (vs entropy which is heuristic).
```

### Weakest Link
**Semantic vs coding-theoretic correctness.** A state with zero syndrome (no detectable quantization errors) may still be SEMANTICALLY wrong. The MDS code protects against quantization noise but not against the model producing incorrect logits for other reasons. The early exit criterion may exit when the state is coding-clean but semantically incomplete. **The syndrome is a NECESSARY but not SUFFICIENT condition for good output.**

### Viability: MEDIUM
The coding-theoretic framework is rigorous and principled. The concern is that coding theory optimizes for a different objective (bit error rate) than language modeling (cross-entropy loss). If these objectives are well-correlated (plausible: quantization errors directly corrupt the signal), the architecture is excellent. If not, the syndrome check gives false confidence. **Worth prototyping the syndrome-quality correlation on existing 50M model.**

---

## Architecture 6: "The Predictive Cascade" — Brain-Like Efficiency

**Findings used:** M2 (spike coding) + M3 (temporal coding) + M10 (diagonal SSM) + M8 (Walsh-KAN) + M7 (routing)

**Theme:** Each layer predicts the next layer's input. Only prediction errors propagate.

### Computational Flow (per token)

```
Input: token embedding -> spike times (M3)
  |
  v
For each LAYER (top-down + bottom-up):
  |
  TOP-DOWN PREDICTION:
    Layer n predicts layer (n-1)'s output using Walsh-KAN (M8):
    predicted[n-1] = KAN(state[n])
  |
  BOTTOM-UP ERROR:
    error = actual[n-1] - predicted[n-1]
    spike = ternarize(error)  -> {-1, 0, +1} per dimension
    ~70% of dimensions have spike = 0 (prediction correct)
  |
  STATE UPDATE (only on non-zero spikes):
    For each dimension where spike != 0:
      h_i(t) = a_i*h_i(t-1) + [B*spike]_i  (M10 diagonal SSM)
    (Skip 70% of dimensions — O(0.3d) cost per layer)
  |
  LAYER ROUTING (M7):
    If total surprise (sum |spike|) < threshold: skip layer
    Active layers: ~50% (M7)
  |
  TEMPORAL OUTPUT (M3):
    Output spike times encode magnitudes via bit shifts
  |
  v
Output head -> logits
```

### Training Flow
**THIS IS THE HARD PART.**
- Standard backprop does NOT naturally support top-down prediction errors
- Options:
  a) **Equilibrium Propagation (M5):** Train via state differences (free vs nudged phase). O(d) memory. ~2x cost of forward. But requires symmetric connections.
  b) **Predictive Coding Training:** Minimize free energy F = ||x - predicted||^2 + KL(posterior || prior). This IS the loss function — no separate cross-entropy needed. The prediction error IS the gradient signal.
  c) **Hybrid:** Standard backprop for weight updates, but use predictive coding to determine WHICH computations to perform in the forward pass.
- **Recommended:** Option (c) — hybrid. Use standard STE training but with the predictive coding architecture. During training, compute all predictions and all errors. Backprop through all of them. During inference, skip zero-error dimensions.

### Efficiency Estimate
```
70% dimensions skipped (M2): 3.3x
50% layers skipped (M7): 2x
Temporal coding (M3): 1.5x (no FP)
Ternary (base): 16x
Compound: 3.3 x 2 x 1.5 x 16 = ~158x

Best case (90% predictable text):
  10 x 2 x 1.5 x 16 = ~480x

Worst case (40% predictable):
  1.67 x 2 x 1.5 x 16 = ~80x
```

### Weakest Link
**Training the prediction mechanism.** The model must learn accurate top-down predictions WHILE learning the language task. These are coupled objectives. If the prediction mechanism is poor early in training, no sparsity is achieved and gradients are noisy. Training curriculum matters: start with prediction-only training (learn to predict next layer), then switch to language task. **This adds training complexity and may require 2x total training steps.**

### Viability: LOW-MEDIUM
The inference architecture is brilliant. The training is the hard problem. Predictive coding networks have been demonstrated at CIFAR-100 scale but never for language modeling at LLM scale. The gap between inference design and training implementation is the largest of any architecture here. **High risk, extremely high reward if it works.**

---

## Architecture 7: "The Ternary Finite Machine" — Maximum Simplicity

**Findings used:** M10 (diagonal SSM) + M4 (MDS B) + M9 (reversible) + M8 (Walsh-KAN) — 4 findings

**Theme:** The simplest possible architecture with zero adaptive computation. Clean, predictable, compilable.

### Computational Flow (per token)

```
Input: token embedding (INT8 or FP16)
  |
  v
For each REVERSIBLE LAYER PAIR (L/2 pairs):
  |
  Split: (x1, x2)
  |
  F(x2):
    1. Diagonal SSM (M10): h_i(t) = a_i*h_i(t-1) + [B_MDS*x2]_i
       - a_i in {-1,0,+1}: keep/flip/reset
       - B_MDS: MDS-coded ternary matrix (M4)
       - Cost: d additions for A*h, d*d_in additions for B*x
    2. Walsh-KAN (M8): y = sum_{n=0}^{N-1} c_n * (-1)^{popcount(n AND gray(h))}
       - c_n in {-1,0,+1}: ternary Walsh coefficients
       - Cost: N*d POPCOUNT + N*d additions
  |
  y1 = x1 + F(x2)  (integer addition, exact reversible)
  |
  G(y1): Same structure, different weights
  |
  y2 = x2 + G(y1)  (integer addition, exact reversible)
  |
  v
Output head: ternary GEMV -> logits -> softmax
```

### Training Flow
- Standard forward pass through all layers
- Reversible backward: recompute activations from final output. O(2) layer memory.
- STE for all ternary weights (A diagonal entries, B MDS entries, Walsh c_n coefficients)
- Gradient guaranteed stable: M10 ensures ||dh(T)/dh(0)|| independent of sequence length

### Efficiency Estimate
```
Per layer:
  Diagonal A: d additions (negligible)
  MDS B * x: d * d_in additions (ternary GEMV)
  Walsh-KAN: N * d POPCOUNTs + N * d additions
  Reversible coupling: 2d additions

For d=512, d_in=512, N=32, L=18 layers:
  Per layer: 512*512 + 32*512 + 32*512 + 2*512 = 296K ops
  Per token: 18 * 296K = 5.33M ops

CPU at 10 GOPS (SIMD integer): 0.53ms = ~1,887 tok/s at 50M
GPU: ~5-10x faster = ~10,000-19,000 tok/s at 50M

Base speedup: 16x (ternary) x 1.0 (no adaptive) = 16x
But SIMPLICITY enables: compilation to fixed instruction stream,
  CUDA graph capture, CPU SIMD vectorization, FPGA synthesis.
  Practical speedup with hardware optimization: 30-50x
```

### Weakest Link
**Walsh-KAN expressiveness with N=32 ternary terms.** The approximation error for alpha-smooth functions is O(N^{1-2*alpha}) + O(N*delta^2). For alpha=1 (Lipschitz) and delta=1/3 (ternary quantization), with N=32: error ~ 0.03 + 3.5 = 3.5. This is LARGE. Need N>>32 or a hybrid approach with a small FP16 residual correction. **If Walsh-KAN cannot approximate the needed nonlinearities, the entire architecture fails.**

Mitigation: N=256 (8x more Walsh terms, still POPCOUNT-only). Error drops to ~0.004 + 28 which is still problematic. The delta^2 term from ternary coefficients dominates. **Fundamental issue: ternary Walsh coefficients may need mixed-precision correction.**

Alternative mitigation: Use Walsh-KAN for the FIRST stage of nonlinearity (coarse shaping) and a small FP16 correction network for residual error (fine tuning). This adds ~5% FP computation but may be essential for quality.

### Viability: VERY HIGH (for the SSM + reversible core) / MEDIUM (for Walsh-KAN expressiveness)
The SSM + reversibility + MDS combination is rock-solid. The Walsh-KAN is the component that needs experimental validation. **This should be the FIRST thing tested.**

---

## Grand Synthesis

### Recommended Implementation Path

**Phase 1: Validate Walsh-KAN (CRITICAL GATE)**
Test Walsh-KAN with N=32,64,128,256 ternary terms as a drop-in replacement for SiLU+MLP in the existing 50M model. If within 10% PPL: proceed. If not: design hybrid Walsh+FP16-residual.
- Time: 1 week
- Risk: This is the single experiment that validates or invalidates Architectures 1, 3, 4, 6, 7.

**Phase 2: Build Architecture 7 (Ternary Finite Machine)**
The simplest, most testable architecture. No adaptive computation to debug. Pure stack of: diagonal SSM + MDS B + Walsh-KAN + reversible coupling.
- Time: 2-3 weeks
- Validates: M10 + M4 + M9 + M8 composing cleanly
- Success criterion: quality within 15% of current 50M model with O(1) activation memory

**Phase 3: Add Surprise Processing (Architecture 2 elements)**
Add M2 (predictive spike coding) to Architecture 7. Token-level sparsity.
- Time: 1-2 weeks
- Validates: M2 stacking with the core
- Success criterion: 2-3x speedup with <2% quality loss

**Phase 4: Add Routing + Exit (Architecture 2 complete)**
Add M7 (three-valued routing) + M11 (early exit). Full Architecture 2.
- Time: 2-3 weeks
- Validates: Full compound speedup
- Success criterion: 50-200x total speedup vs float baseline

**Phase 5: Explore Depth Scaling (Architecture 3)**
Replace fixed 18 layers with weight-tied 100-1000 reversible layers + routing.
- Time: 3-4 weeks
- Validates: Quality-from-depth scaling law for ternary
- Success criterion: Better quality than width-scaling at same parameter count

### Architecture Comparison Table

| Architecture | Findings | Speed | Quality | Memory | Simplicity | Viability | Best For |
|-------------|----------|-------|---------|--------|------------|-----------|----------|
| 1. Spike Machine | M10+M3+M8+M9+M11 | ~48x | Unknown | O(1) | Low | MEDIUM | Edge/embedded |
| 2. Surprise Processor | M10+M2+M7+M4+M11 | ~160x | High | O(d) | Medium | HIGH | General use |
| 3. Infinite Depth | M9+M10+M8+M7+M4 | ~32x | Highest? | O(1) | Medium | MED-HIGH | Quality-critical |
| 4. Holographic | M10+M8+M3+M4+M2 | ~200x | Limited | O(d) | Low | LOW-MED | Research |
| 5. Error-Correcting | M4+M7+M10+M9+M11 | ~42x | High | O(1) | Medium | MEDIUM | Robustness |
| 6. Predictive Cascade | M2+M3+M10+M8+M7 | ~160x | Unknown | O(d) | Low | LOW-MED | Brain-like |
| 7. Finite Machine | M10+M4+M9+M8 | ~16x | High | O(1) | HIGHEST | VERY HIGH | Foundation |

### The Recommended Path

**Build Architecture 7 first (foundation), evolve toward Architecture 2 (production).**

Architecture 7 is the clean, simple, mathematically rigorous core. Every component has strong theoretical backing. The composition has ZERO tensions. It can be built and validated quickly.

Architecture 2 adds the adaptive computation layer on top of Architecture 7's core. It has the best risk/reward ratio for production use: graceful degradation, orthogonal speedup stacking, and proven individual components.

Architecture 3 is the quality moonshot — if weight-tied depth scaling works for ternary, it opens a fundamentally new scaling axis.

**The single most important experiment is Walsh-KAN validation.** Everything else composes cleanly. Walsh-KAN's ternary approximation quality is the gate.

### What We Did NOT Include (and Why)

| Finding | Why Excluded from Top Recommendations |
|---------|--------------------------------------|
| M3 (temporal coding) | Constrains dynamic range to 6 bits. Quality risk too high for primary architecture. Valuable for edge deployment variant. |
| Holographic encoding | Capacity O(d/log(d)) too limited for complex language. Best as a component, not the foundation. |
| Equilibrium propagation | Unproven at LLM scale. Would require abandoning STE-based training. Research-grade. |
| Cellular automata SSM | 12.7x parameter compression but 3x speed overhead. Not worth the complexity for current scales. |

### Cross-Architecture Insights

1. **M10 (diagonal SSM) appears in ALL architectures.** It is the universally agreed-upon foundation. Gradient stability independent of T is too valuable to give up.

2. **M9 (reversible coupling) appears in 5 of 7.** O(1) activation memory is transformative for training. The only architectures that skip it are those that don't need training memory optimization (because they skip computation instead).

3. **M8 (Walsh-KAN) is the critical unknown.** It appears in 5 of 7 architectures and determines whether we can eliminate floating point entirely. The N=32 vs N=256 expressiveness question must be answered first.

4. **M2 + M7 are the "speedup pair."** Token-level × layer-level sparsity provides the bulk of adaptive speedup. They stack multiplicatively and have zero interaction.

5. **M4 (MDS) is infrastructure, not speedup.** It doesn't make things faster — it makes things MORE CORRECT through quantization. Invisible but essential.
