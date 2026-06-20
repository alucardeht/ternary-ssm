# v12 Architecture Specification

## Language: C (C99/C11)

- **SIMD:** AVX2 (x86), NEON (ARM), portable fallback
- **GPU:** Optional CUDA backend
- **HLS:** Compatible for future custom accelerator
- **Build:** CMake or zig cc

## Model Dimensions

- **Layers:** 6
- **d_model:** 1024
- **d_state:** 128
- **n_heads:** 16
- **Vocab:** 32064 (BPE 32K, aligned to 64)
- **Total params:** ~6.3M (6 SSM layers + embeddings + lm_head)

## Weight Representation

- **Primary:** Ternary {-1, 0, +1} stored as INT2 packed (4 weights per byte)
- **Shadow (training):** FP32 for gradient computation
- **Activation:** FP32 or BF16 (bf16 2.1× faster on RTX Ada)

## Core Components

### 1. Ternary MatMul (No Multiplications)
```
Format: CSR-lite (one row per neuron)
  pos_indices[i] = indices where weight = +1
  neg_indices[i] = indices where weight = -1
  n_pos[i], n_neg[i] = counts

Kernel:
  y[i] = sum(x[pos_indices[i]]) - sum(x[neg_indices[i]])
  Cost: 2 * d_in loads + d_in adds (no multiplications)
```

### 2. SSM Scan (Diagonal A, O(1) per step)
```
Per-layer scan:
  h[t] = A ⊙ h[t-1] + B * x[t]
  y[t] = C @ h[t]

Pure sequential loop: for t in 0..T-1: h[t] = ...
No FFT, no matrix inverse. Cost: T * d_state * d_model operations.
```

### 3. RNS Embedding (Exact Integer Arithmetic)
```
Moduli: {8, 9, 11, 13, 7}
Token t → [r_1, r_2, r_3, r_4, r_5] via modular reduction
CRT reconstruction → fp32 embedding vector
Benefit: zero floating-point quantization error in embedding lookup.
```

### 4. MTP — Minimal Task Projection
```
Future prediction at time t:
  h_future = h_t ⊙ A^k + U @ V^T @ h_t
  U ∈ ℝ^{d_state × 20}, V ∈ ℝ^{d_state × 20}
  logits = lm_head(h_future)

Params: 3 layers × 2 matrices × 1024 × 20 = 122K (not 98M with per-layer heads).
Shared lm_head across all predictions.
```

### 5. Loss: GP-KL Unified
```
Welford online accumulators for 5 moments:
  - mean: μ
  - variance: σ²
  - lag-1 autocorrelation: ρ_{t,t-1}
  - stability: E[|x_t - x_{t-1}|²]
  - orthogonality: E[⟨x_i, x_j⟩] for 32 random pairs

Loss = β * (l_mean + l_var + l_autocorr + l_stability + l_orth)
All 5 computed in single pass, no separate losses.
```

### 6. MVUE Gradient (Natural Gradient Implicit)
```
Ternary sample: w ~ P(w|s) where
  P(w=+1|s) = max(s, 0)
  P(w=-1|s) = max(-s, 0)
  P(w=0|s) = 1 - |s|

Gradient: ∇_s = (w_sampled - s) * 1[|s| < 1]
Crystallization: skip updates where |s| > 3 (rare stable states).
LR = 0.1 (333× higher than Adam, stable due to MVUE properties).
```

### 7. Tokenizer: BPE 32K
```
Input: raw text
Output: token IDs [0, vocab)
Load: tokenizer.json (external, e.g., from HuggingFace)
Max sequence length: 512 (or configurable)
```

## Data Flow

```
input_tokens (batch, seq_len)
    ↓
RNS_embed(d_model=1024)
    ↓
Predictive Coding (rank-24):
    surprise = embed(t) - W_pred @ h_prev
    W_pred ternary, rank-24 projection
    ↓
[SSM_layer(6 times)]
    ├─ in_proj(d_in=1024 → d_state=128)
    ├─ conv1d(kernel=4, dilation=1)
    ├─ SSM_scan(diagonal A, B, C)
    ├─ out_proj(d_state → d_model)
    ├─ Ternary Multiplexing (MUX):
    │  W_effective = W ⊙ mask(x), mask = (|x| > τ_mux)
    │  Stays ternary, τ_mux learnable per layer
    └─ residual (skip connection)
    ↓
final_norm(d_model=1024)
    ↓
lm_head(d_model → vocab=32064)
    ↓
logits (batch, seq_len, vocab)
```

## Memory Layout

| Component | Format | Size (6 layers, d=1024, ds=128) |
|-----------|--------|------|
| SSM A matrices | FP32 diagonal | 6 × 128 × 4B = 3.1 KB |
| SSM B, C | FP32 | 6 × (128 + 1024) × 4B = 27 KB |
| W_in (INT2) | INT2 packed | 6 × 1024 × 128 / 4 = 194 KB |
| W_out (INT2) | INT2 packed | 6 × 128 × 1024 / 4 = 194 KB |
| RNS Embedding | INT2 packed | moduli (8,9,11,13,7), ~7K params (~28 KB) |
| lm_head (INT2) | INT2 packed | 1024 × 32064 / 4 = 8.2 MB |
| MTP matrices | FP32 | 3 × 2 × 1024 × 20 × 4B = 490 KB |
| Shadow weights (training) | FP32 | ~6.3M params × 4B = 25 MB |
| Activations (batch=16, seq=512) | FP32 | 16 × 512 × 1024 × 4B = 33 MB |
| **Total inference** | - | **~16.5 MB** |
| **Total training (1 layer active)** | - | **~60 MB** |

## Initialization

```c
// Ternary initialization
for (int i = 0; i < n_weights; i++) {
    float logit = normal(0, 0.5);
    w_shadow[i] = logit;
}

// Eigenspectrum Calibration (A_log from data spectrum)
// Run calibration pass over 2048 tokens at startup
// Identifies 4 timescale clusters: tau=[1, 8, 60, 121]
// Spectral radii: r=[0.37, 0.88, 0.98, 0.99]
// A_log initialized to match empirical autocorrelation peaks
for (int i = 0; i < d_state; i++) {
    A_log[i] = -calibrate_eigenspectrum(i, d_state, data_sample_2048_tokens);
}

// TBD: Temporal Basis Decomposition (geometric A_log spacing)
// A_log uses logarithmically uniform timescale coverage:
// A[k] in (0, 1) covering fast (tau~1) to slow (tau~512)
// A_log[k] = -exp(linear_interp(log(1.0), log(context_length), k, d_state))

// B* Init: Layer 0 B matrix aligned to RNS eigenspace
// B[0] initialized via water-filling assignment from input covariance eigenvectors
B_layer_0 = align_to_rns_eigenspace(embed_covariance);

// RNS seeds: random but fixed per session
rng_seed = timestamp + process_id;
```

## Training Loop Structure

```
for epoch in 0..max_epochs:
    for batch in dataset:
        tokens = batch.tokens  # (batch_size, seq_len)

        # Forward
        logits = model(tokens)
        loss = ce_loss(logits, tokens) + gp_kl_loss(states, moments)

        # Backward (shadow weights only)
        loss.backward()

        # MVUE gradient on ternary quantization
        for each ternary parameter w_shadow:
            grad = (w_sampled - w_shadow) * mask[crystallization]
            w_shadow -= lr * grad

        # Periodically repack INT2
        if step % repack_interval == 0:
            w_int2 = quantize_ternary_to_int2(w_shadow)
```

## Inference

```
tokens → embed → (for each layer: scan + residual) → norm → lm_head
No shadow weights needed. Use INT2 packed directly.
```

## Compile Flags

```cmake
-DSIMD_BACKEND=AVX2      # or NEON, SCALAR
-DUSE_CUDA=OFF            # or ON for GPU
-DBATCH_SIZE=16
-DSEQ_LEN=512
-DVOCAB_SIZE=32064
```
