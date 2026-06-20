# Compilation Pipeline — Technical Specification

**Date:** 2026-03-09
**Version:** R24-spec-v1

---

## Overview

Four-step analytical pipeline to initialize a ternary SSM without gradient descent.
Each step has a theoretical bound, an implementation, and a validation experiment.

---

## Step 1: A Matrix via HiPPO-LegS

### Theory
HiPPO (High-Order Polynomial Projection Operators) computes the optimal polynomial
approximation of an input signal's history. The LegS (Legendre Scaled) basis gives:

```
A_{ij} = -(2i+1)^{1/2} * (2j+1)^{1/2}    if i > j
       = -(i+1)                             if i == j
       = 0                                  if i < j

B_i = (2i+1)^{1/2}
```

### Discretization
Using Zero-Order Hold (ZOH) or bilinear (Tustin):
```
Ā = exp(ΔA)
B̄ = (ΔA)^{-1}(exp(ΔA) - I)ΔB
```

### Ternarization
```python
A_fp = hippo_legs_matrix(d_state)
# Scale to [-1, 1] range
A_fp = A_fp / A_fp.abs().max()
# Ternarize with learned threshold
threshold = A_fp.abs().mean() * 0.7
A_ternary = torch.where(A_fp > threshold, 1.0,
            torch.where(A_fp < -threshold, -1.0, 0.0))
# Hurwitz constraint: force A ∈ {-1, 0} (R19-B)
A_ternary = A_ternary.clamp(max=0)
```

### Validation
- Metric: discrimination ratio = between-class variance / total variance of H
- Expected: HiPPO A > random A discrimination ratio
- File: src/model_v10/experiments/hippo_init.py

---

## Step 2: B, C Matrices via N4SID

### Theory
Numerical Subspace State Space System Identification (N4SID):
Given input-output pairs (u_t, y_t), identifies (A, B, C, D) via SVD of Hankel matrix.

With A fixed (HiPPO), reduces to fitting B and C only.

### PARSIM Bound (He et al., 2025 — arxiv 2501.16639)
With A fixed, burn-in requirement = d tokens (not d² for full A,B,C).
For d=256: need 256 diverse tokens minimum.

### Probe Design
```python
# 256 random sequences, length 512 tokens
# Random tokens (not structured text) = optimal for PE condition
N_probes = 256
seq_len = 512
tokens = torch.randint(0, vocab_size, (N_probes, seq_len))
# Forward pass → collect H: (N_probes * seq_len, d_state)
```

### Identification
```python
# Build Hankel matrix from H
# SVD → top-d singular vectors → B, C
H_matrix = hidden_states.reshape(N_probes, seq_len, d_state)
# H (N_probes*seq_len, d) → via least squares
# B: maps input → state
# C: maps state → output
B_hat = lstsq(H_matrix, inputs)
C_hat = lstsq(outputs, H_matrix)
```

### Validation
- Measure fit quality (reconstruction error) at N = 64, 256, 512, 1024 tokens
- Expected: plateau near N=256
- File: src/model_v10/experiments/n4sid_bc.py

---

## Step 3: W_out via OLS / Online RLS

### Batch OLS
```python
S_hh = torch.zeros(D, D, dtype=torch.float64)  # D = d_model
S_hy = torch.zeros(vocab_size, D, dtype=torch.float64)

for x, y in dataloader:  # 1M tokens total
    with torch.no_grad():
        h = model(x)  # (B, T, D)
    h_flat = h.reshape(-1, D).double()
    y_flat = y.reshape(-1)
    S_hh += h_flat.T @ h_flat
    S_hy.index_add_(0, y_flat, h_flat)

ridge = S_hh.diagonal().mean() * 1e-4
W_out = torch.linalg.solve(S_hh + ridge * torch.eye(D).double(), S_hy.T).T.float()
```

Memory: O(D² + V×D) = O(256² + 50257×256) ≈ 13MB. Never stores full H.

### Online RLS (Sherman-Morrison)
```python
P = torch.eye(D) * initial_ridge_inv  # D×D matrix
W = torch.zeros(vocab_size, D)

def rls_update(h, y):
    # h: (D,), y: scalar token index
    Ph = P @ h
    k = Ph / (1 + h @ Ph)  # Kalman gain
    P -= k.outer(Ph)        # rank-1 update O(D²)
    W[y] += k * (1 - W[y] @ h)  # weight update O(D)
```

### Validation
- Compare OLS W_out vs GD W_out vs Online RLS after same tokens
- Expected: OLS ≈ GD at convergence, OLS faster
- File: src/model_v10/experiments/ols_wout_pipeline.py

---

## Step 4: Forward-Only Fine-Tuning (optional)

### Direct Feedback Alignment (DFA) — R22-WEB-2
```python
# Fixed random ternary feedback matrices (never updated)
B_feedback = {l: torch.randint(-1, 2, (d_model, vocab_size)) for l in range(n_layer)}

def dfa_update(model, x, y):
    # Forward pass
    hidden_states = model.forward_get_hidden(x)
    logits = model.lm_head(hidden_states[-1])

    # Output error
    e_out = logits - F.one_hot(y, vocab_size).float()

    # Direct feedback per layer (parallel, no sequential dependency)
    for l, h_l in enumerate(hidden_states):
        e_l = B_feedback[l] @ e_out.T  # (d_model, B*T)
        # Update: dW = e_l ⊗ h_{l-1}
        grad_W = e_l @ hidden_states[l-1].reshape(-1, d_model)
        model.layers[l].weight.data -= lr * torch.sign(grad_W)  # DSST update
```

### NoProp (local denoising) — R22-WEB-3
```python
# Each block gets local denoising objective
for l in range(n_layer):
    alpha_l = (n_layer - l) / n_layer  # more noise in early layers
    x_noisy = alpha_l * embed(y_true) + (1 - alpha_l) * randn_ternary()
    pred_l = model.blocks[l](hidden_l_minus_1)
    local_loss = F.mse_loss(pred_l, x_noisy)
    local_loss.backward()  # gradient only through block l
    optimizer_l.step()
```

### Validation
- Run each for 100 steps on wikitext
- Metric: loss delta from initial, vs GD baseline 100 steps
- File: src/model_v10/experiments/forward_only_train.py

---

## Full Pipeline Code Sketch

```python
def compile_ternary_ssm(cfg, corpus_path, device='cuda'):
    """
    Full analytical compilation pipeline.
    No gradient descent through the network.
    """
    model = ModelV2(cfg).to(device)

    # Step 1: HiPPO A (0 data)
    for layer in model.layers:
        A_hippo = hippo_legs_matrix(cfg.d_state)
        A_ternary = ternarize_hurwitz(A_hippo)
        layer.ssm.A_log.data = torch.log(A_ternary.abs() + 1e-8)
    print("[COMPILE] Step 1: HiPPO A initialized")

    # Step 2: N4SID B, C (256 probe tokens)
    probe_tokens = torch.randint(0, cfg.vocab_size, (256, 512))
    B_hat, C_hat = n4sid_with_fixed_A(model, probe_tokens)
    inject_bc(model, B_hat, C_hat)
    print("[COMPILE] Step 2: N4SID B,C identified from 256 probe tokens")

    # Step 3: OLS W_out (1M tokens)
    W_out = ols_wout(model, corpus_path, max_tokens=1_000_000)
    model.lm_head.weight.data.copy_(W_out)
    print("[COMPILE] Step 3: OLS W_out from 1M tokens")

    # Step 4: DFA fine-tune (optional, 100 steps)
    dfa_finetune(model, corpus_path, steps=100)
    print("[COMPILE] Step 4: DFA fine-tune complete")

    return model
```

---

## Expected Results vs. Baselines

| Method | Training tokens | VRAM training | Time (RTX 5070) | Perplexity (est.) |
|--------|----------------|---------------|-----------------|-------------------|
| FP16 7B standard | 140B | 70GB | months | ~15 |
| Ternary GD 7B | 140B | ~40GB | months | ~16 |
| Ternary GD v10 6M | 10K steps | 2GB | minutes | ~7.3 |
| **Compiled 6M** | **1M** | **2GB** | **<1 hour** | **TBD (target: ~8-10)** |
| Compiled 1B | 1M | ~8GB | <4 hours | TBD |
| Compiled 7B | 1M | ~12GB | <24 hours | TBD |

Note: Compiled 7B fits in 12GB because compilation = forward passes only (no optimizer states, no activation storage for backward).

---

*Status: Pending R24 experiments*
*Next: src/model_v10/experiments/hippo_init.py*
