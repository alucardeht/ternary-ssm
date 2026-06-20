# v12 C Implementation — Status (2026-04-15)

## 2026-04-15 Status Refresh — R368 M1
- `S4-safe-gpu-only` landed: dead GPU-only Vulkan backend plumbing was removed from the active path and validated by a clean Vulkan rebuild plus a 10-step smoke run.
- `train.c` now supports `--seed`, `--log-every`, and `--loss-log-hex` for deterministic per-step logging without changing training math.
- Strong post-`S4` artifacts live in `.workspace/research/references/r368-m1/`.
- Two 100-step runs with `seed=42` matched on all 100 `[S5]` lines, and `[M1 RECOMPUTE]` appeared 10,800 times per run.
- The strong-validation harness previously surfaced exit code 1, but the direct binary returned 0; the discrepancy was traced to the wrapper or harness, not `train.exe`.
- `N1` is now landed: the default Vulkan backward path uses the recompute path without needing `V12_M1_SLICE_A/B`.
- The stronger `N1` validation is complete: default `USE_VULKAN=1` matched the prior validated reference on 100/100 `[S5]` lines, and old slice flags produced no behavioral diff.
- `N2` is now landed: the legacy forward-side GPU cache path is gone (`fwd_a_eff_gate` / `a_eff_gate_out`), and two new 100-step `seed=42` runs still matched the post-`N1` reference on 100/100 `[S5]` lines.
- The original formal `S5` closure remains historically blocked because no named pre-`S1` reference log exists in the repo.

**Historical note:** The sections below are older baseline status notes and do not fully describe the current Vulkan/runtime cleanup frontier.

## Architecture
- 4 layers, d_model=128, d_state=32, vocab=256 (byte-level), seq=128
- Pure C99, no GPU dependency, portable
- Adam optimizer (beta1=0.9, beta2=0.999)
- Staged Forward Pipeline (FP): layer-by-layer training

## ALL Features Implemented and Verified

| Feature | Status | Verified |
|---------|--------|----------|
| Forward/backward pass | Working | Loss decreases ✓ |
| Adam optimizer | Working | Stable for 5000+ steps ✓ |
| MVUE natural gradient | Working | g = dw*(w-s) into Adam ✓ |
| Staged FP pipeline | Working | 4 stages + joint ✓ |
| Causal conv1d | Working | Between W_in and SSM ✓ |
| RMS Layer Norm | Working | Before lm_head ✓ |
| CSR ternary matvec | Working | Forward + transpose backward ✓ |
| Predictive coding (W_pred) | Working | Layer-0 temporal ✓ |
| Inter-layer PC | Implemented | Shared W_pred at boundaries ✓ |
| MTP auxiliary loss | Working | Even layers, K=4 ✓ |
| GP-KL unified loss | Working | Welford moments ✓ |
| Contraction loss | Working | Log-barrier, grad clamped ✓ |
| Boundary filter | Working | Newline-based h reset ✓ |
| WHT per-timestep | Working | y_ssm mixing ✓ |
| Multiplexing (tau_mux) | Working | Sigmoid gate ✓ |
| Direct-B | Working | sigmoid(x_conv) * B ✓ |
| RNS embedding | Working | 8 moduli, 1/√d scale ✓ |
| Fisher LR scaling | Working | Per-layer at repack ✓ |
| Crystallization | Working | |s|>τ freezes weight ✓ |
| Checkpoint save/load | Working | Save works, resume partial ✓ |
| Gradient clipping | Working | Global norm max=1.0 ✓ |

## Training Results (5000 steps, from scratch)

```
FP Stage 0 (layer 0): 13.0 → 6.8 nats (150 steps)
FP Stage 1 (layer 1): 6.8 → 4.2 nats
FP Stage 2 (layer 2): 4.3 → 2.9 nats
FP Stage 3 (layer 3): 2.4 → 1.8 nats
Joint training:        2.1 → 0.20 nats (4400 steps)
```

Fisher LR at step 4500: [0.619, 0.936, 1.281, 1.857]
Crystal fraction: 3.6% at step 5000

## Known Issues

1. **Checkpoint resume instability**: Adam moments not saved → fresh momentum on resume → divergence. Needs: save m/v buffers in checkpoint format.
2. **Inter-layer PC disabled**: Forward loop disabled (`k < 0`) due to earlier instability. Re-enable after checkpoint fix.
3. **Memory optimizations deferred**: INT2 packing, PCA cache, CPU offload, reversible scan — only needed at 7B scale.
4. **Formal `R368 M1 / S5` closure still blocked**: the repo has strong post-`S4` reproducibility artifacts, but no named pre-`S1` reference log exists for the original zero-delta acceptance check.

## Build & Run
```bash
cd src/model_v12
gcc -O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -I src/ -DUSE_MMAP=0 \
  src/train.c src/core/model.c src/core/ternary_matmul.c src/core/ssm_scan.c \
  src/core/loss.c src/core/optimizer.c src/io/dataset.c src/io/checkpoint.c \
  -lm -o train.exe
./train.exe text_data.bin 5000 checkpoint.bin
```
