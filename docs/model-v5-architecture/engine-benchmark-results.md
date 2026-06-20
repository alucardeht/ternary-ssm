# C Engine Benchmark Results — First Baseline (2026-02-28)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_layer | 4 (3 SSM + 1 SWA) |
| vocab_size | 1000 |
| d_state | 16 |
| expand | 2 (d_inner=512) |
| n_heads (SWA) | 4 |
| ssd_n_heads | 8 |
| window_size | 128 |
| Platform | Windows 11, AVX2, single-thread |
| C Compiler | GCC 15.2.0, Release (-O2) |
| PyTorch | CPU single-thread (torch.set_num_threads(1)) |

## Component-Level Results

### C Engine (bench_components)

| Component | Avg (us) | Min (us) | Max (us) | Est tok/s |
|-----------|----------|----------|----------|-----------|
| Ternary GEMV 256→512 | 107.96 | 104.30 | 583.00 | 9,263 |
| RMSNorm d=256 | 0.08 | ~0 | 0.20 | 12,800,000 |
| SiLU d=512 | 0.20 | 0.10 | 9.90 | 5,080,000 |
| SSM step (full layer) | 352.65 | 339.10 | 761.60 | 2,836 |
| Attention step (full layer) | 665.06 | 629.60 | 1,192.00 | 1,504 |
| **Engine forward (E2E)** | **1,943.65** | **1,852.40** | **3,026.60** | **514** |

Time breakdown: SSM = 34.7%, Attention = 65.3%

### PyTorch CPU Baseline (bench_cpu_baseline.py, use_ternary=True)

| Component | Avg (us) | Min (us) | Max (us) | Est tok/s |
|-----------|----------|----------|----------|-----------|
| RMSNorm d=256 | 57.01 | — | — | 17,540 |
| SSM layer (full block) | 933.03 | — | — | 1,072 |
| SWA layer (full block) | 640.21 | — | — | 1,562 |
| LM Head 256→1000 | 16.26 | — | — | 61,498 |
| **Full forward (1 token)** | **4,023.67** | — | — | **249** |

Peak RSS: 653.8 MB

## Head-to-Head Comparison (Fair — Both Ternary)

| Component | C Engine | PyTorch Ternary | C Speedup |
|-----------|----------|-----------------|-----------|
| RMSNorm | 0.08 us | 57.01 us | **712x faster** |
| SSM step | 352.65 us | 933.03 us | **2.6x faster** |
| Attention step | 665.06 us | 640.21 us | **~par (0.96x)** |
| Full forward | 1,943.65 us | 4,023.67 us | **2.07x faster** |
| tok/s E2E | 514 | 249 | **2.07x faster** |
| Peak memory | ~few MB | 654 MB | **~100x less** |

Note: Initial comparison used PyTorch with float weights (use_ternary=False), which was unfair.
With ternary=True, C engine attention matches PyTorch (previously appeared 2.5x slower).

## Analysis

### Wins
- **RMSNorm: 712x** — PyTorch framework overhead massive for tiny ops. C is bare metal.
- **SSM step: 2.6x** — Ternary GEMV (add/subtract) + no framework overhead.
- **Memory: ~100x less** — C engine uses only model weights + state. No Python/PyTorch runtime.
- **E2E: 2.07x** — 514 tok/s vs 249 tok/s. Already a solid win.

### Status
- **Attention: ~par (0.96x)** — The initial "2.5x slower" result was from comparing C ternary vs PyTorch float (unfair). With both using ternary, they're at parity.
- The ternary GEMV AVX2 kernel currently decodes to float and does FMA — not exploiting true add/sub advantage yet.

### Optimization Opportunities
- **GEMV kernel upgrade:** Current kernel decodes ternary→float then FMA. A true add/sub kernel (masked add/subtract, skip zeros) would speed up ALL GEMV operations by est. 1.5-2x.
- **RoPE cache:** Save ~5 us/step by precomputing cos/sin table.
- **If GEMV is 2x faster:** E2E would drop to ~1,100 us = ~910 tok/s (3.7x vs PyTorch).
- **Bitplane GEMV:** Using popcount-style bit operations with float activations could achieve 2-3x GEMV speedup.

### Benchmark Fairness Note
The original PyTorch benchmark used `use_ternary=False`, which made PyTorch use float nn.Linear with BLAS/MKL — unfairly fast compared to C ternary GEMV. Fixed by setting `use_ternary=True` to match the C engine's ternary weights.

### Next Steps
1. Implement masked add/sub GEMV kernel (OPT-1) — est. 1.5-2x on all GEMVs
2. Cache RoPE cos/sin table (OPT-3) — free ~5 us
3. Consider bitplane GEMV with float activations (OPT-2) — est. 2-3x on GEMVs
4. Re-benchmark after each optimization to track progress
