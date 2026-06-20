# Benchmark Tools Suite

Complete set of tools for P4.3 benchmark: comparing our ternary engine against llama.cpp.

## Files in This Directory

### Core Scripts

#### `create_tiny_gguf.py`
Generates a synthetic 6M-parameter GGUF model for benchmarking.

**Usage:**
```bash
python create_tiny_gguf.py --output models/tiny-llama-6m-f16.gguf
```

**What it creates:**
- GGUF v3 format (llama.cpp compatible)
- Llama architecture: 4 transformer blocks, d_model=256, vocab=1000
- Random float16 weights (~6.5 MB unquantized)
- Full tokenizer metadata
- All weight tensors llama.cpp expects

**Dependencies:**
- `gguf` package: `pip install gguf`
- `numpy`

---

#### `quantize_gguf.sh`
Helper script to quantize GGUF models with llama.cpp.

**Usage:**
```bash
./quantize_gguf.sh models/tiny-llama-6m-f16.gguf models/tiny-llama-6m-q2k.gguf Q2_K
```

**Quantization types:**
- `Q2_K` — 2.5 bpw (recommended, closest to ternary)
- `Q4_0` — 4.5 bpw
- `IQ1_S` — 1.56 bpw (extreme compression)
- `F16` — No quantization

---

#### `check_benchmark_env.py`
Verifies all dependencies and binaries are installed before benchmarking.

**Usage:**
```bash
python check_benchmark_env.py
```

**Checks:**
- Python packages: numpy, gguf
- llama.cpp binaries: llama-quantize, llama-bench, test-backend-ops
- Our engine binaries: bench_gemv, bench_components
- Required directories exist

---

#### `compile_benchmark_results.py`
Parses raw benchmark output and generates formatted comparison report.

**Usage:**
```bash
python compile_benchmark_results.py benchmark_results.txt --output report.md
```

**Parses:**
- Our engine: GEMV latency, E2E throughput, VRAM usage
- llama.cpp: tok/s, latency, memory

**Outputs:**
- Formatted markdown table
- Speedup calculations
- Analysis of fair/unfair comparisons

---

### Documentation

#### `README_GGUF.md`
Complete documentation for GGUF model generation and quantization.

**Contents:**
- Quick start (generate → quantize → test)
- Architecture details (dimensions, parameter count)
- GGUF weight format explanation
- Quantization types comparison
- Troubleshooting common issues
- Full workflow example

---

#### `benchmark_workflow.md`
Step-by-step guide to run the complete benchmark suite.

**Sections:**
- Prerequisites check
- Full workflow (30 minutes)
- Step-by-step instructions with expected outputs
- Troubleshooting
- Expected performance ranges
- Quick command reference

---

#### `BENCHMARK_TOOLS.md`
This file. Overview of all tools and quick reference.

---

## Quick Start (5 minutes)

### 1. Check dependencies
```bash
python src/engine/tools/check_benchmark_env.py
```

### 2. Generate model
```bash
python src/engine/tools/create_tiny_gguf.py \
  --output models/tiny-llama-6m-f16.gguf
```

### 3. Quantize
```bash
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin
llama-quantize.exe ../../../code-ai-workspace/models/tiny-llama-6m-f16.gguf \
                   ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf Q2_K
```

### 4. Run benchmarks
```bash
# Our engine
cd C:/Users/th1rt/Documents/Workspaces/code-ai-workspace/src/engine/build
./bench_components.exe

# llama.cpp
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin
llama-bench.exe -m ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf \
                -t 1 -n 128 -r 5
```

---

## Model Specification

### Parameters
- **Vocabulary:** 1,000 tokens
- **d_model:** 256 (embedding and hidden dim)
- **Layers:** 4 transformer blocks
- **Heads:** 4 attention heads
- **Head dim:** 64
- **FFN hidden:** 683 (2.67x expansion, Llama default)
- **Total params:** ~3.65M

### Weight Shapes (GGUF tensor names)

```
token_embd.weight           (1000, 256)      — 256K params

blk.0-3.attn_norm.weight    (256,) × 4       — 1K params
blk.0-3.attn_q.weight       (256, 256) × 4   — 262K params
blk.0-3.attn_k.weight       (256, 256) × 4   — 262K params
blk.0-3.attn_v.weight       (256, 256) × 4   — 262K params
blk.0-3.attn_output.weight  (256, 256) × 4   — 262K params

blk.0-3.ffn_norm.weight     (256,) × 4       — 1K params
blk.0-3.ffn_gate.weight     (683, 256) × 4   — 699K params
blk.0-3.ffn_up.weight       (683, 256) × 4   — 699K params
blk.0-3.ffn_down.weight     (256, 683) × 4   — 699K params

output_norm.weight          (256,)           — 256 params
output.weight               (1000, 256)      — 256K params
```

---

## Expected Results

### Our Engine (Ternary)
- **E2E throughput:** 300-500 tok/s
- **Peak VRAM:** 50-100 MB
- **Kernel latency (256×256):** 50-100 us
- **File size (packed):** ~1.5 MB

### llama.cpp (Q2_K Quantization)
- **E2E throughput:** 80-150 tok/s
- **Peak VRAM:** 200-400 MB
- **Kernel latency (256×256):** 100-200 us
- **File size:** ~1.2 MB

### Speedup Factors
- **Kernel level:** 1.5-3x our favor
- **E2E inference:** 2-5x our favor
- **Memory footprint:** 2-4x our favor
- **File size:** Similar (1.2-1.5 MB)

---

## Dependencies

### Required
- Python 3.8+
- gguf package: `pip install gguf`
- numpy: `pip install numpy`
- llama.cpp (built): https://github.com/ggerganov/llama.cpp

### Optional
- cloc (source line counter): `apt install cloc` or `brew install cloc`
- psutil (system monitoring): `pip install psutil`

---

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| tiny-llama-6m-f16.gguf | 6.5 MB | Unquantized baseline |
| tiny-llama-6m-q2k.gguf | 1.2 MB | Quantized (Q2_K) |
| tiny-llama-6m-q4.gguf | 2.0 MB | Quantized (Q4_0) |

---

## Benchmarking Strategy

### Tier 1: Kernel-Level (Most Fair)
- Isolates matrix-vector multiply (GEMV) performance
- Removes framework overhead
- Dimensions: 256×256, 256×512, 1024×1024

**Tools:**
- Our: `bench_gemv.exe`
- llama.cpp: `test-backend-ops.exe perf`

### Tier 2: End-to-End (Fair if Models Neutral)
- Measures full inference pipeline: loading, forward, sampling
- Same model size (~6M params)
- Token generation only (not prompt processing)

**Tools:**
- Our: `bench_components.exe`
- llama.cpp: `llama-bench.exe`

### Tier 3: System-Level (Context Only)
- Binary size, source lines of code, build complexity
- Not a performance metric, but engineering effort
- Documentation purposes

**Tools:**
- cloc (count source lines)
- ls (binary sizes)

---

## Troubleshooting

### "gguf module not found"
```bash
pip install gguf
```

### "llama-quantize not found"
Build llama.cpp:
```bash
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON
cmake --build build --config Release -j
```

### Model file is corrupted
Delete and re-create:
```bash
rm models/tiny-llama-6m-f16.gguf
python src/engine/tools/create_tiny_gguf.py -o models/tiny-llama-6m-f16.gguf
```

### Benchmark is slow
Check:
1. Single-thread flag (`-t 1`)
2. No other processes using CPU
3. File is in memory (first run warm-up)

---

## Integration with P4.3 Plan

This suite implements **Phase 1 (Setup, 30 min)** and **Phase 2 (Benchmarks, 20 min)** of the P4.3 benchmark plan.

See: `.workspace/plans/benchmark-p43-design.md` for full plan.

**Files generated:**
- `models/tiny-llama-6m-f16.gguf` — Unquantized model
- `models/tiny-llama-6m-q2k.gguf` — Quantized model
- `.workspace/plans/benchmark-p43-raw-results.txt` — Raw benchmark output
- `docs/engine-benchmark-results.md` — Analysis + summary

---

## Next Steps

After running benchmarks:

1. **Analyze results:** Use `compile_benchmark_results.py` to parse output
2. **Document findings:** Update `docs/engine-benchmark-results.md`
3. **Identify bottlenecks:** Where is our engine slow? Where is llama.cpp slow?
4. **Iterate:** Optimize kernels or framework based on findings
5. **Commit:** Save results and analysis to git

---

## References

- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Llama model architecture: https://arxiv.org/abs/2302.13971
