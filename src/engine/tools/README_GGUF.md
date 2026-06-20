# Tiny GGUF Model Generator for Benchmarking

Quick reference for creating synthetic 6M-parameter GGUF models to benchmark llama.cpp inference engine against our ternary engine.

## Quick Start

### 1. Generate F16 Model (unquantized baseline)

```bash
cd C:/Users/th1rt/Documents/Workspaces/code-ai-workspace

python src/engine/tools/create_tiny_gguf.py \
  --output models/tiny-llama-6m-f16.gguf
```

This creates a ~6.5 MB file with:
- **Vocabulary:** 1,000 tokens
- **d_model:** 256 (embedding and hidden dimensions)
- **Layers:** 4 transformer blocks
- **Heads:** 4 attention heads
- **FFN hidden:** 683 (Llama's 2.67x expansion ratio)
- **Total parameters:** ~3.65M (counting weights only; ~7.3M with biases/norms)
- **Format:** GGUF v3 with float16 weights

**Output:**
```
File size:           6.50 MB
Total parameters:    3,649,000 (~3.6M)
```

### 2. Quantize to Q2_K (2.5 bits per weight)

Q2_K is the closest GGUF quantization to our ternary format (~1.6 bpw packed):

```bash
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin

llama-quantize.exe \
  ../../tiny-llama-6m-f16.gguf \
  ../../tiny-llama-6m-q2k.gguf \
  Q2_K
```

**Result:** ~1.2 MB file (18% of original)

### 3. Test Inference

```bash
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin

# Quick test: generate 10 tokens
llama-cli.exe -m ../../tiny-llama-6m-q2k.gguf \
  -p "hello world" \
  -n 10 \
  -t 1
```

### 4. Benchmark

Single-thread benchmark (same conditions as our engine):

```bash
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin

llama-bench.exe -m ../../tiny-llama-6m-q2k.gguf \
  -t 1 \
  -n 128 \
  -r 5 \
  -p 0
```

Parameters:
- `-t 1` — Single thread (fair comparison with our single-threaded engine)
- `-n 128` — Generate 128 tokens
- `-r 5` — Run 5 repetitions
- `-p 0` — Empty prompt (measure token generation, not prefill)

**Expected output includes:**
```
model_size(MB)  mem_required(MB)  time_to_first_token(ms)  time_per_token(ms)  tokens_per_sec
```

## Architecture Details

### Model Dimensions

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Embedding | 256 × 1000 | 256K |
| Per-layer attention | 5 × 256² | 327K |
| Per-layer FFN | 3 × 256 × 683 | 525K |
| Per-layer norms | 2 × 256 | 512 |
| Per-layer total | — | 853K |
| 4 layers | 4 × 853K | 3.41M |
| Output head | 256 × 1000 | 256K |
| **Total** | — | **3.65M** |

### GGUF Weight Format

All weights stored as **float16** (FP16 / half precision).

Per-weight memory:
- FP16: 2 bytes
- Total uncompressed: 3.65M × 2 = 7.3 MB

After quantization to Q2_K (2.5 bits per weight):
- Effective: 3.65M × 2.5 / 8 ≈ 1.14 MB

### Tensor Names (llama.cpp format)

All tensors added to GGUF follow llama.cpp's naming convention:

```
token_embd.weight             — Embedding matrix (vocab_size × d_model)

blk.0.attn_norm.weight        — Attention layer norm
blk.0.attn_q.weight           — Query projection
blk.0.attn_k.weight           — Key projection
blk.0.attn_v.weight           — Value projection
blk.0.attn_output.weight      — Output projection

blk.0.ffn_norm.weight         — FFN layer norm
blk.0.ffn_gate.weight         — Gate projection (Llama-style gated linear unit)
blk.0.ffn_up.weight           — Up projection
blk.0.ffn_down.weight         — Down projection

output_norm.weight            — Final layer norm
output.weight                 — LM head (vocab_size × d_model)
```

## Quantization Types

### Q2_K (Recommended)

- **Bits per weight:** 2.5 bpw
- **File size (estimated):** ~1.2 MB
- **Accuracy:** ~95% (acceptable for benchmark)
- **Speed:** Fast (llama.cpp optimized)
- **Why:** Closest to our ternary ~1.6 bpw

```bash
llama-quantize.exe model.gguf model-q2k.gguf Q2_K
```

### Q4_0 (Baseline for comparison)

- **Bits per weight:** 4.5 bpw
- **File size (estimated):** ~2 MB
- **Accuracy:** ~99% (minimal loss)
- **Speed:** Comparable to Q2_K
- **Why:** Standard reference quantization

```bash
llama-quantize.exe model.gguf model-q4.gguf Q4_0
```

### F16 (Unquantized)

- **Bits per weight:** 16 bpw
- **File size (estimated):** ~6.5 MB
- **Accuracy:** 100% (exact)
- **Speed:** Fast (no dequantization overhead)
- **Why:** Baseline to measure quantization cost

Don't quantize; use as-is.

### IQ1_S (Extreme compression)

- **Bits per weight:** 1.56 bpw
- **File size (estimated):** ~0.9 MB
- **Accuracy:** ~80% (lossy)
- **Speed:** Slower (complex dequantization)
- **Why:** BitNet-equivalent density

```bash
llama-quantize.exe model.gguf model-iq1s.gguf IQ1_S
```

## Comparison: Our Engine vs llama.cpp

### What We're Testing

| Metric | Our Engine | llama.cpp |
|--------|-----------|-----------|
| Model | SSM + Attention hybrid | Pure Transformer (Llama) |
| Weights | Ternary {-1, 0, +1} packed | Float16 or quantized |
| Format | `.trnm` (custom, 2 bpw) | GGUF (standard) |
| Operations | Add/subtract only | Matmul + dequant |
| Thread model | Single-threaded | Single-threaded (for fairness) |

### Fair Comparison

**KERNEL LEVEL (most fair):**
- Our: `bench_gemv.exe` — measure ternary GEMV (256×256, etc.)
- llama.cpp: `test-backend-ops.exe perf` — measure Q2_K GEMV at same sizes
- Isolates OPERATION speed

**E2E INFERENCE (fair if model-neutral):**
- Both: Generate 128 tokens from same 6M model
- Measures: Forward pass + sampling overhead
- Caveat: Architecture differs (SSM vs Transformer), so quality also differs

**NOT FAIR:**
- Prompt processing (we don't batch, llama.cpp does)
- Multi-threaded (we're single-only)
- Large dimensions (llama.cpp has years of tiling optimization)

## Troubleshooting

### "gguf module not found"

```bash
pip install gguf
```

Or with conda:
```bash
conda install -c conda-forge gguf
```

### "llama-quantize.exe not found"

Build llama.cpp first:

```bash
cd C:/Users/th1rt/Documents/Workspaces

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j
```

Quantize tool will be at: `llama.cpp/build/bin/llama-quantize.exe`

### "GGUF file corrupted" or "Invalid file"

GGUF format is strict. If creation fails midway:
1. Delete output file
2. Re-run script
3. If still fails, check Python version (needs 3.8+)

### Quantization very slow

Q2_K is slower than Q4_0. This is normal (complex bit-packing). Use `-t N` to parallelize:

```bash
llama-quantize.exe -t 4 input.gguf output.gguf Q2_K
```

### "Tokens per second" is very low

Expected for a 6M model on CPU:
- Ternary GEMV kernel: ~514 tok/s
- llama.cpp Transformer on CPU: ~50-150 tok/s (depends on quantization)
- Our engine should be faster due to simpler arithmetic

If `< 10 tok/s`, check:
- Single-thread flag (`-t 1` in llama-bench)
- No other programs using CPU
- File I/O not cached

## References

- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp quantization: https://github.com/ggerganov/llama.cpp#quantization
- llama-bench docs: https://github.com/ggerganov/llama.cpp#quantize

## Files in This Directory

```
create_tiny_gguf.py      — Main script to generate GGUF model
quantize_gguf.sh         — Helper script to quantize with llama.cpp
README_GGUF.md           — This file
```

## Example Full Workflow

```bash
# 1. Generate
cd C:/Users/th1rt/Documents/Workspaces/code-ai-workspace
python src/engine/tools/create_tiny_gguf.py -o models/tiny-6m-f16.gguf

# 2. Quantize
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin
llama-quantize.exe ../../../code-ai-workspace/models/tiny-6m-f16.gguf \
                   ../../../code-ai-workspace/models/tiny-6m-q2k.gguf Q2_K

# 3. Benchmark our engine
cd C:/Users/th1rt/Documents/Workspaces/code-ai-workspace/src/engine/build
./bench_components.exe

# 4. Benchmark llama.cpp
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin
llama-bench.exe -m ../../../code-ai-workspace/models/tiny-6m-q2k.gguf \
                -t 1 -n 128 -r 5

# 5. Compare results in docs/engine-benchmark-results.md
```
