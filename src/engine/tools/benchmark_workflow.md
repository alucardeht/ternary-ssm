# Complete Benchmarking Workflow

Step-by-step guide to run the full P4.3 benchmark comparing our ternary engine with llama.cpp.

## Prerequisites Check

Before starting, verify all dependencies are installed:

```bash
cd C:\Users\th1rt\Documents\Workspaces\code-ai-workspace

python src/engine/tools/check_benchmark_env.py
```

If any are missing, install them first (see output of checker).

---

## Full Workflow (30 minutes)

### Step 1: Generate Tiny Model (2 min)

```bash
cd C:\Users\th1rt\Documents\Workspaces\code-ai-workspace

python src/engine/tools/create_tiny_gguf.py \
  --output models/tiny-llama-6m-f16.gguf
```

**Output should show:**
```
Creating tiny GGUF model...
  Architecture: Llama (Transformer)
  d_model: 256
  n_layers: 4
  n_heads: 4
  vocab_size: 1000
  d_ff: 683

Adding tokenizer metadata...
Adding embedding layer...
Adding 4 transformer blocks...
  ... layer 2/4
  ... layer 4/4
Adding output layers...
Finalizing GGUF file...

============================================================
✓ GGUF file created: models/tiny-llama-6m-f16.gguf
============================================================
File size:           6.50 MB (6656.0 KB)
Total parameters:    3,649,000 (~3.6M)

Parameter breakdown:
  Embedding:         256,000
  Transformer:       3,137,024 (4 × 784,256)
    Per-layer:
      Attention:     327,680
      FFN:           447,648
      Norms:         8,928
  Output head:       256,000
```

### Step 2: Quantize Model (3 min)

```bash
cd C:\Users\th1rt\Documents\Workspaces\llama.cpp\build\bin

llama-quantize.exe \
  ../../../code-ai-workspace/models/tiny-llama-6m-f16.gguf \
  ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf \
  Q2_K
```

**Expected output:**
```
llama_model_loader: loaded meta data with 20 key-value pairs and 115 tensors from models/tiny-llama-6m-f16.gguf ...
...
main: quantizing 'tiny-llama-6m-f16.gguf' to 'tiny-llama-6m-q2k.gguf' as Q2_K
...
main: total time: 2.34 seconds
```

**File sizes:**
- Input:  6.5 MB
- Output: ~1.2 MB (18% of original)

### Step 3: Verify Model Works (1 min)

Quick sanity check:

```bash
cd C:\Users\th1rt\Documents\Workspaces\llama.cpp\build\bin

llama-cli.exe \
  -m ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf \
  -p "hello" \
  -n 10 \
  -t 1
```

**Expected:** Generates 10 random tokens (model has random weights, so output is gibberish).

### Step 4: Benchmark Our Engine (5 min)

Terminal 1: Monitor system resources
```bash
# Keep this running in background to watch CPU/VRAM
nvidia-smi
# For CPU-only: Task Manager or psutil
```

Terminal 2: Run engine benchmarks

```bash
cd C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\src\engine\build

echo "=== GEMV Kernel Benchmark ===" > ../../../bench_results.txt
./bench_gemv.exe >> ../../../bench_results.txt 2>&1

echo "" >> ../../../bench_results.txt
echo "=== Full Component Benchmark ===" >> ../../../bench_results.txt
./bench_components.exe >> ../../../bench_results.txt 2>&1
```

**Saves results to:** `C:\Users\th1rt\Documents\Workspaces\bench_results.txt`

### Step 5: Benchmark llama.cpp (5 min)

```bash
cd C:\Users\th1rt\Documents\Workspaces\llama.cpp\build\bin

echo "=== llama.cpp test-backend-ops (Kernel Benchmark) ===" >> ../../../bench_results.txt
./test-backend-ops.exe perf 2>&1 | grep -i "mul_mat" >> ../../../bench_results.txt || \
  echo "(test-backend-ops may take 2-3 min, skipping for now)" >> ../../../bench_results.txt

echo "" >> ../../../bench_results.txt
echo "=== llama.cpp bench (Q2_K) ===" >> ../../../bench_results.txt
./llama-bench.exe \
  -m ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf \
  -t 1 \
  -n 128 \
  -r 5 \
  -p 0 >> ../../../bench_results.txt 2>&1

echo "" >> ../../../bench_results.txt
echo "=== llama.cpp bench (F16) ===" >> ../../../bench_results.txt
./llama-bench.exe \
  -m ../../../code-ai-workspace/models/tiny-llama-6m-f16.gguf \
  -t 1 \
  -n 128 \
  -r 5 \
  -p 0 >> ../../../bench_results.txt 2>&1
```

**Total llama.cpp benchmark time:** 10-15 minutes

### Step 6: Compile Results (5 min)

```bash
cd C:\Users\th1rt\Documents\Workspaces\code-ai-workspace

cat bench_results.txt

# View in editor
notepad .workspace/plans/benchmark-p43-raw-results.txt
```

### Step 7: Analyze and Document (5 min)

Update: `docs/engine-benchmark-results.md`

Add a new section:

```markdown
## Benchmark Round [DATE]

### Models
- Our engine: 6M ternary (d=256, 4 layers)
- llama.cpp: 6M Transformer (d=256, 4 layers)

### Hardware
- CPU: Intel Core Ultra 5 225F (6P cores)
- RAM: 32GB DDR5
- SIMD: AVX2, FMA, POPCNT

### Results

#### GEMV Kernel (256×512)
| Engine | Implementation | Latency | GOPS | Notes |
|--------|----------------|---------|------|-------|
| Our | Ternary AVX2 | X.XX us | Y.Y | (actual result) |
| llama.cpp | Q2_K | X.XX us | Y.Y | (actual result) |

#### E2E Inference (6M model, 128 tokens)
| Model | Quantization | tok/s | Peak RSS | Time to first token |
|-------|--------------|-------|---------|-------------------|
| Our engine | N/A (packed ternary) | X.X | X MB | X ms |
| llama.cpp | Q2_K | X.X | X MB | X ms |
| llama.cpp | F16 | X.X | X MB | X ms |

#### Analysis
- Where we win: [kernel latency, memory footprint, etc.]
- Where we lose: [large dimensions, prompt processing, etc.]
- Fair/unfair comparisons: [methodology notes]
```

### Step 8: Commit Results

```bash
cd C:\Users\th1rt\Documents\Workspaces\code-ai-workspace

git add docs/engine-benchmark-results.md .workspace/plans/benchmark-p43-raw-results.txt

git commit -m "bench: P4.3 benchmark results — 6M ternary vs llama.cpp Q2_K"
```

---

## Troubleshooting

### llama-bench crashes immediately

Usually means the model is corrupted or too small. Try:

```bash
llama-cli.exe -m model.gguf -p "test" -n 1
```

If it crashes, re-create model:

```bash
python src/engine/tools/create_tiny_gguf.py -o models/tiny-llama-6m-f16.gguf
```

### Benchmark results vary wildly between runs

This is normal on a general-purpose PC. Mitigations:

1. Close other applications (especially Chrome, VS Code)
2. Disable background tasks (update checkers, antivirus scans)
3. Run multiple times and average

For scientific comparison, run at least 3 times and report mean ± std dev.

### Memory usage spikes to 100%

Our engine or llama.cpp may be allocating all system RAM. This is expected behavior:

- Our engine: Allocates model weights + activation buffers upfront
- llama.cpp: Similar allocation strategy

Reduce model size or batch size if needed. For this benchmark, single-token generation shouldn't exceed 500 MB.

### "Tokens per second" seems too low

Benchmarks on CPU are intentionally single-threaded for fairness. Expected ranges:

- Our ternary GEMV: 200-500 tok/s
- llama.cpp on small model: 50-200 tok/s

If significantly lower (< 10 tok/s), check:

1. Single-thread flag is set (`-t 1`)
2. No thermal throttling (check CPU temps with HWiNFO)
3. Model loaded correctly (verify with llama-cli test)

### "test-backend-ops" produces no output

This tool has a verbose mode but no filtering built-in. Try:

```bash
./test-backend-ops.exe perf 2>&1 | head -100
```

If output is empty, the tool may not support your hardware. Skip it and rely on llama-bench instead.

---

## Expected Performance

Based on our analysis in the design spec:

### GEMV Kernel Level

| Operation | Our Ternary | llama.cpp Q2_K | Ratio |
|-----------|-----------|----------------|-------|
| 256×256 | ~50-100 us | ~100-200 us | 1.5-2x our favor |
| 1024×1024 | ~800 us | ~1500 us | 1.8x our favor |
| 4096×4096 | ~50 ms | ~40 ms | 1.2x llama.cpp favor (their optimization) |

### End-to-End Inference (128 tokens)

| Metric | Our Engine | llama.cpp Q2_K | Difference |
|--------|-----------|----------------|-----------|
| tok/s | ~400-500 | ~80-120 | 3-5x our favor |
| Peak RSS | ~50 MB | ~200 MB | 4x less our favor |
| Startup | ~10 ms | ~5 ms | 2x our favor |

---

## Files Modified/Created

- `src/engine/tools/create_tiny_gguf.py` — GGUF generator
- `src/engine/tools/quantize_gguf.sh` — Quantization helper
- `src/engine/tools/check_benchmark_env.py` — Dependency checker
- `src/engine/tools/README_GGUF.md` — Full GGUF documentation
- `models/tiny-llama-6m-f16.gguf` — Generated model (not in git)
- `models/tiny-llama-6m-q2k.gguf` — Quantized model (not in git)
- `.workspace/plans/benchmark-p43-raw-results.txt` — Raw benchmark output
- `docs/engine-benchmark-results.md` — Analysis and summary

---

## Quick Commands Reference

```bash
# Generate model
python src/engine/tools/create_tiny_gguf.py -o models/tiny-llama-6m-f16.gguf

# Quantize
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin
llama-quantize.exe ../../../code-ai-workspace/models/tiny-llama-6m-f16.gguf \
                   ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf Q2_K

# Test
llama-cli.exe -m ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf -p "hello" -n 10 -t 1

# Our engine
cd C:/Users/th1rt/Documents/Workspaces/code-ai-workspace/src/engine/build
./bench_gemv.exe
./bench_components.exe

# llama.cpp benchmark
cd C:/Users/th1rt/Documents/Workspaces/llama.cpp/build/bin
llama-bench.exe -m ../../../code-ai-workspace/models/tiny-llama-6m-q2k.gguf -t 1 -n 128 -r 5
```
