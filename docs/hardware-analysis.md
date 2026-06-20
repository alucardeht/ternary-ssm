# Hardware Analysis — RTX 5070 + Intel Core Ultra 5 225F

## RTX 5070 Specifications

| Spec | Value |
|------|-------|
| Architecture | Blackwell (GB205) |
| VRAM | 12GB GDDR7 (192-bit, 28 Gbps) |
| Memory Bandwidth | 672 GB/s (+33% vs RTX 4070) |
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th gen) |
| AI TOPS | 988 (INT8/FP8) |
| FP32 | 31 TFLOPS |
| FP4 (Dense) | ~988 TFLOPS |
| FP4 (Peak) | ~1,406 TOPS |
| TDP | 250W |
| MSRP | $549 |

### Blackwell Advantages for LLMs

- **Native FP4/FP6 Support** — First consumer GPU with hardware FP4. Models requiring 20GB+ at FP16 can fit in 12GB at FP4 with minimal quality loss.
- **5th-gen Tensor Cores** — Optimized for transformer operations and scaled dot-product attention.
- **2nd-gen FP8 Transformer Engine** — 10-20% real inference gains over unoptimized FP16 without quality regression.
- **GDDR7 Memory** — 672 GB/s bandwidth critical for token generation (LLM inference is memory-bandwidth bound, not compute bound).
- **PCIe 5.0 Support** — Faster CPU-GPU communication for batch processing.

### LLM Performance Benchmarks

- **7B models (Q4 quantization):** 50-100+ tokens/sec (framework dependent)
- **Peak performance (optimized TensorRT-LLM, 7B):** up to 140-360 tok/s
- **13B models:** Requires aggressive quantization, not ideal for 12GB VRAM
- **Sweet spot:** 7-8B models with Q4_K_M or AWQ quantization

Practical token rates:
- Ollama: 50-70 tok/s (conservative, CPU overhead)
- llama.cpp: 40-60 tok/s (portable, accurate)
- vLLM: 200+ tok/s (batch processing, production)
- SGLang: 180+ tok/s (RadixAttention optimization)

### Fine-Tuning Capabilities

- **7B QLoRA:** Works reliably (~11-12GB VRAM, batch_size=1)
- **13B QLoRA:** Possible with extreme optimization (gradient_checkpointing, reduced sequence_length)
- **Full 16-bit fine-tuning:** NOT viable (needs 2-4x model size in VRAM)

### Comparison: RTX 5070 vs Previous Generations

| Metric | RTX 5070 | RTX 4070 | RTX 4080 |
|--------|----------|----------|----------|
| VRAM | 12GB GDDR7 | 12GB GDDR6X | 16GB GDDR6X |
| Memory Bandwidth | 672 GB/s | 504 GB/s | 717 GB/s |
| Native FP4 Support | Yes | No | No |
| AI TOPS (INT8) | 988 | ~490 | ~780 |
| TDP | 250W | 200W | 320W |
| LLM inference (7B Q4) | 50-100+ tok/s | 30-50 tok/s | 40-70 tok/s |
| Cost | $549 | $599-699 | $1,199 |

RTX 5070 delivers **best value for LLM workloads** — superior to RTX 4070 in performance while maintaining same VRAM, priced $50-100 cheaper.

## Intel Core Ultra 5 225F

| Spec | Value |
|------|-------|
| Architecture | Arrow Lake |
| P-Cores (Performance) | 6 cores |
| E-Cores (Efficiency) | 4 cores |
| Total Cores | 10 (6P + 4E) |
| NPU (Neural Processing Unit) | NPU3, 13 TOPS (INT8) |
| iGPU (Integrated Graphics) | **None (F-suffix)** |
| Max System RAM Support | DDR5-6400, up to 192GB |
| Platform AI TOPS | 36 (P-cores: 23 + NPU: 13) |

### Critical Limitation: No Integrated GPU

The "F" suffix designation means this CPU has **no integrated Xe-LPG GPU**. This eliminates several capabilities that other Arrow Lake processors have:

- **Cannot use shared GPU memory override** — No ability to access system RAM as GPU VRAM
- **No GPU for graphics** — Requires discrete GPU for any display output
- **No GPU acceleration** — Cannot run GPU-accelerated inference alongside NPU

**Implication:** The 16GB of available system RAM from your 32GB total **cannot be used as GPU VRAM** for the RTX 5070. GPU and system memory are completely isolated.

### NPU Capabilities

**Performance:**
- 13 TOPS INT8 compute
- Capable of running 7B INT8 models at ~15-20 tokens/sec
- Power draw: ~5W (vs 250W for RTX 5070)
- Excellent power efficiency (2.6 TOPS/W)

**Supported Frameworks:**
- OpenVINO (Intel's framework, best support)
- IPEX-LLM (Intel PyTorch extension)
- NITRO (lightweight)
- Not supported in Ollama (yet)

**Supported Models:**
- Llama 2 7B (INT8)
- Mistral 7B (INT8)
- Qwen 2 7B (INT8)
- Phi-3 (INT8)
- Gemma 7B (INT8)

**Quantization Notes:**
- INT8 is the optimal format for NPU
- INT4 quantization shows **no speed improvement** on NPU vs INT8 (memory bandwidth bottleneck — NPU VRAM is limited)
- Precision loss: minimal (95%+ quality vs INT8)

**Framework Limitation:**
- No framework currently supports **NVIDIA GPU + Intel NPU hybrid inference** (model splitting, dynamic routing)
- This remains an open research area

### NPU Practical Use Cases

1. **Always-on code completion assistant** — Lightweight model, 5W power draw
2. **Background inference** — While RTX 5070 handles main workload
3. **Task specialization** — Different model on NPU for specialized domains
4. **Efficient batching** — NPU for light requests, RTX 5070 for heavy requests

**Not recommended for:** High-throughput serving, complex reasoning, large context processing.

## Memory Architecture Summary

```
┌──────────────────────────────────────────┐
│ System RAM: 32GB DDR5                    │
│ ├── ~16GB: OS + System Applications      │
│ └── ~16GB: Available (but NOT usable as  │
│           GPU VRAM — no iGPU!)           │
├──────────────────────────────────────────┤
│ RTX 5070 VRAM: 12GB GDDR7               │
│ ├── Dedicated, high-speed (672 GB/s)    │
│ ├── Isolated from system RAM             │
│ └── ONLY GPU memory available            │
├──────────────────────────────────────────┤
│ Intel NPU: Uses System RAM               │
│ ├── Draws from available DDR5            │
│ ├── 13 TOPS, ~15-20 tok/s               │
│ └── Separate pipeline from GPU           │
└──────────────────────────────────────────┘
```

## Practical Implications for Project

**For Inference (Phase 1):**
- RTX 5070 can handle all 7-8B models with 50-100+ tok/s
- 12GB is sufficient for Q4 quantization
- No workarounds needed; VRAM is adequate

**For Fine-Tuning (Phase 3):**
- QLoRA at batch_size=1 fits in 12GB (barely)
- Sequence length starts at 512, can extend to 1024-2048
- System RAM available for caching datasets, checkpoints

**For Dual-Device Setup:**
- NPU can run lightweight model independently (always-on)
- RTX 5070 and NPU cannot share models or inference workload
- Task specialization (different models) preferred over model splitting

**Scaling Path:**
- For 13B models: Upgrade GPU to RTX 6000 (24GB) or RTX 5080 (16GB)
- For 70B models: Enterprise GPU (H100/H200, 80GB+)
- Current setup (RTX 5070 + 225F) is optimal for 7-8B fine-tuning
