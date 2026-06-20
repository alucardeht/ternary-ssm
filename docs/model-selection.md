# Model Selection Guide — February 2026

## What's Outdated from the Original Plan

Original recommendations from 2024 have been superseded by newer models (2025-2026):

| Original (2024) | Replacement (2025-2026) | Reason |
|---|---|---|
| Mistral 7B v0.3 | **Ministral 3 8B** (Dec 2025) | New release, 3 size variants, improved performance |
| Qwen 2.5 7B | **Qwen 3 8B** (Apr 2026) | Newest release, 140+ languages, dual-mode (thinking + fast) |
| Gemma 2 9B | Gemma 3 exists but only 27B | Too large for 12GB VRAM |
| Llama 3.1 8B | **Still solid** (2024) | Mature ecosystem, widely supported, no replacement yet |
| _(not listed)_ | **DeepSeek-R1-Distill-Qwen-7B** (Jan 2025) | Game-changer: reasoning distilled to 7B, MIT license |

## Comprehensive Model Comparison

| Model | Params | Released | Context | License | Multilingual | Code | Fine-tune | VRAM (Q4) |
|---|---|---|---|---|---|---|---|---|
| **Qwen 3 8B** | 8.2B | Apr 2026 | 128K | Apache 2.0 | 140+ langs | Strong | Excellent | ~5GB |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | Jan 2025 | 128K | MIT | Good | Excellent | Excellent | ~4.5GB |
| **DeepSeek-R1-Distill-Llama-8B** | 8B | Jan 2025 | 128K | MIT | Good | Good | Excellent | ~5GB |
| **Qwen 2.5 7B Coder** | 7B | Sept 2024 | 128K | Apache 2.0 | Good | #1 Ranked | Excellent | ~4.5GB |
| **Llama 3.1 8B Instruct** | 8B | Aug 2024 | 128K | Apache 2.0 | Good | Good | Excellent | ~5GB |
| **Ministral 3 8B** | 8B | Dec 2025 | 32K/128K | Apache 2.0 | European | Good | Good | ~5GB |

### Model Details

**Qwen 3 8B** (Recommended for general purpose)
- Released: April 2026 (latest)
- Unique feature: Dual-mode with built-in thinking token
- Multilingual: 140+ languages including Portuguese
- Performance: Outperforms Qwen 2.5 14B on 15 benchmarks
- Fine-tuning: Excellent support, community growing
- Use case: Default choice for balanced performance

**DeepSeek-R1-Distill-Qwen-7B** (Recommended for reasoning)
- Released: January 2025
- Architecture: Distilled from 671B parameter model
- Benchmarks: 92.8% on MATH-500, 49.1% on GPQA Diamond
- License: MIT (most permissive)
- Advantages: Superior reasoning, logic, math — 7B size
- Fine-tuning: Excellent, maintains reasoning capability
- Use case: Math, logic problems, complex reasoning

**DeepSeek-R1-Distill-Llama-8B** (Alternative reasoning)
- Similar performance to Qwen variant
- Better Llama ecosystem compatibility
- Slightly worse on code tasks

**Qwen 2.5 7B Coder Instruct** (Recommended for coding)
- Released: September 2024
- Rankings: #1 in 2026 coding benchmarks (14/15 points)
- Specialization: Logic, debugging, code explanations
- Context: 128K tokens
- Fine-tuning: Excellent for code-domain adaptation
- Use case: Code completion, debugging, programming assistance

**Llama 3.1 8B Instruct** (Stability/ecosystem)
- Released: August 2024
- Advantages: Largest community, most tools, best documentation
- Performance: Solid across all tasks, no specific weaknesses
- Ecosystem: vLLM, llama.cpp, Ollama all have optimized paths
- Use case: Production fallback if others unavailable
- Fine-tuning: Excellent, established patterns

**Ministral 3 8B** (European focus)
- Released: December 2025
- Niche: Optimized for European languages (French, German, Spanish, Italian)
- Context: Flexible (32K/128K variants)
- Performance: Good general purpose, better on European tasks
- Use case: Multilingual European deployments
- Note: If Portuguese support is critical, prefer Qwen 3 over Ministral

## Recommendation by Use Case

### General Purpose + Portuguese Support
**→ Qwen 3 8B**
- Newest model (April 2026)
- Dual-mode: fast chat mode + thinking mode for complex queries
- 140+ languages, native Portuguese support
- Outperforms larger models on benchmarks
- Strong fine-tuning community emerging

### Reasoning, Logic, Mathematics
**→ DeepSeek-R1-Distill-Qwen-7B**
- Superior on MATH benchmarks (92.8% MATH-500)
- Reasoning distilled down to 7B size
- MIT license (no restrictions)
- Excellent for: solving equations, proofs, complex logic
- Growing community

### Code Completion, Debugging, Programming
**→ Qwen 2.5 7B Coder Instruct**
- #1 ranked coding model in 2026 (14/15 benchmarks)
- Excels at understanding programmer intent
- Excellent code review, debugging, explanations
- 128K context for large codebases
- Mature fine-tuning patterns

### Production/Stability/Fallback
**→ Llama 3.1 8B Instruct**
- Largest ecosystem
- Best documentation
- Most tools available
- Proven in production
- If main model unavailable, Llama 3.1 is safe alternative

### Multi-Device Setup
**RTX 5070 (primary):** Qwen 3 8B
**Intel NPU (background):** Qwen 2 7B or Phi-3 (smaller INT8 variant)

## Models NOT Recommended for 12GB VRAM

| Model | Size | Why Not |
|---|---|---|
| Llama 4 Scout | 17B active | Too large even with Q2 quantization |
| Gemma 3 27B | 27B | Only fits as Q2_K_M, severe quality loss |
| DeepSeek-R1 (full) | 671B / 37B active | Way too large, requires 80GB+ GPU |
| Devstral Small 2 | 24B | Exceeds 12GB even with aggressive quantization |
| Mixtral 8x7B | 56B | Mixture of Experts model, too large |

**Rule of thumb:** Stick to 7-8B parameter models for 12GB VRAM. Anything larger requires Q2_K_M quantization (poor quality) or better hardware.

## Quantization Guide (2026 State of the Art)

128K context is now standard for 7-8B models across all quantization formats.

| Format | Quality vs FP32 | Speed | File Size | Best For |
|---|---|---|---|---|
| **AWQ (4-bit)** | 95% fidelity | Fast | 4GB (7B) | Quality-focused work (creative, research, code) |
| **GGUF Q4_K_M** | 92% fidelity | Good | 4-5GB (7B) | Best balance, CPU/GPU flexible, portable |
| **GGUF Q5_K_M** | 94% fidelity | Good | 5-6GB (7B) | Higher quality if VRAM allows |
| **GPTQ 4-bit** | 90% fidelity | Fastest | 4GB (7B) | Maximum throughput on NVIDIA (vLLM) |
| **EXL2 4.65bpw** | Configurable | Fast | ~3.8GB (7B) | Fine-grained size/quality control |

### Practical Quantization Strategy

1. **Start with Q4_K_M (GGUF)**
   - Reason: Works in Ollama native (no extra setup)
   - Format: Universal, portable
   - Quality: 92% is acceptable for most tasks
   - Speed: 50-70 tok/s on RTX 5070

2. **Upgrade to AWQ if needed**
   - When: Quality is critical (fine-tuning, research)
   - Reason: 95% quality, still fast
   - Setup: More frameworks, slightly more complex
   - Speed: 60-80 tok/s on RTX 5070

3. **Use GPTQ for production API**
   - When: Serving 50+ requests/min with vLLM
   - Speed: 200+ tok/s with batching
   - Setup: Requires vLLM server (not Ollama)
   - Cost: Higher GPU utilization

4. **Avoid Q2/Q3 quantization**
   - Quality loss is severe
   - Speed gains marginal
   - Only if 12GB VRAM truly constrains (it shouldn't for 7B)

## Ollama Pull Commands

```bash
# General purpose (newest, recommended)
ollama pull qwen3:8b

# Reasoning and logic
ollama pull deepseek-r1:7b

# Code-focused
ollama pull qwen2.5-coder:7b

# Stable fallback
ollama pull llama3.1:8b

# European languages
ollama pull ministral:8b
```

All models above are available in GGUF Q4_K_M format through Ollama (largest community source).

## Quantization Download Sizes (Q4_K_M in Ollama)

- Qwen 3 8B: ~5.0GB
- DeepSeek-R1 7B: ~4.5GB
- Qwen 2.5 7B Coder: ~4.5GB
- Llama 3.1 8B: ~4.7GB
- Ministral 8B: ~4.8GB

**Total for all 5 models:** ~23GB (fits in user's system with room to spare)

## Fine-Tuning Capability Matrix

| Model | QLoRA Support | LoRA Support | Data Requirements | Community Resources |
|---|---|---|---|---|
| Qwen 3 8B | Excellent | Excellent | Growing | Small (new) |
| DeepSeek-R1 7B | Excellent | Excellent | Well-documented | Growing |
| Qwen 2.5 7B Coder | Excellent | Excellent | Abundant | Large |
| Llama 3.1 8B | Excellent | Excellent | Abundant | Very large |
| Ministral 8B | Good | Good | Emerging | Small |

All modern 7-8B models are equally fine-tunable with QLoRA. Choose based on inference performance, not fine-tuning capability.
