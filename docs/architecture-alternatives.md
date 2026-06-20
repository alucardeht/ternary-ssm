# Alternative Architectures to Transformers — Research Notes (February 2026)

## Status: Active Research — Seeking "Model v2"

The goal is to find architectures that achieve Transformer-level "thinking" quality with dramatically less hardware (especially VRAM). The Transformer has 3 fundamental wastes:

1. **Quadratic attention O(n²)** — 128K context = 16 billion attention ops, most useless
2. **KV Cache grows unbounded** — 7B model + 128K context = 67GB cache alone
3. **All layers process all tokens** — "the" gets same compute as "consequently"

---

## 1. BitNet b1.58 — Ternary Weights, No Matmul

### How It Works
Weights quantized to {-1, 0, +1} via absmean quantization. Matrix multiplication becomes addition/subtraction/skip. Activations remain BF16. log2(3) = 1.58 bits per weight.

```
Standard:  Y = X @ W   (float multiply-accumulate)
BitNet:    if W==1: add X; if W==-1: subtract X; if W==0: skip
```

### Hardware Implications
- No tensor cores needed — integer ALUs only
- CPU can be FASTER than GPU (integer addition is CPU-native)
- 10x less energy per operation (addition vs FP multiply)
- Paradigm inversion: GPU becomes unnecessary

### Numbers
| Metric | Transformer 7B (FP16) | BitNet 7B |
|--------|----------------------|-----------|
| Weight memory | 14 GB | ~1.1 GB (12x less) |
| Energy | 100% | 18-28% (71-82% savings) |
| Speed (CPU) | 1x | 2.4-6.2x faster |
| Needs GPU? | Yes | No — CPU is faster |

### Quality
- Sub-1B: significant quality loss
- 2B: competitive (68.5% ARC-Challenge = matches Llama 3 3B)
- 3B+: matches full precision across benchmarks
- 70B: 4.1x faster than Llama 70B, comparable quality

### Training
- Can train on Google Colab T4 (free tier) via AutoBitnet
- onebitllms toolkit for training/fine-tuning
- BitLinear: drop-in replacement for nn.Linear
- Must train natively (can't convert FP16 model to BitNet)

### Production Status: YES
- Microsoft bitnet.cpp (Oct 2024) — official inference framework
- microsoft/bitnet-b1.58-2B-4T on HuggingFace (Apr 2025)
- Supports x86 (Intel/AMD), ARM (Apple Silicon)
- Falcon-Edge family available

### Key Limitation
Requires specialized bitnet.cpp kernels. Standard frameworks show no gains. Limited model selection compared to Transformers.

---

## 2. Mamba Hybrid (1:8 Transformer:Mamba Ratio)

### How It Works
State Space Model (SSM) compresses sequence into fixed-size hidden state instead of KV cache. Selective mechanism: SSM parameters become functions of input, allowing content-based filtering.

```
Transformer: stores K,V for ALL tokens = O(n) memory
Mamba: compresses into fixed state = O(1) memory
Hybrid (1:8): 87.5% Mamba + 12.5% Attention = efficiency + reasoning
```

Mamba-2 improvements: 25 lines of code (vs 100+), 2x throughput over Mamba-1, larger hidden states.

### Why Hybrid?
Pure Mamba fails on:
- Multi-hop reasoning (can't "jump back" to earlier context)
- In-context learning (5-shot MMLU, Phonebook Lookup)
- Copy and recall tasks (fundamental SSM limitation)

1 attention layer per 8 layers handles these cases. Rest is Mamba for efficiency.

### Numbers (IBM Granite 4.0-H 8B)
| Metric | Transformer 8B | Mamba Hybrid 8B |
|--------|---------------|-----------------|
| VRAM (256K ctx) | 90 GB | 15 GB (6x less) |
| KV Cache | 100% | 12.5% (8x less) |
| Throughput | 1x | 4-5x faster |
| Quality (12 benchmarks) | baseline | +2.65 points ABOVE |

Quality is BETTER, not just comparable.

### Production Status: YES — Multiple Enterprise Deployments
- IBM Granite 4.0 (Nov 2025) — Apache 2.0
- AI21 Jamba 1.5 — beats Claude 3 Haiku, Mixtral 8x22B on Arena Hard
- NVIDIA Nemotron-H (Dec 2025)
- Mistral Codestral Mamba 7B — beats CodeLlama 34B at code
- Falcon Mamba 7B — open source

### Key Limitation
Pure Mamba can't do content-based reasoning. Hybrid solves this but still has small KV cache (12.5% of Transformer).

---

## 3. xLSTM 7B — Sepp Hochreiter's Comeback

### How It Works
Two variants:
- **sLSTM**: scalar memory, explicit state tracking
- **mLSTM**: matrix memory, FULLY PARALLELIZABLE during training (the breakthrough)

Plus exponential gating (replaces sigmoid, fixes vanishing gradients, enables 100+ layer depth).

### Numbers
- Constant memory (no KV cache, O(1) per token)
- Lower perplexity than Transformer at same scale
- Extrapolates beyond training context WITHOUT degradation (Transformers degrade)
- NXAI claims "fastest and most efficient 7B LLM"
- For fixed FLOP budget, xLSTM achieves lower validation loss than Transformer

### Production Status: YES
- xLSTM 7B released March 2025, trained on 2.3T tokens
- HuggingFace: NX-AI/xLSTM-7b
- GitHub: NX-AI/xlstm
- Open source: weights + code + training code

### Key Limitation
Only 7B size available. Newer ecosystem, fewer tools. Claims need independent validation. Requires mlstm_kernels package.

---

## 4. RWKV-7 "Goose" — RNN That Trains Like Transformer

### How It Works
RNN-style but parallelizable during training. Receptance-weighted mechanism controls info flow into fixed hidden state. RWKV-7 adds dynamic state evolution, in-context learning rates, recognizes all regular languages.

### Numbers
- O(1) memory: any context length, even 1 million tokens
- With INT8 compression: 10x memory reduction (~1.4GB for 7B)
- RWKV-7 2.9B: 71.4 MMLU (matches Qwen2.5-3B, beats LLaMA3.2-3B)
- Runs on embedded ARM at 16 tok/s
- 206K tok/s training speed on 4x8xH100

### Production Status: YES
- RWKV-7 released March 2025
- All earlier versions (RWKV-6/5/4) are EOL — use RWKV-7 only
- llama.cpp integration available

### Key Limitation
Passkey retrieval degrades rapidly past 28K tokens. Prompt format sensitivity. "Infinite context" is misleading — recall weakens over distance.

---

## 5. MatMul-Free LM — No Matrix Multiplication At All

### How It Works
Replaces ALL components:
- Dense layers: ternary weights (like BitNet)
- Token mixing: MLGRU with element-wise (Hadamard) products only
- Channel mixing: GLU with BitLinear, element-wise only

No matmul anywhere. Just addition, subtraction, element-wise multiply.

### Numbers (2.7B tested)
- 10x less memory than Transformer
- 4x faster inference
- 10x more energy efficient
- 61% memory reduction during training
- Quality comparable to Transformer at 2.7B (higher ARC-Challenge accuracy)

### Production Status: NO — Research Only
- Paper: June 2024 (arxiv.org/abs/2406.02528)
- GitHub: ridgerchu/matmulfreellm
- Only tested up to 2.7B
- No pre-trained models, no ecosystem integration
- Theoretically CPU > GPU but unvalidated

### Key Limitation
No models > 2.7B. Research code only. Unproven at scale. But if validated at 7B+, this could be transformative.

---

## 6. TTT (Test-Time Training) — Learning During Inference

### How It Works
Hidden state IS a machine learning model (linear or MLP) that trains on context in real-time.
- TTT-Linear: linear model as hidden state
- TTT-MLP: MLP as hidden state
- TTT-E2E (Dec 2025): final 25% of blocks are mutable (static + dynamic layers)

### The Breakthrough
Perplexity KEEPS DECREASING with more context. Mamba saturates at 16K. Transformer goes OOM. TTT just gets better indefinitely.

### Numbers (1.3B tested)
- 2.7x faster than Transformer on 128K context
- Matches Mamba-2 speed
- Achieves Transformer accuracy on long context tasks
- Linear O(n) complexity

### Production Status: NO — Research Only
- TTT-E2E paper: December 31, 2025 (2 months old)
- GitHub implementations available (PyTorch + JAX)
- Only tested at 1.3B scale
- No production serving infrastructure
- Open question: how to handle concurrent requests with stateful updates?

### Key Limitation
Research only. No pre-trained models. Complex meta-learning training. Untested at 7B+. But most promising future direction.

---

## The "Model v2" Concept — Combining Everything

### No One Has Combined These Yet

| Technique | Eliminates | Status |
|-----------|-----------|--------|
| BitNet weights | Matmul, GPU dependency | Production |
| SSM/Mamba core | KV cache, O(n) memory | Production |
| MoD (Mixture of Depths) | Wasted compute on easy tokens | Research |
| Early Exit (LayerSkip) | Processing all 32 layers | Production (Meta) |
| Speculative decode | Slow sequential generation | Production |

**The combination "BitNet + SSM" does not exist yet.** Each technique is studied in isolation.

### Theoretical Model v2

```
Model v2 = BitNet weights (no matmul, 1.58-bit)
         + Mamba/SSM core (no KV cache, O(1) memory)
         + MoD routing (easy tokens skip layers)
         + Early Exit (simple tokens exit at layer 8)
```

### Theoretical Performance vs Standard Transformer 7B

```
                        Transformer 7B     Model v2 (7B equiv)
                        (Q4, Ollama)       (theoretical)
────────────────────────────────────────────────────────────
Weight memory           4.5 GB             ~0.5 GB (BitNet)
KV Cache (32K)          4.0 GB             ~0 (SSM, no cache)
Total VRAM              ~10 GB             ~1-2 GB
Needs GPU?              Yes                No (CPU faster)
Easy token compute      32 layers (100%)   ~8 layers (~25%)
Avg compute/token       100%               ~30% (MoD+exit)
Power consumption       250W               ~15W (CPU only)
Max context             53K (VRAM limit)   Unlimited (O(1))
```

### What's Needed to Build It
1. Implement BitLinear layers (exists: Microsoft's code)
2. Implement SSM blocks with BitNet weights (novel — no one has done this)
3. Add MoD routing (exists in research)
4. Add early exit heads (exists: Meta LayerSkip)
5. Train from scratch (BitNet must be trained natively)

### Risk
No one has validated BitNet + SSM convergence during training. Each piece works alone. The combination MIGHT have numerical instabilities.

### Feasibility
HIGH — it's integration of existing pieces, not invention. Each component is well-understood individually.

---

## Decision Matrix

```
Need production deployment NOW?
  → Mamba Hybrid (IBM Granite 4.0, AI21 Jamba)

Need CPU-only inference?
  → BitNet b1.58

Need edge/embedded deployment?
  → RWKV-7

Want to experiment with post-Transformer?
  → xLSTM 7B

Researching long-context breakthrough?
  → TTT (if willing to use research code)

Want to build something truly novel?
  → Model v2 (BitNet + SSM + MoD + Early Exit)
```

---

*Sources listed in research agent outputs.*
*Last updated: February 2026*
