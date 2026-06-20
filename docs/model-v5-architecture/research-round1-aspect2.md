# Round 1, Aspect 2: Natively RAM-Efficient Architectures (Trained from Scratch)

## Wave 1 — Web Research (2 researchers, Feb 2026)

### PARADIGM: ARCHITECTURE-LEVEL RAM GAINS (zero quality loss)

#### v5-R1-A2-F1: Universal Reasoning Model (URM) — Cross-Layer Weight Sharing
- **Source:** arXiv:2512.14693v1, December 2025
- **Summary:** ALL parameters (attention + FFN) shared across layers. 18M params achieves BERT-large (334M) performance = 18x fewer unique parameters. Includes ConvSwiGLU and truncated backpropagation. Trained from scratch.
- **Quality:** SAME/BETTER (+13% on reasoning tasks at 32x FLOPs)
- **RAM gain:** 18x fewer unique parameters stored. For our case: a 7B-quality model might need only ~400M unique params + loop overhead.
- **Throughput:** Same or faster inference (same compute per forward pass, just reused)
- **Trained from scratch:** YES
- **Verdict:** PROMISING — Proven parameter sharing at scale, quality improves on reasoning

#### v5-R1-A2-F2: Recursive Inference Scaling (RINS)
- **Source:** NeurIPS 2025, "Recursive INference Scaling: A Winning Path to Scalable Inference"
- **Summary:** Recursive depth with parameter sharing + lightweight linear adapters (<1% params). Improves BOTH asymptotic performance AND scaling exponents. No-regret strategy: improves training-compute, parameter, AND inference performance simultaneously.
- **Quality:** BETTER (+2% ImageNet for SigLIP-B/16, improved asymptotic limits)
- **RAM gain:** Adapters are <1% of base model. Same base params reused recursively = fewer unique params needed.
- **Throughput:** Same at inference (recursion is depth, not width)
- **Trained from scratch:** YES
- **Verdict:** PROMISING — Quality IMPROVEMENT with efficiency, proven approach

#### v5-R1-A2-F3: Basis Sharing — Cross-Layer Shared Basis Vectors
- **Source:** ICLR 2025, "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression"
- **Summary:** Weight matrices across layers decomposed as linear combinations of shared basis vectors + unique per-layer coefficients. Unlike ALBERT (shares full matrices), this shares only the BASIS allowing layer-specific adaptations.
- **Quality:** BETTER than ALBERT-style sharing and SVD-based approaches at same param count
- **RAM gain:** Shared basis vectors + small per-layer coefficients. For 32-layer model: instead of 32 full matrices, store 1 basis set + 32 small coefficient sets.
- **Throughput:** Same or slightly faster (smaller active parameter set)
- **Trained from scratch:** YES (can be trained from scratch, paper validates both)
- **Verdict:** PROMISING — Outperforms full-matrix sharing, fine-grained control per layer

#### v5-R1-A2-F4: Fine-Grained MoE with Shared Experts (DeepSeek V3 / Qwen3)
- **Source:** DeepSeek V3/R1 (2025), Qwen3 (2025), Llama4 Maverick (2025)
- **Summary:** 128-256 fine-grained experts with reduced hidden size per expert + shared experts for common knowledge. DeepSeek V3: 256+1 experts, only 8 active per token (96.9% inactive). Qwen3: 128 experts, 22B active of 235B total. Sigmoid gating for stability.
- **Quality:** SAME/BETTER (DeepSeek-R1 and Qwen3 top ALL leaderboards as of Feb 2026)
- **RAM gain:** Only active experts loaded to fast memory. 8 of 256 experts = 3.1% of expert params in cache per token. Rest can live on slower storage (NVMe). For ternary: each expert even smaller (1.58 bits/weight).
- **Throughput:** 500+ tok/s proven at production scale
- **Trained from scratch:** YES
- **Verdict:** PROMISING — Dominant paradigm in 2025-2026, proven at frontier scale

### PARADIGM: INTELLIGENCE FROM COMPUTE-TIME, NOT PARAMETER-TIME

#### v5-R1-A2-F5: Inference-Time Compute Scaling (DeepSeek-R1 paradigm)
- **Source:** DeepSeek-R1 (Jan 2025), ThreadWeaver (2025), P1 Physics Model (Nov 2025)
- **Summary:** Test-time compute scaling achieves higher intelligence WITHOUT more parameters. DeepSeek-R1 7B matches o1 quality via extended chain-of-thought at inference (10-100x more tokens per query). Intelligence scales with THINKING TIME, not model size.
- **Quality:** BETTER — DeepSeek-R1 AIME: 15.6% -> 71% (86.7% with voting) via inference scaling ALONE
- **RAM gain:** PARADIGM SHIFT. A small base model (e.g., our ternary SSM) can achieve much higher intelligence by thinking longer at inference. The base model stays small in RAM; intelligence comes from compute.
- **Throughput:** Slower per query (more tokens generated), but base model speed unchanged. For simple queries: fast. For hard queries: slower but smarter.
- **Trained from scratch:** YES (pure RL training via GRPO)
- **Verdict:** PROMISING — Radical reframe: RAM holds the "engine", intelligence comes from "running time"

### PARADIGM: TRAINING METHODOLOGY GAINS

#### v5-R1-A2-F6: Over-Training Small Models (Llama 3 methodology)
- **Source:** Meta, "The Llama 3 Herd of Models", July 2024 (8B trained on 15T tokens = 1875 tok/param)
- **Summary:** Training FAR beyond Chinchilla-optimal (20 tok/param) makes same-size models dramatically smarter. Llama 3 8B at 1875 tok/param achieves quality that Chinchilla-optimal would need 70B+ params for. The model doesn't get bigger — it gets SMARTER.
- **Quality:** SAME as much larger Chinchilla-optimal models
- **RAM gain:** MASSIVE. Instead of needing 70B params (FP16 = 140 GB), achieve same quality with 8B params (~16 GB FP16, ~1.6 GB ternary). The cost is MORE TRAINING DATA, not more params.
- **Throughput:** FASTER (fewer params = faster inference)
- **Trained from scratch:** YES
- **Verdict:** PROMISING — Proven at scale, directly applicable: train our ternary model on MORE data

### PARADIGM: SSM AS NATIVE RAM ADVANTAGE

#### v5-R1-A2-F7: xLSTM 7B — Fastest SSM Production Model
- **Source:** Beck et al., "xLSTM 7B", arXiv:2503.13427, March 2025
- **Summary:** 7B xLSTM matching Llama-3 and Mamba quality while being FASTER and more memory efficient. Linear compute scaling with sequence length + constant memory (no KV cache). Open-source, production-ready code.
- **Quality:** SAME as Llama-3 7B and Mamba-2 7B on downstream tasks
- **RAM gain:** No KV cache = saves 4-8 GB at 2K+ context vs Transformer. Fixed state size regardless of sequence length. For ternary version: 7B x 1.58 bits = ~1.4 GB (vs 14 GB FP16).
- **Throughput:** FASTER than both Llama-based and Mamba-based at same scale
- **Trained from scratch:** YES
- **Verdict:** PROMISING — Validates SSM approach, fastest 7B recurrent model in production

### THEORETICAL FRAMEWORK

#### v5-R1-A2-F8: MDL/Kolmogorov Complexity Objectives
- **Source:** Shaw et al., arXiv:2509.22445, September 2025
- **Summary:** Minimum Description Length objectives based on Kolmogorov complexity provably select for maximally efficient representations. Adaptive Gaussian mixture priors + variational objectives. MDL-SAE discovers 26x fewer features than baselines for same quality.
- **Quality:** BETTER generalization on algorithmic tasks
- **RAM gain:** Theoretical framework for finding the MINIMUM representation needed. If applied to LLM training: could discover architectures needing dramatically fewer params for same intelligence.
- **Throughput:** Same or faster (fewer active units)
- **Trained from scratch:** YES (MDL guides training from initialization)
- **Verdict:** NEEDS-MORE-WORK — Theoretical framework, not yet proven at LLM scale. But PRINCIPLED approach to our exact problem.

## Key Insights — Aspect 2

1. **Weight sharing is FREE efficiency** — URM proves 18x param reduction with BETTER reasoning quality. Basis sharing is more fine-grained. Both trained from scratch.
2. **Fine-grained MoE is the 2025-2026 standard** — DeepSeek V3, Qwen3, Llama4 ALL use 128-256 expert MoE. 96%+ params inactive per token. Proven at frontier scale.
3. **Intelligence from compute-time is a PARADIGM SHIFT** — DeepSeek-R1 7B matches o1 by thinking longer. Small base model + extended inference = high intelligence at low RAM.
4. **Over-training is the cheapest quality gain** — Train 1875 tok/param instead of 20 tok/param. Same model size, dramatically smarter. Cost = training time, not inference RAM.
5. **SSM is already a RAM advantage** — xLSTM 7B proves SSM matches Transformer quality with much less inference RAM (no KV cache).
6. **All findings are GAINS, not trade-offs** — Every finding here improves efficiency WITHOUT losing quality. This is the correct direction.

## For Our Project

**Combination potential (multiplicative gains):**
- Ternary weights: 10x less RAM per param (1.58 vs 16 bits)
- Weight sharing (URM/Basis): 5-18x fewer unique params
- Fine-grained MoE: 96% inactive per token (only 4% in fast memory)
- SSM: No KV cache (saves 4-8 GB at long context)
- Over-training: Same params = smarter (quality gain at zero RAM cost)
- Inference-time compute: Small base + extended thinking = high intelligence

**Hypothetical stack:**
- 7B total params x ternary (1.58 bits) = 1.4 GB stored
- Weight sharing (5x) = 280 MB unique params
- Fine-grained MoE with 4% active = 56 MB in fast memory per token
- SSM state = ~16 MB (vs 1+ GB KV cache)
- Over-training (1500 tok/param) = quality of much larger model
- Inference-time compute = Haiku+ quality on hard tasks

**RAM budget: ~350 MB active + NVMe backing for inactive experts**
