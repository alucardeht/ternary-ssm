# Round 1, Aspect 5: Memory-Efficient Training & Inference

## Wave 1 — Web Research (2 researchers, Feb 2026)

### MEMORY-EFFICIENT TRAINING

#### v5-R1-A5-F1: AdaPM — Adaptive Partial Momentum
- **Source:** Zhang et al., ICLR 2026 (submitted Sept 2025)
- **Summary:** Assigns different momentum designs to different parameter blocks instead of uniform momentum. Most blocks don't require full momentum acceleration. Uses error-feedback bias correction. Reduces momentum memory by up to 90% while maintaining training efficiency.
- **Memory Savings:** 90%+ reduction in momentum memory, 30%+ GPU hours saved
- **Quality:** Maintains performance parity with full AdamW
- **Maturity:** Production-ready (tested on GPT-2, LLaMA 60M-1.5B, RLHF)
- **Relevance:** Momentum is a major memory component — adaptive block-level assignment could dramatically reduce our training memory.
- **Verdict:** PROMISING

#### v5-R1-A5-F2: Q-Adam-mini — 8-bit Quantized Adam-mini
- **Source:** Han et al., ICML 2025 Workshop (ES-FoMo III)
- **Summary:** Builds on Adam-mini (50% savings) by quantizing first-order momentum (m) to INT8 while keeping second-order (v) in FP32 (<1% memory). Stochastic rounding applied ONLY to embedding layer to avoid weight norm instability. Net result: 8× reduction in GPU optimizer memory vs AdamW.
- **Memory Savings:** 8× reduction in optimizer state memory
- **Quality:** Performance parity with full-precision AdamW (tested 60M-8B)
- **Maturity:** Production-ready (pre-training + fine-tuning validated)
- **Relevance:** EXTREMELY relevant — INT8 momentum is ternary-compatible. Our ternary weights need minimal optimizer state. Could enable training 7B ternary on consumer GPU.
- **Verdict:** PROMISING

#### v5-R1-A5-F3: CompAct — Compressed Activations via Random Projections
- **Source:** Shamshoum et al., 2024 (updated 2025)
- **Summary:** Projects activations X ∈ ℝ^(bl×n) as Z = XP (P ∈ ℝ^(n×r)) using random projection, stores only Z + random seed. Reconstructs gradients via same projection during backprop. Convergence tied to preserving singular values.
- **Memory Savings:** 25-30% GPU memory (pre-training), 50% (fine-tuning)
- **Quality:** <5% quality drop
- **Maturity:** Production-ready
- **Relevance:** Random projections align perfectly with our ternary architecture — P could be a ternary random matrix (JL lemma applies to ternary projections, proven in A3-P3).
- **Verdict:** PROMISING

#### v5-R1-A5-F4: TriTera — Ternary LLMs at Scale (1.2T Tokens)
- **Source:** Vaidhya et al., ACL 2025 (July 2025)
- **Summary:** Suite of ternary LLMs trained on 1.2T tokens. Scaling law analysis shows ternary models benefit MORE from data than parameter scaling (critical insight). Novel 2-bit and 1.6-bit packing schemes for ternary weights. TriRun GPU kernel achieves 5× inference speedup vs FP baselines.
- **Memory Savings:** Ternary = 1.58 bits/param vs 16 bits for FP16
- **Quality:** Sustained performance gains at scale (scales with data, not params)
- **Maturity:** Production-ready (open suite + inference kernels)
- **Relevance:** CRITICAL reference for v5. Confirms ternary training at 1.2T tokens is viable. Scaling law insight ("more data > more params for ternary") directly impacts our architecture sizing decisions.
- **Verdict:** PROMISING

#### v5-R1-A5-F5: Joint Reversible-Quantized Optimizer
- **Source:** Liu et al., IEEE 2025 (Nov 2024 published)
- **Summary:** Combines reversible layers (invertible activations — reconstruct inputs from outputs) with quantized optimizer states. Reversible layers eliminate activation storage during backprop. Quantized states reduce optimizer memory. Combined: 16.2× memory savings.
- **Memory Savings:** Up to 16.2× (combined technique)
- **Quality:** Nearly identical model quality maintained
- **Maturity:** Production-ready (validated on deep embeddings)
- **Relevance:** Reversible ternary layers are particularly clean — ternary quantization after inversion stays ternary. The 16.2× combined savings could enable training our full model on single GPU.
- **Verdict:** PROMISING

#### v5-R1-A5-F6: ZenFlow — Gradient Priority Offloading
- **Source:** PyTorch Blog / Xinnor benchmarks, Feb 2026
- **Summary:** Keeps "important" gradients (top 10% by norm) on GPU for immediate update. Offloads "unimportant" 90% to CPU, updates asynchronously. Exploits observation that small gradient subset contributes most update signal.
- **Memory Savings:** Enables significantly larger batch sizes on same VRAM
- **Quality:** Minimal convergence impact (slightly more steps needed)
- **Maturity:** Production-ready (DeepSpeed extension)
- **Relevance:** For ternary training, gradient magnitude distribution may be bimodal (near threshold vs far from threshold). Priority offloading could keep only threshold-critical gradients on GPU.
- **Verdict:** PROMISING

### MEMORY-EFFICIENT INFERENCE

#### v5-R1-A5-F7: MHA-to-GQA Conversion via Procrustes Analysis
- **Source:** Jin et al., ACL EMNLP 2025 ("Align Attention Heads Before Merging")
- **Summary:** Converts Multi-Head Attention to Grouped-Query Attention by applying Procrustes analysis to align heads before merging, then L0 regularization to prune redundant parameters. Compresses up to 87.5% of KV heads in LLaMA2-7B with acceptable degradation.
- **Memory Savings:** Up to 87.5% KV-cache reduction
- **Speed Impact:** Reduces memory traffic proportionally
- **Maturity:** Production-ready (EMNLP 2025)
- **Relevance:** Applicable to our hybrid SSM+attention layers. Could reduce the attention component's KV-cache from bottleneck to negligible.
- **Verdict:** PROMISING

#### v5-R1-A5-F8: EvolKV — Evolutionary KV-Cache Budget Optimization
- **Source:** Yu et al., EMNLP 2025 Findings
- **Summary:** Uses evolutionary search to find optimal per-layer KV-cache budgets. Discovers that different layers need vastly different cache sizes. Achieves better performance than FULL KV-cache on some tasks while using only 1.5% of original budget.
- **Memory Savings:** 98.5% KV-cache reduction (1.5% of full budget)
- **Speed Impact:** Positive (less cache to manage)
- **Maturity:** Research (EMNLP 2025)
- **Relevance:** Per-layer optimization could apply to SSM state size allocation — not all layers need the same state dimension.
- **Verdict:** PROMISING

#### v5-R1-A5-F9: ChunkKV — Semantic Chunk Compression
- **Source:** Liu et al., February 2025
- **Summary:** Treats semantic chunks rather than individual tokens as compression units. Preserves linguistic structure while improving throughput by 26% over token-level methods.
- **Memory Savings:** Moderate (chunk-level granularity)
- **Speed Impact:** +26% throughput vs token-level
- **Maturity:** Research (2025)
- **Relevance:** Could apply to SSM state compression — compress state at semantic boundaries rather than fixed intervals.
- **Verdict:** PROMISING

#### v5-R1-A5-F10: BitNet.cpp ELUT — Element-wise Lookup Table for Ternary
- **Source:** Wang et al., ACL 2025 (Microsoft)
- **Summary:** Three ternary inference kernels: I2_S (lossless 2-bit), TL2_0 (element-wise LUT at 2.00 bpw), TL1_0 (group=3, 1.67 bpw). ELUT addresses spatial inefficiencies of bit-wise methods. Achieves 6.25× speedup vs FP16, 2.32× vs generic low-bit baselines on CPU.
- **Memory Savings:** 8× vs FP16 (1.58 bpw effective)
- **Speed Impact:** 6.25× faster than FP16
- **Maturity:** Production-ready (Microsoft GitHub)
- **Relevance:** DIRECTLY applicable — ELUT technique is the reference implementation for ternary CPU inference. Must study for our production inference engine.
- **Verdict:** PROMISING

#### v5-R1-A5-F11: Hybrid SSM-Transformer Industry Convergence (2025-2026)
- **Source:** Multiple: Jamba (AI21), Bamba (Samsung+UIUC), Nemotron-H (NVIDIA), Zamba2 (2025-2026)
- **Summary:** Major labs converged on SSM-Attention hybrids replacing 92%+ of attention with Mamba2 blocks. Nemotron-H achieves 3× higher throughput than similar-size Transformers. SSM layers eliminate KV-cache entirely, attention layers provide precise retrieval for remaining 8%.
- **Memory Savings:** Major (no KV-cache for SSM layers = constant memory regardless of sequence length)
- **Speed Impact:** 3× throughput vs pure Transformer
- **Maturity:** Production-ready (NVIDIA, AI21, Samsung)
- **Relevance:** VALIDATES our hybrid SSM+attention architecture. Industry consensus in 2025-2026 confirms this is the right direction.
- **Verdict:** PROMISING

#### v5-R1-A5-F12: Gated Attention — NeurIPS 2025 Best Paper
- **Source:** Alibaba Qwen team, NeurIPS 2025 Best Paper
- **Summary:** Adds sigmoid gate per attention head after Scaled Dot-Product Attention. Introduces nonlinearity and sparsity, reduces "attention sink" phenomenon (excessive attention on special tokens). Improves training stability and context extension (32k→128k with less degradation).
- **Memory Savings:** Modest (reduces attention sinks)
- **Speed Impact:** Neutral (minimal gate overhead)
- **Maturity:** Production-ready (NeurIPS 2025)
- **Relevance:** Easy win for our hybrid attention layers. Sigmoid gate is a single-element multiply — trivial overhead for stability improvement.
- **Verdict:** PROMISING

#### v5-R1-A5-F13: Sub-1GB Mobile Best Practices (2026 Consensus)
- **Source:** "On-Device LLMs: State of the Union" — Meta AI Research, Vikas Chandra, 2026
- **Summary:** Industry consensus for <1GB mobile: (1) Memory bandwidth is bottleneck, NOT compute. (2) 4-bit quantization standard, 2-bit viable with QAT. (3) Sub-1B params sufficient for utility tasks. (4) Deep-thin architectures (more layers, smaller d_model) outperform wide-shallow below 1B. (5) Data quality matters more than param count at small scale.
- **Relevance:** CRITICAL guidance for v5 architecture sizing. "Deep-thin" confirms we should prioritize more layers over larger d_model. "Bandwidth bottleneck" confirms ternary's advantage (1.58 bits vs 4 bits = 2.5× less traffic).
- **Verdict:** PROMISING

#### v5-R1-A5-F14: KV-Cache 3-bit Quantization
- **Source:** 2025 research cited in "On-Device LLMs" 2026
- **Summary:** KV-cache can be quantized down to 3 bits with negligible quality drop. For long context (>8K tokens), KV-cache often exceeds model weights in total memory — making cache compression MORE impactful than further weight compression.
- **Memory Savings:** ~5× for KV-cache (3-bit vs 16-bit)
- **Relevance:** Complements our ternary weights. Even with ternary weights, attention layers' KV-cache can dominate memory. 3-bit cache keeps the entire pipeline ultra-low memory.
- **Verdict:** PROMISING

## Key Insights — Aspect 5, Wave 1

1. **Optimizer memory is solvable** — Q-Adam-mini (8×) and AdaPM (90% momentum reduction) combined could reduce optimizer state from 2× model size to <0.25× model size.
2. **TriTera validates ternary at scale** — 1.2T tokens trained, and crucially: ternary benefits more from DATA than PARAMS. This means our 50M-param ternary model should be trained on as much data as possible, not scaled up in params.
3. **KV-cache is the hidden memory killer** — For long context, KV-cache exceeds model weights. Our SSM-heavy architecture (no KV-cache for SSM layers) + 3-bit cache for remaining attention layers solves this.
4. **Industry converged on SSM-Attention hybrids** — Jamba, Bamba, Nemotron-H all use 92%+ SSM + 8% attention. Our architecture is aligned with industry consensus.
5. **Deep-thin wins for sub-1B** — More layers with smaller d_model outperforms wide-shallow. Our 48-layer design is validated.
6. **ELUT is the ternary inference standard** — BitNet.cpp's element-wise LUT achieves 6.25× vs FP16. Must study and potentially adopt.
7. **Activation compression via random ternary projections** — CompAct + our JL-ternary proof (A3-P3) = ternary activation compression during training.
8. **Memory bandwidth is THE bottleneck** — Not compute. Ternary's 10× less memory traffic is the primary speed advantage, not the add/subtract simplification.

## For Our Project

**IMMEDIATE WINS:**
- Q-Adam-mini for training memory reduction (8×)
- Gated Attention for stability (trivial to add)
- KV-cache 3-bit quantization for attention layers
- ELUT technique from BitNet.cpp for inference

**ARCHITECTURAL VALIDATION:**
- SSM-Attention hybrid: confirmed by industry (Jamba, Bamba, Nemotron-H)
- Deep-thin: confirmed for sub-1B models (Meta consensus)
- Ternary scaling: data > params (TriTera finding)
- Bandwidth-bound: ternary's 10× less traffic is the real win

**Open questions for math-paradigm-breakers:**
- What is the theoretical minimum optimizer state for ternary QAT?
- Can reversible ternary layers achieve zero-activation-storage backprop?
- What is the optimal depth-width tradeoff for ternary specifically (not just generic sub-1B)?
- How does ELUT interact with our SSM recurrence? Can SSM steps be LUT-accelerated?
