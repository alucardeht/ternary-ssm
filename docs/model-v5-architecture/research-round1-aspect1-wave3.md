# Round 1, Aspect 1: Extreme Sparse Activation — Wave 3 (Paradigm-Breakers)

## PB1: Radical Alternatives Beyond Neural Networks

### v5-R1-A1-P1: RATR — Retrieval-Augmented Tiny Reasoner
- **Paradigm:** Factor model into small REASONING engine (50-100 MB RAM) + large KNOWLEDGE store (NVMe)
- **How:** SSM reasoning core (30-60M params, 60-120 MB) outputs HIGH ENTROPY when it needs external knowledge. Knowledge index (2-10 GB on NVMe, 50-100 MB hot index in RAM) provides factual associations.
- **RAM:** ~270 MB total (120 MB reasoner + 100 MB hot index + 50 MB integration)
- **Quality:** 40-60% of Haiku (knowledge retrieval is bottleneck)
- **Throughput:** 100-300 tok/s (retrieval limited), up to 10K+ tok/s for fluent generation without retrieval
- **Key insight:** Shannon separation theorem — source coding (knowledge) and channel coding (reasoning) can be optimized independently
- **Biggest risk:** The reasoning-knowledge interface. Poor gating → oscillation between incoherent reasoning and disconnected facts.
- **Viability:** HIGH — most practical path, reuses existing ternary SSM

### v5-R1-A1-P2: Computation-as-Lookup (Product Quantized Inference)
- **Paradigm:** Replace ALL matrix-vector computation with table lookups. Zero arithmetic at inference.
- **How:** Split d=512 input into G=64 groups of 8 dims. Each group has lookup table with 3^8=6561 entries. Inference = G table lookups + concatenation per layer.
- **RAM:** ~164 MB (24 layers × 64 groups × 6561 entries × 8 output × 2 bits)
- **Quality:** 30-50% of original (independence assumption between groups is very strong)
- **Throughput:** 10,000+ tok/s CPU (pure memory reads, zero arithmetic!)
- **Biggest risk:** Cross-group interactions carry critical information. PQ destroys this.
- **Viability:** MEDIUM — extraordinary speed but uncertain quality

### v5-R1-A1-P3: Dendritic Trees (Decision Tree Neurons)
- **Paradigm:** Replace standard neurons (weighted sum) with binary decision trees. Depth-8 tree = 6561 leaves = expressive as 256-1024 unit MLP layer.
- **How:** Each "neuron" is a depth-8 ternary decision tree. 512 neurons × 20 layers. Training via distillation from neural teacher.
- **RAM:** ~95 MB (10,240 neurons × 5 KB + embeddings)
- **Quality:** 30-60% of teacher (depends on distillation quality)
- **Throughput:** 5,000-12,000 tok/s CPU (branch operations = fastest CPU ops, ~8 ns per tree)
- **Biggest risk:** Trainability. Decision trees are not differentiable.
- **Viability:** MEDIUM — extraordinary speed if distillation works

### v5-R1-A1-P4: HDC Ternary Composition Module
- **Paradigm:** Hyperdimensional Computing with ternary vectors. BIND (element-wise multiply), BUNDLE (majority vote), PERMUTE (shift).
- **As standalone:** INFEASIBLE — capacity bound requires D proportional to N×log(N), language needs millions of patterns → impractical dimensions
- **As component:** D=4096 codebook + neural prediction head = ~48 MB
- **Benefit:** Provides algebraically EXACT positional encoding, LOSSLESS composition up to capacity bound
- **Quality:** +5-10% on compositional reasoning tasks (when combined with neural base)
- **Viability:** LOW standalone, MEDIUM as composition module in hybrid

### v5-R1-A1-P5: Sparse Distributed Memory with Ternary Addressing
- **Paradigm:** Content-addressable memory with high-dimensional ternary addresses
- **RAM:** ~305 MB (compressed variant with LSH acceleration)
- **Problem:** 10B ternary comparisons per read → needs LSH for feasibility
- **Benefit:** Online learning (can update knowledge at inference time, no gradient descent)
- **Viability:** LOW standalone, MEDIUM as knowledge store component (combines with P1)

### v5-R1-A1-P6: Frequency-Domain NTT Processing
- **Paradigm:** NTT over GF(3) keeps everything ternary. Convolution → element-wise multiplication.
- **RAM:** 50-80 MB (extremely small but limited expressiveness)
- **Problem:** Context length restricted to powers of 3 (3, 9, 27... 729, 2187)
- **Viability:** LOW standalone, MEDIUM as attention replacement

## PB2: Concrete Architecture Proposals

### v5-R1-A1-P7: Architecture Proposals — Comparative Analysis

**The Bandwidth-Intelligence Bound (fundamental constraint):**
```
N_active_max = Memory_Bandwidth / bits_per_weight / target_tok_s
CPU DDR5 @ 500 tok/s: 410M active ternary params → MMLU ~30-40%
GPU GDDR7 @ 500 tok/s: 5.4B active ternary params → MMLU ~60-70%
Haiku quality needs ~7B active → CPU @ 7B: only ~30 tok/s
```

**POPCOUNT breakthrough:** INT8 activation quantization + bit-serial ternary POPCOUNT reduces effective memory reads by ~4x. This is THE key enabler for CPU throughput.

**Four proposals evaluated:**

| Proposal | Total Params | Working RAM | CPU tok/s (POPCOUNT) | GPU tok/s | MMLU est. | Training Cost |
|----------|-------------|-------------|---------------------|-----------|-----------|---------------|
| A: Dense Tower | 2.85B | 793 MB | 285 | 980 | 48-55% | ~$0 (local) |
| B: SSM-MoE Stream | 19.9B | 720 MB | 918 | 3000 | 50-58% | $500-1500 |
| C: Mega-MoE | 42.4B | 555 MB | 568 | 1450 | 63-70% | $3K-8K |
| D: Kronecker-MoE | 28B | 513 MB | 800 | 2630 | 50-58% | $1K-4K |

### v5-R1-A1-P8: Proposal A — Dense Ternary Tower (Baseline)
- **Spec:** d=3072, 54 layers (48 SSM + 6 SWA), 2.85B params, 100% dense
- **RAM:** 793 MB (688 MB ternary weights + 60 MB FP16 + 19 MB SSM state + 26 MB buffers)
- **Quality:** Phi-2-class (MMLU 48-55%), NOT Haiku
- **Speed:** 285 tok/s CPU+POPCOUNT, 980 tok/s GPU
- **Training:** Fits SINGLE RTX 5070 12 GB with activation checkpointing. Cost: ~$0
- **Value:** Baseline validation, inference engine development, zero cost
- **Verdict:** PROMISING as stepping stone, insufficient quality for target

### v5-R1-A1-P9: Proposal C — Mega-MoE Deep Tower (Quality Maximizer)
- **Spec:** d=2048, 48 layers, 256 effective experts (4 sub-proj × 64 experts, top-2), 42.4B total params
- **Active per token:** 1.84B params, N_eff ~7.36B → 6.84B FP16-equiv
- **RAM:** 555 MB working set (142 MB shared + 360 MB hot cache + 53 MB state/buffers)
- **Full model mmap:** 10.6 GB in DRAM (fits 32 GB DDR5)
- **Quality:** MMLU 63-70%, potentially 70-75% with Phi-3-quality data. CLOSEST TO HAIKU.
- **Speed:** 568 tok/s CPU+POPCOUNT, 1450 tok/s GPU
- **Training:** 8× A100-80GB, $3K-8K
- **Risk:** Fine-grained expert routing at 256 experts, potential expert collapse
- **Verdict:** VERY PROMISING — best quality-RAM tradeoff discovered

### v5-R1-A1-P10: Proposal D — Kronecker-MoE Hybrid (Compression Pioneer)
- **Spec:** d=2048, 32 layers, shared backbone = Kronecker K=32 ternary (36.4× compression), experts = standard ternary, 28B total
- **Key insight:** Kronecker backbone reads only 2.1 MB per token (vs 218 MB for experts). Bandwidth spent on experts, not backbone.
- **RAM:** 513 MB working set (smallest of all proposals)
- **Quality:** MMLU 50-58% (Kronecker quality loss is the wildcard)
- **Speed:** 800 tok/s CPU+POPCOUNT, 2630 tok/s GPU
- **Risk:** Kronecker ternary training stability unknown. MUST validate at small scale first.
- **Verdict:** PROMISING if Kronecker validates, RISKY if it doesn't

## Key Insights — Aspect 1 Wave 3

1. **Honest truth: Haiku quality in sub-1GB CPU is EXTREMELY hard.** The bandwidth-intelligence bound limits CPU (DDR5) to ~410M active ternary params at 500 tok/s → MMLU 30-40%. POPCOUNT (4×) pushes to ~1.6B active → MMLU 55-65%.
2. **Mega-MoE (Proposal C) is the most credible path** — 42.4B total, 555 MB working set, MMLU 63-70%. With excellent data, could approach 75%.
3. **POPCOUNT is the secret weapon** — 4× effective bandwidth = 4× more active params within same tok/s budget. This is NOT well-known in the literature.
4. **RATR (knowledge factoring) is the most practical radical alternative** — separate reasoning from knowledge, only keep reasoning in RAM.
5. **Computation-as-Lookup is the speed king** — 10K+ tok/s by eliminating arithmetic entirely. But quality is uncertain.
6. **Kronecker ternary is the compression dark horse** — 36× storage compression, only 2.1 MB backbone reads. Needs small-scale validation BEFORE investing.
7. **Critical validation experiments (2 hours, $0):**
   - Kronecker approximation quality of existing trained weights
   - Expert routing statistics (hot/cold split validation)
   - POPCOUNT throughput measurement on existing model

## Recommended Implementation Path

**Phase 1 (Local, $0):** Build Dense Tower (Proposal A) on RTX 5070. Validate inference engine, POPCOUNT, training pipeline.

**Phase 2 (Cloud, $3K-8K):** Build Mega-MoE (Proposal C) on 8× A100. Push toward Haiku quality.

**Phase 3 (Conditional):** If Kronecker validates at small scale → build Kronecker-MoE hybrid for smallest possible RAM.

**Before anything:** Run 3 critical experiments (~2 hours, $0) on existing 6M model to validate core assumptions.
