# Round 1, Aspect 6: Training Speed & Throughput for Ternary Models

## Wave 1 — Web Research (2 researchers, Feb 2026)

### GRADIENT COMPRESSION & COMMUNICATION

#### v5-R1-A6-F1: EDGC — Entropy-Driven Dynamic Gradient Compression
- **Source:** Yi et al., arXiv:2511.10333, Nov 2025
- **Summary:** Dynamically adjusts gradient compression rate during training based on evolving gradient entropy. Uses downsampling for efficient entropy estimation + window-based adjustment across pipeline stages.
- **Speedup:** 46.45% communication latency reduction, 16.13% training time reduction
- **Quality:** Zero degradation — preserves accuracy perfectly
- **Applicability:** HIGH — Ternary weights create inherently sparse gradients (66%+ zeros). Entropy-based adaptation tracks ternary quantization patterns per-layer.
- **Verdict:** PROMISING

#### v5-R1-A6-F2: TAGC — Transformer-Aware Gradient Compression
- **Source:** Polyakov et al., arXiv:2504.05638, Apr 2025 (EuroMLSys '25)
- **Summary:** Lossless homomorphic compression for sharded models (FSDP/ZeRO). Selective compression of non-attention linear layers. Uses Reduce instead of All-Reduce (2× communication savings).
- **Speedup:** 15% training acceleration, 32.1% communication time reduction, 6.15× data transfer reduction
- **Quality:** 3.6% loss increase at 10× compression (98.75% sparsification)
- **Applicability:** VERY HIGH — SSM layers vs routing layers can be selectively compressed. Ternary creates natural 66%+ gradient sparsity.
- **Verdict:** PROMISING

#### v5-R1-A6-F3: PacTrain — Pruning + Adaptive Sparse Gradient Compression
- **Source:** Wang et al., arXiv:2505.18563, May 2025
- **Summary:** Combines neural network pruning with sparse gradient compression. Global gradient sparsity awareness across workers. Compatible with all-reduce primitive.
- **Speedup:** 1.25-8.72× throughput under bandwidth-constrained conditions
- **Quality:** Near-optimal compression preserves accuracy
- **Applicability:** MEDIUM — Ternary already quantized; pruning on top might over-constrain. Gradient sparsity awareness aligns with STE patterns.
- **Verdict:** NEEDS-MORE-WORK

### LEARNING RATE & OPTIMIZER INNOVATIONS

#### v5-R1-A6-F4: WSM — Warmup-Free Learning Rate Schedule
- **Source:** Tian et al., arXiv:2507.17634, Jul 2025
- **Summary:** Eliminates LR decay phase via checkpoint merging. Maintains constant LR after warmup, enabling faster convergence without traditional decay.
- **Speedup:** Saves 10-20% of total training steps (eliminates decay phase entirely)
- **Quality:** No degradation
- **Applicability:** HIGH — Ternary training already uses simplified schedules. Constant LR after warmup could simplify pipeline significantly.
- **Verdict:** PROMISING

#### v5-R1-A6-F5: Muon Optimizer for Pretraining
- **Source:** Shah et al., arXiv:2505.02222, May 2025
- **Summary:** Novel optimizer designed for pretraining efficiency. Uses linear warmup + cosine decay. Designed for large-scale LM pretraining.
- **Speedup:** Faster convergence than Adam/AdamW at equivalent quality
- **Quality:** Maintains or improves quality
- **Applicability:** MEDIUM — May interact poorly with STE gradients (STE introduces bias). Needs testing with quantized training.
- **Verdict:** NEEDS-MORE-WORK

#### v5-R1-A6-F6: Critical Batch Size Scaling Laws
- **Source:** Bergsma et al., arXiv:2505.13738, May 2025
- **Summary:** Defines B_crit as point where training requires 2× minimum data. Provides scaling laws for batch size + weight decay in LLM pre-training.
- **Speedup:** Enables optimal batch size selection to maximize throughput without quality loss
- **Quality:** Maintained at optimal batch size
- **Applicability:** MEDIUM — Ternary models smaller, may hit memory before compute limits. Helps optimize current batch=64 setting.
- **Verdict:** NEEDS-MORE-WORK

### KERNEL OPTIMIZATION

#### v5-R1-A6-F7: Hardware-Aware Kernel Fusion
- **Source:** Multiple (Snider 2023, Huang Dec 2025, Zheng 2020, FlashFuser)
- **Summary:** Merges multiple kernels into single execution units optimized for memory hierarchy. Reduces off-chip traffic, launch overhead, increases arithmetic intensity. FlashFuser extends to patterns exceeding shared memory.
- **Speedup:** 2.3-10.56× for specific workloads, 4.1× on H100 (FlashFuser)
- **Quality:** No degradation — lossless optimization
- **Applicability:** MEDIUM-HIGH — SSM sequential scan is Python-loop bound (30-50% GPU util). Fusing scan + ternary quantization + router into single kernel could help. SSM sequential dependency limits parallelism.
- **Verdict:** PROMISING (with caveats — SSM-specific fusion needed)

### DATA EFFICIENCY

#### v5-R1-A6-F8: Curriculum Learning & Data Pipeline Optimization
- **Source:** BabyLM Team, arXiv:2602.20092, Feb 2026
- **Summary:** Optimized data mixture selection + tokenization strategies + curriculum learning for small models. Byte Premium adjustment for multilingual balancing.
- **Speedup:** 10-30% reduction in total training tokens needed via curriculum (easy→hard)
- **Quality:** Improves quality per token processed
- **Applicability:** MEDIUM-HIGH — Currently using simple data ordering. Curriculum learning could reduce total tokens to convergence. Especially relevant given M7's 150:1 data/param ratio for ternary.
- **Verdict:** PROMISING

### STE ALTERNATIVES & GRADIENT ESTIMATORS

#### v5-R1-A6-F9: FOGZO — First-Order-Guided Zeroth-Order Training
- **Source:** NeurIPS 2025 poster
- **Summary:** Combines STE (biased but fast) with zeroth-order estimates (unbiased but slow). Reduces STE bias while maintaining speed. 796× faster than pure ZO methods.
- **Speedup:** 796× faster than pure ZO (n-SPSA); 1-8% accuracy improvement over pure STE at same iterations
- **Quality:** Better than STE alone (1-22 PPL improvement on LLaMA-0.3B)
- **Applicability:** HIGH — Ternary training suffers from STE bias. FOGZO directly addresses this. Proven on LLaMA quantization-aware pretraining.
- **Verdict:** PROMISING

#### v5-R1-A6-F10: QZO-Scale — Quantized ZO via Scale Perturbation
- **Source:** ICLR 2026 poster
- **Summary:** Perturbs quantization SCALE instead of weights for ZO training on quantized models. Avoids de-quant/re-quant cycles entirely. Different from Phase 10.E.1's QZO (which perturbed weights directly and failed).
- **Speedup:** 18× memory reduction vs FP16 fine-tuning; enables 13B on single 24GB GPU
- **Quality:** Comparable to FP16 fine-tuning
- **Applicability:** HIGH — Our Phase 10.E.1 tested weight-perturbation QZO (FAILED +110%). Scale-perturbation is fundamentally different — perturbs the continuous scale parameter, not the discrete ternary weights.
- **Verdict:** PROMISING (distinct from previously failed QZO)

### SPARSITY & STRUCTURE

#### v5-R1-A6-F11: SRigL — Structured N:M Sparsity Training
- **Source:** ICLR 2024, OpenReview
- **Summary:** Dynamic sparse training maintaining constant fan-in (N:M) + neuron ablation. Hardware-friendly structure enables real-world speedups.
- **Speedup:** 3.4× CPU (online), 1.7-13× GPU (batched) at 90% sparsity
- **Quality:** Matches dense and unstructured sparse accuracy
- **Applicability:** HIGH — Ternary already has ~33% zeros (unstructured). SRigL enforces STRUCTURED pattern → SIMD exploitable. Combination: ternary values + structured sparsity = extreme efficiency.
- **Verdict:** PROMISING

### ARCHITECTURE-SPECIFIC

#### v5-R1-A6-F12: Phased Specialization for SSM-Attention Hybrids
- **Source:** Medium article "Phased Specialization" Jan 2026
- **Summary:** Differential learning rates for heterogeneous modules (Mamba vs Transformer). Attention-lead phase then SSM-specialization phase.
- **Speedup:** 34-point PPL improvement (191.5→157.5) over joint training — not throughput, but convergence quality
- **Quality:** 6% better than baseline (vs 14% worse with naive joint training)
- **Applicability:** MEDIUM — Our architecture is mostly SSM, but has some SWA layers. Phased LR per module type could help.
- **Verdict:** NEEDS-MORE-WORK

#### v5-R1-A6-F13: ZO Subspace Gradient Projection
- **Source:** arXiv 2602.17155, Feb 2026
- **Summary:** Projects ZO gradient estimates onto low-dimensional subspaces to reduce variance and computation. Makes ZO training practical for larger models.
- **Speedup:** Not directly quantified, but enables larger-scale ZO training
- **Quality:** Matches first-order methods in some settings
- **Applicability:** MEDIUM — Could combine with QZO-Scale (F10) or FOGZO (F9) for efficient gradient-free ternary training.
- **Verdict:** NEEDS-MORE-WORK

---

## Key Insights — Aspect 6 Wave 1

1. **Communication is the primary distributed training bottleneck** — 47-96% of training time. Ternary's natural gradient sparsity (66%+) is a free advantage for gradient compression.
2. **STE alternatives are maturing** — FOGZO combines best of STE (speed) + ZO (unbiased). Could improve our convergence without changing architecture.
3. **Structured sparsity + ternary is unexplored** — SRigL's N:M patterns could combine with ternary zeros for hardware-exploitable sparsity.
4. **Kernel optimization lags for training** — BitNet.cpp/TABv2 focus on inference. Training kernels (fused scan + quantize + route) are the gap.
5. **Curriculum learning reduces token cost** — With ternary's 150:1 data/param ratio, even 10-30% token reduction is significant.
6. **Scale-perturbation ZO is fundamentally different from weight-perturbation ZO** — Our Phase 10.E.1 QZO failure was weight-perturbation. Scale-perturbation bypasses the discrete weight problem entirely.

## For Our Project

**Immediate wins:**
- F9 (FOGZO): ~50 lines of code change, directly improves STE. Test at 6M first.
- F4 (WSM): Simplifies training schedule. Easy to test.
- F11 (SRigL): Structured sparsity on ternary could unlock hardware speedups.

**Medium-term:**
- F1/F2 (EDGC/TAGC): Gradient compression for distributed training (when we scale beyond single GPU)
- F10 (QZO-Scale): If FOGZO works, QZO-Scale is the next step toward gradient-free training
- F8 (Curriculum): Reduce the 150:1 token burden via smart data ordering
