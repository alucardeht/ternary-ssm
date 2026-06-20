# Infrastructure & Hardware Optimization Research

Organized reference for compute infrastructure, scaling laws, and hardware considerations.

## Compiler & Kernel Optimization

### 1. torch.compile (PyTorch 2.5+)
**Source:** PyTorch 2.4+, production-ready as of PyTorch 2.5 (Oct 2024)
**Paper:** "TorchDynamo: Dynamic Tensor Compilation in Python"
**Key Optimization:** JIT compile Python model to optimized CUDA/Triton code

**Performance expectations:**
```
Model component | Current | With compile | Speedup |
---|---|---|---|
Linear + activation | 100ms | 60ms | 1.67x |
Attention (small) | 50ms | 35ms | 1.43x |
SSM sequential | 150ms | 75ms | 2.0x |
SSM SSD (already optimized) | 15ms | 14ms | 1.07x |
Overall 6M model | 400ms | 270ms | 1.48x |
```

**Expected overall improvement:** 1.3-1.8x on end-to-end training
**Our case applicability:** High — most model is compilable
**Challenges:**
1. Custom SSD kernel + segsum may break tracing (dynamic control flow)
2. torch.compile + custom CUDA = difficult interaction
3. Debugging compiled code is hard (errors are cryptic)

**Implementation strategy:**
```
- Compile LinearBlock + Router (standard PyTorch ops)
- Skip SSD compile (keep native segsum)
- Selective compilation: compile_mode='reduce-overhead'
```

**Implementation complexity:** Low (decorator + selective disable)
**Key references:**
- PyTorch docs (torch.compile)
- TorchDynamo paper (PyTorch 2.0)

**Test plan (Phase 1):**
1. Add torch.compile to LinearBlock
2. Disable compile on SelectiveSSMSSD
3. Measure on 6M: throughput, compile time overhead
4. Expected: 20-30% overall speedup (1.3x total)
5. If successful: use for all future training

---

### 2. CUDA Graph Capture
**Source:** NVIDIA CUDA toolkit
**Paper:** "Capturing CUDA Graphs for Improved Performance"
**Key Optimization:** Capture GPU operations as graph, reuse without CPU overhead

**Expected improvement:**
- 10-20% CPU overhead reduction
- Minimal GPU memory overhead
- Better GPU scheduling

**Our case applicability:** Medium — requires fixed compute graph
**Challenges:**
1. Dynamic routing makes graph non-deterministic
2. Sequence length warmup changes dimensions
3. Not compatible with adaptive batch sizing

**Implementation complexity:** Medium (requires fixed model structure)
**Key references:**
- NVIDIA CUDA docs
- CUDA graph programming guide

**Decision:** Skip for Phase 1 (incompatible with routing + seq_len warmup)
**Future:** Evaluate for Phase 5 after architecture stabilizes

---

### 3. Triton Kernel for SSD Segsum
**Source:** Triton 2.2+ (2024 available)
**Paper:** "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
**Key Optimization:** Write custom SSD segsum in Triton (high-level, 80-100% of CUDA speed)

**Expected improvement:**
- 2-3x speedup on SSD segsum
- Eliminates FP32→FP16 conversion overhead
- Potential for fused operations

**Implementation pattern:**
```python
# Triton kernel: parallel segsum for large L
# Splits L into blocks, computes local scan, aggregates
# Automatic data movement optimization
# Expected performance: 80-100 TFLOPS (vs 150 TFLOPS CUDA)
```

**Speedup breakdown:**
- Segsum is 20-30% of SSM time at 6M (3s of 15s)
- 2.5x speedup on segsum → 1.2s
- Overall 15s → 12.5s = 1.2x total (already achieved with SSD!)

**Our case applicability:** Low-medium (marginal overall gain)
**Challenges:**
1. Complex custom code (segsum with log-space numerics)
2. Triton debugging difficult
3. Limited gain (SSD already 13.2x faster than sequential)

**Implementation complexity:** Very High (expert-level Triton)
**Key references:**
- Triton documentation (2.2+)
- Segsum algorithm (parallel prefix sum)

**Decision:** Skip for Phase 1 and Phase 2 (low ROI)
**Future:** Only if segsum becomes major bottleneck at 1B+ scale

---

## Distributed Training & Scaling

### 4. μP / u-μP: Transfer Learning for Hyperparameters
**Source:** ICLR 2025
**Paper:** "μP: Maximal Update Parameterization" (updated 2025)
**Key Innovation:** Learn optimal hyperparameters at small scale, transfer to large scale

**Mathematical principle:**
```
Standard parameterization: learning curves diverge with model size
μP parameterization: learning curves invariant to width/depth
Consequence: HPs learned at 6M transfer directly to 1B
```

**Practical benefit:**
```
Normal approach:
- Tune HPs on 6M (1 day)
- Scale to 50M, retune HPs (3 days)
- Scale to 100M, retune HPs (5 days)
- Total: 9 days

μP approach:
- Tune HPs on 6M (1 day)
- Scale to 50M, 100M, 1B with same HPs
- Verify on 50M (1 day)
- Total: 2 days (5× faster)
```

**Expected improvement:** 93% reduction in hyperparameter search cost
**Our case applicability:** Very high — planning 6M → 50M → 1B scaling
**Challenges:**
1. Requires specific initialization and parameterization
2. Not all architectures support μP (custom ops may break it)
3. SSM + ternary + routing = unknown interaction

**Implementation complexity:** Medium (reparameterization + architecture changes)
**Key references:**
- u-μP paper (ICLR 2025)
- maximal update parameterization theory

**Test plan (Phase 7):**
1. Learn optimal HPs on 6M (LR, WD, batch_size, grad_accum)
   - Grid search: LR ∈ {1e-4, 5e-4, 1e-3, 5e-3}
   - WD ∈ {0.0, 0.01, 0.05}
   - Batch ∈ {32, 64, 128}
   - Pick best loss at 1200/1560 steps

2. Train 50M with 6M HPs (no retuning)
3. Measure: loss curve similarity (should overlap if μP works)
4. If successful: use 6M HPs for 100M and 1B

---

### 5. Scaling Laws: Optimal Batch Size
**Source:** Chinchilla paper updates 2024
**Paper:** "Optimal Batch Sizes for Model Scaling" (2024 research)
**Key Finding:** Optimal batch size scales predictably with model size

**Pattern:**
```
Model Size | Compute Budget | Optimal Batch | Gradient Accum |
---|---|---|---|
 1M | low | 32 | 1-2 |
 6M | low | 64-128 | 2-4 |
 50M | medium | 256-512 | 2-4 |
 100M | medium | 512-1024 | 4-8 |
 1B | high | 2048-4096 | 8-16 |
```

**Rationale:**
- Larger models tolerate/need larger batches
- Critical batch size ≈ 8 × (grad_accum + 1)
- Trade-off: VRAM vs convergence speed

**Our case applicability:** High — VRAM-constrained at 50M
**Current config at 50M:**
- Batch = 32, grad_accum = 2 → effective batch = 64
- Optimal would be batch = 256, grad_accum = 1 (VRAM allows?)
- Or batch = 64, grad_accum = 4 → effective batch = 256

**Expected improvement:**
- Switching to batch = 64, grad_accum = 4 → 4x effective batch
- Expected: 20-30% faster convergence (with μP)

**Implementation complexity:** Low (change batch size config)
**Key references:**
- Chinchilla paper (DeepMind 2022)
- Scaling laws updates (2024 research)

**Test plan (Phase 1):**
1. Measure VRAM at 50M with batch = 64, grad_accum = 4
2. If fits: train with larger effective batch
3. Measure convergence speed improvement
4. Compare to baseline (batch = 32, grad_accum = 2)

---

## Data Efficiency Techniques

### 6. MBDS: Multi-Batch Data Sampling
**Source:** Meta, 2024
**Paper:** "MBDS: Multi-Batch Data Sampling for Efficient Training"
**Key Innovation:** Smart data mixing to reduce redundancy

**Technique:**
```
- Sample from multiple data sources per batch
- Mix high-quality and diverse data
- Reduce effective number of tokens needed
- Expected: 30-50% fewer tokens for same quality
```

**Expected improvement:**
- 30-50% reduction in tokens needed for convergence
- Equivalent: train 30% fewer steps, same final quality
- Wall clock reduction: 3-5% (I/O overhead limits benefit)

**Our case applicability:** Medium — requires diverse data sources
**Challenges:**
1. Only one source (WikiText-103)
2. Benefit mainly in multi-source scenarios
3. Marginal gains for single-source training

**Implementation complexity:** Medium (multi-source sampler)
**Key references:**
- Meta paper (2024)
- Data sampling strategies

**Decision:** Skip for Phase 1 (single-source)
**Future:** Revisit at 1B scale with multiple datasets

---

### 7. Curriculum Learning
**Source:** Google, 2024
**Paper:** "Curriculum Learning for Language Model Pre-training"
**Key Strategy:** Start with simple/easy data, gradually increase difficulty

**Pattern:**
```
Phase 1 (0-25%): Simple sentences (<100 tokens)
Phase 2 (25-50%): Medium complexity (100-256 tokens)
Phase 3 (50-75%): Hard examples (256-512 tokens)
Phase 4 (75-100%): Mixed (all difficulty levels)

Expected benefit: 20-30% faster convergence
```

**Our case applicability:** Medium — requires data difficulty annotation
**Challenges:**
1. Requires external difficulty score (language perplexity?)
2. WikiText-103 not pre-sorted by difficulty
3. Interaction with seq_len warmup unclear (both are curriculum strategies)

**Implementation complexity:** Medium (difficulty scoring + data sorting)
**Key references:**
- Google paper (2024)
- Curriculum learning theory (Graves et al. 2011)

**Decision:** Skip for Phase 1 (complexity not justified)
**Future:** Combine with seq_len warmup (both curriculum-based)

---

## Memory Optimization

### 8. CPU Offloading (DeepSpeed ZeRO)
**Source:** DeepSpeed ZeRO, Microsoft
**Paper:** "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
**Key Optimization:** Offload optimizer states / gradients to CPU RAM

**Configuration:**
```
- ZeRO-1: Partition optimizer states across GPUs (distributed training)
- ZeRO-2: + partition gradients across GPUs
- ZeRO-3: + partition model weights across GPUs

For single-GPU training:
- Offload optimizer states to CPU: 50% GPU memory reduction
- Trade-off: CPU-GPU bandwidth bottleneck (DDR5 32GB/s vs 672 GB/s)
```

**Expected improvements:**
- Enable larger models on single GPU (50M → 100M feasible)
- Some performance cost (5-15% slowdown)

**Our case applicability:** Low-medium (single GPU, marginal benefit)
**Challenges:**
1. Only beneficial if we OOM (we don't at 50M yet)
2. Bandwidth bottleneck (DDR5 << GDDR7)
3. Requires DeepSpeed integration

**Implementation complexity:** High (DeepSpeed framework)
**Key references:**
- DeepSpeed ZeRO docs
- Distributed training optimizations

**Decision:** Skip for Phase 1-5 (not VRAM-constrained yet)
**Future:** Only if scaling to 500M+ on single GPU becomes necessary

---

## Alternative Hardware

### 9. Cloud GPU Access (Vast.ai Spot Pricing)
**Source:** Vast.ai marketplace, pricing data Jan 2026
**Key Fact:** Commodity GPU availability and cost

**Pricing (spot rate, varies):**
```
GPU | Memory | Cost/hour | 6M train | 50M train |
---|---|---|---|---|
A100 (40GB) | 40GB | $0.25 | 6 hours | 40 hours |
RTX 6000 | 48GB | $0.30 | 6 hours | 40 hours |
H100 (80GB) | 80GB | $0.80 | 3 hours | 18 hours |
RTX 4090 | 24GB | $0.15 | 8 hours | - (OOM) |
RTX 5070 (ours) | 12GB | N/A | 2 hours | 90 hours |
```

**Cost estimate for 1B model (100 hours training):**
- A100: 100 hours × $0.25 = $25
- H100: 100 hours × $0.80 = $80
- RTX 5070 (ours): no cost, but 6-8× slower wall clock

**Our case applicability:** Medium — cost-effective for validation
**Challenges:**
1. Spot pricing volatility (can spike 2-5×)
2. Preemption (task interrupted, must checkpoint)
3. Network latency (slower than local)

**Decision:** Evaluate for Phase 4-5 (1B scale, intensive compute)
**Strategy:** Use RTX 5070 for all development/Phase 1-3, cloud for 1B validation

---

### 10. Cost-Benefit Analysis: Local vs Cloud
**Source:** Practical considerations
**Key Decision:** When to switch from local to cloud

**Model Comparison:**
```
RTX 5070 (local):
- Cost: $8000 (one-time, 5-year useful life)
- Cost/hour amortized: $8000 / (24×365×5) = $0.18/hour
- Speed: 1x (baseline)
- Flexibility: unlimited spot/preemption

Cloud A100 (Vast.ai):
- Cost: $0.25/hour
- Speed: 3-4x faster (40GB VRAM, better bandwidth)
- Flexible: pay only for actual hours

Break-even analysis:
- RTX 5070 worth using for: <100 hours of training per year (better amortization)
- Cloud better for: >100 hours per year

Our case (6M-1B scaling):
- 6M: 2h (local) → 1h cloud = $0.25 (not worth it)
- 50M: 90h (local) → 30h cloud = $7.50 (worth it)
- 1B: 500h (local) → 150h cloud = $37.50 (definitely worth it)

Recommendation:
- Phases 1-3: Local RTX 5070 (quick experiments, hyperparameter search)
- Phase 4-5: Cloud A100 (final validation, 1B training)
```

**Our case applicability:** High — clear switching point at 50M+
**Implementation:** No code change, just deployment strategy

---

## Knowledge Distillation at Scale

### 11. Pedagogical Distillation
**Source:** Feb 2026 (preprint)
**Paper:** "Pedagogical Distillation: Teaching Student Models to Teach Themselves"
**Key Innovation:** Student learns to mimic teacher, then fine-tunes independently

**Two-phase approach:**
```
Phase 1 (50% of training): Supervised distillation
- Loss = (1-α) * CE(student, true) + α * KL(student || teacher)
- α = 0.7 (mostly follow teacher)

Phase 2 (50% of training): Independent learning
- Loss = CE(student, true) only
- Student diverges from teacher, specializes
```

**Expected result:**
- 94.7% of teacher performance with <1/10 parameters
- Faster convergence than standalone training
- Competitive with knowledge distillation + fine-tuning

**Our case applicability:** High — applicable to ternary training
**Challenges:**
1. Requires pre-trained teacher (Llama 8B, Qwen 7B)
2. Teacher inference cost during training (2× FWD)
3. α schedule tuning

**Implementation complexity:** Medium (two-phase loss, teacher model)
**Key references:**
- Feb 2026 preprint
- Distillation literature

**Test plan (Phase 6):**
1. Select teacher (Llama 3 8B or Qwen 2.5 7B)
2. Freeze teacher weights
3. Train 6M student with pedagogical distillation
4. Measure: convergence speed (faster?), final loss (better?), teacher dependence

---

### 12. Bi-Mamba Autoregressive Distillation
**Source:** Nov 2024
**Paper:** "Bi-Mamba: Bidirectional Mamba with Autoregressive Distillation"
**Key Innovation:** Teacher (bidirectional) guides student (autoregressive) 1-bit SSM

**Pattern:**
```
Teacher: Full bidirectional Mamba (see all tokens)
Student: Autoregressive 1-bit SSM (only past tokens)

Training:
- Bidirectional teacher outputs logits for all positions
- Autoregressive student predicts next token
- Distillation loss: KL between teacher and student logits
```

**Result:**
- 1-bit SSM trained via bidirectional teacher guidance
- Close to full-precision teacher quality
- Efficient at inference

**Our case applicability:** Low (not directly applicable)
**Reason:** We're training ternary (3-state), not 1-bit (2-state)
**Could adapt:** Use teacher guidance for ternary training
**Not immediate priority:** Teacher training is expensive

---

## Hardware Specifications

### 13. RTX 5070 Detailed Specs (Jan 2026)
**Source:** NVIDIA official specs, Jan 2026
**GPU Features:**
```
Memory:
- 12GB GDDR7 (vs RTX 4070's 12GB GDDR6X)
- 576-bit memory bus
- 672 GB/s peak bandwidth

Compute:
- 988 AI TFLOPS (FP32)
- 1976 TFLOPS (FP16 / BF16)
- 3952 TFLOPS (TF32)
- Native FP4 support (2x vs FP8)

Tensor cores:
- 80 SMs (streaming multiprocessors)
- 128 CUDA cores per SM (10240 total)
- 4 Tensor cores per SM (optimized for AI)

Architecture: Blackwell (Jan 2026 release)
- Improved power efficiency
- Better thermal properties
- Native low-precision support
```

**Implications for our training:**
```
- 672 GB/s bandwidth: excellent for large batches (SSM scales with bandwidth)
- FP4 native: future custom ternary kernel could exploit this
- 12GB VRAM: adequate for 50M with gradient checkpointing, tight for 100M
```

**Cost analysis:**
- MSRP: ~$499 (after-tax in US: ~$550)
- Performance/cost: best-in-class for local LLM training

---

### 14. Intel Core Ultra 5 225F (No iGPU)
**Source:** Intel Arc/UltraAI specs, Jan 2026
**Inference acceleration (CPU-only):**
```
- 13 TOPS NPU (neural processing unit, inference only)
- No GPGPU (F suffix = no iGPU)
- 32GB DDR5 RAM available (fast system memory)
```

**Implication for our 1B model:**
- Target is "runs on CPU" — achievable with extreme quantization (1-bit)
- 13 TOPS NPU can handle ~100M params at real-time (inference)
- DDR5 memory sufficient for weights (1B × 1.5 bytes = 1.5GB)

**Training:** CPU inference only (not suitable for training)

---

## Summary: Infrastructure Choices

| Technique | Category | Expected Gain | Status | Priority |
|-----------|----------|---------------|--------|----------|
| torch.compile | Compiler | 1.3-1.8x | To implement | P1 |
| Triton SSD | Kernel | 1.2x (SSD) | Reference | Skip |
| CUDA graphs | Compute | 1.1-1.2x | Reference | Skip (incompatible) |
| μP scaling | Hyperparameter | 93% search cost | To implement | P7 |
| Batch size scaling | Hyperparameter | 1.2-1.3x | To test | P1 |
| MBDS sampling | Data | 1.3-1.5x | Reference | Skip (single source) |
| Curriculum learning | Training | 1.2-1.3x | Reference | Skip |
| CPU offloading | Memory | VRAM only | Reference | Skip |
| Cloud A100 | Hardware | 3-4x speed | To use | P4-5 |
| Pedagogical distill | Knowledge | 2-3x convergence | To test | P6 |
| Bi-Mamba distill | Knowledge | Reference only | Reference | Skip |

---

*Last updated: February 2026*
