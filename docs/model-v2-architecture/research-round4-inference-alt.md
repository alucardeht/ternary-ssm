# Research Round 4: Inference Optimization (Alternative Angles)

**Date:** 2026-02-26  
**Phase:** 27-D Cycle 2  
**Focus:** Serving systems, hardware-specific tricks, edge deployment, decoding strategies, compilation, batching

This document captures research findings on inference optimization from different angles than the primary researcher, focusing on deployment infrastructure, hardware features, and novel serving patterns.

---

## Finding 98: T-SAR — CPU Ternary Inference via In-Register LUT (Nov 2025)

**Source:** "T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization" (arXiv:2511.13676, Nov 2025)

**Technique:** Repurpose SIMD register file for dynamic in-register LUT generation to eliminate memory bottleneck in ternary LLM inference on CPUs.

**Mechanism:**
- Ternary-to-binary decomposition: Split ternary weights {-1,0,+1} into dense binary {-1,+1} and sparse binary {0,1}
- Replace one 3^c ternary LUT with two 2^c binary LUTs (matches SIMD register width)
- Generate LUTs on-the-fly in SIMD registers instead of loading from DRAM
- Two ISA extensions: TLUT_c×s (LUT generation) and TGEMV_k×m (matrix-vector multiply)
- Minimal hardware changes: 3.2% power, 1.4% area overhead in SIMD units
- Adaptive kernel scheduling: activation-persistent (AP) vs output-persistent (OP) dataflows

**Results:**
- **GEMM latency:** 5.6-24.5× improvement over BitNet.cpp/T-MAC
- **GEMV throughput:** 1.1-86.2× improvement
- **Memory requests:** 8.7-13.8× reduction (87.6% → <10% of total traffic)
- **Energy efficiency:** 2.5-4.9× better than NVIDIA Jetson AGX Orin GPU on Llama-8B/Falcon3-10B
- **End-to-end prefill:** 8.8× (Workstation), 8.4× (Laptop), 12.4× (Mobile) speedup
- **End-to-end decode:** 6.4× (Workstation), 4.1-4.2× (Laptop/Mobile)

**Applicability to Our Model:**
- **HIGHLY RELEVANT** — Our model is 84% ternary weights (50M model = ~42M ternary params)
- Current bottleneck: GPU memory bandwidth (672 GB/s saturated at high batch sizes)
- T-SAR shows CPU inference can be FASTER and MORE EFFICIENT than edge GPUs for ternary models
- Our 50M model fits entirely in L3 cache (48MB available on Intel 225F)
- SSM decode is memory-bound (not compute-bound) — same profile as GEMV in T-SAR

**Implementation Path:**
1. **Immediate (x86 CPU fallback):** Port T-SAR kernel approach to our ternary SSM
   - Decompose ternary A/B matrices into dense+sparse binary
   - Use AVX2 SIMD (256-bit) on Intel 225F
   - Target: 10-50 tok/s on CPU (for deployment on devices without GPU)
2. **Future (ARM edge devices):** Adapt to ARM NEON (128-bit) for mobile/Raspberry Pi
   - Paper confirms NEON compatibility with parameter retuning
   - Target: 5-20 tok/s on ARM devices
3. **Research (custom ISA):** If deployment at scale, consider custom SIMD extensions
   - Only 1.4% area, 3.2% power overhead
   - Could be integrated into future CPU generations

**Complexity:** Medium (kernel implementation) to High (ISA extensions)

**Key Insight:** Ternary models on CPUs can BEAT edge GPUs in energy efficiency. Our focus on GPU-only inference may be misguided for edge deployment. A dual-target approach (GPU for server, CPU for edge) maximizes deployment flexibility.

---

## Finding 99: ARM NEON Mixed-Precision Optimization (IEEE 2025)

**Source:** "Enhancing LLM Inference Performance on ARM CPUs Through Software and Hardware Co-Optimization Strategies" (IEEE 2025, paper 10994252)

**Technique:** Mixed-precision quantization + SIMD optimization + fused attention for ARM CPUs.

**Mechanism:**
- Preserve critical weights (attention Q/K/V) at higher precision (INT8)
- Quantize non-essential weights to INT8 (8× compression)
- Leverage ARM NEON SIMD for parallel INT8 operations
- Fuse attention computation components (reduce kernel launches)
- Streamline dequantization by modifying scaling factor computation

**Results (Qwen-1.8B on ARMv9):**
- **Accuracy:** Only 0.66% decrease
- **Memory usage:** 58.8% of baseline (41.2% reduction)
- **Prefill speedup:** 4.09× over baseline
- **Decode speedup:** 15.23× over baseline

**Applicability:**
- Moderate — Our model already uses mixed precision (84% ternary, 16% FP16)
- ARM NEON optimizations applicable if we target ARM edge devices
- Attention fusion technique applicable to our sliding window attention (4 token window)
- Dequantization streamlining could apply to our ternary → FP16 conversion

**Implementation Complexity:** Medium-High (requires ARM-specific intrinsics)

**Trade-off:** Our ternary quantization (2-bit) is more aggressive than INT8 (8-bit), but achieves similar memory reduction with different compute patterns.

---

## Finding 100: Disaggregated Prefill-Decode Serving (2025-2026 trend)

**Source:** BentoML LLM Handbook, NVIDIA Dynamo, vLLM disaggregated architecture

**Technique:** Separate prefill (compute-bound) and decode (memory-bound) phases onto dedicated hardware pools.

**Mechanism:**
- **Prefill cluster:** GPU-heavy, optimized for throughput (parallel token processing)
- **Decode cluster:** Memory-optimized, high-bandwidth, optimized for latency (sequential generation)
- KV cache transfer between clusters via fast interconnect (NIXL, NVMe-oF, CXL)
- Independent scaling: add prefill GPUs for TTFT, add decode GPUs for ITL
- Prefix cache reuse benefits decode cluster (multi-turn conversations)

**Results:**
- TTFT and ITL can be optimized independently (no interference)
- Parallel execution: prefill and decode run concurrently for different requests
- Better resource utilization (prefill doesn't starve decode and vice versa)

**Applicability to SSM Models:**
- **HIGHLY RELEVANT BUT DIFFERENT** — SSMs don't have KV cache
- SSM state is MUCH smaller (707MB at 1B context vs 25.3GB KV cache at 64K context)
- SSM prefill is ALSO compute-bound (recurrent scan)
- SSM decode is memory-bound (weight fetch dominates)
- State transfer cost is minimal compared to Transformer KV cache

**SSM-Specific Disaggregation Strategy:**
1. **Prefill pool:** High-compute GPUs (RTX 4090, A100) for recurrent scans
2. **Decode pool:** Memory-optimized (RTX 5070 with high bandwidth, or T-SAR CPUs)
3. **State transfer:** 707MB state @ 1B tokens → 5.3 ms transfer on PCIe 5.0 (negligible)
4. **Benefit:** Decode can be 10× cheaper hardware (ternary on CPU via T-SAR)

**Thresholds:**
- BentoML reports 20-30% degradation if workload too small
- SSM state transfer is ~1000× smaller than Transformer KV cache → lower threshold
- Viable at lower request rates than Transformer disaggregation

**Implementation Complexity:** High (requires serving infrastructure, load balancing, inter-node communication)

**Key Insight:** SSMs are BETTER suited for disaggregation than Transformers due to tiny state size. Decode pool can use cheap CPUs (T-SAR) while prefill uses GPUs → massive cost reduction.

---

## Finding 101: WebAssembly + WebGPU for Browser Inference (2026)

**Source:** "WebAssembly for LLM Inference: Running Models in Browsers" (dasroot.net, Jan 2026), ONNX Runtime Web v2.6

**Technique:** Deploy quantized LLMs in browsers using WebAssembly (compute) + WebGPU (GPU acceleration).

**Mechanism:**
- Compile model to ONNX format, run via ONNX Runtime Web v2.6
- WebAssembly provides near-native execution speed (10× faster than JavaScript)
- WebGPU provides GPU acceleration in browser (compute shaders)
- Quantization to INT-8 or INT-4 reduces model size (75% reduction for INT-4)
- Zero server-side dependencies, fully client-side execution

**Results:**
- **WebLLM (Llama-3.1-8B INT-4):** 7 tok/s prefill, 4.27 tok/s decode
- **ONNX Runtime Web:** <150 KB footprint
- **Latency:** 12.3ms/token (browser) vs 11.8ms/token (native CPU) = only 4.2% gap
- **Privacy:** 100% offline, zero data transmission
- **Memory:** Quantized models fit in browser memory (INT-4 = 75% reduction)

**Applicability:**
- **EXPLORATORY** — Our 50M ternary model is ~12.5 MB (50M × 2 bits / 8 / 1024^2)
- Adding FP16 embeddings/layernorms: ~100 MB total model size
- Easily fits in browser memory, suitable for offline edge AI
- WebGPU supports custom compute shaders → could implement ternary matmul
- Privacy-preserving deployment (medical, legal, sensitive data)

**Use Cases:**
- Offline personal assistant (runs in browser, no server)
- Edge AI on devices without native runtime (Chromebooks, tablets)
- Privacy-critical applications (healthcare, finance)
- Demos and prototypes (instant deployment, no backend)

**Implementation Path:**
1. Export model to ONNX format (currently PyTorch)
2. Write custom WebGPU compute shader for ternary matmul (replacement for standard GEMM)
3. Integrate with ONNX Runtime Web v2.6 via custom operator
4. Deploy as static HTML+JS (single-file distribution)

**Complexity:** Medium (ONNX export + custom WebGPU shader)

**Performance Expectation:** 10-30 tok/s in browser (our model is 10× smaller than Llama-8B benchmarked)

**Key Insight:** Browser deployment opens entirely new use cases impossible with server-based models. Our small model size (100 MB) is ideal for this.

---

## Finding 102: NVIDIA Async Copy Engine for PCIe-GPU Overlap (NVIDIA DevBlog)

**Source:** "Advanced API Performance: Async Copy" (NVIDIA Developer Blog)

**Technique:** Use dedicated async copy engine to overlap PCIe transfers with compute.

**Mechanism:**
- NVIDIA GPUs have dedicated async copy hardware (separate from compute)
- Schedule copy work onto async copy queue (separate from graphics/compute queue)
- Full parallelism: PCIe transfer runs concurrently with GPU compute
- Use fences/semaphores to synchronize between queues
- Best for: host-to-device, device-to-host transfers; NOT for device-to-device

**Best Practices:**
- **DO:** Put PCIe transfers on async copy queue (saturates PCIe, not GPU bandwidth)
- **DO:** Use sufficient latency to cover variable update costs
- **DON'T:** Move local GPU copies to async queue (overhead not worth it)
- **DON'T:** Use for GPU bandwidth saturation (async copy engine is for PCIe)
- **DON'T:** Overuse fences (creates idle time)

**Applicability:**
- **MODERATE** — Our current inference is fully GPU-resident (no PCIe transfers)
- Relevant for batched inference with streaming inputs/outputs
- SSM state loading from host memory (if state is cached on CPU)
- Multi-GPU scenarios (model sharding, disaggregated serving)

**Potential Use Cases:**
1. **Streaming batched inference:** Load inputs while previous batch decodes
2. **State caching:** If SSM state is too large for GPU, cache on host, async load
3. **Disaggregated serving:** Transfer state from prefill GPU to decode GPU/CPU

**Implementation:** Requires compute queue management, fencing logic

**Complexity:** Medium (API changes, synchronization)

**Expected Speedup:** 10-20% if PCIe transfers become bottleneck (not current bottleneck)

---

## Finding 103: Prefill-Decode Latency Characteristics (BentoML)

**Source:** BentoML LLM Inference Handbook — Prefill-Decode Disaggregation

**Observation:** Co-locating prefill and decode creates latency interference.

**Mechanism:**
- Prefill is compute-bound, decode is memory-bound
- When GPU processes prefill (high compute utilization), decode requests wait
- When GPU processes decode (high memory bandwidth), prefill requests wait
- Colocation prevents independent optimization of TTFT (prefill) and ITL (decode)

**Measured Impact:**
- Image from paper shows 20-30% latency increase when colocated
- Interference worst when workload has mixed prompt lengths (some long, some short)

**SSM Implications:**
- **LESS SEVERE THAN TRANSFORMERS** — SSM prefill is recurrent scan (not parallelizable like attention)
- SSM prefill is already sequential, no parallel advantage
- BUT decode is still memory-bound (same as Transformers)
- Disaggregation still benefits: prefill on high-compute GPU, decode on high-bandwidth device

**Key Insight for Our Model:**
- Current single-GPU inference is fine for low request rates
- Multi-request serving WILL benefit from disaggregation
- Decode can move to CPU (T-SAR) or cheap high-bandwidth GPU (RTX 5070)
- Prefill stays on powerful GPU (future A100/H100 for larger models)

---

## Finding 104: ONNX Runtime Web WebGPU Backend (2026)

**Source:** ONNX Runtime Web v2.6 documentation

**Technique:** WebGPU execution provider for GPU-accelerated inference in browsers.

**Features (v2.6):**
- Direct WebAssembly + WebGPU backend (no TensorFlow.js bridge)
- Cross-platform (Chrome, Edge, Firefox experimental)
- Model conversion from PyTorch, TensorFlow, HuggingFace, Scikit-learn
- SIMD optimizations in WebAssembly
- <150 KB runtime footprint

**Performance:**
- Near-parity with CPU inference (4.2% gap for Llama-7B)
- GPU acceleration via WebGPU compute shaders
- Supports quantized models (INT-8, INT-4)

**Applicability:**
- Same as Finding 101 — browser deployment path
- WebGPU allows custom ternary kernels (compute shaders)
- ONNX is standard format (good interoperability)

**Advantage over WebLLM:**
- ONNX is vendor-neutral (WebLLM is MLC-AI specific)
- Easier model export (PyTorch → ONNX is one-liner)
- Better browser support (WebLLM requires cutting-edge browsers)

**Disadvantage:**
- Custom operators (ternary matmul) require more integration work
- WebLLM has better optimizations for LLM-specific patterns

**Recommendation:** Use ONNX Runtime Web for initial browser deployment, migrate to WebLLM if performance critical.

---

## Finding 105: Profile-Guided Optimization (PGO) for Inference (2026)

**Source:** "From Profiling to Optimization: Unveiling the Profile Guided Optimization" (arXiv:2507.16649, 2025)

**Technique:** Use runtime profiling to guide compiler optimizations.

**Mechanism:**
1. **Instrumentation build:** Compile with profiling enabled
2. **Profile collection:** Run representative workload, collect execution traces
3. **Optimization build:** Recompile with profile data, compiler makes informed decisions
4. **Typical optimizations:** Hot path inlining, branch prediction, code layout, register allocation

**Results (general software):**
- 10-30% speedup typical for CPU-bound workloads
- Diminishing returns for memory-bound (already at hardware limits)
- Best for: irregular control flow, polymorphic calls, branching

**Applicability to Inference:**
- **LOW FOR CURRENT MODEL** — Our inference is kernel-bound, not CPU-bound
- Most time in Triton/CUDA kernels (already heavily optimized)
- Python overhead is <5% of total time
- PGO benefits Python dispatch, not kernel execution

**Where PGO Might Help:**
1. **Serving framework:** FastAPI, request routing, batch scheduling (CPU-bound)
2. **Pre/post-processing:** Tokenization, sampling, output formatting
3. **Multi-model ensembles:** If we add retrieval, reranking, etc.

**Implementation:**
- Compile Python/C++ extensions with `gcc -fprofile-generate`, run inference, recompile with `gcc -fprofile-use`
- Minimal effort (build system change only)
- Expected speedup: 5-10% on non-kernel code (negligible overall impact)

**Complexity:** Low (build flag change)

**Priority:** Low (kernel optimization >> PGO for our workload)

---

## Summary: Alternative Angles Coverage

This research round explored 8 additional findings (98-105) across 6 focus areas:

### 1. Edge/Mobile Deployment (Findings 98, 99, 101, 104)
- **T-SAR:** CPU ternary inference BEATS edge GPUs (2.5-4.9× energy efficiency)
- **ARM NEON:** 15.23× decode speedup on ARMv9 with mixed-precision
- **WebAssembly/WebGPU:** Browser deployment, 100% offline, 4.27 tok/s achievable
- **ONNX Runtime Web:** Standard path for browser export

### 2. Serving Systems (Findings 100, 103)
- **Disaggregated serving:** Separate prefill (GPU) and decode (CPU/cheap GPU)
- **SSM advantage:** 1000× smaller state than Transformers → easier disaggregation
- **Latency interference:** Co-location causes 20-30% overhead

### 3. Hardware-Specific Tricks (Finding 102)
- **Async copy engine:** Overlap PCIe with compute (10-20% if PCIe bottleneck)
- **Currently not bottleneck** but relevant for multi-GPU, streaming

### 4. Novel Decoding Strategies
- No major findings in this search (would require speculative decoding, draft models)
- Deferred to future research

### 5. Compilation/Graph Optimization (Finding 105)
- **PGO:** 10-30% CPU speedup, but negligible for kernel-bound workload
- Low priority for current inference pipeline

### 6. Batching Optimization
- Covered implicitly in disaggregated serving (Finding 100)
- SSM batching is simpler than Transformer (no KV cache management)

---

## Cross-References with Primary Researcher

This research complements the primary researcher's findings by covering:
- **CPU inference** (T-SAR, ARM NEON) vs GPU-only focus
- **Browser deployment** (WebAssembly/WebGPU) vs server deployment
- **Disaggregated serving** (architecture pattern) vs single-node optimization
- **PCIe overlap** (system-level) vs kernel-level optimization

Avoids duplication on:
- Flash Attention / KV cache (SSM doesn't use KV cache)
- Speculative decoding (needs separate research)
- Batching strategies (SSM is simpler, covered in serving section)

---

## Top 3 Actionable Findings

**1. T-SAR CPU Inference (Finding 98) — IMMEDIATE IMPACT**
- Our 50M ternary model can run faster on CPU than edge GPU
- Enables deployment on devices without GPU (Raspberry Pi, laptops, servers)
- 2.5-4.9× better energy efficiency than Jetson
- Implementation: Port T-SAR kernel to our SSM ternary ops

**2. Disaggregated Serving (Finding 100) — SCALING STRATEGY**
- SSM state is 1000× smaller than Transformer KV cache
- Decode can run on cheap hardware (T-SAR CPUs) while prefill uses GPU
- Massive cost reduction for production serving
- Implementation: Multi-node serving with FastAPI + state transfer

**3. WebAssembly Deployment (Finding 101) — NEW MARKET**
- 100 MB model fits easily in browser
- Privacy-preserving offline AI (medical, legal, personal)
- Instant deployment (no backend infrastructure)
- Implementation: ONNX export + WebGPU ternary kernel

---

## Next Steps for Implementation

**Phase 23 Continuation (Inference Engine):**
1. Complete Step 4 (Fused SSM mega-kernel) — PRIMARY FOCUS
2. Benchmark browser deployment (ONNX export + WebGPU PoC)
3. Prototype T-SAR CPU inference (AVX2 kernel for ternary SSM)

**Phase 24+ (Deployment):**
4. Implement disaggregated serving (prefill GPU + decode CPU/cheap GPU)
5. Full browser inference engine (WebAssembly production build)
6. ARM NEON port for Raspberry Pi deployment

---

**Research Complete: 2026-02-26**  
**Findings: 98-105 (8 findings)**  
**Focus: Edge deployment, serving systems, browser inference, CPU optimization**
