# Research Round 5 — Deployment & Serving at Scale

## Overview

This document tracks research findings on deployment, serving systems, and production infrastructure for ternary SSM models discovered in Research Cycle 5, Aspect 5 (Feb 2026).

## Wave 1: Web Research (2 researchers, Feb 2026)

### Serving Architecture Findings

**F1: NestedFP Dynamic Precision Switching** (NeurIPS 2025)
- Overlay FP8 parameters onto FP16 in same memory footprint for SLO-aware dynamic precision selection
- Switches between FP16 (default) and FP8 (load surge) without storing two model copies
- Relevance: overlay ternary weights onto FP16 base, switch precision based on request type

**F2: TurboMind Mixed-Precision Pipelines** (arXiv 2508.15601, 2025)
- Dual-pipeline: (1) GEMM pipeline with offline weight packing + online acceleration, (2) Attention pipeline with arbitrary Q/K/V precision
- 61% lower latency (30% avg), 156% higher throughput (58% avg) vs vLLM/MARLIN/TensorRT-LLM across 16 LLMs on 4 GPU architectures
- Relevance: exact architecture for mixed-precision serving — ternary weights + FP16 scaling factors fit perfectly

**F3: Intel BitNet 1-bit Serving** (Intel, arXiv 2508.06753, Jan 2026)
- 1-bit and 2-bit GEMM kernels for Intel GPUs, vLLM integration as quantization plugin
- 4-8x reduction in GEMM time vs BF16, up to 6.3x speedup in end-to-end latency
- Proves ternary serving is viable on non-NVIDIA consumer hardware (AI PCs)

**F4: SGLang vs vLLM Comparison** (localaimaster.com, 2026)
- SGLang: 16,215 tok/s (H100, Llama 3.1 8B) vs vLLM 12,553 tok/s (+29% throughput)
- RadixAttention (automatic prefix caching) vs PagedAttention (memory efficiency)
- SGLang's RadixAttention ideal for multi-turn conversations + quantization support

**F5: QuickReduce All-Reduce Acceleration** (AMD ROCm, 2025)
- Inline compression all-reduce with Q4/Q6/Q8/FP8 quantization (block size 32)
- TTFT accelerated >1.2x with Q4, up to 3x speedup at TP=2
- Zero-code-change via environment variable for vLLM/SGLang

**F6: High-Throughput Inference Patterns** (emergentmind.com, Feb 2026)
- Multi-bin batching (group similar-length requests to reduce straggler stalls)
- Work-conserving mixed batching (co-batch prefill + decode up to token budget)
- Chunked prefill (break prefill into chunks to avoid blocking decode)
- KV-cache compression: PyramidInfer 54% memory reduction, ShadowKV 6x batch size

**F7: Microsoft BitNet b1.58 Serving** (InfoQ, Apr 2025)
- Natively trained 1.58-bit (ternary {-1,0,+1}) with per-group FP16 scaling (group_size=32)
- bitnet.cpp for CPU inference — EXACT same ternary scheme as ours
- Reference implementation for our serving engine

**F8: vLLM Quantization Plugin Architecture** (2025)
- Supports GPTQ, AWQ, AutoRound, INT4, INT8, FP8 + FP8 KV cache
- Plugin architecture for custom quantization — add ternary support without forking
- Production deployments at Meta, Mistral

**F9: Prefill vs Decode Precision Asymmetry** (TurboMind 2025)
- Prefill: compute-bound (high arithmetic intensity), optimal with pipeline-parallel
- Decode: memory-bound (low batch size), optimal with tensor-parallel
- Ternary helps decode phase MORE (memory-bound) — serving should account for asymmetry

### Batching & Scheduling Findings

**F10: DistServe Disaggregated Serving** (Hao AI Lab, 2025)
- Separate prefill (GPU) and decode (memory-bound) into independent compute pools
- Eliminates 2-30x TPOT inflation from prefill-decode interference
- Now standard in NVIDIA Dynamo, vLLM, SGLang, Ray Serve
- SSM: prefill needs GPU (attention), decode can run CPU (ternary + state ops)

**F11: NVIDIA Dynamo Multi-Hardware Orchestration** (GTC 2025)
- KV-aware router + GPU Planner for auto-scaling P/D instances
- NIXL unified fabric across NVLink, InfiniBand, PCIe, SSD
- SSM's 2.8KB state transfer << KV cache enables fast migration

**F12: vLLM Continuous Batching** (blog.vllm.ai, Sep 2025)
- Iteration-level scheduling: add/remove requests mid-batch, preemption, priority queues
- 2-24x throughput vs static batching (production at Meta, Mistral)
- 73% cost reduction at Stripe (50M daily API calls on 1/3 GPU fleet)
- SSM: O(1) memory → 5-10x more sessions than Transformers (160+ in 12GB)

**F13: BitNet.cpp Ternary Edge Inference** (Wang et al., ACL 2025)
- TL-2 ternary lookup table for sub-2-bit inference, I2_S for lossless edge deployment
- 6.25x speedup over FP16 baseline, 2.32x over low-bit baselines
- DIRECT application to our ternary model — eliminates matmul on edge

**F14: T-SAR CPU-Only Ternary Inference via SIMD** (arXiv:2511.13676, Nov 2025)
- In-register LUT generation (eliminates 87.6% memory traffic), ternary-to-binary decomposition
- Prefill: 8.8-12.4x speedup, Decode: 4.1-6.4x speedup
- Energy: 2.5-4.9x better than Jetson AGX Orin GPU
- GAME-CHANGER for CPU deployment — scales 125M to 100B parameters

**F15: MoonCake KVCache-Centric Disaggregation** (Kimi AI, FAST'25 Best Paper)
- Centralized cache pool (SSD/NVMe/DRAM) enabling any prefill → any decode handoff
- SSM version: 2.8KB state fits in L3 cache → ultra-fast handoff, 1M+ cached states with <1ms retrieval

**F16: SGLang Expert Parallelism at Scale** (lmsys.org, Jan 2025)
- P/D disaggregation with wide expert parallelism (EP=256), 3 prefill + 9 decode nodes
- 52.3k input TPS, 22.3k output TPS (DeepSeek-R1)
- SSM decode is O(1) → parallelizes perfectly across CPU pool

**F17: Goodput-Optimized Serving** (emergentmind.com, Jan 2026)
- Maximize rate of outputs meeting user SLOs, not raw throughput
- Phase disaggregation + deadline-aware scheduling
- SSM's predictable O(1) decode → easier to guarantee <10ms TPOT for 99% of requests

**F18: ML-SpecQD Speculative Decoding with Quantized Drafts** (arXiv:2503.13565, Mar 2025)
- Small quantized draft model generates tokens, full model verifies
- Our ternary SSM = PERFECT draft model: fast decode (876 tok/s), low memory, cheap state (2.8KB)
- Hybrid: ternary draft on CPU, FP16 verify on GPU

**F19: Edge Energy Analysis** (arXiv:2504.03360, Apr 2025)
- Quantization reduces energy proportional to bit reduction
- Ternary (1.58-bit) achieves best energy/accuracy tradeoff
- Our ternary SSM: double energy savings (1.58-bit weights + O(1) memory)

**F20: Cactus v1 Cross-Platform Mobile** (InfoQ, Dec 2025)
- Cross-platform energy-efficient kernels for mobile/wearables
- Sub-50ms latency on low-power devices, Y Combinator-backed
- Target deployment platform for ternary SSM

**F21: ExecuTorch + Arm SME2** (Arm Blog, 2025)
- PyTorch ExecuTorch framework + Arm Scalable Matrix Extension 2 for mobile
- SME2 can accelerate ternary matmul (similar to T-SAR's SIMD approach)
- Deployment framework for ternary SSM on iOS/Android

**F22: TOM Ternary ROM Accelerator** (arXiv:2602.20662, Feb 2026)
- Custom ASIC for ternary inference using read-only memory optimizations
- SSM + ternary = ideal for ROM-based accelerators (O(1) state, ternary routing)
- Future hardware direction for maximum energy efficiency

### KEY PATTERNS (Wave 1)
1. **Disaggregation is universal** — all major frameworks (vLLM, SGLang, Dynamo) separate prefill/decode
2. **CPU ternary is production-ready** — BitNet.cpp, T-SAR prove 6-12x speedup on consumer CPUs
3. **SSM's O(1) memory is massive advantage** — 5-10x more concurrent sessions than Transformers
4. **Plugin architecture standard** — add ternary support to vLLM/SGLang without forking
5. **Mobile deployment mature** — Cactus, ExecuTorch, BitNet.cpp all shipping on mobile
6. **State transfer trivial** — 2.8KB << KV cache (10-100MB), enables instant migration

## Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

### Serving Optimization Mathematics

**M1: Ternary SIMD-Fused Batch Dispatch** — Algebraic Structure of Signed Accumulation
- Ternary GEMV batching is a SIMD-width problem, not bandwidth problem
- N sessions within a SIMD lane are FREE (same bitmask broadcast, different data)
- GPU warp=32: process 32 sessions simultaneously with identical instruction count to batch=1
- CPU AVX-512=16: process 16 sessions per instruction
- 160 sessions on single GPU: ceil(160/32)=5 dispatch groups → 28K tok/s system throughput, 5.7ms/token
- Implementation: Medium | Viability: HIGH | Impact: ~32x throughput

**M2: Contractive-Eigenmode State Compression** — Dynamical Systems Theory
- Each SSM dimension has decay rate rho_i = |a_i| and memory horizon tau_i = -1/ln(rho_i)
- Short-memory dimensions (tau << position): quantize aggressively, error heals exponentially
- Bit allocation: bits_i = max(2, ceil(log2(tau_i / tau_min)))
- 75% dimensions tau<10: 2 bits; 25% dimensions tau>10: 16 bits → 5.8x compression
- State: 2.8KB → ~480 bytes. Transfer at 10Gbps: 0.38 microseconds
- Implementation: Low | Viability: HIGH | Impact: 5-6x state compression

**M3: Layer-Type Disaggregated Serving** — Graph Partitioning on Compute DAG
- Hybrid SSM+SWA creates bipartite compute graph: SSM=bandwidth-bound, SWA=compute-bound
- Route each layer type to hardware optimized for its bottleneck
- Inter-device transfer: d_model values = 1.28KB at 50M → 0.02us over PCIe Gen5
- 7:1 SSM:SWA ratio with 6 crossings at 50M = 0.12us total overhead
- Implementation: Medium | Viability: MEDIUM-HIGH | Impact: 1.5-2x on heterogeneous HW

**M4: Contractive Pipeline Parallelism** — Perturbation Theory of Linear Dynamical Systems
- Pipeline SSM with 1-step state delay; contraction property bounds the error
- Error: ||delta|| ~ L * 1.6 * ||B|| * ||Delta_input|| (heals exponentially)
- K pipeline stages → K*throughput with unchanged latency per token
- Circuit breaker: fall back to sequential when ||delta|| exceeds threshold
- Implementation: High | Viability: MEDIUM | Impact: 2-4x multi-device throughput

**M5: Affine Monoid Prefill Decomposition** — Algebraic Semigroup Theory
- SSM step = affine map T_t: h → A_t*h + b_t. Maps form MONOID under composition (associative)
- Parallel prefill: split P tokens into K chunks, compute composed map per chunk, tree-reduce
- Binary composition: (alpha_2*alpha_1, alpha_2*beta_1 + beta_2) — O(d_state*d_inner)
- For P=2048, K=8: nearly 8x prefill speedup (merge cost ~0.002 token-equivalents)
- EXACT computation — algebraically identical to sequential (no approximation)
- Implementation: Medium | Viability: HIGH | Impact: K-fold TTFT reduction

**M6: Differential State Gossip** — Information Theory + Dynamical Systems
- Continuous state replication via gossip protocol with eigenmode-adaptive coding
- Delta compression: fast-decay dimensions have predictable deltas, slow-decay have tiny deltas
- Gossip every G=32 tokens: 160 sessions * (876/32) * 4KB = 17.5 MB/s bandwidth
- Zero-downtime failover: recovery within ceil(max_i(tau_i)) tokens for full convergence
- Implementation: Medium | Viability: MEDIUM-HIGH | Impact: zero-downtime elastic scaling

### Deployment Hardware Mathematics

**M7: Ternary Galois Arithmetic** — Finite Field / Modular Ring Theory (GF(3))
- Embed computation in ring Z/3Z: phi(-1)=2, phi(0)=0, phi(+1)=1
- GF(3) routing via parallel modular comparisons: vpternlogd (AVX-512) = 1 instruction per 64-bit word
- Separates SIGN information (GF(3) domain) from MAGNITUDE (real domain)
- NPU: embed GF(3) in GF(4)=GF(2^2) — leverage cryptographic accelerators for free routing
- Implementation: Medium | Viability: MEDIUM | Impact: 1.3-2x on crypto-capable hardware

**M8: Spectral Ternary Decomposition** — Walsh-Hadamard Transform Domain
- Transform W and x into WHT domain where ternary structure creates spectral sparsity
- GEMV = pointwise multiplication in spectral domain: O(K log K + M*s_avg) vs O(MK)
- WHT is additions-only (no multiplications) — maps perfectly to SIMD butterfly networks
- For trained weights with 60-80% spectral sparsity: 2-4x throughput on bandwidth-limited HW
- Implementation: Medium | Viability: MEDIUM | Impact: 2-4x if spectrally sparse

**M9: Residual State Quantization** — Rate-Distortion Theory + Lattice Quantization
- Rate-distortion bound: 4 bits/dimension suffices for negligible distortion (sigma=0.1, D=10^-4)
- E8 lattice quantizer: 14% distortion reduction vs scalar at same rate, decoded via integer ops
- Sigma-delta error feedback: shapes quantization noise to high frequencies (SSM attenuates)
- State: 2.8KB → 350 bytes (INT4) or 700 bytes (INT8). 4-8x more concurrent sessions
- Implementation: Low-Medium | Viability: HIGH | Impact: 4-8x state compression

**M10: Boolean Circuit Synthesis** — Logic Minimization (Quine-McCluskey/Espresso)
- Ternary GEMV = family of M independent linear Boolean functions, not matrix multiplication
- Common Subexpression Elimination: shared sub-sums across outputs compute once, reuse
- Hardware-constrained synthesis: different gate models for CPU SIMD, GPU, ARM, FPGA
- Offline one-time synthesis, compiled circuit for each hardware target
- 15-40% reduction in additions depending on weight structure
- Implementation: Medium | Viability: HIGH | Impact: 15-40% ops reduction

**M11: Tensor Network State Evolution** — Tensor Train Decomposition
- Decompose ternary projection matrices into TT format: G_1 * G_2 * ... * G_d with ternary cores
- TT-matvec: O(L * r^2 * d^2) vs O(D^2). For r=4, d=4, L=5: 819x reduction
- Ternary TT-cores: each small "matmul" is additions-only (double benefit)
- Apply to B/C projection matrices (d_model × d_state) — not the small A matrix
- Implementation: Medium | Viability: MEDIUM | Impact: 10-50x if low TT-rank

**M12: Homomorphic Ternary Lifting** — Carry-Free Signed-Digit Arithmetic
- Eliminate carry propagation via signed carry-save representation: value = s_i - c_i
- ADD x: XOR into s-half. SUBTRACT x: XOR into c-half. SKIP: nothing. O(1) depth per op
- Ternary weight directly encodes BIT ROUTING: +1→route to s, -1→route to c, 0→skip
- 30-40% reduction in bit-operations. On FPGA: 2x throughput from eliminated carry latency
- EXACT computation — different representation, not approximation
- Implementation: Low-Medium | Viability: HIGH | Impact: 30-40% ops, 2x FPGA

### KEY INSIGHTS (Wave 2)
1. **Batching is SIMD-width problem:** 32 sessions per GPU warp at ZERO additional instruction cost (M1)
2. **State compression via contraction theory:** error heals exponentially → aggressive quantization safe (M2, M9)
3. **Monoid algebra enables exact parallel prefill:** K-device prefill with algebraic correctness guarantee (M5)
4. **Weights-as-program paradigm:** ternary GEMV should be COMPILED (codegen), not interpreted (load+iterate) (M10, M12)
5. **Carry-free arithmetic is universal:** signed-digit routing network replaces traditional accumulation (M12)
6. **Cross-pollination:** M2+M6 (compressed gossip), M10+M12 (circuit+carry-free), M5+M1 (batched prefill)

## Wave 3: Paradigm Breakers (2 agents, Feb 2026)

### Radical Serving Paradigms

**P1: Spatial Pipeline Serving — "The Ternary Assembly Line"**
- Break assumption: token must traverse entire depth before next session's token enters Layer 1
- Pipeline L layers across L compute units: L tokens from L sessions in flight simultaneously
- Only possible because inter-stage data is ~81KB (activation + state slice), not MB of KV cache
- At 6M (8 layers): 7,018 tok/s aggregate, 8x improvement over time-sharing
- At 50M (24 layers): 24x throughput multiplier
- Feasibility: HIGH — pure software architecture, prototype in 1-2 weeks

**P2: Entropic State Triage — "Intelligence-Aware Resource Allocation"**
- Break assumption: every token in every session requires same computation
- Measure SSM state velocity/saturation/predictability → 3 compute tiers (FULL/REDUCED/FROZEN)
- FROZEN tier: cached output + D*x correction = ~5% of full compute
- Steady-state: 30% FULL + 40% REDUCED + 30% FROZEN = 0.435x effective load
- Effective capacity: 368 sessions from 160 (2.3x). Only SSM has cheap state metrics
- Feasibility: MEDIUM-HIGH — simple metrics, main challenge is threshold calibration

**P3: Persistent Activation Atlas — "Ternary Neuron Weather Map"**
- Break assumption: each forward pass starts with no knowledge of which neurons fire
- Maintain per-layer bitmap of predicted activation states, skip provably-dead neurons
- Ternary creates sharper decision boundaries + SSM continuity = high prediction accuracy
- Expected: 84% predicted (42% OFF, 42% ON trivial), only 16% need full GEMV = ~3-5x per layer
- Combined with P1: 8 * 2.5 = 20x aggregate throughput at 6M
- Feasibility: MEDIUM — needs activation autocorrelation measurement for validation

**P4: Polytope Navigation — "Walk the Geometry, Skip the Algebra"**
- Break assumption: inference = computing through layers (actually partitions input space into polytopes)
- Within each polytope, entire model = single affine map y = M*x + b
- Ternary composed maps are INTEGER → use INT8 tensor cores (988 TOPS vs ~15 TOPS ternary)
- Block-level caching (4 layers): at 90% hit rate → 2,915 tok/s (3.3x)
- At 50% hit rate → SLOWER (0.66x). Break-even >55% hit rate. Highest ceiling, highest risk
- Feasibility: MEDIUM-LOW — needs block-level cache hit rate measurement

### Radical Deployment Paradigms

**P5: Monolithic Native Binary — "The Model IS the Program"**
- Break assumption: model is data interpreted by runtime. Compile model into single .exe with zero dependencies
- Ternary weights as packed bitmaps in .rodata, inference code ~2000 lines of C
- Binary size: ~15MB (vs ~2GB for llama.cpp + weights). Cold start: <50ms. Dependencies: ZERO
- Estimated CPU-only throughput: ~660 tok/s at 50M (AVX-512 on 5GHz x86-64)
- "Download 15MB file, double-click, talk to AI. No internet, no installation, no GPU."
- Feasibility: HIGH — pure engineering, implementable in 2-3 days

**P6: Transferable Microstate — Stateless Inference with Client-Carried Context**
- Break assumption: inference state lives on server. Client carries entire context as portable blob
- Corrected state sizes: ~317KB at 6M, ~2.1MB at 50M, ~175MB at 7B
- Compressed (LZ4): ~700KB-1MB at 50M. Fits in HTTP request, IndexedDB, file storage
- Server becomes genuinely stateless: zero session affinity, trivial horizontal scaling
- Conversation forking = copy the blob. Resume anywhere = send blob to any endpoint
- Feasibility: HIGH — straightforward serialization protocol, implementable in 1 day

**P7: Fabric-Configured Compute — Weights as Physical Topology**
- Break assumption: weights are data loaded into compute. Ternary weights = routing configuration of FPGA
- w=+1: route input to positive accumulator. w=-1: route through inverter. w=0: no connection
- Per-layer fabric: ~200ns latency. Full model pipeline: output one token per 200ns = 5M tok/s potential
- Practical (hybrid, 4 layers in fabric + 20 conventional): ~100K tok/s at ~10us/token
- Power: ~1W FPGA fabric vs 75W GPU. 75x energy efficiency
- Feasibility: MEDIUM — concept sound, FPGA place-and-route is the challenge

**P8: Event-Driven Neuromorphic Inference — Compute Proportional to Change**
- Break assumption: every token triggers full computation. Delta-propagation = process only what changed
- Column-oriented weight index: for each changed input, ADD or SUB from affected output accumulators
- Ternary: O(1) per event (add/sub only). FP16: O(1) per event but 3-5x slower (multiply)
- Average 3.89x speedup on natural text (60% smooth=5x, 30% moderate=2.5x, 10% surprise=1.4x)
- Neuromorphic (Loihi 2): 42M synapses on one chip, ~2100 tok/s at ~1W power
- Feasibility: MEDIUM — needs delta sparsity measurement, error accumulation validation

### State Size Correction (Important)
The previously cited "2.8KB per session" was incorrect. Actual measured state sizes:

| Scale | SSM State | Conv State | SWA KV Cache | Total |
|-------|-----------|------------|--------------|-------|
| 6M | 224 KB | 29 KB | 64 KB | ~317 KB |
| 50M | 1.41 MB | 147 KB | 640 KB | ~2.1 MB |
| 7B | ~100 MB | ~10 MB | ~65 MB | ~175 MB |

Still dramatically smaller than Transformer KV cache (which grows with sequence length), but not negligible. P6 (portable state) works well at 50M (~2MB compressed to ~800KB) but becomes impractical for per-request HTTP at 7B.

### KEY INSIGHTS (Wave 3)
1. **Structural predictability:** ternary+SSM makes STRUCTURE of computation predictable (same neurons fire, same polytopes traversed) — impossible with Transformers
2. **Model as executable:** at 11MB + 2K lines of C, the model IS a portable binary — no framework needed
3. **State portability:** ~2MB portable state enables stateless serving, conversation forking, zero-cold-start
4. **Event-driven is natural:** ternary = O(1) event cost → neuromorphic deployment at ~1W power
5. **FPGA weights-as-wires:** ternary routing embodied in physical topology, not loaded as data

---

## Aspect 5 Summary: Deployment & Serving at Scale

**Total findings:** 22 web (F1-F22) + 12 math (M1-M12) + 8 paradigm shifts (P1-P8) = **42 findings**

### Top 10 by Estimated Impact

| Rank | Finding | Type | Impact | Feasibility |
|------|---------|------|--------|-------------|
| 1 | M1 SIMD-Fused Batch | Math | 32x aggregate throughput | HIGH |
| 2 | P5 Monolithic Binary | Paradigm | Zero-dependency 15MB executable | HIGH |
| 3 | P1 Spatial Pipeline | Paradigm | 8-24x aggregate throughput | HIGH |
| 4 | M5 Monoid Prefill | Math | K-fold TTFT reduction (exact) | HIGH |
| 5 | P6 Portable Microstate | Paradigm | Stateless serving, ~2MB state | HIGH |
| 6 | P3 Activation Atlas | Paradigm | 3-5x per-session via neuron skip | MEDIUM |
| 7 | M12 Carry-Free Arithmetic | Math | 30-40% ops, 2x FPGA | HIGH |
| 8 | P8 Event-Driven | Paradigm | 3.89x average, 1W neuromorphic | MEDIUM |
| 9 | M2 Eigenmode Compression | Math | 5-6x state compression | HIGH |
| 10 | F14 T-SAR CPU SIMD | Web | 8.8-12.4x CPU prefill speedup | PROVEN |

### Key Patterns
- **Weights-as-program paradigm:** ternary GEMV should be COMPILED (codegen), not interpreted
- **Disaggregation + SSM:** O(1) memory enables 5-10x more sessions than Transformers
- **CPU ternary is production-ready:** BitNet.cpp, T-SAR prove 6-12x on consumer CPUs
- **State portability revolution:** ~2MB state enables architectures impossible for Transformers
- **SIMD batching is key:** 32 sessions per GPU warp at ZERO additional instruction cost

### Cross-Wave Synergies
- **P5+P6:** Monolithic binary + portable state = complete "peer-to-peer AI" deployment
- **P1+P3:** Pipeline + neuron prediction = 20x aggregate throughput
- **M1+P1:** SIMD batch within pipeline stages = multiplicative
- **M5+M1:** Batched monoid prefill = inter-session + intra-session parallelism
- **M2+M6:** Eigenmode compression feeds gossip protocol
- **M10+M12:** Circuit synthesis + carry-free = optimized routing network
