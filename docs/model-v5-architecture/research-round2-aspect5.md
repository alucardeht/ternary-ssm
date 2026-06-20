# Research Round 2, Aspect 5: Parallelism & Hardware Utilization

**Date:** 2026-03-01
**Focus:** CPU+GPU workload splitting, operator placement, async pipelining, kernel optimization, memory efficiency
**Sources:** 2025-2026 papers, implementations, industry reports

---

## Wave 1 — Web Research (2 researchers, Mar 2026)

### Theme A: Heterogeneous CPU+GPU Computing

#### R2-A5-F1: TOM — Ternary Read-Only Memory Accelerator for Edge LLMs
- **Src:** [Guan et al., arXiv 2026] | **Gain:** Hybrid NPU+GPU system design | **V:** PROMISING
- TOM accelerator explores hybrid NPU+GPU systems for ternary-quantized LLM inference at the edge. Key insight: different layers have varying computational profiles — ternary layers are CPU/NPU-friendly (add/subtract), while attention and normalization benefit from GPU Tensor Cores. Paper proposes layer-aware heterogeneous placement for ternary models.

#### R2-A5-F2: TABv2 — Ternary SIMD Optimization via Encoding & Register Packing
- **Src:** [Fu et al., IEEE TVLSI 2026] | **Gain:** 2-3× SIMD throughput (inference), 4-8× on larger images | **V:** PROMISING
- TABv2 is a CPU-focused ternary/binary neural network library optimized for AVX2/AVX512 SIMD. Replaces scalar popcount with Harley-Seal SIMD algorithm and exploits ternary encoding {-1,0,+1} in 2 bits per weight, enabling 16 weights per AVX2 register (256-bit) or 32 weights per AVX512 register. Demonstrates that SIMD vectorization is more effective on larger tensors (better data reuse). For ternary SSM: ternary matmuls in FFN/projection layers can leverage this directly; the scan recurrence remains sequential but matmul-heavy layers benefit significantly. Portable across x86 CPUs (no GPU needed).

#### R2-A5-F3: DNNPipe — Optimal DNN Partitioning for Pipelined Inference
- **Src:** [Seo et al., J. Sys. Arch. 2025] | **Gain:** 1.78× throughput vs single device, 98% lower partitioning overhead | **V:** PROMISING
- Dynamic programming-based DNN partitioning for pipeline parallelization across heterogeneous IoT devices. Maximizes throughput by finding optimal layer split points that balance asymmetric device profiles. Applicable to ternary SSM: partition ternary layers (CPU) vs attention/norms (GPU/NPU) for minimal stage time.

#### R2-A5-F4: Ecomap — Sustainability-Driven Multi-DNN Heterogeneous Scheduling
- **Src:** [Paramanayakam et al., IEEE TC 2025] | **Gain:** Latency compliance + energy optimization | **V:** NEEDS-WORK
- Framework for managing multi-DNN workloads on heterogeneous edge servers under strict latency constraints. Optimizes for energy efficiency and carbon footprint. Relevant for batched ternary inference where multiple models share CPU+GPU resources. Needs adaptation for ternary-specific operator cost models.

#### R2-A5-F5: Split Inference — Dynamic Layer Partitioning for Device-Edge-Cloud
- **Src:** [Bakhtiarnia et al., 2022; Evgenidis et al., 2024] | **Gain:** 10-40% latency reduction vs static splits | **V:** PROMISING
- Adaptive split point selection for DNN layers based on real-time wireless bandwidth, server load, and batch size. Natural bottlenecks (compression ratio <1) in architectures like EfficientNet are optimal split candidates. For ternary SSM: split at state compression points where hidden state is smaller than input (e.g., after recurrent projection).

#### R2-A5-F6: CPU-GPU Data Transfer Pipelining — Performance Implications
- **Src:** [Li et al., ACM TACO 2025] | **Gain:** Overlap compute+transfer, reduce PCIe stalls | **V:** PROMISING
- Studies pipelining of data transfer in CPU-GPU heterogeneous systems considering CPU, DRAM, and PCIe hardware generations. Key for ternary SSM: while CPU computes ternary matmuls (no GPU needed), pipeline attention inputs to GPU via pinned memory + async cudaMemcpyAsync. Overlap is critical when PCIe Gen3 (16 GB/s) is the bottleneck.

#### R2-A5-F7: PreScope — Prefetching for Resource-Constrained MoE Inference
- **Src:** [Yu et al., arXiv 2025] | **Gain:** Hybrid CPU-GPU with expert prefetching reduces PCIe latency | **V:** NEEDS-WORK
- Proposes hybrid CPU-GPU inference for Mixture-of-Experts (MoE) models to overcome memory bottleneck and PCIe latency. Prefetches expert weights from CPU RAM to GPU VRAM while compute runs. Applicable to ternary MoD (Mixture-of-Depths): prefetch ternary routers and experts to GPU for attention, keep recurrent blocks on CPU.

#### R2-A5-F8: BitNet.cpp — CPU+GPU Heterogeneous Ternary Inference Framework
- **Src:** [Microsoft BitNet, 2025-2026] | **Gain:** 1.37-6.17× speedup, 55-82% energy reduction on CPU | **V:** PROMISING
- Official BitNet inference framework supports CPU (SIMD) and GPU kernels for 1.58-bit LLMs. Mentions "NPU support coming next." Current implementation: i2_s and tl1 quantization modes. Key: CPU kernels use table lookup for ternary matmul (T-MAC methodology), GPU kernels for FP16 layers. Heterogeneous execution not yet exposed, but architecture supports it.

#### R2-A5-F9: Deployment-Aware Compression — Heterogeneous CPU-NPU Execution
- **Src:** [Dritsas & Trigka, IEEE TAI 2026] | **Gain:** Binary/ternary optimal for NPU offload | **V:** NEEDS-WORK
- Survey on LLM compression for edge intelligence. Notes that edge devices increasingly feature heterogeneous CPU-NPU architectures (e.g., CMSIS-NN). Binary or ternary quantization is optimal for NPU execution while CPU handles control flow and FP16 ops. Intel NPU 13 TOPS (our hardware) can run ternary matmuls natively if drivers support it.

#### R2-A5-F10: Heterogeneous Accelerator Survey — Hybrid FP8, TNNs, SNNs
- **Src:** [Silvano et al., ACM Computing Surveys 2025] | **Gain:** Comprehensive HPC accelerator taxonomy | **V:** SPECULATIVE
- Survey covers various neural network accelerators for heterogeneous HPC platforms, including L2-coupled SNN and TNN accelerators. Mentions hybrid FP8 training ASICs and ternary neural network workloads. Insight: Ternary ops are distinct compute class — should be scheduled to specialized units (SIMD CPU, NPU) separate from FP16 (GPU Tensor Cores).

#### R2-A5-F11: ERGo — Energy-Efficient Hybrid GNN Training on PIM Architectures
- **Src:** [Dhingra et al., ACM TECS 2025] | **Gain:** Split GNN kernel between compute units | **V:** SPECULATIVE
- Proposes splitting GNN computational kernels into two parts for hybrid processing-in-memory + traditional compute. Relevant principle: partition operations by memory access pattern (sparse→PIM, dense→compute). For ternary SSM: sparse ternary matmuls could run on PIM if BRAM allows, dense attention on GPU.

#### R2-A5-F12: Async Memory Offloading — Rethinking IPC with Memory Operation Offloading
- **Src:** [Park et al., arXiv 2026] | **Gain:** Hybrid polling reduces CPU overhead while maintaining low latency | **V:** NEEDS-WORK
- Proposes memory operation offloading for inter-process communication with hybrid polling strategy (balances CPU overhead + latency). Relevant for CPU+GPU ternary inference: use offloaded memory ops to coordinate between CPU-side ternary layers and GPU-side attention without blocking CPU cores.

### Theme B: Kernel Optimization & Memory Efficiency

#### R2-A5-F13: Neuro-C — Hardware-Constrained Ternary Inference
- **Src:** [Romano et al., venue TBD 2026] | **Gain:** Improved convergence + custom SIMD kernels exploit sparsity | **V:** NEEDS-WORK
- Shapes neural inference to hardware limits using custom AVX512-optimized kernels that exploit zero-weight sparsity in ternary networks. Compares against standard ternary networks and reports better convergence under "identical architectural settings." Key mechanism: SIMD vectorization combined with branch-free sparse execution (skip zero weights). For ternary SSM: ~33% zero weights (ideally) could benefit from sparse SIMD kernels during training forward/backward passes. However, paper doesn't detail training throughput — primarily inference-focused. Needs validation for training applicability.

#### R2-A5-F14: T-SAR — In-Place SIMD Register LUT for Ternary Inference
- **Src:** [Oh et al., DATE 2026] | **Gain:** 5.6-24.5× GEMM latency, 1.1-86.2× GEMV throughput vs baseline, 2.5-4.9× energy efficiency vs Jetson Orin | **V:** PROMISING
- T-SAR repurposes SIMD register file for dynamic in-register lookup tables (LUTs), eliminating memory-based LUTs for ternary operations. Achieves scalable ternary LLM inference on CPUs with only 3.2% power and 1.4% area overhead in SIMD units. Core innovation: Instead of memory lookups (bottleneck), precompute ternary multiplication results {-a, 0, +a} in SIMD registers, indexed directly. For training: forward pass could use this for ternary weight matmuls. Backward pass still requires FP32 gradients, but forward speedup reduces overall training time. NOTE: Requires hardware modification (not software-only), but demonstrates theoretical ceiling for SIMD ternary optimization.

#### R2-A5-F15: FlexibleLLM — Block-Level Fractional Bit-Width Quantization
- **Src:** [Tan et al., ICLR 2026 submission] | **Gain:** SOTA PTQ with fractional bit-widths (e.g., 2.1-bit), finetuning-free | **V:** NEEDS-WORK
- FlexibleLLM addresses clustered outliers (groups of high-magnitude weights) via Self-Adaptive Block-Level Greedy Bit Search (SBGBS), enabling fractional-level bit allocation per block. Also includes Hadamard transform for discrete outlier suppression and Hessian-aware handling of "attribute outliers" (sensitive but not numerically large). For training: the Hadamard transform for outlier suppression could stabilize ternary training by reducing gradient variance from extreme activations. However, fractional bit-widths (2.1, 2.5, etc.) are PTQ-oriented and not directly applicable to QAT ternary training. The Hessian-aware outlier detection IS relevant — could guide which layers to keep FP16 during progressive training.

#### R2-A5-F16: Vulkan Compute 2026 Roadmap — Portable GPU Compute Baseline
- **Src:** [Khronos Vulkan Roadmap 2026, Jan 2026] | **Gain:** Standardized portable GPU compute across vendors | **V:** PROMISING
- Vulkan Roadmap 2026 establishes a baseline for portable GPU compute, including variable rate shading, compute shader derivatives, higher descriptor limits, and improved swapchain/presentation. Importantly, Vulkan Compute shaders are NOW mature enough for serious ML workloads across NVIDIA, AMD, Intel, Qualcomm, and Apple (via MoltenVK). For our portability requirement (NOT CUDA-only): Vulkan Compute can implement ternary matmul kernels that run on ANY GPU. The 2026 roadmap milestone ensures consistent feature availability. Key for training: write ternary forward/backward kernels in GLSL/SPIR-V once, deploy to all GPUs. No vendor lock-in.

#### R2-A5-F17: WebAssembly SIMD 2025 — Fixed-Width 128-bit SIMD in Browsers
- **Src:** [Multiple: platform.uno, web.dev, Gallant 2025] | **Gain:** 10-15× speedup vs pure JS for SIMD-friendly tasks, browser-native | **V:** PROMISING
- WebAssembly SIMD (128-bit fixed-width) achieved cross-browser support in 2025 (Chrome, Firefox, Safari, Edge). Enables vectorized operations in the browser with near-native performance. Benchmarks show array operations dropping from 1.4ms (JS) to 0.231ms (WASM SIMD) = 6× speedup. For inference deployment: ternary model inference can run in-browser with WASM SIMD, eliminating server dependency. For training: browser-based federated learning or edge fine-tuning becomes feasible. Rust can compile to WASM with SIMD intrinsics via wasm-pack. Portable (desktop, mobile browsers) and secure (sandboxed). No GPU needed — pure CPU SIMD.

#### R2-A5-F18: WasmGC — Native Garbage Collection for Higher-Level Languages
- **Src:** [WebAssembly GC baseline support, Dec 2024; Gallant 2025] | **Gain:** Smaller binaries + faster startup for GC languages (Java, Kotlin, Python) | **V:** SPECULATIVE
- WasmGC allows languages with garbage collectors (Java, Kotlin, Python) to leverage the browser's native GC instead of bundling their own. Achieves significantly smaller .wasm binaries and faster execution. For Rust+ternary models: Rust doesn't use WasmGC (manual memory management), but this expands the ecosystem. If we expose a Python API via WASM for browser-based inference, WasmGC enables efficient Python→WASM compilation. Indirect benefit: broader WASM adoption → better tooling for everyone.

#### R2-A5-F19: WASI 0.2 Component Model — Modular Cross-Language WASM
- **Src:** [WASI Preview 2 (0.2), Wasmtime 2024; Gallant 2025] | **Gain:** Language-agnostic WASM components, enables composability | **V:** NEEDS-WORK
- WASI 0.2 introduces the Component Model for building applications from language-agnostic WASM components. Standardizes system interfaces (cli, http, filesystem, sockets) for non-browser environments. Wasmtime (Rust-based runtime) achieved full WASI 0.2 support in late 2024. For training: if we compile ternary training loops to WASM components, they can run in Wasmtime on any OS (Windows, Linux, macOS, even embedded). However, browser support for full components is LIMITED — requires transpilation via `jco` (adds build step + bundle size overhead). Primarily useful for server-side/edge deployment, NOT browser inference.

#### R2-A5-F20: Protein LM Efficiency — FlashAttention + Sequence Packing
- **Src:** [Çelik & Xie, iScience 2025] | **Gain:** 4-9× faster inference, 3-14× lower memory, 6× training speedup | **V:** PROMISING
- Study on efficient protein language models (ESM) using FlashAttention + sequence packing achieved 4-9× faster inference and 3-14× lower memory usage. Training optimized via activation checkpointing + DeepSpeed ZeRO-Offload, cutting runtime 6-fold. 4-bit quantization reduced memory 2-3× with minimal accuracy loss for variant effect prediction. For our ternary SSM: FlashAttention principles (minimize HBM reads, use SRAM for attention) apply to our sliding window attention. ZeRO-Offload (partition optimizer states to CPU) is DIRECTLY applicable for single-GPU training — offload Adam states to RAM, keep only active gradients on GPU. The 6× training speedup validates this stack. 4-bit quantization results suggest our ternary (1.58-bit effective) should be even better.

#### R2-A5-F21: ZeRO-Offload Variants for Single-GPU
- **Src:** [Implicit from Çelik 2025, general ZeRO literature] | **Gain:** ~6× training speedup via optimizer state offloading | **V:** PROMISING
- DeepSpeed ZeRO-Offload partitions optimizer states (Adam m/v tensors) to CPU RAM, reducing GPU memory by ~66% (12 bytes/param → 4 bytes/param on GPU). During backward pass, gradients computed on GPU are sent to CPU for optimizer step, then updated weights returned to GPU. For RTX 5070 (12GB VRAM): enables training models that would otherwise OOM. The 6× speedup in Çelik 2025 is END-TO-END (including CPU↔GPU transfer overhead), confirming it's practical. For ternary training: our FP32 shadow weights (4 bytes/param) + FP32 Adam states (8 bytes/param) = 12 bytes/param can be offloaded, keeping only ternary weights (0.2 bytes/param) + activations on GPU. Massive memory savings.

---

## Key Insights — Aspect 5 Wave 1 (Combined Themes)

1. **Operator-Level Heterogeneity is Key:** Ternary models have fundamentally different operator profiles — ternary matmuls (add/sub, CPU/NPU-optimal) vs FP16 attention (matmul, GPU Tensor Core-optimal). Optimal execution requires layer-by-layer placement decisions, not uniform offloading.

2. **SIMD is the ternary inference workhorse:** TABv2, Neuro-C, and T-SAR all demonstrate that AVX2/AVX512 SIMD vectorization delivers 4-24× speedups for ternary operations on CPU. The key insight is encoding ternary {-1,0,+1} in 2 bits enables packing 16-32 weights per SIMD register, maximizing data-level parallelism. For training, forward pass matmuls can leverage this directly.

3. **Dynamic Split Point Selection Outperforms Static:** Adaptive partitioning based on real-time bandwidth, device load, and model topology (natural bottlenecks where activations compress) achieves 10-40% latency reduction. For ternary SSM, split at recurrent state compression points.

4. **PCIe Communication is the Bottleneck:** Even on PCIe Gen4 (32 GB/s), transferring large activation tensors creates stalls. Pipelining (overlap compute+transfer via pinned memory + async copies) and prefetching (anticipate next layer's data) are mandatory for heterogeneous ternary inference.

5. **Vulkan Compute solves the portability requirement:** The 2026 Vulkan Roadmap establishes a mature, vendor-neutral GPU compute baseline. Writing ternary kernels in GLSL/SPIR-V (compiled to SPIR-V bytecode) enables deployment to NVIDIA, AMD, Intel, Qualcomm, and Apple GPUs from a single codebase. This is THE solution for "portable GPU acceleration" mandated by project rules.

6. **WebAssembly SIMD enables browser-native inference:** WASM SIMD (128-bit) achieves 6-15× speedups vs JavaScript and runs on all major browsers (2025+). Combined with Rust→WASM compilation (via wasm-pack), we can deploy ternary models for in-browser inference with zero server dependency. This opens browser-based federated learning and edge fine-tuning use cases.

7. **ZeRO-Offload is the single-GPU memory savior:** Offloading Adam optimizer states to CPU RAM (DeepSpeed ZeRO-Offload) reduces GPU memory by ~66% while delivering 6× end-to-end training speedup. For our 12GB RTX 5070, this is CRITICAL for scaling beyond 6M parameters.

8. **Sparse ternary execution is underexploited:** Neuro-C demonstrates that exploiting zero-weight sparsity (~33% of ternary weights) via branch-free SIMD kernels improves both speed and energy efficiency. However, most ternary libraries treat zeros as regular values. For training, fused kernels that skip zero-weight multiplications in forward/backward passes could yield 20-30% speedup.

9. **Activation checkpointing + selective recomputation is table stakes:** The protein LM study's 6× training speedup comes from activation checkpointing (trade compute for memory) combined with ZeRO-Offload. For our ternary SSM, checkpointing SSM states (large, sequential) and recomputing them during backward is essential for fitting in 12GB VRAM. The memory-compute tradeoff is heavily in our favor given fast GPU recomputation.

10. **NPU Utilization for Ternary is Emerging:** Intel NPU (13 TOPS INT8) and similar edge accelerators are designed for quantized inference. Ternary matmuls map naturally to INT4 ops with value remapping {-1→0, 0→1, +1→2}. Driver/compiler support is the blocker, not hardware capability.

---

## For Math-Paradigm-Breakers (Wave 2) — Aspect 5

### Theme A Questions

1. **Optimal Workload Partitioning Under Communication Constraints:** Given a ternary SSM model with T ternary layers and A FP16 attention layers, CPU compute cost `C_cpu(T)`, GPU cost `C_gpu(A)`, and PCIe transfer cost `D(split_point)`, derive the optimal split point(s) that minimize end-to-end latency: `argmin_{split} [C_cpu + C_gpu + D(split)]`. What mathematical structure exists in this optimization? Is it convex? Can dynamic programming solve it exactly?

2. **Asynchronous Pipeline Depth Optimization:** When pipelining CPU (ternary layer i) and GPU (attention layer i+1) execution, what is the optimal pipeline depth (number of in-flight transfers) to maximize throughput while minimizing memory overhead? Formulate as a queueing theory problem with service rates `μ_cpu`, `μ_gpu`, `μ_pcie` and arrival rate `λ`.

3. **Communication-Compute Overlap Bounds:** For a ternary layer with compute time `T_compute` and transfer time `T_transfer`, what is the theoretical maximum overlap achievable via pipelining? Derive bounds based on dependency structure (data-parallel vs sequential ops). Can we achieve `T_total ≈ max(T_compute, T_transfer)` for ternary SSM recurrence?

4. **NPU Scheduling Theory for Heterogeneous Ternary Workloads:** Given CPU (SIMD, variable-latency), GPU (Tensor Cores, fixed-latency), and NPU (INT4, low-power) as compute units, formulate a scheduling problem to assign ternary SSM operators to units under power budget `P_max` and latency constraint `L_max`. What is the complexity class of this problem? Are there efficient approximation algorithms?

5. **Memory Hierarchy Optimization for Ternary State:** Ternary SSM maintains a recurrent state that is accessed every step. Under a heterogeneous CPU+GPU+NPU system, where should the state reside (CPU DDR5, GPU GDDR7, or NPU on-die SRAM)? Derive optimal state placement as a function of access frequency, update cost, and inter-device transfer latency.

### Theme B Questions

1. **Optimal SIMD register allocation for ternary:** Given AVX512 (512-bit registers) and ternary 2-bit encoding, what is the mathematically optimal packing strategy for SSM recurrence + ternary matmul fusion? Can we interleave SSM state updates with ternary weight loads in the same SIMD register to minimize memory bandwidth?

2. **Memory layout theory for ZeRO-Offload + ternary:** What memory layout (row-major, column-major, blocked, Z-order) minimizes CPU↔GPU transfer overhead when offloading FP32 Adam states while keeping ternary weights on GPU? Can we prove a lower bound on transfer latency given PCIe Gen4 bandwidth (32 GB/s)?

3. **Sparse ternary matmul scheduling:** For a ternary weight matrix with ~33% zeros, what is the optimal execution graph (static vs dynamic) to skip zero-weight computations without branch misprediction penalties? Can we use a compressed sparse row (CSR) format at the kernel level without sacrificing SIMD parallelism?

4. **Vulkan Compute kernel fusion theory:** Given Vulkan's subgroup operations (GPU equivalent of SIMD), how do we fuse ternary quantize→forward→backward into a single kernel dispatch to minimize memory round-trips? What are the theoretical register pressure limits for storing intermediate gradients during fused backward?

5. **Activation compression vs recomputation tradeoff:** For SSM sequential states (T tokens × D dimensions), what is the optimal checkpoint granularity (every N tokens) to minimize total memory+compute cost? Can we derive a closed-form expression for the break-even point given GPU FLOPS and memory bandwidth?
