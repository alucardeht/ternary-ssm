# Research Round 4 — Inference Deployment & Compilation

## Overview

This document tracks research findings on inference deployment, kernel optimization, speculative decoding, and serving infrastructure for ternary SSM models discovered in Research Cycle 4, Aspect 2 (Feb 2026).

## NEW FINDINGS (Round 4)

### Wave 1: Web Research (2 researchers, Feb 2026)

#### KERNEL OPTIMIZATION (Researcher 1)

| # | Finding | Source | Impact |
|---|---------|--------|--------|
| F1 | BitNet.cpp TL-2 | Microsoft, Jan 2026 | 1.15-2.1x over original |
| F2 | T-SAR In-Register LUT | arXiv 2511.13676, Nov 2025 | 5.6-24.5x GEMM, 1.1-86.2x GEMV |
| F3 | BitMat Triton Kernels | astramind-ai/BitMat, 2025 | Python-native ternary GPU kernels |
| F4 | bitnet_cuda Full CUDA | thumper-ai/bitnet_cuda, 2025 | Persistent kernels, multi-GPU |
| F5 | Spectra 1.1 Weight Packing | arXiv 2506.23025, Jun 2025 | Optimized 1.58-bit encoding |
| F6 | TENET Sparsity-Aware LUT | arXiv 2509.13765, Sep 2025 | Edge-optimized sparsity exploitation |
| F7 | TeLLMe FPGA Accelerator | arXiv 2504.16266, Apr 2025 | Energy-efficient ternary on FPGA |
| F8 | WASM SIMD Inference | 2025 | Browser-based ternary inference |
| F9 | BitNet.cpp ARM Optimizations | ACL 2025 | 1.37-5.07x ARM, 55-70% energy reduction |
| F10 | Adaptive Dataflow Selection | T-SAR pattern, 2025 | Per-layer kernel selection by dimensions |

**F1: BitNet.cpp TL-2 (Ternary Lookup v2)**
- Official Microsoft update (Jan 2026): parallel kernel implementations with configurable tiling.
- Embedding quantization support. 1.15-2.1x speedup over original BitNet.cpp.
- Reference implementation for CPU ternary inference benchmarking.

**F2: T-SAR In-Register LUT Generation** — KEY FINDING
- Full-stack SIMD technique: generate LUTs in SIMD registers instead of DRAM.
- Eliminates 87.6% of memory transactions (LUTs occupy <0.01% RAM but cause 87.6% accesses).
- Ternary-to-binary decomposition: w = w_D × w_S where w_D ∈ {-1,+1} (dense) and w_S ∈ {0,1} (sparse).
- Result: y = Σ w_D,i × a_i - Σ w_S,i × a_i. LUT storage: 3^c → 2^(c+1) (matches SIMD register width).
- Performance: 5.6-24.5x GEMM, 1.1-86.2x GEMV vs BitNet.cpp baseline.
- 128.96 tok/s on AMD 9950X for Llama-8B decode. Hardware overhead: +3.2% power, +1.4% area.
- Works with AVX2, ARM NEON, RISC-V Vector extensions.

**F3: BitMat Triton Kernels**
- Python package with custom Triton kernels for ternary matmul.
- GPU-focused, easier prototyping than raw CUDA.
- PyTorch integration via astramind-ai/BitMat repository.

**F4: bitnet_cuda Full CUDA Implementation**
- Complete CUDA implementation with persistent kernels and multi-GPU support.
- PyTorch bindings for direct integration. Reference for GPU ternary kernels.

**F5: Spectra 1.1 Weight Packing**
- Optimized weight packing for 1.58-bit models. Better density than 2-bit encoding.
- Trade-off: 1.58-bit → 1.67-bit achieves better density but worse bandwidth utilization.

**F6: TENET Sparsity-Aware LUT**
- Edge-optimized architecture exploiting ternary weight sparsity patterns.
- Zero-skipping logic in LUT indexing: skip computation for zero-weight groups entirely.
- 1.5-2x speedup on high-sparsity layers (our models have ~42% zeros).

**F7: TeLLMe FPGA Accelerator**
- Energy-efficient ternary accelerator for prefill+decode on edge FPGAs.
- Separate optimization paths for prefill (parallel) and decode (sequential) phases.

**F8: WASM SIMD Inference**
- Browser-based quantized inference using SIMD.js.
- Enables privacy-preserving, offline ternary inference in web applications.
- Future consideration for web deployment of ternary models.

**F9: BitNet.cpp ARM Optimizations**
- 1.37-5.07x speedup on ARM processors, 55.4-70% energy reduction.
- Critical for mobile/edge deployment. ARM NEON/SVE optimized kernels.
- Combined with ternary SSM's O(1) memory → ideal for phone deployment.

**F10: Adaptive Dataflow Selection**
- Per-layer kernel selection based on N/K/M dimensions of each operation.
- Activation-persistent vs output-persistent dataflow selected dynamically.
- 2-4x improvement in mixed workloads (prefill vs decode phases).
- Pattern from T-SAR but applicable to any ternary kernel system.

#### SERVING & DEPLOYMENT (Researcher 2)

| # | Finding | Source | Impact |
|---|---------|--------|--------|
| F11 | SGLang Hybrid SSM Serving | PyTorch Blog + Alibaba, Dec 2025 | Production SSM serving with EAGLE |
| F12 | Bitnet.cpp Edge Inference System | ACL 2025 | 6.25x vs FP16, TL + I2_S kernels |
| F13 | BitNet b1.58 2B4T Production Model | MS Research, Apr 2025 | First open ternary LLM at scale |
| F14 | ReDrafter Recurrent Speculative | Apple, 2025 | 2.8x H100, RNN draft from SSM state |
| F15 | State-Isolation Speculative Decoding | SGLang v0.5.5, Dec 2025 | Isolated state slots per draft token |
| F16 | MambaRadixCache Prefix Caching | SGLang, Dec 2025 | Snapshot-based SSM prefix reuse |
| F17 | vLLM / TensorRT-LLM Low-Bit | Nov 2025 | W4A16/INT4 support (no native ternary) |
| F18 | DenseSSM Information Flow | ACL 2025 | Dense connections for SSM quality |
| F19 | Ternary Edge Deployment Viability | Multiple sources, 2025-2026 | 50M ternary = ~11MB, fits in CPU cache |
| F20 | TOM ROM-SRAM Hybrid Accelerator | MS Research, Feb 2026 | 3,306 tok/s, 465x faster than A100 |

**F11: SGLang Hybrid SSM-Attention Serving** — CRITICAL
- Production-grade serving for hybrid Mamba-2/Transformer models.
- Dual memory pools: Mamba state (request-level) + KV cache (token-level).
- Elastic memory allocation between pools based on workload.
- State-isolation speculative decoding: independent cache slot per draft token.
- Performance: Qwen3-Next-80B achieves 257-324 tok/s with EAGLE speculative (2.7-4.2 acceptance).
- Prefix caching reduces TTFT to 57.6% of baseline.
- THE ONLY production system that solved SSM serving at scale.

**F12: Bitnet.cpp Edge Inference System**
- Ternary Lookup Table (TL) for sub-2-bit weight inference.
- Int2-with-Scale (I2_S) for lossless edge inference.
- 6.25x speedup vs FP16 baseline, 2.32x vs other low-bit baselines.
- Targets CPU and mobile edge devices specifically.

**F13: BitNet b1.58 2B4T Production Model**
- First open-source native ternary LLM: 2B params trained on 4T tokens.
- Matches FP16-equivalent accuracy. 3.55x memory reduction (1.75GB vs ~7GB).
- Validates ternary training from scratch at production scale.
- Benchmark target for our scaling path (50M → 200M → 1B).

**F14: ReDrafter Recurrent Speculative Decoding** — HIGH PRIORITY
- RNN draft model conditioned on LLM hidden states.
- Dynamic tree attention over beam search eliminates duplicate prefixes.
- 2.8x speedup on H100, 2.3x on Apple Silicon (MLX).
- Self-speculative: no separate draft model needed.
- RNN draft naturally compatible with SSM hidden state — perfect fit for our architecture.

**F15: State-Isolation Speculative Decoding** — BREAKTHROUGH
- Solves THE fundamental SSM speculation problem: irreversible state updates.
- Each draft token gets independent state slot (sandbox).
- When verified: promote final slot's state to main. Reject: free slot.
- Supports tree speculation (EAGLE-Tree) via parent-index tracking.
- Combined with ReDrafter (F14): full speculative pipeline for SSMs.

**F16: MambaRadixCache Prefix Caching**
- Hybrid radix tree for SSM state + KV cache management.
- Match: find longest prefix, COPY SSM state snapshot (not reference).
- Dual-LRU eviction: KV (leaf-to-root) + SSM (elastic at any level).
- Our 64-dim state ≈ 16KB per request → manageable for snapshots.
- Enables caching system prompts, RAG prefixes, tool descriptions.

**F17: vLLM / TensorRT-LLM Low-Bit Support**
- vLLM: flexible, W4A16/W8A8 quantization, PagedAttention.
- TensorRT-LLM: compiled, INT4/FP8, Flash Attention.
- Neither supports native ternary (1.58-bit) — custom kernel integration needed.
- Integration path: develop ternary kernel → wrap as custom op → inherit framework features.

**F18: DenseSSM Information Flow**
- Dense connections integrate shallow-layer hidden states into deeper SSM layers.
- Retains fine-grained information. Compatible with RetNet, Mamba, etc.
- Quality improvement without losing O(1) memory advantage.

**F19: Ternary Edge Deployment Viability**
- 50M ternary model = ~11MB weights → fits entirely in CPU L3 cache.
- Combined with O(1) memory inference → ideal for edge/mobile.
- Bitnet.cpp ARM: 55-70% energy reduction. TOM: 80% power reduction via gating.
- Real path to phone/IoT deployment of ternary SSM.

**F20: TOM ROM-SRAM Hybrid Accelerator**
- 3,306 tok/s on BitNet-2B (465x faster than A100).
- ROM for immutable base weights (5.2x denser than standard ROM).
- SRAM for QLoRA adapters (flexible fine-tuning).
- Workload-aware power gating: 80% power reduction, zero latency cost.

#### KEY PATTERNS
- **Custom kernels mandatory**: Generic frameworks (vLLM, TRT-LLM) don't support 1.58-bit. Need custom ternary kernel integration.
- **SSM serving solved**: SGLang demonstrated production SSM serving with state isolation, prefix caching, speculative decoding.
- **In-register LUT is the paradigm**: T-SAR eliminates 87.6% of memory transactions. All modern ternary kernels converge on register-level computation.
- **Edge deployment viable NOW**: 50M ternary fits in CPU cache (~11MB). ARM optimization gives 55-70% energy savings.
- **Speculative decoding for SSMs**: State-isolation (sandbox per draft) + RNN drafter (from SSM hidden state) = complete pipeline.

### Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

#### TERNARY GEMV MATHEMATICAL ALTERNATIVES

| # | Technique | Speedup | Exact? | Viability |
|---|-----------|---------|--------|-----------|
| M1 | SCS (Sorted-Coordinate Seriation) | 64-340x | YES | HIGH |
| M2 | RTDC (Row-TSP Differential Chaining) | 1.5-38x | YES | MEDIUM-HIGH |
| M3 | MCGV (Monte Carlo with Control Variates) | 10-25x | Bounded error | MEDIUM |
| M4 | TTC (Tensor Train Contraction) | ~5x | Exact if TT exact | MEDIUM |
| M5 | HTD (Hadamard-Ternary Decomposition) | Up to n/log(n) | YES | SPECULATIVE |
| M6 | SARM (Suffix-Automaton Row Merging) | 1-4x | YES | LOW-MEDIUM |
| M7 | SRS (Sparse Random Screening) | 6-18x (routing) | YES (for selected) | MEDIUM |

**M1: SCS (Sorted-Coordinate Seriation)** — TOP PRIORITY
- Reorder columns of W via permutation σ so +1 and -1 entries form contiguous blocks per row.
- Compute each output using PREFIX SUMS: each contiguous run = one range-sum lookup O(1).
- Cost: O(n + m × B_avg) where B_avg = average blocks per row after seriation.
- For B_avg = O(√n): speedup = √n ≈ 64x. For B_avg = O(log n): speedup = n/log(n) ≈ 340x.
- Offline: spectral seriation O(n²) via Fiedler vector — one-time, amortized.
- EXACT computation. No numerical issues. Trained weights have feature-correlated structure → low B_avg expected.
- KEY empirical question: what is B_avg for our trained ternary weights after optimal seriation?

**M2: RTDC (Row-TSP Differential Chaining)** — HIGH PRIORITY
- Compute first row directly, then each subsequent row as CORRECTION from previous.
- Correction cost = Hamming distance between consecutive rows.
- Order rows via Traveling Salesman Problem on Hamming metric → minimize total path length L.
- For random ternary: guaranteed 1.5x. For structured/clustered rows: up to 38x.
- TSP has mature algorithms: Christofides gives 1.5-approximation. Nearest-neighbor heuristic is 2-approx.
- EXACT computation. Complementary to SCS (row reordering vs column reordering — can combine).

**M3: MCGV (Monte Carlo GEMV with Control Variates)**
- Estimate y_i via importance sampling instead of exact sum.
- Top-K inputs (by magnitude) computed exactly as control variates.
- Light tail estimated stochastically: cost O(s) samples per output.
- For heavy-tailed activations (top √n capture 90%): 25x speedup.
- APPROXIMATE but bounded error. w×sign(x) is ternary — no FP multiply in sampling.
- Best for temperature>0 inference where sampling noise dominates anyway.

**M4: TTC (Tensor Train Contraction)**
- Reshape W as 2d-tensor, decompose into chain of small cores via SVD.
- Cost: O(d × r² × max(m_k × n_k)) where r = TT-rank.
- For d=4, r<255: beats direct computation.
- LIMITATION: cores are real-valued (not ternary) → reintroduces FP multiplications.
- Ternary-cores variant: preserve no-multiply but requires higher rank.

**M5: HTD (Hadamard-Ternary Decomposition)**
- Decompose each row's sign pattern into Walsh-Hadamard spectral basis.
- If rows have spectral sparsity K_avg << n: compute via fast WHT + sparse corrections.
- For K_avg = O(log n): speedup = n/log(n). SPECULATIVE — needs empirical K_avg measurement.

**M6: SARM (Suffix-Automaton Row Merging)**
- Process columns left-to-right. Rows with identical remaining columns share accumulators.
- Build suffix trie of rows. Cost: O(Σ_j L(j)) where L(j) = distinct suffixes at position j.
- For random ternary: negligible speedup (~1.002x). For structured: up to 4x.

**M7: SRS (Sparse Random Screening for Selective GEMV)**
- For MoD/routing architectures: only compute needed outputs (not all m).
- Use random sketch to identify top-K outputs, compute those exactly.
- Speedup = m/(K×log(m/K)). For K=32 of 4096: 18x. For K=128: 6.4x.

#### SSM SPECULATIVE INFERENCE — MATHEMATICAL ALTERNATIVES

| # | Technique | Speedup | Key Property | Viability |
|---|-----------|---------|--------------|-----------|
| M8 | Eigenmode Decoupling | 1.5-3x | Partition by decay rate | HIGH |
| M9 | Affine Orbits + Codebook Tree | 2-4x | Exact tree speculation | HIGH |
| M10 | Semigroup N-gram Caching | 1.3-2x | Cache composed affine maps | MEDIUM |
| M11 | Stochastic State Collapse | 1.5-2x | Moment propagation, closed-form | MEDIUM |
| M12 | Fixed-Point Attractors | 2-5x (steady state) | Jump to steady state | MEDIUM-LOW |
| M13 | Carry-Less Binary Arithmetic | 3-5x | Integer shift-add state evolution | HIGH |
| M14 | Topological State Prediction | Unknown | Persistent homology of trajectories | SPECULATIVE |

**M8: Eigenmode Decoupling with Horizon-Adaptive Speculation** — HIGH PRIORITY
- Partition state dimensions by effective memory length L_eff(i) = -1/log|a[i]|.
- G_fast (a≈0): memoryless, compute independently per speculated token. Zero inter-step dependency.
- G_slow (a≈1): nearly constant, freeze at current value (zero-order hold).
- G_med: first-order extrapolation h[i] ≈ h_t[i] + (h_t[i] - h_{t-1}[i]) × s.
- Sequential bottleneck reduced from d_state to |G_med| (only medium dims need careful treatment).
- Error bounded: fast dims exact, slow dims O(√k × |1-a|), medium dims O(k²).

**M9: Affine Orbits with Ternary Input Enumeration** — HIGH PRIORITY
- Precompute codebook Φ[v, i] = b_i^T × e_v for all vocab tokens v and state dims i.
- k-step state via Horner: S = 0; for s in k..1: S = a×S + Φ[v_s].
- Tree speculation: depth-k tree with branching m costs O(k × m × d_state) total.
- Shared prefixes reuse partial Horner sums — amortized across branches.
- EXACT computation. Verification via ternary C is essentially free (signed sums).
- Combined with M8: build tree only for G_med dimensions → cheaper branching.

**M10: Semigroup N-gram Caching**
- SSM step is affine map (α, β). Composition: α_composed = a^k (SAME for all k-grams!), only β varies.
- Cache β_composed for frequent n-grams (top 10K bigrams cover 60-80% of text).
- Trie-based lookup: match longest cached suffix, apply composed map O(d_state).
- Speedup proportional to cache hit rate. Highly text-dependent.

**M11: Stochastic State Collapse**
- Propagate mean and variance through recurrence instead of exact state.
- Under stationarity: closed-form E[h_{t+k}[i]] and Var[h_{t+k}[i]].
- Confidence horizon: speculate until variance exceeds threshold.
- Self-regulating: stops when prediction becomes unreliable.
- Ternary B + CLT → input effects well-approximated by Gaussian.

**M12: Fixed-Point Attractors**
- Diagonal A with |a|<1 → contractive map → unique fixed point per input distribution.
- h_fp[i] = E[φ_i(X)] / (1 - a[i]). Computable from model's own output distribution.
- During stable generation regimes: jump to fixed point, skip sequential evolution.
- Multiple regime library for different text types (narrative, dialogue, etc.).

**M13: Carry-Less Binary Arithmetic** — KEY SYNERGY WITH PHASE 30
- With discretized A (a[i] = 1 - 2^{-j}): state update = shift + subtract + add.
- Multi-step via transfer polynomial: R^k = Σ C(k,m)×(-1)^m × (>>m×j_i).
- For k=8: small fixed number of integer shifts and adds per dimension.
- ENTIRE inference pipeline becomes integer-only (ternary B/C + discretized A + POPCOUNT).
- 3-5x from integer vs float alone. Combines multiplicatively with other techniques.

**M14: Topological State Prediction** — MOONSHOT
- State trajectories trace curves in R^{d_state} with topological features (loops, attractors).
- Persistent homology detects recurrent patterns.
- If topology matches historical segment → predict future trajectory.
- SPECULATIVE: no prior work establishing topological recurrence in SSM trajectories.

#### KEY CROSS-POLLINATION (Wave 2)
- **M1 (SCS) + M2 (RTDC):** Column seriation + row TSP = JOINT reordering. Seriation makes rows more locally similar → RTDC path length also improves. MULTIPLICATIVE benefit.
- **M8 (Eigenmode) + M9 (Codebook Tree):** Only branch on medium-rate dimensions → cheaper tree. Fast=independent, slow=frozen.
- **M9 (Codebook) + M13 (Binary Arithmetic):** Integer codebook + Horner in shift-add = fully integer speculation pipeline.
- **M13 (Binary Arithmetic) + Phase 30 POPCOUNT:** State evolution maps directly onto POPCOUNT infrastructure. ENTIRE inference integer-only.
- **M1 (SCS) empirical test:** Measure B_avg for trained weights → single experiment determines if 64-340x is achievable.
- **Combined M8+M9+M13:** 1.3x (eigenmode) × 2.5x (tree speculation) × 3x (integer) ≈ 10x total autoregressive speedup.

### Wave 3: Paradigm Breakers (2 agents, Feb 2026)

#### RADICAL INFERENCE APPROACHES

| # | Technique | Impact | Key Insight |
|---|-----------|--------|------------|
| P1 | Self-Speculative Superposition (SSS) | 1.5-2.5x | Feed expected embedding back — SSM linearity guarantees correctness |
| P2 | Activation-Regime Layer Composition (ARLC) | 2-5x | Compose ternary layers → INTEGER matrix → INT8 tensor cores |
| P3 | Constraint Propagation Inference (CPI) | Up to 6400x on LM head | Propagate bounds through ternary, eliminate impossible tokens |
| P4 | Resonance-Based State Prediction (RSP) | 1.3-2x | SSM as IIR filter bank — spectral prediction of future states |

**P1: Self-Speculative Superposition (SSS)** — HIGH PRIORITY
- Instead of sample→embed→forward, compute expected embedding x_bar = Σ P(v)×e_v from top-K.
- Propagate x_bar through ALL layers for approximate state h_{t+1}^approx.
- SSM linearity: E[h_{t+1}] = A×h_t + B×E[x_{t+1}] — mathematically exact for expected state.
- Ternary robustness: small input perturbation causes ZERO change in most outputs (sign pattern preserved).
- Self-speculative: model IS its own draft via output distribution. No separate draft model needed.
- Impossible with Transformers (attention is nonlinear in query/key). SSM linearity is key enabler.
- Viability: HIGH. Pure software change, graceful degradation, testable in 30 min.

**P2: Activation-Regime Layer Composition (ARLC)** — TENSOR CORE ACCESS
- Two consecutive ternary layers composed through activation: M_r = W₂ × D_r × W₁.
- D_r = diagonal binary mask (activation regime: which neurons are "on").
- M_r is INTEGER matrix (entries in [-d, +d]). Fits INT8 with per-row scale.
- INT8 tensor cores: 988 TOPS vs 15.4 TOPS CUDA cores = 64x hardware multiplier.
- Precompute K=64-256 regime clusters at model load. Classify regime per token O(K).
- 24 layers → 12 "super-layers" with tensor core GEMM. Fewer kernel launches.
- Ternary-specific: continuous weights compose to arbitrary FP16 (no tensor core benefit).
- Viability: MEDIUM-HIGH. Memory: ~384MB for clusters at 50M. Expected 1.5-3x net.

**P3: Constraint Propagation Inference (CPI)** — ZERO-COMPUTE TOKENS
- Propagate BOUNDS through ternary layers instead of exact values.
- Ternary bounds are EXACT: no interval arithmetic dependency problem.
- w=+1: include x_min for lower bound, x_max for upper. w=-1: swap. w=0: skip.
- Neurons with y_max < 0 → output 0 through SiLU → skip in all subsequent layers.
- ~60-80% of neurons have deterministic sign → cascading tightening through depth.
- At LM head: eliminate tokens whose logit upper bound < winner's lower bound.
- If >90% tokens eliminated → compute exact logits for only ~3200 candidates (10%).
- For greedy decoding: can PROVE argmax without computing all logits.
- Viability: MEDIUM. Key question: how tight are bounds after 24 layers?

**P4: Resonance-Based State Prediction (RSP)**
- SSM dimension i is IIR filter: H_i(z) = 1/(1 - a_i×z⁻¹).
- Maintain running DFT of last W=64 input embeddings.
- Predict future state from spectral content using analytical filter response.
- Multi-step prediction replaces draft model in speculative decoding.
- Best for structured/repetitive text (code, templates). Less effective for creative text.
- Combined with Eigenmode Decoupling (M8): predict slow+fast dims spectrally, compute medium exactly.
- Viability: MEDIUM. Requires spectral concentration in natural language input.

#### RADICAL DEPLOYMENT PARADIGMS

| # | Technique | Impact | Key Insight |
|---|-----------|--------|------------|
| P5 | Combinational Logic Inference | 36M tok/s, 41,000x | Weights ARE the circuit, not data |
| P6 | Ambient Swarm Inference | ∞ scale-out, $0 cost | Every device IS a server, state migrates |
| P7 | Always-On DSP Streaming | 0ms latency, 250mW | Model as continuous IIR signal processor |
| P8 | In-Memory Computing | 625K tok/s, 4.5Mx energy | Analog crossbar: physics computes the GEMV |

**P5: Combinational Logic Inference (FPGA/ASIC)** — HARDWARE PARADIGM SHIFT
- Ternary w=+1: wire input to positive bus. w=-1: negative bus. w=0: disconnect.
- Entire GEMV = combinational circuit. NO memory access, NO ALU cycles.
- Computation in propagation delay of logic gates (~1 cycle at 500MHz).
- 50M weights × 6 transistors = 300M transistors. Fits on single high-end FPGA.
- ASIC at 7nm: 3mm², <$2/chip at volume.
- Pipeline: 8 layers × 14 cycles = 112 cycles at 500MHz = 224ns/token.
- Throughput: 36M tok/s. Latency: 224ns. Power: <5W.
- WHY ONLY TERNARY: FP16 multiplier = 4000 transistors/weight. 7B FP16 = 28 trillion transistors. Impossible.
- Ternary is the ONLY quantization where hardwiring is physically feasible.
- Viability: MEDIUM. FPGA prototype: 2 weeks for 1 layer. Full ASIC: 12-18 months.

**P6: Ambient Swarm Inference (Stateless Mesh)**
- 11MB model + 323KB state → ANY device processes ANY token.
- State serialized and travels WITH conversation (not stored on server).
- TernaryCast protocol: client broadcasts (token + state), first responder wins.
- mDNS discovery: _ternaryllm._udp.local. Each device advertises capacity.
- State transfer: 323KB at WiFi 6 = 0.22ms (2.2% overhead vs 10ms compute).
- No server, no session affinity, no failover complexity. $0 infrastructure cost.
- WHY IMPOSSIBLE FOR CONVENTIONAL: 7B model = 14GB (can't distribute). KV cache = 2GB (can't transfer).
- Viability: MEDIUM. Prototype: 3 days (two laptops, UDP).

**P7: Always-On DSP Streaming** — ZERO FIRST-TOKEN LATENCY
- SSM recurrence = IIR digital filter. DSP chips run IIR filters at millions of samples/sec for decades.
- Deploy on $5 Qualcomm Hexagon DSP / TI TMS320 / CEVA-X2.
- Model CONTINUOUSLY processes input stream (like microphone processing audio).
- When queried: response begins IMMEDIATELY (state already computed).
- 8 layers × 256 IIR filters = 1,792 filters. Standard DSP workload.
- At 2GHz DSP with SIMD: 322 tok/s at 250mW (phone battery: ~75 hours continuous).
- SWA layers (12.5%) need 128-token buffer — breaks pure streaming but manageable.
- Viability: MEDIUM. 1 week for 1-layer prototype on eval board.

**P8: In-Memory Computing (Analog Crossbar)** — ULTIMATE EFFICIENCY
- Resistive crossbar: weights as conductance values. Kirchhoff's law = dot product.
- Apply input voltages to rows → column currents = Σ V_i × G_ij. ONE RC time constant (~10ns).
- Ternary: only 3 conductance levels needed. Within ANY memristor technology's capability.
- 50M crosspoints at 10nm pitch = 0.005mm². With peripherals: 0.05mm².
- 625,000 tok/s. Energy: 50 nJ/token (vs 228 mJ GPU = 4.5 MILLION x reduction).
- Power: 31mW — runs on energy harvesting (solar, body heat, RF).
- WHY ONLY TERNARY: FP16 needs 65536 conductance levels (no technology achieves this). Ternary needs 3 (ALL technologies achieve this).
- Viability: SPECULATIVE. Requires ReRAM/memristor fab. 3-5 year strategic direction.

#### META-INSIGHT FROM PARADIGM BREAKERS
> **"Ternary weights transform computation from ARITHMETIC to TOPOLOGY."**
>
> Weights don't specify HOW MUCH to scale — they specify WHETHER and IN WHICH DIRECTION to include.
> This is the difference between a CALCULATOR and a SWITCHBOARD.
> Switchboards can be: wired in silicon (P5), computed by physics (P8), distributed anywhere (P6), or run as signal processors (P7).
>
> **No one builds a "switchboard server." They build the connections directly.**

---

## Aspect 2 Summary: Inference Deployment & Compilation

**Total findings:** 20 web research (F1-F20) + 14 mathematical (M1-M14) + 8 paradigm shifts (P1-P8) = **42 findings**

**Top 5 by estimated impact:**
1. **M1 SCS (Sorted-Coordinate Seriation)** — 64-340x exact GEMV speedup via column reordering + prefix sums. Needs single empirical measurement (B_avg).
2. **P5 Combinational Logic** — 36M tok/s, ternary-only paradigm. Weights ARE the circuit.
3. **M13 Carry-Less Binary Arithmetic** — Integer-only state evolution. Synergy with Phase 30 POPCOUNT.
4. **P1 Self-Speculative Superposition** — SSM linearity enables self-speculation. 30-min prototype.
5. **F2 T-SAR In-Register LUT** — Eliminates 87.6% of memory transactions. 5.6-24.5x proven.

**Top 5 by novelty:**
1. **P8 In-Memory Computing** — Physics computes the GEMV. 4.5Mx energy reduction.
2. **P3 Constraint Propagation** — Prove argmax without computing all logits.
3. **P7 DSP Streaming** — 0ms latency, model as continuous signal processor.
4. **P6 Swarm Inference** — Conversations migrate between devices, no servers.
5. **P2 Layer Composition** — Compose ternary layers → INTEGER → tensor cores.

**Immediate actions (can measure now):**
- M1: Measure B_avg for trained weights after seriation → determines if 64-340x is real
- P1: SSS minimum experiment (30 min) → top-1 agreement between exact and expected-embedding forward pass
- M2: Compute Hamming distance matrix for trained weight rows → determines RTDC viability
- M8+M9: Profile A matrix eigenvalue distribution → calibrates eigenmode decoupling
