# Round 1, Aspect 3: Compute-on-Demand & Adaptive Architectures

## Wave 1 — Web Research (2 researchers, Feb 2026)

### STREAMING & OFFLOADING

#### v5-R1-A3-F1: FlexGen — Block-Scheduled Offloading
- **Source:** Stanford et al., arXiv:2303.06865, 2023 (updated 2024)
- **Summary:** Aggregates GPU/CPU/disk memory via LP-optimized block scheduling. 175B on single 16GB GPU (1 tok/s batch=144). Streaming weights in blocks amortizes I/O across batches.
- **Quality:** SAME (lossless)
- **RAM gain:** 27x intelligence-to-RAM ratio (175B/16GB vs conventional 7B/16GB)
- **Throughput:** Low for interactive (1 tok/s at 175B), high for batch (7.3 tok/s at 30B)
- **For us:** Block scheduling concept applies to ternary expert streaming from NVMe.
- **Verdict:** PROMISING (batch; architecture concept useful)

#### v5-R1-A3-F2: Speculative Decoding with Draft Models
- **Source:** Song et al., arXiv:2511.11733, Nov 2025 + EAGLE-3 (NAACL 2025, NVIDIA TensorRT integration)
- **Summary:** Tiny draft model (~200M params, 400 MB) proposes tokens, full model verifies in parallel. EAGLE-3: multi-layer fused features, 111% higher throughput. 80% token acceptance rate. Rejection sampling = LOSSLESS.
- **Quality:** SAME (mathematically lossless via rejection sampling)
- **RAM gain:** Draft in RAM (400 MB), target streams from NVMe. 70B effective intelligence with <2 GB resident.
- **Throughput:** 2-3x over autoregressive baseline. 4 tokens avg per verification cycle.
- **For us:** Ternary SSM draft model could be TINY (~50 MB ternary). Target model streams expert params from NVMe. Draft predicts expert routing = expert prefetching for free.
- **Verdict:** PROMISING — Best-in-class speedup, zero quality loss, production-ready

#### v5-R1-A3-F3: Layer Streaming on Unified Memory
- **Source:** oLLM library, Feb 2026; Strix Halo APU benchmarks
- **Summary:** Layer-by-layer streaming with async prefetch. On unified memory (Apple Silicon, AMD APU): 75% offload = 1.75x FASTER than full GPU (counterintuitive — less VRAM contention). Per-layer load: 4ms DDR5, 108ms NVMe.
- **Quality:** SAME (bit-identical)
- **RAM gain:** Run 70B with 30% in VRAM. Ternary layers are 16x smaller → 6ms NVMe load vs 108ms.
- **Throughput:** Unified memory: 3.36 tok/s (75% offloaded) > 1.92 tok/s (0% offloaded). Discrete GPU: -33% penalty.
- **For us:** Ternary weights + unified memory = sweet spot. 16x faster layer streaming. Mobile/edge deployment.
- **Verdict:** PROMISING — Counterintuitive gain on unified memory, ternary amplifies advantage

### ADAPTIVE COMPUTATION

#### v5-R1-A3-F4: Mixture of Depths (MoD)
- **Source:** Raposo et al., Google DeepMind, arXiv:2404.02258, Apr 2024
- **Summary:** Tokens dynamically route through blocks OR skip via residual. Learned top-k selection with static budget (e.g., 12.5% tokens per layer). Matches baseline at 50% faster inference, or +1.5% better at same FLOPs. Trained from scratch (60M-3B).
- **Quality:** SAME to BETTER (+1.5% at isoFLOP)
- **RAM gain:** Fewer active tokens per layer = smaller intermediate state. Static compute graph = hardware-friendly.
- **Throughput:** 50% faster sampling, 60% faster training
- **For us:** "MoD for SSM" — tokens skip state updates when predictable. Zero state modification for easy tokens. Static k budget = predictable memory. COMBINES with entropy-gating (P2).
- **Verdict:** PROMISING — Proven at scale, quality SAME/BETTER, directly applicable to SSM

#### v5-R1-A3-F5: Phase-Level Early Exit (AdaInfer)
- **Source:** Fan et al., 2024; emergentmind.com analysis
- **Summary:** Track logit stability across layers. Exit when top-prob + gap exceeds threshold. SVM classifier on logit features. 43% layer pruning on easy tasks, <1% accuracy loss. 50% FLOP reduction on mixed workloads.
- **Quality:** SAME (<1% accuracy loss on easy tasks, full depth on hard)
- **RAM gain:** Only active layers loaded. 43% avg pruning → 2.3x effective intelligence-to-RAM.
- **Throughput:** 1.8-2x on mixed workloads
- **For us:** SSM early exit = stop when state stops changing (low delta norm). Ternary weights make exit detection trivial (discrete outputs stabilize clearly).
- **Verdict:** PROMISING — Proven quality preservation, ternary amplifies exit detection

#### v5-R1-A3-F6: Mamba-Shedder — SSM-Specific Structured Pruning
- **Source:** Mamba-Shedder, Jan 2025
- **Summary:** 50% SSM weight pruning = zero accuracy loss. Input-dependent gating (σ(W_g·x)) suppresses/activates state updates per token. O(LD) complexity maintained.
- **Quality:** SAME (zero loss at 50% pruning)
- **RAM gain:** 11.5% memory reduction. Pruned SSM = smaller footprint.
- **Throughput:** 1.14x speedup
- **For us:** Ternary SSM can use binary gates (0 or 1) to skip/include state updates. Even simpler than continuous gating. Gate itself can be ternary.
- **Verdict:** PROMISING — Validates that SSMs have massive redundancy. 50% prunable with zero loss.

### TEST-TIME INTELLIGENCE SCALING

#### v5-R1-A3-F7: First-Finish Search (FFS) — Efficient Test-Time Compute
- **Source:** arXiv:2512.02008, Dec 2025; Raschka Jan 2026 survey
- **Summary:** Sample N outputs, majority-vote on the k shortest. "Short-horizon" models (concise reasoning) benefit from FFS-1@N = 90% fewer tokens. Optimal strategy depends on model type: short-horizon → FFS, long-horizon → majority-vote.
- **Quality:** SAME to BETTER (15% → 52% accuracy via inference scaling alone)
- **RAM gain:** FFS reduces sequential tokens = lower peak generation memory. 90% token reduction.
- **Throughput:** 90% fewer tokens per output = massive speedup for short-horizon models
- **For us:** Train ternary SSM as SHORT-HORIZON (reward concise reasoning) → naturally lower RAM/latency. FFS layered on top = free quality boost.
- **Verdict:** PROMISING — Zero training cost, applies on top of any model

#### v5-R1-A3-F8: HALT-CoT — Adaptive Early Stopping
- **Source:** Referenced in arXiv:2512.02008, 2025
- **Summary:** Estimate answer uncertainty during generation. Halt early when confidence is high. Internal compute modulation — model decides when to stop reasoning.
- **Quality:** SAME (stops when confident)
- **RAM gain:** Fewer tokens generated = lower peak memory
- **For us:** Ternary SSM state stabilization = natural confidence signal. When ternary state stops changing between tokens, model is confident. Binary detection: delta_h == 0 in ternary space.
- **Verdict:** PROMISING — Ternary makes confidence detection trivial

## Key Insights — Aspect 3

1. **MoD for SSM is the key combination** — Skip state updates for predictable tokens. Static budget = hardware-friendly. Quality SAME/BETTER. (F4)
2. **Speculative decoding with ternary draft** — Tiny ternary draft (~50 MB) + full model streaming = 70B intelligence at <2 GB resident. Draft predicts routing = free expert prefetch. (F2)
3. **Unified memory is a ternary superpower** — 16x smaller layers → 16x faster streaming. On Apple Silicon/AMD APU: offloading is FASTER than full GPU. (F3)
4. **SSMs are 50% redundant** — Mamba-Shedder proves 50% pruning = zero loss. Massive room for conditional computation. (F6)
5. **Train short-horizon + FFS = free intelligence** — Concise reasoning models + first-finish search = 90% fewer tokens at same quality. Zero extra training. (F7)
6. **Ternary makes exit/confidence detection trivial** — In discrete space, "state stopped changing" is EXACT (delta == 0), not approximate. (F5, F8)

## For Our Project

**Compute-on-demand stack (combines with A1/A2 findings):**
- ES-SSM with MoD routing (F4): tokens skip blocks they don't need
- Speculative decoding (F2): ternary draft proposes, full model verifies
- Early exit (F5): stop when ternary state stabilizes
- FFS at inference (F7): sample concise answers, vote on shortest
- Layer streaming (F3): unified memory deployment for mobile/edge

**This stack is ORTHOGONAL to the compression stack (GF(3), Kronecker, PLA) — they multiply.**
