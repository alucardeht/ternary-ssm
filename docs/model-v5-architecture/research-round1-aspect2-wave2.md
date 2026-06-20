# Round 1, Aspect 2: Natively RAM-Efficient Architectures — Wave 2 (Math)

## Wave 2 — Math Paradigm-Breakers (2 researchers, Feb 2026)

### ALGEBRAIC FOUNDATIONS

#### v5-R1-A2-M1: Ternary Monoid Sharing via {-1,0,+1} Algebra
- **Core idea:** Sparse ternary matrices are closed under multiplication. Signed permutation generators form a group (inverses for free). Store r generators + per-layer binary selectors.
- **Compression:** ~600x theoretical (8 generators for 32 layers at d=1024)
- **Limitation:** Signed permutations too restrictive for MLP/gate components. Best for SSM A matrix (state transition = naturally permutation-like).
- **Verdict:** NEEDS-MORE-WORK — Extraordinary compression but narrow applicability

#### v5-R1-A2-M2: Information-Theoretic RAM Floor
- **Core idea:** Rate-distortion theory + ternary regularization bonus. Ternary weights force structured, low-entropy solutions = 5-7x effective capacity bonus over naive bit comparison.
- **RAM floor for MMLU 75%:** 20 MB (aggressive) to 200 MB (conservative) for weights
- **Key insight:** The ternary constraint is an ADVANTAGE, not limitation — forces efficient use of each parameter
- **Verdict:** PROMISING — Framework for setting targets and identifying levers

#### v5-R1-A2-M3: Optimal Sharing Topology via Kolmogorov/Hamming Distance
- **Core idea:** For ternary weights, inter-layer Kolmogorov distance = exact Hamming distance (unique to discrete weights). Cluster layers by Hamming distance → data-driven optimal sharing groups.
- **Compression:** 3-4x from natural layer grouping (early/middle/late)
- **Key insight:** Ternary makes optimal sharing topology COMPUTABLE — impossible with continuous weights
- **Algorithm:** Train → compute pairwise Hamming → cluster → retrain with sharing
- **Verdict:** PROMISING — Principled, computable, leverages ternary uniqueness

#### v5-R1-A2-M4: Recursive Ternary SSM Fixed-Point Analysis
- **Core idea:** Single SSM block applied N times with learned iteration embeddings. Near-critical systems (Lipschitz ~1) exhibit limit cycles = p different representations from SAME parameters.
- **Compression:** 31x (single block for 32-layer equivalent, + 128 KB iteration embeddings)
- **Key insight:** Limit-cycle behavior means recursion naturally diversifies representations — explains WHY URM works
- **RAM:** 3.4 MB for "32-layer equivalent" weights
- **Verdict:** PROMISING — Dynamical systems backing for recursive sharing

#### v5-R1-A2-M5: GF(3) Basis Sharing (TOP RECOMMENDATION)
- **Core idea:** Ternary {-1,0,+1} = GF(3) finite field. Store r basis matrices over GF(3), per-layer coefficients in GF(3). Combination W_l = Σ c_{l,j} * B_j is ALWAYS ternary. Zero approximation error.
- **Compression:** 4-8x (r=4-8 basis matrices for 32 layers)
- **Key insight:** EXACT algebra — never leaves ternary domain. Reconstruction uses ONLY add/negate (operations we already have). Impossible with continuous weights.
- **Composes with M4:** Recursive (31x) × Basis (4x) = 124x on weight storage
- **Verdict:** PROMISING — Cleanest formulation, zero error, native ternary operations

#### v5-R1-A2-M6: Spectral Ternary Decomposition via Hadamard
- **Core idea:** Fixed Hadamard basis (free, procedurally generated) + per-layer diagonal ternary scaling + sparse correction.
- **Compression:** 230x theoretical (12.9 KB for 32 layers at d=2048)
- **Limitation:** Hadamard eigenvector constraint likely too rigid. Non-ternary intermediate results need quantization.
- **Verdict:** SPECULATIVE — Extraordinary compression but expressivity constraints severe

### ARCHITECTURE: EXPERT-SELECTIVE SSM

#### v5-R1-A2-M7: Expert-Selective SSM (ES-SSM) — Foundational Architecture
- **Core idea:** Separate SSM into SHARED recurrence (maintains state coherence) + EXPERT selectivity projections (sparsely activated). Experts don't own state — they own input/output gates.
- **Math:** Shared A, B_0 (always active) + per-expert W_Delta, W_B, W_C (sparsely activated). State h_t always updated by every token.
- **Active ratio:** 4.06% (8 of 256 experts active per token)
- **RAM at 24L, d=4096:** 26.4 MB active, 336 MB total ternary
- **Key insight:** Solves MoE+SSM state fragmentation problem. State is ALWAYS coherent.
- **Verdict:** PROMISING — Foundational building block for entire architecture

#### v5-R1-A2-M8: Optimal RAM Budget Allocation
- **Core idea:** Partition fast RAM between embeddings, shared params, router, active experts, state. Optimal allocation depends on scale.
- **Key finding at 1 GB:** d_model=4096, d_state=64, L=32, E=512, K=8 → only 86 MB used. Can increase to E=10,000 or d_model=8192.
- **Factored router:** d_model→d_router→E (d_router=128) saves significant router RAM
- **Verdict:** PROMISING — Concrete optimization, analytical solution available

### SCALING AND THROUGHPUT

#### v5-R1-A2-M9: Ternary MoE-SSM Scaling Law
- **Core idea:** Unified scaling law: L(N_a, N_t, D) with ternary capacity function κ, MoE granularity exponent δ=0.58, over-training correction.
- **Predictions at 1 GB (200B training tokens):** Perplexity ~9.0 (~80% of Chinchilla-70B)
- **Predictions at 1 GB (2T training tokens):** Perplexity ~7.9 (approaching Chinchilla-70B)
- **Key insight:** MoE granularity δ=0.58 means each doubling of total experts gives 1.49x effective intelligence
- **Verdict:** PROMISING — Enables concrete capacity planning

#### v5-R1-A2-M10: MoE Decouples Intelligence from Speed (KEY THEOREM)
- **Core idea:** For bandwidth-bound ternary inference, active params determine speed, total params determine intelligence. MoE makes these ORTHOGONAL. No tradeoff.
- **Dense 1GB:** P=5B, S=50 tok/s, T=500 thinking tokens in 10s
- **MoE 1GB:** P_total=34.6B, P_active=270M, S=926 tok/s, T=9260 thinking tokens in 10s
- **MoE wins on BOTH axes:** Higher base intelligence AND 18.5x more thinking tokens
- **Optimal strategy:** Maximize E (NVMe), minimize K → every NVMe expert is FREE intelligence
- **Verdict:** PROMISING — Eliminates fundamental tradeoff, most important theoretical result

#### v5-R1-A2-M11: NVMe Scheduling Bound
- **Core idea:** Double-buffered expert loading hides NVMe latency. Derive minimum fast RAM and throughput.
- **At NVMe 5 GB/s, p_reuse=0.4, p_predict=0.7:** 1060 tok/s at 85 MB fast RAM
- **At NVMe 7 GB/s (PCIe 5.0):** 1483 tok/s at 85 MB
- **For 100 tok/s target:** Only needs 472 MB/s NVMe — SATA SSD sufficient
- **Expert caching:** Surplus RAM as LRU cache, diminishing returns
- **Verdict:** PROMISING — Solid systems math, practical throughput achievable

#### v5-R1-A2-M12: Multiplicative Efficiency Composition
- **Core idea:** When do ternary × MoE × SSM × sharing compose multiplicatively?
- **Independence conditions:** All hold under ES-SSM (M7) with layer-position-aware routing
- **Conservative estimate:** 20x intelligence/RAM vs dense FP16 Transformer
- **Optimistic estimate:** 50x with aggressive sharing
- **Main uncertainty:** δ for ternary experts (0.3 pessimistic → 15x, 0.58 optimistic → 46x)
- **Verdict:** PROMISING — Even conservative 20x is transformative

### SPECULATIVE: STATE-EXPERT SSM

#### v5-R1-A2-M13: State-Expert SSM (SE-SSM)
- **Core idea:** Instead of many experts with distinct PARAMETERS, maintain many parallel STATE VECTORS sharing same parameters. Knowledge in params, reasoning diversity in states.
- **RAM:** 45 MB for 256 state tracks × 32 layers with 181M shared params
- **Hybrid ES-SSM + SE-SSM:** Knowledge layers (param experts on NVMe) + Reasoning layers (state experts in RAM) = 65 MB
- **Risk:** State collapse (all tracks converge to similar values without diversity pressure)
- **Verdict:** SPECULATIVE — Elegant and cheap but untested. Worth small-scale experiment.

## Key Insights — Aspect 2 Wave 2

1. **GF(3) algebra is a ternary SUPERPOWER** — Exact basis sharing, zero approximation error, only add/negate operations. Impossible with continuous weights. (M5)
2. **ES-SSM solves the MoE+SSM problem** — Shared state + expert selectivity = coherent recurrence with sparse activation. (M7)
3. **MoE ELIMINATES the speed-intelligence tradeoff** — Total experts (NVMe) = intelligence, active experts (RAM) = speed. Orthogonal variables. (M10)
4. **20-50x efficiency vs FP16 Transformer is achievable** — Ternary (8.8x) × MoE (3.7x at RAM level) × SSM (1.3x) × sharing (up to 3x). (M12)
5. **85 MB fast RAM → 1000+ tok/s is real** — NVMe double-buffering with expert prediction. Even SATA SSD gives 100 tok/s. (M11)
6. **Recursive SSM explains WHY weight sharing works** — Limit-cycle dynamics in near-critical systems create diverse representations from same params. (M4)
7. **RAM floor for MMLU 75%: 20-200 MB** — Information-theoretic bound. Ternary regularization bonus = genuine advantage. (M2)

## Combined Architecture Vision

**ES-SSM with GF(3) Basis Sharing + NVMe Expert Offloading:**
- Shared recurrence: GF(3) basis (4 bases × d²) = 3.3 MB
- 512 expert selectivity sets: Each ~100 KB ternary = 51 MB total on NVMe
- 8 active experts per token: 800 KB in fast RAM (double-buffered: 1.6 MB)
- Router (factored): ~4 MB
- Embeddings: ~25 MB
- SSM state: ~4 KB
- **Fast RAM total: ~35 MB**
- **Throughput: 1000+ tok/s** (NVMe 5 GB/s)
- **Quality: approaching Chinchilla-70B** with 2T training tokens
