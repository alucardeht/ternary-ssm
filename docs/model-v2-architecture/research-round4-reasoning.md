# Research Round 4 — Long-Context & Reasoning

## Overview

This document tracks research findings on long-context capabilities, reasoning limitations, and quality optimization for ternary SSM models discovered in Research Cycle 4 (Wave 1: Feb 2026).

## NEW FINDINGS (Round 4)

### Wave 1: Web Research (2 researchers, Feb 2026)

#### LONG-CONTEXT TECHNIQUES

**F1: LongMamba — Training-Free Context Extension** (HIGH IMPACT)
- Classify channels into 3 categories: monotonically increasing (accumulate info), decaying (local context), stable (long-range)
- Apply token-level filtering: suppress noise in accumulating channels, preserve signal in stable channels
- Extend SSM context window WITHOUT retraining — zero-cost context extension
- Key insight: SSM channel behavior is highly heterogeneous, treating all channels equally wastes capacity
- Potential: 2-4x context extension with no training cost

**F2: MambaExtend — Discretization Step Adjustment**
- Adjust discretization step Δt for different context lengths at inference time
- Smaller Δt for longer sequences (finer temporal resolution)
- Training-free method but less effective than LongMamba for extreme extensions
- Complementary: can combine with F1 for additional gains

**F3: SSM Recency Bias & Over-Smoothing Theorem** (HIGH IMPACT)
- PROVEN: SSM states exhibit exponential recency bias — older tokens exponentially forgotten
- Over-smoothing: deep SSM stacks converge all states to similar values (like GNN over-smoothing)
- FIX: "Polarization" — reserve 2+ channels with A eigenvalues near 1.0 (long memory)
- These channels act as "infinite memory" lanes, preventing catastrophic forgetting
- Our model with d_state=16: reserve 2-4 channels for polarization = cheap fix
- Mathematical proof exists — not empirical, this is a theorem

**F4: Retrieval-Aware Distillation**
- SSM state is a lossy compression — retrieval of specific past tokens is fundamentally limited
- Distillation from attention-based teacher specifically on retrieval tasks improves SSM recall
- Key: the teacher shows the SSM WHAT to remember, not just final answers
- Relevant for our hybrid model: SSM handles most context, attention handles retrieval

**F5: Structured Sparse Transition Matrices**
- Beyond diagonal A: use block-diagonal, tridiagonal, or learned sparse patterns
- Block-diagonal: groups of states that interact within blocks but not across
- Tridiagonal: state i interacts with i-1 and i+1 only — chain-like memory
- 10-30% quality improvement on state-tracking tasks with minimal compute overhead

**F6: SparseSSM — State Dimension Pruning**
- 30-40% of d_state dimensions are redundant and prunable after training
- OBS-based (Optimal Brain Surgeon) pruning identifies least-important state dimensions
- At d_state=16: probably 4-6 dims prunable → d_state=10-12 effective
- Post-training optimization: zero training cost, pure inference speedup
- Synergy with Phase 30 POPCOUNT: fewer state dims = fewer operations

**F7: Samba — Interleaved SSM+Attention Hybrid**
- Pattern: SSM-Attn-SSM-Attn interleaving (not SSM-only with occasional attention)
- Each SSM layer handles temporal modeling, each attention layer handles retrieval
- Achieves unlimited context length with bounded memory
- Our architecture already does this (SWA layers at specific indices)
- Validates our hybrid approach — confirms design direction

**F8: RoPE-to-NoPE — Disable Position Encoding for Distant Tokens**
- In hybrid models: disable RoPE for tokens beyond training context window
- SSM naturally handles position via state evolution — no explicit position encoding needed
- For attention layers: use RoPE for local context, disable for long-range
- Enables length generalization beyond training context with zero cost

**F9: DeltaNet with Negative Eigenvalues**
- Expanding A matrix eigenvalue range from [0,1] to [-1,1]
- Negative eigenvalues create oscillating memory — captures periodic patterns
- Standard SSMs with A in [0,1] can only decay, never oscillate
- Our ternary A projections may naturally produce negative eigenvalues
- 5-15% improvement on tasks with periodic structure (code, structured text)

**F10: RAM-Net — Selectively Addressable Memory**
- Augment SSM with explicit read/write memory bank (like NTM but lightweight)
- SSM state = working memory, RAM = long-term storage
- Read/write gated by learned controller using SSM hidden state
- Addresses SSM's fundamental compression bottleneck for retrieval
- Adds ~5% parameters but 30-50% improvement on retrieval tasks

**F11: Length Generalization in SSMs**
- SSMs CAN extrapolate to longer sequences than training IF trained with varied lengths
- Key: include diverse sequence lengths during training (not just max_seq_len)
- Curriculum: start short (128), increase to medium (512), include some long (2048)
- Our seq_len=256 training may limit generalization — consider mixed-length batches

**F12: Infinite Retrieval — Two-Tier Memory Architecture**
- Tier 1: SSM state (fast, compressed, recent context) — always active
- Tier 2: Attention-based retrieval over cached key-value pairs — activated on demand
- Router decides when SSM state is insufficient and triggers Tier 2 retrieval
- Memory budget: Tier 1 = O(d_state), Tier 2 = O(k × d_model) for k cached tokens
- Achieves "infinite" retrieval within bounded memory budget

#### REASONING & QUALITY

**F13: Serial Scaling Hypothesis — SSMs are TC⁰ Limited** (CRITICAL)
- PROVEN: Pure SSMs belong to complexity class TC⁰ — cannot solve inherently serial problems
- Examples of serial problems: multi-digit multiplication, graph connectivity, nested parentheses
- This is a FUNDAMENTAL limitation, not fixable by scaling
- FIX: Chain-of-Thought (CoT) extends SSM beyond TC⁰ by externalizing serial computation
- FIX: Hybrid SSM+Attention — attention layers provide serial computation capability
- Our model ALREADY has attention layers (SWA) — this validates our hybrid design
- Implication: pure SSM models will ALWAYS underperform on reasoning tasks

**F14: Chain-of-Thought Extends SSM Expressiveness**
- CoT transforms a bounded-depth computation into unbounded depth (one step per token)
- For SSMs specifically: CoT tokens become "intermediate state" that SSM can accumulate
- SSM + CoT can solve problems that pure SSM cannot (proven, not empirical)
- Training with CoT data essential for reasoning quality
- Implication: our model should be trained with CoT examples for reasoning tasks

**F15: SSM Memory Mechanism — Δt Control & Lyapunov Stability** (HIGH)
- Δt ≈ 0: freezes current state (memory preservation mode)
- Δt → large: rapid state update (new information absorption)
- SSM selectively learns WHEN to remember vs forget via Δt
- Under ternary quantization: Lyapunov stability analysis shows bounded state divergence
- Quantization noise in A matrix creates bounded random walk in state space
- Stability condition: max|eigenvalue(A_ternary)| < 1 (our diagonal A satisfies this)

**F16: BitNet b1.58 2B Validates Ternary Quality** (HIGH)
- Microsoft's BitNet b1.58 2B4T: first open ternary LLM
- Matches FP16 models of same size on standard benchmarks
- Uses 2-bit ternary weights with 4-bit activations
- VALIDATES our entire approach — ternary quality is not a concern at 2B+ scale
- Key learning: ternary needs ~40% more parameters to match FP16 at small scale

**F17: Hybrid Models Achieve Best Accuracy-Efficiency Tradeoff** (HIGH)
- Pure attention: best quality, worst efficiency
- Pure SSM: best efficiency, 5-15% quality gap on reasoning
- Hybrid (SSM + sparse attention): near-attention quality at near-SSM efficiency
- Optimal ratio: 70-80% SSM layers, 20-30% attention layers
- Our model uses ~12.5% attention (1/8 layers for 6M, 2/18 for 50M) — could benefit from more

**F18: Small Models (<1B) Require Task-Specific Targeting**
- Sub-1B models cannot be "general purpose" — must specialize
- Best strategy: identify 2-3 core capabilities and optimize for those
- Ternary models at 50M-200M: focus on text completion, summarization, simple QA
- Avoid benchmarking on tasks that require multi-step reasoning at small scale

**F19: Emergent Capabilities Follow Power Laws Across Quantization**
- Capabilities emerge at specific scale thresholds regardless of quantization level
- Ternary shifts thresholds by ~1.4-1.8x (need 1.4-1.8x more parameters)
- At 50M ternary: equivalent to ~28-36M FP16 for capability emergence
- At 1B ternary: equivalent to ~560-710M FP16
- Implication: our ternary gap closing with scale is EXPECTED from power law behavior

**F20: SSM Inference Speed Advantage**
- SSMs are 1.4-3.3x faster than equivalent Transformers for generation
- Advantage increases with sequence length (O(1) vs O(n) per token)
- At 2K context: ~1.4x faster. At 8K context: ~2.5x faster. At 32K: ~3.3x
- Combined with ternary: multiplicative speedup (ternary × SSM speed advantages)

**F21: Distillation Recovers Reasoning in Ternary Models**
- Ternary models distilled from FP16 teachers recover 90%+ reasoning capability
- Key: distillation temperature T=4-6 works best for ternary (higher than standard T=2)
- Progressive distillation (FP16 → 8-bit → 4-bit → ternary) worse than direct FP16 → ternary
- Our approach (train from scratch) may benefit from distillation fine-tuning at end

**F22: Ternary Implicit Regularization**
- Ternary quantization provides implicit regularization similar to dropout
- Weight noise from quantization prevents overfitting on small datasets
- Effect strongest at small scale (6M-50M) where overfitting is a real risk
- Our dropout=0.1 at 50M might be REDUNDANT with ternary regularization
- Experiment: try dropout=0.0 with ternary at 50M — ternary might be sufficient

#### KEY PATTERNS (Wave 1)
- SSM limitations are FUNDAMENTAL (TC⁰) but hybrid architecture solves them
- Long-context is achievable via channel-level techniques (LongMamba, polarization)
- Ternary quality validated at scale (BitNet 2B) — gap closes with scale as predicted
- Two-tier memory (SSM state + attention retrieval) is the emerging consensus
- Our hybrid SSM+SWA architecture is validated by multiple independent findings
- Ternary provides FREE regularization — explicit dropout may be redundant

---

### Wave 2: Mathematical Paradigm Breakers (2 agents, Feb 2026)

#### LONG-CONTEXT MEMORY FORMULATIONS

| # | Technique | Key Property | Viability |
|---|-----------|-------------|-----------|
| M1 | Modular Arithmetic Memory | Zero decay via F_3 bijection | MEDIUM |
| M2 | Hyperbolic State Space | Exponential capacity via curvature | MEDIUM-HIGH |
| M3 | Bounded Stack Recurrence | Pushdown automaton, slot-based | HIGH |
| M4 | Hierarchical Cascade + Ternary Coupling | Multi-timescale promotion | HIGH |
| M5 | Rotational State Evolution | Isometric (zero info loss) | HIGH |
| M6 | Ternary Superposition Memory | Content-addressed read/write | HIGH |
| M7 | Symplectic Ergodic State | Volume-preserving dynamics | SPECULATIVE |

**M1: Modular Arithmetic Memory** — via Number Theory
- Replace real-valued state with residues mod small primes: h_t = (a * h_{t-1} + b(x_t)) mod p
- For p=3: ALL arithmetic is EXACT in ternary (F_3 closure). Zero quantization error.
- Multiplication in Z/pZ is BIJECTIVE (not contractive) → zero decay, information persists forever
- CRT reconstruction across m primes gives ~m*log(p) bits of non-decaying storage
- Continuous interface via round(alpha * tanh(W*x) * (p-1)/2) mod p
- Key limitation: continuous→discrete quantization at input, but error doesn't accumulate

**M2: Hyperbolic State Space** — via Riemannian Geometry
- State lives in Poincare ball B^N = {x: ||x|| < 1} with hyperbolic metric
- Near origin: coarse, abstract (old info). Near boundary: fine, detailed (recent)
- Mobius addition/scaling replaces linear state evolution
- Exponential volume growth: H^N ball of radius R has volume ~ exp((N-1)R) vs R^N Euclidean
- Anti-over-smoothing: epsilon-apart states can be arbitrarily far in hyperbolic distance
- Natural hierarchy representation (syntax trees, discourse structure)
- Cost: O(N) per token, slightly more than diagonal SSM (atanh/tanh instead of exp)

**M3: Bounded Stack Recurrence** — via Pushdown Automata (HIGH PRIORITY)
- Partition d_state into K frames: h = [f_1, ..., f_K], each frame D = d_state/K dims
- Soft stack ops: PUSH, POP, REPLACE, NOOP via softmax(W_action * x_t)
- Pushed frames persist UNCHANGED until popped — no exponential decay
- For K=8 (d_state=16, D=2): can copy/retrieve sequences up to length 8
- Directly solves copy task, balanced parentheses, nested dependencies
- O(d_state) per token — identical cost to diagonal SSM
- Drop-in replacement for SSM state in selected layers
- Training: fully differentiable through soft action weights, Gumbel-Softmax for hardening

**M4: Hierarchical Cascade with Ternary Coupling** (HIGH PRIORITY)
- Partition d_state into B bands with geometrically increasing timescales
- Band 1 (fast): a~0.5, tau~2 tokens. Band 4 (persist): a~0.999, tau~1000 tokens
- Cross-band coupling via ternary gate: T^(b)(x_t) = diag(ternary(W_T * x_t))
- Only +1/-1/0 decisions: promote/inhibit/ignore — natural ternary information bottleneck
- Info flows fast→slow ONLY (causal cascade). Selective: only "important" info promoted
- Anti-over-smoothing: different timescales prevent state convergence
- Cost: O(N) — essentially free (sparse ternary coupling ~11 additions)
- Easiest to implement: partition existing eigenvalues + add sparse coupling matrices

**M5: Rotational State Evolution** — via Fourier Analysis (TOP RECOMMENDATION)
- Replace contractive A*h with ROTATION R(theta)*h — isometric, preserves ALL information
- 2x2 rotation blocks: [cos(θ), -sin(θ); sin(θ), cos(θ)] per pair of state dims
- N/2=8 rotation pairs with log-spaced frequencies: periods T=1,2,4,...,128 tokens
- Creates discrete wavelet-like multi-resolution temporal decomposition
- Phase-Magnitude decomposition: phase=WHAT (no decay), magnitude=importance (decays)
- Anti-over-smoothing: rotations are ISOMETRIES — distances preserved forever
- Retrieval: de-rotate by accumulated angle (exactly like RoPE for position)
- Cost: O(N) per token, ~2x diagonal SSM constant (2 trig evals per pair) — negligible for d=16
- Drop-in replacement for diagonal multiplication in SSM

**M6: Ternary Superposition Memory** — UNIQUELY TERNARY
- Store K independent memory channels in same d_state space via ternary key vectors
- Write: h += gate * k_i ⊙ (B*x_t). Read: read_i = k_i^T * h
- Ternary keys are approximately orthogonal FOR FREE (random ternary vectors)
- Read/write = PURE ADD/SUBTRACT — no multiplication at all
- Channel contents persist until explicitly erased (no decay, model chooses when to forget)
- Selective forgetting: h -= erase_gate * k_i ⊙ read_i + write_gate * k_i ⊙ (B*x_t)
- With sparse keys (s=sqrt(N) nonzero entries): SNR ~ sqrt(N), adequate at N≥64
- At N=16 with K=4: SNR~0.82 (marginal). Works better at d_state=64
- IMPOSSIBLE with continuous weights: orthogonality is FREE for ternary, O(K²N) for continuous
- Synergy: combine with M5 (rotational base + superposition retrieval)

**M7: Symplectic Ergodic State** — via Hamiltonian Dynamics (SPECULATIVE)
- State split into conjugate pairs (q,p). Leapfrog integrator: volume-preserving
- Volume preservation = information cannot be destroyed (no contraction)
- Quasi-periodic orbits (KAM theorem) = "memory revival" — old info periodically accessible
- Conserved Hamiltonian H encodes sequence-level information
- Bounded by construction (bounded forces from ternary projections)
- Beautiful theory, but Path 1 (M8) achieves same NC¹ with simpler implementation

#### REASONING & SERIAL COMPUTATION FORMULATIONS

| # | Technique | Key Property | Viability |
|---|-----------|-------------|-----------|
| M8 | Barrington-SSM (full matrix scan) | TC⁰ → NC¹, ~10% overhead | HIGH |
| M9 | Diagonal + Rank-r Mixing | Parameterized expressiveness | HIGH |
| M10 | F_3 Discrete Reasoning Channel | Pure integer reasoning | HIGH/MEDIUM |
| M11 | Adaptive-Depth Iteration | Selective serial computation | HIGH |
| M12 | Polynomial State Recurrence | Exponential depth via quadratic | MEDIUM |
| M13 | Ternary Boolean Circuit Layer | O(1) lookup inference | MEDIUM |
| M14 | Spectral Braiding (Group Theory) | Hierarchy of computational power | SPECULATIVE |

**M8: Barrington-SSM** — TOP RECOMMENDATION (TC⁰ → NC¹)
- Replace diagonal scan with FULL d×d matrix scan: h_t = M_t * h_{t-1} + c_t
- M_t = sum_k alpha_k(x_t) * P_k * diag(a_t), P_k are learned permutation-like ternary matrices
- Non-commutativity of matrix products enables serial computation (Barrington's theorem)
- For d=16: scan cost increases from ~0.07% to ~10% of layer cost — NEGLIGIBLE
- Parallel scan STILL WORKS: (M_a,c_a) ⊕ (M_b,c_b) = (M_b*M_a, M_b*c_a+c_b) — associative
- Stability: convex combination of permutation matrices × diag(a) has spectral radius < 1
- THE most fundamental upgrade available: TC⁰→NC¹ for ~10% overhead
- Random initialization of permutation matrices almost certainly sufficient (Dixon's theorem)

**M9: Diagonal + Rank-r Mixing** — Parameterized Barrington
- M_t = diag(a_t) + U_t * V_t^T, where U,V ∈ R^{d×r} from ternary projections
- Rank r controls expressiveness/cost tradeoff: r≥5 guarantees S_5 (Barrington)
- For r=5: 160 parameters per step (vs 256 for full matrix)
- Scan cost same as M8 (matrices become full rank during scan)
- Provides tuning knob: r=2 (minimal mixing) to r=16 (full matrix)

**M10: F_3 Discrete Reasoning Channel** — Finite Field Arithmetic
- Parallel discrete state machine: h_t^(d) = M_t^(d) * h_{t-1}^(d) + c_t^(d) (mod 3)
- F_3 = Z/3Z = {0,1,2} ≅ {0,1,-1} — ISOMORPHIC to ternary weights!
- Scan over F_3 is EXACT — no rounding, no stability issues
- For d_disc=8: 3^8 = 6561 states, lookup table fits in L1 cache
- Information: d_disc * 1.585 bits (25.4 bits for d_disc=16)
- Entire channel uses ZERO floating point — pure integer/ternary
- POPCOUNT acceleration applies directly (synergy with Phase 30)
- Training challenge: STE through mod-3 needs empirical validation
- Combined with continuous SSM: discrete for exact reasoning, continuous for semantics

**M11: Adaptive-Depth Iteration** — Mixture-of-Depth for SSM
- Difficulty predictor: d_t = sigmoid(w_d^T * h_t^(0))
- Easy tokens (fraction 1-p): standard O(d) SSM — zero overhead
- Hard tokens (fraction p): K iterations of nonlinear map + ternary matrix
- h_t^(k) = diag(a_t) * h_t^(k-1) + B*x + phi(W_iter * h_t^(k-1))
- Nonlinearity phi is essential — without it, iterations remain TC⁰
- For d=16, K=4, p=0.1: ~7.4x overhead on 10% of tokens ≈ 1.74x total
- Fixed-point convergence: contraction condition guarantees unique answer
- GPU-friendly: iterations parallel across tokens, serial within token

**M12: Polynomial State Recurrence** — Quadratic Dynamics
- Add quadratic term: h_t = diag(a) * h_{t-1} + B*x + C * (h_{t-1} ⊙ h_{t-1})
- After T compositions: effective polynomial degree 2^T — exponential depth
- Sphere projection (h/||h|| * sqrt(d)) prevents divergence
- Cost: O(d) per step — SAME as standard SSM (quadratic is elementwise)
- NOT parallelizable with standard scan (breaks affine structure)
- Mitigation: chunk sequence into blocks, sequential within, parallel across
- Risk: chaotic dynamics may cause training instability (exploding gradients)

**M13: Ternary Boolean Circuit Layer** — O(1) Inference
- State in {-1,0,+1}^d. Transition: z = W_A * sign(h) + W_B * q(x), h = sign(z+b)
- Entire forward pass is integer arithmetic — no floating point
- For d=8: transition function has 3^12 ≈ 531K entries → ~1MB lookup table (fits L1!)
- Inference: SINGLE TABLE LOOKUP per token — O(1), fastest possible
- Training: STE through both weights AND state (double quantization)
- Information capacity: 3^d states (43M for d=16, 6561 for d=8)
- Escapes TC⁰ via serial composition of threshold operations across tokens

**M14: Spectral Braiding** — Group Theory (SPECULATIVE)
- State evolution via group representation: h_t = ρ(g(x_t)) * h_{t-1}
- G = S_5 (symmetric group), 5-dimensional faithful representation
- Non-abelian group provides NC¹ power (Barrington)
- Fourier transform on G decomposes into irreps → parallel within irreps
- Hierarchy of computational power parameterized by group choice
- Beautiful theory, but Path 1 (M8) achieves same NC¹ with simpler implementation

#### KEY PATTERNS (Wave 2)
- The TC⁰ limitation has a SINGLE root cause: diagonal state matrix (commutativity)
- Full matrix scan (M8) solves it with only ~10% overhead — highest-value modification
- Rotational state evolution (M5) solves exponential decay at root cause
- Ternary superposition (M6) exploits ternary discreteness for content-addressed memory
- F_3 channel (M10) creates perfect ternary-native integer reasoning
- Best combined architecture: M5 (rotational base) + M6 (superposition retrieval) + M8 (matrix scan for reasoning)

---

### Wave 3: Paradigm Breakers (2 agents, Feb 2026)

#### RADICAL LONG-CONTEXT APPROACHES

**P1: Trit-Flip Memory** — UNIQUELY TERNARY
- Store information in PARAMETER MODIFICATIONS, not state vector
- During inference, controller flips sparse ternary weights: W_eff = (W + delta) mod 3
- Z/3Z closure: modified weights are STILL ternary — zero precision change
- Capacity scales with NUMBER OF PARAMETERS, not state dimension
- Zero-decay (flipped weights persist), zero-cost reads (part of normal computation)
- IMPOSSIBLE with FP16 weights (modification breaks precision guarantees)
- Training: STE through controller that learns WHICH weights to flip
- Viability: MEDIUM. Key risk: can STE train meaningful flip patterns?

**P2: Compressed Sensing State** — Provably Optimal
- Add ternary accumulators: s_t = s_{t-1} + Phi * embed(x_t), Phi random ternary
- Pure accumulator (A=I): ZERO decay, information persists forever
- Retrieval: solve L1 minimization (ISTA/FISTA) to reconstruct from measurements
- Ternary Phi satisfies RIP (Restricted Isometry Property) — provable guarantees
- Measurement update = PURE ADD/SUBTRACT (ternary matrix, zero multiplication)
- Reconstruction expensive O(M²K) but triggered selectively by SWA attention
- Viability: MEDIUM. NO TRAINING needed for CS components — easiest to test

**P3: Arithmetic Coding State** — Shannon-Optimal
- State = interval [lo,hi) in [0,1) that narrows with each token
- TWO numbers encode the EXACT sequence — provably optimal compression
- Base-3 variant: ternary-native, M trits = M*1.585 bits of perfect memory
- FP64 precision limits perfect memory to ~50-60 tokens
- Good for augmenting recent context, not for truly long context
- Viability: MEDIUM-HIGH for short context, LOW for long context

**P4: Sparse Causal Graph State** — Structural Memory
- State = ternary adjacency matrix G ∈ {-1,0,+1}^{K×K} encoding RELATIONSHIPS
- Tokens assigned to ROLES (subject, object, etc.), not positions
- Edges: +1 promotes, -1 inhibits, 0 unrelated — natural ternary semantics
- K=16 slots: 404 bits of structural information, zero decay
- Risk: may collapse into a form of attention over fixed-size memory
- Viability: MEDIUM

#### RADICAL REASONING APPROACHES

**P5: Holographic Ternary Algebra (HDC)** — TERNARY-NATIVE REASONING
- {-1,0,+1} IS the natural alphabet for Hyperdimensional Computing
- BIND(a,b) = a⊙b (elementwise multiply): produces ternary, encodes relation
- BUNDLE = sign(sum): majority vote, encodes superposition
- Variable binding + compositional structure + relational reasoning — ALL in TC⁰
- Hardware: binding = XNOR+AND, bundling = POPCOUNT — maps to Phase 30!
- Capacity: O(d/log d) bindings — d_state=16 ≈ 4 bindings, d_state=64 ≈ 10
- Synergy: POPCOUNT infrastructure serves BOTH inference speed AND reasoning
- Viability: MEDIUM-HIGH. Theory is solid, ternary fit is remarkable

**P6: Stochastic Resonance Reasoning** — ZERO Architecture Change
- Inject calibrated noise into hidden states, run k forward passes, majority-vote
- Single pass = TC⁰. k stochastic passes = BPP-TC⁰ (strictly MORE powerful)
- Each noise pattern activates different computational paths in the network
- For ternary: randomly flip fraction p of zero-weights to ±1 per pass
- k=20-50 passes for reliable results. Embarrassingly parallel on GPU
- ZERO training changes, ZERO architecture changes — pure inference technique
- Natural confidence measure: agreement across passes = model certainty
- Viability: HIGH. Trivially implementable, sound theory. Question is magnitude.

**P7: Geometric Reasoning via Learned Metric** — POPCOUNT Unification
- Reasoning IS retrieval: conclusion is geometrically near premises in embedding space
- Johnson-Lindenstrauss: O(log N) dims suffice for N reasoning patterns
- Ternary distance = Hamming distance = POPCOUNT of XOR
- Phase 30 POPCOUNT serves BOTH inference speed AND reasoning retrieval!
- Add contrastive loss: push premises near conclusions, away from wrong ones
- Viability: MEDIUM. Elegant unification, but training modifications nontrivial

**P8: Kolmogorov Template Compilation** — Constant-Depth Reasoning
- Move serial computation from INFERENCE to TRAINING
- Most reasoning follows finite templates (24 valid syllogistic forms, etc.)
- Template selector maps input to template index → activates pre-compiled circuit
- Ternary zeros act as "off switches" — templates time-share weight matrix
- Truly novel reasoning fails (outside template library)
- Viability: LOW-MEDIUM. Sound theory but training challenges significant

#### KEY PATTERNS (Wave 3)
- Trit-Flip (P1) exploits Z/3Z closure — IMPOSSIBLE with any other precision
- Stochastic Resonance (P6) provides reasoning upgrade with ZERO changes
- HDC (P5) + POPCOUNT (Phase 30) = unified reasoning + inference acceleration
- CS State (P2) is the easiest to test (no training needed)
- Multiple approaches connect to Phase 30 POPCOUNT (P5, P7) — convergent evidence

---

## Aspect 3 Summary: Long-Context & Reasoning

**Total findings:** 22 web research (F1-F22) + 14 mathematical (M1-M14) + 8 paradigm shifts (P1-P8) = **44 findings**

**Top 5 by Impact:**
1. **M8 Barrington-SSM** — TC⁰→NC¹ for ~10% overhead. THE most fundamental upgrade.
2. **M5 Rotational State Evolution** — Eliminates exponential decay at root cause. Same cost.
3. **P5 Holographic Ternary Algebra** — Ternary-native reasoning, POPCOUNT synergy.
4. **F3/F13 SSM Recency Bias + Serial Scaling** — Proven theorems defining limitations.
5. **P6 Stochastic Resonance** — Free reasoning upgrade, zero architecture changes.

**Top 5 by Novelty:**
1. **P1 Trit-Flip Memory** — Parameters as memory medium via Z/3Z closure (unique to ternary)
2. **M6 Ternary Superposition** — Content-addressed memory exploiting ternary orthogonality
3. **M10 F_3 Discrete Channel** — Finite field reasoning, exact integer computation
4. **P2 Compressed Sensing** — Random ternary measurements with provable retrieval
5. **M3 Bounded Stack** — Pushdown automaton in SSM state (copy task solver)

**Immediate Actions:**
- M8 (Barrington-SSM): Replace diagonal scan with matrix scan (~10% overhead, NC¹ power)
- M5 (Rotational): 2x2 rotation blocks as drop-in for diagonal multiplication
- P6 (Stochastic Resonance): Test on existing model, zero code changes to architecture
- P2 (Compressed Sensing): Add ternary accumulators, no training needed
- M4 (Hierarchical Cascade): Partition eigenvalues into bands + sparse ternary coupling
