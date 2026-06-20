# Research Round 1 — Paradigm Shift Exploration

## Objective
Find fundamentally different approaches to building AI systems that can match frontier LLM intelligence (DeepSeek, Claude Opus, GPT-4o) with dramatically less compute and memory.

We pursue TWO TRACKS in parallel, each with distinct research goals.

### Track A — Complete Paradigm Shift
We are NOT looking for:
- Transformer variants or improvements
- Incremental efficiency gains
- Better next-token prediction mechanisms

We ARE looking for:
- Entirely different computation paradigms (not next-token prediction)
- Alternative knowledge representations (not dense weights)
- Non-autoregressive generation methods
- Fundamentally different training objectives
- Mathematical frameworks that replace or transcend the LLM paradigm
- Systems that achieve intelligence through structure, not parameter scale

### Track B — SSM Revolution
We are NOT looking for:
- Attention mechanisms (even hybrid)
- Transformer-like variants of SSM
- Incremental token-level efficiency

We ARE looking for:
- Radical improvements to SSM core (state compression, parallel inference, richer state)
- Ways to eliminate attention entirely while maintaining quality
- Ternary-native SSM designs (not ternary as a bolted-on layer)
- Breaking the sequential inference bottleneck
- More intelligence per state-vector element
- Hardware-native SSM implementations that exploit SSM's O(1) structure

## 6 Research Aspects

### TRACK A — COMPLETE PARADIGM SHIFT (Aspects 1-3)

### Aspect 1: Non-Autoregressive & Alternative Generation Paradigms
Can we generate entire sequences/responses without sequential token prediction?
- Diffusion models for text (MDLM, SEDD, Plaid, etc.) — generate all tokens in parallel
- Non-autoregressive transformers (NAT, CMLM, etc.) — how close to AR quality?
- Energy-based models for text generation — iterative refinement instead of sequential
- Parallel decoding and set prediction — treat generation as structured prediction, not sequence
- Latent variable models — compress meaning before decoding
- Recent 2025-2026 advances — what's the quality gap now? Can it match autoregressive?

### Aspect 2: Knowledge Without Dense Weights & Neurosymbolic Reasoning
Can intelligence be stored and retrieved as structure rather than implicitly in parameters?
- Retrieval-augmented generation at architecture level (not bolted-on adapter)
- Knowledge graphs as primary representation + neural reasoning over structure
- Learned databases, differentiable data structures, content-addressable memory
- Program synthesis — model learns algorithms and stores them as code/circuits
- Neurosymbolic architectures (DeepProbLog, NeurASP) — neural perception + symbolic reasoning
- Differentiable reasoning frameworks — theorem proving, logic, rule application as differentiable ops

### Aspect 3: Radical Architecture Departures (NOT Transformer/SSM Variants)
What fundamentally different compute substrates exist for language intelligence?
- Kolmogorov-Arnold Networks (KAN) — function approximation without dense layers
- Liquid Neural Networks — continuous-time dynamics instead of discrete tokens
- Hyperdimensional computing — semantic vectors and symbolic operations at massive scale
- Spiking neural networks — event-driven computation instead of dense activations
- Cellular automata and rule-based computation — parallel local rules generating global intelligence
- Graph neural networks for language — structure-first instead of sequence-first
- Quantum or analog computing approaches to language — fundamentally different physics

### TRACK B — SSM REVOLUTION (Aspects 4-6)

### Aspect 4: Ultra-Efficient SSM State Representation
Can we compress SSM state to dramatically reduce memory footprint?
- Quantized state vectors (mixed-precision, ternary, binary state)
- Factored state representations (low-rank decompositions of SSM state)
- State pruning and selective state updates (only update relevant state elements)
- Hierarchical state — multi-level state representations with compression
- Sub-linear overall memory (not just O(1) per token, but compressed state vectors themselves)
- State sharing and factorization across layers — can we reuse state computationally?

### Aspect 5: SSM Performance Breakthroughs — Dramatically Faster
Can we make SSM inference dramatically faster while maintaining quality?
- Parallel SSM scans (fused scan kernels, hardware-native implementations)
- Chunked/block-level processing — process multiple tokens in parallel within SSM
- Ternary-native SSM kernels (add/subtract only, no matmul for SSM)
- Hardware acceleration specific to SSM (Triton, custom CUDA, CPU SIMD)
- Eliminating sequential bottleneck — can we batch/parallelize the serial recurrence?
- Training throughput improvements — SSM training faster than Transformer at same scale?

### Aspect 6: Novel SSM State Representations — More Intelligence Per Bit
Can SSM state encode richer, denser information per element?
- Richer state formulations (hypercomplex numbers, structured state, recursive state)
- State-space alternatives to attention for retrieval (retrieval from state, not token-based)
- Infinite context via state compression (memory of entire context compressed into fixed state)
- State transformations and projections (learning how to squeeze more meaning into state)
- Multi-scale state (state at different timescales for different phenomena)
- State amplification and boosting (can we learn to extract more signal from same state size?)

## Research Protocol

Each aspect follows the 7-step workflow:
1. Orchestrator organizes search themes (2 distinct directions)
2. 2 web researchers in PARALLEL (distinct themes) → WAIT
3. Organize + document findings → COMMIT
4. 2 math-paradigm-breakers in PARALLEL → WAIT
5. Organize + document → COMMIT
6. 2 paradigm-breakers in PARALLEL → WAIT
7. Organize + document + summary → COMMIT

All agents receive known-findings-index.md to avoid re-discovering known results.
