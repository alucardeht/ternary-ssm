# Round 1, Aspect 4: Alternative Knowledge Representations

## Wave 1 — Web Research (2 researchers, Feb 2026)

### RETRIEVAL-AS-COMPUTATION PARADIGM

#### v5-R1-A4-F1: StructGPT — LLMs as Query Generators Over Structured Data
- **Source:** Jiang et al., EMNLP 2023 (568 citations), widely adopted 2025
- **Summary:** Trains LLMs to generate executable queries (SQL, Cypher, SPARQL) to reason over external databases/knowledge graphs. Model stores QUERY SKILLS (compact) not FACTUAL KNOWLEDGE (massive). Iterative reasoning: query → result → next query. A 1B StructGPT + 100GB database can match 70B pure-parametric model on knowledge tasks.
- **Quality:** SAME or BETTER on knowledge-grounded QA, multi-hop reasoning
- **RAM gain:** MASSIVE — query generation model can be tiny (~100M-1B params). Knowledge lives in structured DB on disk (~1 KB/fact vs ~100 KB in parameters).
- **Throughput:** SLOWER (query parsing + execution adds latency). Deterministic and cacheable.
- **For us:** Ternary SSM as query generator + SQLite on NVMe = factual knowledge offloaded entirely. Model stores only reasoning/language skills. DB query is deterministic = no quality loss on facts.
- **Verdict:** PROMISING

#### v5-R1-A4-F2: Memory Layers at Scale — Sparse Retrieval Replaces Dense FFN
- **Source:** Mu et al., arXiv:2412.09764, December 2024
- **Summary:** Replaces dense FFN layers with millions of (key, value) pairs + top-k sparse retrieval. Stores 100x more facts than dense FFN in same memory. Only top-k pairs retrieved per token (k << total). 10x parameter reduction in FFN layers with SAME or BETTER perplexity.
- **Quality:** SAME or BETTER than dense FFN baseline
- **RAM gain:** 10x for FFN component (~2/3 of model params). With disk-based storage: potentially 100x.
- **Throughput:** NEUTRAL (ANN retrieval ≈ dense matmul latency for reasonable k)
- **For us:** Replace MLP experts with Memory Layers. Each "expert" becomes a retrieval index over ternary key-value pairs. Ternary keys enable Hamming distance for ultra-fast ANN.
- **Verdict:** PROMISING

#### v5-R1-A4-F3: RETRO — Retrieval-Enhanced Transformer (Proven at Scale)
- **Source:** DeepMind (2021), production guides 2025-2026
- **Summary:** Interleaves retrieval and prediction during forward pass. Chunks input, retrieves documents per chunk, cross-attends. 7B RETRO matched 175B baseline on knowledge tasks. Retrieval happens DURING forward pass, not just at prompt time.
- **Quality:** SAME as 10x larger models on knowledge tasks
- **RAM gain:** 25x intelligence-to-RAM ratio (7B RETRO ≈ 175B pure model). Datastore on disk (10-50 GB).
- **Throughput:** 2-3x SLOWER (retrieval + cross-attention overhead)
- **For us:** RETRO-style retrieval + ternary SSM base = tiny model with massive effective knowledge. SSM state could drive retrieval queries (similar to A3-M15 SSM predictor).
- **Verdict:** PROMISING

#### v5-R1-A4-F4: kNN-LMs — "Great Memory, Shallow Reasoning" Analysis
- **Source:** Geng et al., NAACL 2025
- **Summary:** k-Nearest Neighbor LMs retrieve (context, token) pairs from datastore. BETTER than neural on factual recall, WORSE on multi-step reasoning. 2025 analysis identifies critical limitation: retrieval stores EXAMPLES not RULES. Datastore = 20x more memory-efficient per fact than parameters.
- **Quality:** BETTER (facts) / WORSE (reasoning) — complementary to neural
- **RAM gain:** Base model can be much smaller. 1B + 100GB datastore ≈ 7B pure model on knowledge tasks.
- **Throughput:** SLOWER (ANN search adds 10-30ms/token with FAISS optimization)
- **For us:** CONFIRMS the hybrid approach: retrieval for facts, ternary SSM for reasoning/generation. The "great memory, shallow reasoning" gap is exactly what the SSM should fill.
- **Verdict:** NEEDS-MORE-WORK (as standalone — PROMISING as component of hybrid)

### EXTERNAL MEMORY & CONTINUAL LEARNING

#### v5-R1-A4-F5: MemoryLLM — Self-Updatable External Memory Pool
- **Source:** Wang et al., arXiv:2402.04624, Feb 2024; production analysis Feb 2026
- **Summary:** External memory pool (N vectors × d dimensions) persists across conversations. Model attends to [local_context; memory_pool]. Memory updated via k-gram conditional compression after each text segment. 29.7% perplexity reduction, +9-11% dialogue coherence.
- **Quality:** BETTER on long-context and continual learning. SAME on single-document.
- **RAM gain:** 10K vectors × 4096 dim × 2 bytes = 80 MB. Enables smaller base model by offloading long-term memory.
- **Throughput:** SLOWER (attention over expanded context). Mitigated by top-k retrieval over memory.
- **For us:** Ternary SSM + external memory pool = persistent state across sessions. Memory vectors could be ternary too (Hamming retrieval). Combines with SSM hidden state for dual-memory architecture.
- **Verdict:** PROMISING

#### v5-R1-A4-F6: Memory-Augmented LLMs (2025 Ecosystem)
- **Source:** MemInsight (Mar 2025), MMAG (Dec 2025), MemLoRA (Dec 2025)
- **Summary:** Ecosystem of external memory modules: flat vector memory, hierarchical (episodic/semantic/long-term), semantic attribute-value stores. MemLoRA distills memory into low-rank adapters for on-device deployment. Extends context 16K → 160K tokens.
- **Quality:** MATCHES or EXCEEDS baselines with well-managed memory
- **RAM gain:** MemLoRA: distills memory to on-device adapter (~10-50 MB). Hierarchical memory: separates hot/cold knowledge for tiered storage.
- **Throughput:** Adds 100-200ms retrieval latency per turn. FAISS mitigates.
- **For us:** MemLoRA is directly applicable — distill external knowledge into ternary low-rank adapter. Hierarchical memory maps to our NVMe/DRAM/VRAM tiering from A3-M10.
- **Verdict:** PROMISING

### NEUROMORPHIC / SPIKING APPROACHES

#### v5-R1-A4-F7: SpikeGPT — Spiking Neural Network Language Model
- **Source:** Zhu et al., arXiv:2302.13939, Feb 2023 → TMLR 2024 → ICLR 2025
- **Summary:** Largest backprop-trained SNN language model (up to 260M params). RWKV-inspired: replaces O(N²) attention with O(N) linear spiking mechanism. Binary event-driven activations (1-bit). 22x fewer operations on neuromorphic hardware.
- **Quality:** COMPETITIVE with non-spiking models on tested benchmarks (but not transformer-level at scale)
- **RAM gain:** Binary activations (1-bit vs 16/32-bit). Inference state: minimal. Weights still parametric.
- **Throughput:** 22x fewer operations BUT requires neuromorphic hardware for gains. On conventional hardware: sequential processing bottleneck.
- **For us:** Ternary weights + binary spiking activations = BOTH weight and activation compression. SSM is already sequential (like spiking). If combinable: ternary-spiking SSM with ~0.3 bits effective per weight×activation.
- **Verdict:** PROMISING (architecture concept; hardware-dependent for full gains)

#### v5-R1-A4-F8: Sigma-Delta Neural Networks (SDNN) — ANN-to-SNN Conversion
- **Source:** Brehove et al., May 2025 (Loihi 2 research)
- **Summary:** ANN-to-SNN conversion using sigma-delta coding with graded (integer) spikes. Achieves ~6% of original MACs (94% reduction). Match or slightly exceed ANN baselines. Demonstrated on video processing on Loihi 2 (200x energy-delay improvement).
- **Quality:** SAME (quality parity with ANN after conversion)
- **RAM gain:** Activations = graded spikes (4-8 bits). Weights unchanged. Inference state: minimal.
- **Throughput:** 16x theoretical speedup (6% MACs). 200x EDP on neuromorphic hardware.
- **For us:** Train ternary model conventionally → convert to SDNN for neuromorphic deployment. Preserves training flexibility + unlocks neuromorphic efficiency. Could be the deployment path for Loihi 2/3.
- **Verdict:** PROMISING (as deployment conversion path)

### LOOKUP-BASED INFERENCE

#### v5-R1-A4-F9: LUT-NN — Lookup Table Neural Networks
- **Source:** ACM MM'23 (2023), NPU extensions 2025
- **Summary:** Learns "typical features" per operator, pre-computes lookup tables to replace dense matmul. 2025: extended to NPU-specialized execution with flexible quantization. Tables eliminate weight loading during inference — computation becomes table lookup.
- **Quality:** Claims maintained accuracy at operator level. No zero-loss proof at full LLM scale.
- **RAM gain:** Fixed tables proportional to input dimensionality × table size. 3-5x compression potential.
- **Throughput:** 5-10x on NPU hardware. On CPU: moderate gains (table lookup vs arithmetic).
- **For us:** For ternary, tables are TINY (3^k entries for k-input lookup). Combines with A3-P8 (bitplane POPCOUNT) — both replace matmul with simpler operations.
- **Verdict:** NEEDS-MORE-WORK (no LLM-scale zero-loss validation)

## Key Insights — Aspect 4

1. **Retrieval-as-Computation is PROVEN** — RETRO (25x), StructGPT, Memory Layers all demonstrate that retrieval can REPLACE parameters with same/better quality on knowledge tasks
2. **"Great Memory, Shallow Reasoning"** — kNN-LMs expose the fundamental split: retrieval for FACTS, neural for REASONING. This CONFIRMS our hybrid approach.
3. **Structured > Unstructured** — StructGPT (databases) beats unstructured retrieval. Structure = compression = efficiency.
4. **External memory is production-ready** — MemoryLLM, MemLoRA, MMAG all deployed 2025. Not RAM reduction per se, but enables smaller base model.
5. **Spiking + Ternary = natural combination** — Binary activations + ternary weights = both dimensions compressed. SSM is already sequential (spiking-compatible).
6. **Neuromorphic gains require specialized hardware** — 22x-200x gains on Loihi 2, but consumer availability = zero. Concept useful, hardware path uncertain.
7. **Lookup tables for ternary are trivial** — 3^k entries. But scaling to full LLM dimensions unproven.

## For Our Project

**The Hybrid Architecture Pattern (confirmed by 4+ independent findings):**

| Component | Handles | Storage | Technology |
|-----------|---------|---------|------------|
| Ternary SSM core | Reasoning, language patterns, generation | RAM (~100-500 MB) | Our architecture |
| Structured DB | Factual knowledge (entities, relations) | NVMe (~1-50 GB) | SQLite + query gen |
| Memory Layers | Pattern memory (frequent patterns) | RAM (sparse retrieval) | Replaces dense FFN |
| External Memory | Conversation history, adaptation | RAM/NVMe (vectors) | MemLoRA-style |
| Retrieval Index | Long-tail knowledge | NVMe (ANN index) | RETRO-style |

**Key principle:** The ternary SSM stores SKILLS (reasoning, generation, query formulation). Knowledge is in external structures. This is fundamentally more efficient — skills are compact, knowledge is unbounded.
