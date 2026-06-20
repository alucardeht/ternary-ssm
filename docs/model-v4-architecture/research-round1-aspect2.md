# Round 1, Aspect 2: Knowledge Without Dense Weights + Retrieval-Based Models

## Wave 1 — Web Research (2 researchers, Feb 2026)

### NEUROSYMBOLIC APPROACHES

#### v4-R1-A2-F1: Symbolic Knowledge Distillation — 100x Compression
- **Source:** PMC 2025, Neurosymbolic AI as Antithesis to Scaling Laws
- **Summary:** Distills common sense knowledge from GPT-3 into knowledge graph + tiny neural model. 100x fewer parameters, SUPERIOR performance on common sense reasoning.
- **Quality:** Better than GPT-3 on common sense tasks
- **Efficiency:** 100x parameter reduction
- **Maturity:** Proof-of-concept (2025)
- **Verdict:** PROMISING — Exactly the "knowledge separated from computation" paradigm

#### v4-R1-A2-F2: LightRAG — 10x Token Reduction via KG
- **Source:** October 2024, production-ready
- **Summary:** Knowledge graph-based retrieval with dual-level (structure + detail). Comparable accuracy to full GraphRAG with 10x less tokens.
- **Quality:** Comparable to GraphRAG
- **Efficiency:** 10x token reduction, dramatically lower compute
- **Maturity:** Production-ready (2024)
- **Verdict:** PROMISING

#### v4-R1-A2-F3: Paths-over-Graph (PoG) — KG Empowered LLM Reasoning
- **Source:** ACM WWW 2025
- **Summary:** Dynamic multi-hop path exploration through knowledge graphs with three-step pruning (graph structure + LLM prompting + pre-trained LM). KG handles reasoning structure, LLM only processes pruned relevant paths.
- **Quality:** SOTA on multi-document synthesis
- **Efficiency:** Dramatic search space reduction via pruning
- **Maturity:** Published research (2025)
- **Verdict:** PROMISING

#### v4-R1-A2-F4: ReKnoS Super-Relations Framework
- **Source:** OpenReview 2025
- **Summary:** Introduces "super-relations" that summarize relational paths within KG, enabling bidirectional reasoning.
- **Quality:** Improved over standard KG querying
- **Efficiency:** Claims improved retrieval efficiency (unvalidated at scale)
- **Maturity:** Research paper
- **Verdict:** NEEDS-MORE-WORK

#### v4-R1-A2-F5: Hybrid SyGuS + LLM Program Synthesis
- **Source:** Edinburgh 2025
- **Summary:** Combines correctness guarantees of classical Syntax-Guided Synthesis with LLM speed. LLM generates candidate programs, SyGuS verifies correctness.
- **Quality:** Correctness guaranteed (symbolic verification)
- **Efficiency:** Faster than pure SyGuS, more reliable than pure LLM
- **Maturity:** Research system
- **Verdict:** PROMISING — Separates generation (fast neural) from verification (symbolic guarantee)

#### v4-R1-A2-F6: SOAR Self-Improving Synthesis
- **Source:** arXiv July 2025
- **Summary:** Alternates evolutionary search (LLM samples/refines programs) and learning (hindsight finetuning).
- **Verdict:** NEEDS-MORE-WORK — Unclear if efficiency persists at scale

#### v4-R1-A2-F7: ReMindRAG — Train-Free KG Traversal
- **Source:** OpenReview 2025
- **Summary:** LLM-guided KG traversal with node exploration, exploitation, memory replay. Zero training cost — memorizes in KG edge embeddings.
- **Quality:** Competitive with fine-tuned approaches
- **Efficiency:** Zero training cost
- **Maturity:** Research system
- **Verdict:** PROMISING — Train-free + KG-based knowledge = strong efficiency

#### v4-R1-A2-F8: Neurosymbolic AI Industrial Adoption (Amazon Vulcan/Rufus)
- **Source:** Singularity Hub / Cogent 2025-2026
- **Summary:** Amazon deployed neurosymbolic AI in warehouse robots (Vulcan) and shopping assistant (Rufus). Neural perception + symbolic reasoning layers.
- **Quality:** Enhanced accuracy vs pure neural
- **Efficiency:** More energy efficient (less data storage needed)
- **Maturity:** PRODUCTION (deployed 2025)
- **Verdict:** PROMISING — Real-world validation of neural+symbolic separation at scale

### SPARSE & RETRIEVAL-BASED MODELS

#### v4-R1-A2-F9: Infini-Attention — Bounded Memory Infinite Context
- **Source:** Google Research, arXiv 2404.07143, April 2024
- **Summary:** Compressive memory integrated into attention. Combines local masked attention + long-term linear attention in single block. Associative memory matrix M updated incrementally via delta rule.
- **Quality:** 1B model matches GPT-3/Jurassic-1 on Pile with 25x fewer params. SOTA book summarization (500K length)
- **Efficiency:** 114x compression vs Memorizing Transformers. Constant memory regardless of sequence length. 1M passkey retrieval.
- **Maturity:** Research proof-of-concept (2024)
- **Verdict:** PROMISING — Perfect alignment with SSM (both O(1) memory). Delta rule aligns with our gradient work.

#### v4-R1-A2-F10: GraphRAG (Microsoft)
- **Source:** Microsoft Research, arXiv 2404.16130, 2024
- **Summary:** Builds knowledge graph from corpus before retrieval. Local search (graph neighborhoods) + Global search (community summaries).
- **Quality:** Excellent on multi-document synthesis where baseline RAG completely fails
- **Efficiency:** VERY expensive indexing (LLM processes entire corpus)
- **Verdict:** NEEDS-MORE-WORK — Relational reasoning excellent but indexing cost prohibitive

#### v4-R1-A2-F11: Hybrid RAG Search (RRF)
- **Source:** SIGIR 2009 → production standard 2025
- **Summary:** Dense vector search + sparse BM25 in parallel, fused via Reciprocal Rank Fusion.
- **Quality:** Consistently outperforms individual ranking
- **Efficiency:** Two parallel searches + fast fusion. Native in modern vector DBs.
- **Maturity:** Production standard (2025)
- **Verdict:** PROMISING — Battle-tested, low cost, proven gains

#### v4-R1-A2-F12: Self-RAG + Corrective RAG
- **Source:** arXiv 2310.11511 (Self-RAG 2023), 2401.15884 (C-RAG 2024)
- **Summary:** RAG systems that verify own quality before using retrieved context. Self-RAG uses special tokens for self-assessment. C-RAG uses lightweight classifier (T5-large 770M) to score retrieval quality.
- **Quality:** Prevents silent failures, self-correcting
- **Efficiency:** Inference overhead from verification, but prevents hallucination
- **Maturity:** Research prototypes (2023-2024)
- **Verdict:** PROMISING — Graceful degradation essential for production

#### v4-R1-A2-F13: RDR2 — Document Structure-Aware RAG
- **Source:** EMNLP 2025
- **Summary:** RAG that navigates document STRUCTURE trees instead of treating passages as isolated chunks. LLM-based router evaluates content relevance + hierarchical relationships.
- **Quality:** SOTA on 5 challenging datasets, significant multi-document synthesis improvement
- **Maturity:** Research (EMNLP 2025)
- **Verdict:** PROMISING — Structural awareness resolves "orphan chunk" problem

#### v4-R1-A2-F14: Activation Caching + Hybrid KV Caching
- **Source:** arXiv 2501.01792, January 2025
- **Summary:** Efficient LLM inference combining activation checkpointing (recompute vs store tradeoff) with smart KV cache management (offload to NAND when overflow).
- **Quality:** Maintains accuracy while reducing memory
- **Efficiency:** Trades compute for memory. Critical for long-context inference.
- **Maturity:** Research (2025)
- **Verdict:** PROMISING — KV cache is bottleneck #1 in inference

#### v4-R1-A2-F15: RETRO Follow-ups (2023-2025)
- **Source:** Multiple follow-ups to DeepMind RETRO 2021
- **Summary:** Frozen BERT retriever + 2T token database. Retrieves chunks, uses chunked cross-attention.
- **Quality:** GPT-3 performance with 25x fewer params (original claim)
- **Efficiency:** 2T token database = massive storage. Frozen retriever limits adaptability.
- **Verdict:** NEEDS-MORE-WORK — Database size prohibitive. Infini-Attention supersedes with bounded memory.

#### v4-R1-A2-F16: RAG Survey 2025 — State of the Field
- **Source:** arXiv 2506.00054, May 2025
- **Summary:** Comprehensive survey. Key trade-offs: retrieval precision vs generation flexibility, efficiency vs faithfulness. Future: adaptive retrieval, real-time integration, privacy-preserving.
- **Verdict:** CONTEXTUAL — Confirms RAG still has open challenges (robustness, efficiency, multi-hop)

## Key Insights — Aspect 2

1. **Knowledge-Computation Separation is THE 2025-2026 trend** — every finding separates knowledge storage from neural computation
2. **100x compression achievable** (F1) when knowledge lives in graphs, not weights
3. **Bounded memory solves infinite context** — Infini-Attention (F9) + SSM = both O(1), perfect match
4. **Train-free knowledge integration emerging** (F7) — zero-cost knowledge addition
5. **Production validation exists** — Amazon (F8), LightRAG (F2) deployed at scale
6. **Self-correction layers essential** — Self-RAG/C-RAG (F12) prevent hallucination from bad retrieval
7. **Structure > flat chunks** — RDR2 (F13) shows hierarchical document navigation beats naive chunking

## For Our Project

**CRITICAL MATCH: Infini-Attention (F9) + Ternary SSM**
- Both achieve O(1) memory per token
- Compressive memory matrix M can be ternary (associative memory via Hamming distance)
- Delta rule update aligns with our STE/gradient work
- 114x compression ratio exceeds our 5-10x efficiency target

**OPEN QUESTION: Should SSM state BE the knowledge representation?**
- Current: SSM state = dense vector evolved via recurrence
- Alternative: SSM state = sparse graph-like structure with ternary routing
- Hybrid: SSM evolution + periodic knowledge consolidation into graph structure
