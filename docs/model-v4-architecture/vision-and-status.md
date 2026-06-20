# Model v4 — Beyond Autoregressive Token Prediction

## The Question

We've optimized ternary SSM + sliding-window attention (model v2/v3) to near its limits:
- RLEC rank=16: -5.91% PPL improvement
- SSM: O(1) memory per token
- Ternary: no matmul, add/subtract only

But we've been optimizing HOW to predict the next token more efficiently.

**The real question: Is "predict the next token" even the right paradigm?**

LLMs work by predicting the next word in a sequence. This is incredibly general but potentially incredibly wasteful:
- Every token prediction activates ALL parameters
- Knowledge is stored implicitly in dense weight matrices
- The model "re-derives" everything from scratch for each forward pass
- Inference cost scales linearly with output length, regardless of difficulty

What if there's a fundamentally different way to build a system that:
1. Answers questions intelligently (matches DeepSeek, Claude Opus quality)
2. Uses dramatically less memory and compute
3. Doesn't necessarily predict tokens one at a time

## Quality Baseline — INVIOLABLE CONSTRAINT

**This is NOT a trade-off. This is a HARD REQUIREMENT.**

Whatever paradigm we adopt MUST deliver:

1. **Same quality or BETTER than frontier models** (DeepSeek-V3, Claude Opus, GPT-4o)
   - At equivalent effective intelligence, not parameter count
   - Preferably BETTER quality, not just "close enough"

2. **Zero quality regression in ANY dimension:**
   - Complex multi-step reasoning (math, logic, code)
   - Factual accuracy — no increase in hallucination
   - Instruction following fidelity
   - Long-context coherence
   - Creative and nuanced language generation
   - Tool use and structured output

3. **Disqualification criteria** — a paradigm is REJECTED if it:
   - Produces more hallucinations than autoregressive baselines
   - Loses complex reasoning ability (chain-of-thought, multi-step)
   - Degrades fluency or coherence
   - Cannot handle long contexts
   - Trades quality for speed in ANY measurable way

**The goal is: SAME or BETTER intelligence with DRAMATICALLY less compute/memory.**
**NOT: worse intelligence that's faster.**

Research agents MUST evaluate every finding against this constraint. If a technique is efficient but quality-degrading, mark it as REJECTED immediately.

## What We're Exploring

### Quality-First Filter
Every paradigm candidate MUST pass this filter BEFORE being considered:
- Does published research show quality matching autoregressive at equivalent scale?
- If quality is lower, is the gap closing rapidly (trajectory matters)?
- Is there a theoretical argument that quality CAN match with further development?
- If none of the above: REJECT immediately, don't waste research time.

We pursue **TWO EQUALLY IMPORTANT TRACKS** in parallel:

### Track A — Complete Paradigm Shift (Beyond Autoregressive LLM)

Radical departure from next-token prediction entirely:
1. **Non-autoregressive generation** — predict multiple/all tokens at once instead of sequentially
2. **Knowledge as structure, not weights** — graphs, databases, programs instead of dense matrices; retrieval-first intelligence
3. **Neurosymbolic approaches** — neural perception + symbolic reasoning combined; differentiable reasoning frameworks
4. **Radical architecture departures** — NOT transformer/SSM variants; explore KAN, liquid nets, hyperdimensional computing, spiking NNs, cellular automata
5. **Energy-based / diffusion models for text** — non-autoregressive generation via diffusion, energy-based, or alternative training objectives
6. **Mixture-of-paradigms** — different computation pathways for different types of thinking

### Track B — SSM Revolution (Radical Improvements WITHIN SSM Paradigm)

Keep the SSM core but fundamentally transform what it can do:
1. **Ultra-efficient SSM state** — compress state representation to use dramatically less RAM (sub-linear memory not just per-token, but overall)
2. **SSM variants that eliminate attention** — pure SSM with retrieval capability; richer state formulations that replace attention entirely
3. **Parallel/batched SSM inference** — break the sequential bottleneck; hardware-native SSM implementations with chunked processing
4. **Novel state-space formulations** — pack more intelligence per bit; state-space alternatives to attention for retrieval; infinite context via state compression
5. **Ternary-native SSM** — SSMs designed FROM THE GROUND UP for ternary (not ternary bolted onto SSM)
6. **SSM performance breakthroughs** — dramatically faster training and inference through specialized SSM kernels and optimizations

**Priority:** Both tracks have equal status. Track A explores if there's a fundamentally better way. Track B explores how far we can push SSM if we stop trying to make it look like Transformers.

### What We Already Know Works (from v2/v3)
- Ternary quantization is viable for all layers
- SSM provides efficient sequence processing
- RLEC compensates quantization error effectively
- STE gradient flow is fragile — orthogonal techniques work best
- C-level optimizations only matter at 1B+ scale

## Research Organization

### Folder Structure
```
docs/model-v4-architecture/
├── vision-and-status.md           # THIS FILE — always read first
├── known-findings-index.md        # Master dedup list (ALL findings across ALL rounds)
├── research-roundN-aspectM.md     # Research findings per round/aspect
└── paradigm-evaluation-*.md       # Deep evaluation of specific paradigms
```

### Rules for Research Agents
1. **READ known-findings-index.md BEFORE researching** — if it's listed, DON'T re-discover it
2. **Max 400 lines per document** — split if larger
3. **Each finding gets a unique ID** — format: v4-RN-AM-FX (version-round-aspect-finding)
4. **Document source, date, and relevance** for each finding
5. **Focus on PARADIGM SHIFTS, not incremental improvements** — we don't want "better attention mechanism", we want "alternative to attention entirely"

### Research Loop (per MEMORY.md rule #12)
- 3 tasks per round: A (current), B (next, blocked), C (loop controller)
- 6 aspects per round
- 7 steps per aspect: organize → 2 web researchers → document → 2 math-paradigm-breakers → document → 2 paradigm-breakers → document

## Current Status

- **Phase:** Research Round 1 — STARTING
- **Paradigm candidates evaluated:** 0
- **Total findings:** 0

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-27 | Start model-v4 paradigm research | v2/v3 optimizations exhausted at PyTorch level. User wants fundamental paradigm shift. |
