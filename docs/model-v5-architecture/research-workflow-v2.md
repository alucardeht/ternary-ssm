# Research Workflow v2 — Optimized Protocol

## Overview

This document defines the EXACT research workflow for all future research rounds. After any /compact, the orchestrator MUST read this file to restore workflow knowledge.

## File Structure

```
.workspace/research/
├── dedup-registry.txt       # All finding IDs + known techniques (persistent)
├── r{N}-a{N}-w{M}-themeA.md # Temp output from agent A (cleaned after merge)
├── r{N}-a{N}-w{M}-themeB.md # Temp output from agent B (cleaned after merge)
└── (temp files only, cleaned after each wave)

docs/model-v5-architecture/
├── research-round{N}-aspect{N}.md          # Final Wave 1 doc (permanent)
├── research-round{N}-aspect{N}-wave2.md    # Final Wave 2 doc (permanent)
├── research-round{N}-aspect{N}-wave3.md    # Final Wave 3 doc (permanent)
└── research-workflow-v2.md                  # THIS FILE (permanent)
```

## Dedup Registry

**Path:** `C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.workspace\research\dedup-registry.txt`

**Format:**
```
# Section: FINDINGS
R2-A1-F1:Hypernetwork-Threshold
R2-A1-F2:Annealing-Adaptive-Step
...

# Section: KNOWN_TECHNIQUES (do not report these as findings)
cosine-annealing
gradient-accumulation
...
```

**Maintained by:** code-writer appends new IDs after each wave's documentation is written.

## Agent Writing Protocol

### WHO writes WHERE:

| Agent Type | Can Write Temp Files? | Method | Writes To |
|---|---|---|---|
| paradigm-breaker | YES | mcp__may-la__write | .workspace/research/ ONLY |
| math-paradigm-breaker | YES | mcp__may-la__write | .workspace/research/ ONLY |
| research-reader | MAYBE | mcp__may-la__write (test) | .workspace/research/ ONLY |
| code-writer | YES | Write tool | docs/ (final) + .workspace/ (registry) |

### SAFETY RULES for agents writing temp files:
1. Agents write ONLY to `.workspace/research/` — NEVER to `docs/`, `src/`, or root
2. File names follow pattern: `r{round}-a{aspect}-w{wave}-theme{A|B}.md`
3. If agent cannot write (tool unavailable), return output as text — orchestrator passes to code-writer
4. Temp files are DELETED by code-writer after merge into final doc

### FALLBACK if agent can't use mcp__may-la__write:
The agent returns its full output as text. The orchestrator does NOT deeply read it — just tells code-writer:
"Agent returned research output. Write it to [temp path], then proceed with merge."
This keeps agent output out of orchestrator context as much as possible.

## Wave Workflow — Step by Step

### WAVE 1: Web Research (2 research-reader agents)

**Orchestrator does:**
1. Formulates 2 distinct themes (A and B)
2. Launches 2 agents in parallel with this prompt template:

```
## Research Task: [Aspect Name] — Theme [A/B]: [Theme Description]

### BEFORE RESEARCHING:
1. Read the dedup registry at:
   C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.workspace\research\dedup-registry.txt
   - FINDINGS section: Do NOT re-discover any listed finding
   - KNOWN_TECHNIQUES section: Do NOT report basic techniques as findings

2. [If previous waves exist for this aspect]:
   Read the previous wave doc at:
   C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\docs\model-v5-architecture\research-round{N}-aspect{N}[-wave{M}].md
   Pay attention to "Key Insights" and seed questions.

### YOUR RESEARCH FOCUS:
[Theme-specific instructions, 3-5 sentences]

### OUTPUT FORMAT:
Write your findings as a markdown document. Each finding:

### R{round}-A{aspect}-F{N}: SHORT_NAME — One-line description
- **Src:** [Author, venue, year] | **Gain:** [quantified] | **V:** PROMISING/NEEDS-WORK/SPECULATIVE
- [2-3 sentences: What is it? Why ternary SSM? Key math insight?]

End with:
## Key Insights — [Aspect] Wave 1
[5-6 bullet points synthesizing themes]

## For Math-Paradigm-Breakers (Wave 2)
[4-5 seed questions]

### WRITING OUTPUT:
Write your complete output to:
C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.workspace\research\r{N}-a{N}-w1-theme{A|B}.md
Use the mcp__may-la__write tool with this exact path.

Then return to me ONLY:
- Number of findings
- List: "F1:NAME, F2:NAME, ..." (IDs and short names only)
- Top 3 most impactful (1 sentence each)
```

3. Waits for both agents
4. Delegates to code-writer: "Read temp files themeA and themeB, merge+dedup, write final doc to docs/model-v5-architecture/research-round{N}-aspect{N}.md. Then append new finding IDs to dedup-registry.txt."
5. Commits

### WAVE 2: Math-Paradigm-Breakers (2 math-paradigm-breaker agents)

Same template but:
- Finding IDs use M instead of F: `R{N}-A{N}-M{N}: NAME`
- Agents read Wave 1 doc file for context
- Focus: novel mathematical formulations, not literature review
- Seed questions come from Wave 1's "For Wave 2" section
- Output to: `.workspace/research/r{N}-a{N}-w2-theme{A|B}.md`

### WAVE 3: Paradigm-Breakers (2 paradigm-breaker agents)

Same template but:
- Finding IDs use P instead of M: `R{N}-A{N}-P{N}: NAME`
- Agents read Wave 1 AND Wave 2 doc files
- Focus: radical paradigm alternatives, not incremental improvements
- Include summary section: "ASPECT COMPLETE — SUMMARY ACROSS ALL WAVES"
- Output to: `.workspace/research/r{N}-a{N}-w3-theme{A|B}.md`

## Code-Writer Merge Instructions

When orchestrator says "merge wave files":

1. Read both temp files from `.workspace/research/`
2. Combine findings, renumber sequentially (F1, F2, ...), remove exact duplicates
3. Keep the wave summary sections (Key Insights, Seed Questions, Composites)
4. Write final merged doc to `docs/model-v5-architecture/research-round{N}-aspect{N}[-wave{M}].md`
5. Append ALL new finding IDs to dedup-registry.txt in compact format:
   ```
   R{N}-A{N}-{F|M|P}{N}:{SHORT_NAME}
   ```
6. Delete the temp files from `.workspace/research/`

## Commit Flow

1 commit per wave (3 per aspect):
- After Wave 1 merge: commit final doc + updated registry
- After Wave 2 merge: commit wave2 doc + updated registry
- After Wave 3 merge: commit wave3 doc + updated registry

## Known Techniques Exclusion List

These are in the KNOWN_TECHNIQUES section of dedup-registry.txt. Agents skip reporting these as findings:

- cosine-annealing, warmup-schedulers, OneCycleLR
- gradient-accumulation, gradient-clipping (basic)
- STE (straight-through-estimator, basic concept)
- knowledge-distillation (basic concept)
- data-augmentation (basic concept)
- mixed-precision-training (basic FP16/BF16)
- dropout, stochastic-depth (basic regularization)
- AdamW, SGD, basic optimizers
- batch-normalization, layer-normalization
- weight-decay, L1/L2-regularization
- progressive-quantization (basic concept)
- curriculum-learning (basic concept — proven INEFFECTIVE for NLP)
- gradient-checkpointing (basic concept)
- model-parallelism, data-parallelism (basic concepts)
- quantization-aware-training (basic concept)

## Post-/compact Restoration

After ANY /compact:
1. Read THIS file: `docs/model-v5-architecture/research-workflow-v2.md`
2. Read dedup registry: `.workspace/research/dedup-registry.txt`
3. Check which aspect/wave was last committed (git log)
4. Resume from next wave/aspect
