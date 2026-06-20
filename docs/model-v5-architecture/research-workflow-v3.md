# Research Workflow v3 — Token-Optimized Protocol

## Changes from v2
- **2 waves instead of 3**: Wave 1 (Web Research) + Wave 2 (Math+Paradigm combined). Paradigm breakers and math breakers overlap ~60% — combining saves 12 agent calls per round.
- **Single merge per aspect**: code-writer merges once after both waves, not after each wave.
- **1 commit per aspect**: Batch wave1+wave2 into single commit. Reduces 18 commits to 6.
- **Strict output budget**: Max 8 findings per agent, max 5 lines per finding. Agents return ONLY IDs to orchestrator.
- **Architecture brief**: Agents read .workspace/implementation/architecture-brief.md instead of re-reading code.

## File Structure

```
.workspace/research/
  dedup-registry.txt         # Persistent
  r{N}-a{N}-w{W}-{A|B}.md   # Temp (deleted after merge)

docs/model-v5-architecture/
  research-round{N}-aspect{N}-findings.md   # Final merged doc (permanent)
  research-workflow-v3.md                    # THIS FILE
```

## Round Structure

Each round has 6 aspects. Each aspect has 2 waves.
- **Wave 1**: 2 research-reader agents (parallel, themes A+B)
- **Wave 2**: 2 math-paradigm-breaker agents (parallel, themes A+B)
- **Merge**: 1 code-writer call (merge 4 temp files into 1 final doc)
- **Commit**: 1 git commit per aspect

Total per round: 6 aspects x (4 agents + 1 merge) = 30 calls + 6 commits
(vs v2: 36 agents + 18 merges + 18 commits)

## Agent Prompt Template — Wave 1 (Web Research)

```
Research: R{N} Aspect {N} — [Aspect Name] — Theme {A|B}: [Description]

CONTEXT: Read architecture brief at:
  C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.workspace\implementation\architecture-brief.md
DEDUP: Read and respect:
  C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.workspace\research\dedup-registry.txt
[If wave 1 exists]: Read previous findings at docs/model-v5-architecture/research-round{N}-aspect{N}-findings.md

FOCUS: [3-5 sentences, theme-specific]
CONSTRAINT: Only findings with ZERO tradeoffs — pure benefits. Deep algorithmic/mathematical detail required.
DATE FILTER: 2024-2026 papers/techniques only.

OUTPUT FORMAT — max 8 findings, max 5 lines each:
### R{N}-A{N}-F{N}: SHORT_NAME
- **Src:** [ref] | **Gain:** [quantified] | **V:** PROMISING/SPECULATIVE
- **Principle:** [1 sentence: the mathematical principle]
- **Application:** [1 sentence: how it applies to ternary SSM training]
- **Key insight:** [1 sentence: why this has zero tradeoffs]

End with:
## Key Insights (max 5 bullets)
## Seed Questions for Wave 2 (max 4 questions)

WRITE OUTPUT TO:
  C:\Users\th1rt\Documents\Workspaces\code-ai-workspace\.workspace\research\r{N}-a{N}-w1-{A|B}.md

RETURN TO ORCHESTRATOR ONLY:
  - Count: N findings
  - IDs: F1:Name, F2:Name, ...
  - Top 2 impactful (1 sentence each)
```

## Agent Prompt Template — Wave 2 (Math + Paradigm Combined)

Same as Wave 1 but:
- Agent type: math-paradigm-breaker
- Finding IDs: M instead of F
- Reads Wave 1 doc for context + seed questions
- Focus: Novel mathematical formulations AND radical paradigm alternatives
- Not literature review — DERIVE new approaches from principles
- Output to: r{N}-a{N}-w2-{A|B}.md

## Code-Writer Merge Instructions

When orchestrator says "merge aspect {N}":
1. Read all temp files: r{N}-a{N}-w1-A.md, w1-B.md, w2-A.md, w2-B.md
2. Merge + dedup (remove findings with same principle, keep best version)
3. Renumber sequentially within aspect
4. Write to: docs/model-v5-architecture/research-round{N}-aspect{N}-findings.md
5. Append new IDs to dedup-registry.txt
6. Delete temp files

## Aspect Selection for Speed/Efficiency Focus

For rounds focused on training speed and efficiency, use these 6 aspects:

1. **Gradient Computation**: Faster ways to compute/estimate gradients through ternary quantization
2. **Training Decomposition**: Split training into independent parts, accumulative training, checkpoint-efficient methods
3. **Optimizer Efficiency**: Reduce optimizer state memory and computation, ternary-aware optimizers
4. **Data Pipeline**: Zero-overhead data loading, prefetch optimization, sequence-level parallelism
5. **Forward/Backward Pass**: Fused operations, computation reordering, memory-compute tradeoffs
6. **Convergence Acceleration**: Fewer steps to same loss, better initialization, training shortcuts

## Post-/compact Restoration

After ANY /compact:
1. Read THIS file
2. Read dedup-registry.txt
3. Check git log for last committed aspect
4. Resume from next wave/aspect
