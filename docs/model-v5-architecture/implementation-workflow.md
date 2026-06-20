# v5 Implementation Workflow

## Overview
File-based pipeline for implementing research findings. Designed for token efficiency, /compact resilience, and systematic validation.

## File Structure
```
.workspace/implementation/
├── architecture-brief.md      — Compact v5 reference for agents (~150 lines)
├── baseline-metrics.md        — Performance snapshot before implementations
├── tracker.md                 — Status per finding (persistent across /compact)
├── triage-aspect-N.md         — Batch analysis output per aspect
└── recipe-FINDING-ID.md       — Implementation recipe per finding
```

## Phases

### Phase 0: Setup (once)
- Create architecture-brief.md (architecture-enforcer audits src/model_v5/)
- Capture baseline metrics (200 steps at 6M, document loss/throughput/VRAM)
- Create empty tracker

### Phase 1: Triage (parallelizable, no GPU)
- Input: 1 aspect doc (research findings) + architecture-brief
- Agent: architecture-enforcer
- Output: triage file with per-finding verdict
- Verdicts: IMPLEMENT (math justification) | SKIP (reason) | EXPLORE (needs thought)
- Can run 2-3 agents on different aspects simultaneously
- After all triage: consolidate into tracker.md

### Phase 2: Recipe Creation (per IMPLEMENT finding)
- Input: finding details + architecture-brief + relevant source code
- Agent: architecture-enforcer
- Output: recipe file in .workspace/implementation/recipe-XXX.md
- Recipe contains:
  - Finding ID and name
  - Mathematical principle being exploited
  - Exact files to modify (with paths)
  - Exact changes in implementation terms
  - Expected behavior after implementation
  - Debug logging points (MANDATORY)
  - Test criteria (success/failure)
  - Rollback plan

### Phase 3: Implementation (SEQUENTIAL, per finding)
1. code-writer reads recipe → implements changes
2. basher runs smoke test (15 steps, no GPU warning needed)
3. Verify debug logs show technique activated
4. If crash/fail → architecture-enforcer analyzes → code-writer fixes
5. If pass → update tracker to IMPLEMENTED
6. Commit code changes

### Phase 4: Validation (needs user permission for GPU benchmark)
1. WARN USER before running performance benchmark
2. Run benchmark against baseline (same config)
3. Compare: loss, throughput, VRAM
4. Update tracker with results
5. Commit with results

## Tracker Format

| ID | Finding | Status | Verdict | Result | Commit |
|----|---------|--------|---------|--------|--------|
| R2-A6-P2 | TADA | IMPLEMENT | 10x capacity via iteration | - | - |

Status flow: PENDING → SKIP | IMPLEMENT → RECIPE → IMPLEMENTED → TESTING → VALIDATED | REJECTED

## Recipe Template

```markdown
# Recipe: [FINDING-ID] — [NAME]

## Principle
[1-2 sentences: mathematical principle being exploited]

## Applicability
[Why this applies to ternary SSM specifically]

## Changes
### File: src/model_v5/xxx.py
- Function/class: XXX
- Change: [description]

## Debug Logging
- Add print at [location]
- Expected output: [what logs should show]

## Test Criteria
- Success: [measurable condition]
- Failure: [what indicates not working]
- Red flags: [suspicious results]

## Rollback
- Revert [files] to pre-implementation
- Config: set technique=None
```

## Rules
1. NEVER implement without recipe first
2. NEVER implement 2+ techniques without validating each
3. ALWAYS check debug logs before declaring results
4. ALWAYS compare against baseline-metrics.md
5. Tracker updates after EVERY status change
6. Commit after EVERY successful implementation

## Post-/compact Restoration
1. Read `.workspace/implementation/tracker.md`
2. Read `.workspace/implementation/architecture-brief.md`
3. Read `docs/model-v5-architecture/implementation-workflow.md` (this doc)
4. Check for in-progress recipe files
5. Resume from last tracked status
