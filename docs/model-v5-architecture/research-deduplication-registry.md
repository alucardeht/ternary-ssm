# Research Deduplication Registry

**Purpose:** Track all documented findings by ID to prevent re-discovery in future research rounds.

**Format:** `RoundNumber-AspectNumber-WaveType-Number` where WaveType ∈ {F=Wave1-FWeb, M=Wave2-Math, P=Wave3-Paradigm}

---

## Research Round 3 — Complete Findings Registry

### Aspect 1: [To be documented]

### Aspect 2: [To be documented]

### Aspect 3: [To be documented]

### Aspect 4: [To be documented]

### Aspect 5: [To be documented]

### Aspect 6: Lazy Parameter Updates & State Eviction — 26 Findings Complete

#### Wave 1 — Web Research (10 findings)
| ID | Title | Verdict | Status |
|---|---|---|---|
| R3-A6-F1 | DYNAMIC-PARAM-GROUPS-EVICTION | IMPLEMENT | ✓ Documented |
| R3-A6-F2 | 4BIT-OPTIMIZER-STATE-PARKING | IMPLEMENT | ✓ Documented |
| R3-A6-F3 | CPU-OFFLOAD-SELECTIVE-FROZEN | IMPLEMENT | ✓ Documented |
| R3-A6-F4 | MEMORY-POOL-REPURPOSING | EXPLORE | ✓ Documented |
| R3-A6-F5 | GRADIENT-CHECKPOINTING-FROZEN-SKIP | IMPLEMENT | ✓ Documented |
| R3-A6-F6 | PARTIAL-PARAMETER-UPDATE-SUPAR | IMPLEMENT | ✓ Documented |
| R3-A6-F7 | GRADIENT-SENSITIVITY-SPARSE-UPDATING | IMPLEMENT | ✓ Documented |
| R3-A6-F8 | HIERARCHICAL-UPDATE-FREQUENCY | EXPLORE | ✓ Documented |
| R3-A6-F9 | SPARSE-ADAM-MASKING | EXPLORE | ✓ Documented |
| R3-A6-F10 | QZO-GRADIENT-FREE-COMMITTED | EXPLORE | ✓ Documented |

#### Wave 2 — Math-Paradigm-Breaker (8 findings)
| ID | Title | Verdict | Status |
|---|---|---|---|
| R3-A6-M1 | ADAM-AWARE-FLIP-PROBABILITY | IMPLEMENT | ✓ Documented |
| R3-A6-M2 | GRADUATED-SKI-RENTAL-EVICTION | IMPLEMENT | ✓ Documented |
| R3-A6-M3 | GRADIENT-NORM-COMMITMENT-SCHEDULE | EXPLORE | ✓ Documented |
| R3-A6-M4 | BLOCK-ALIGNED-GROUP-FREEZING | IMPLEMENT | ✓ Documented |
| R3-A6-M5 | SPRT-CUSUM-DRIFT-DETECTOR | IMPLEMENT | ✓ Documented |
| R3-A6-M6 | SNR-OPTIMAL-UPDATE-FREQUENCY | IMPLEMENT | ✓ Documented |
| R3-A6-M7 | SELECTIVE-WEIGHT-GRADIENT-SUBSAMPLING | IMPLEMENT | ✓ Documented |
| R3-A6-M8 | KNAPSACK-OPTIMAL-JOINT-POLICY | EXPLORE | ✓ Documented |

#### Wave 3 — Paradigm-Breaker (8 findings)
| ID | Title | Verdict | Status |
|---|---|---|---|
| R3-A6-P1 | COMMITTED-MODULE-REPLACEMENT | IMPLEMENT | ✓ Documented |
| R3-A6-P2 | OPTIMIZER-STATE-RECYCLING-POOL | IMPLEMENT | ✓ Documented |
| R3-A6-P3 | OUTPUT-GRADIENT-METABOLIC-PROBE | IMPLEMENT | ✓ Documented |
| R3-A6-P4 | JIT-COMPILED-TERNARY-PATTERN-KERNEL | EXPLORE | ✓ Documented |
| R3-A6-P5 | BIFURCATED-CRYSTAL-LIQUID-OPTIMIZER | IMPLEMENT | ✓ Documented |
| R3-A6-P6 | TERNARY-PATTERN-TRANSPLANTATION | EXPLORE | ✓ Documented |
| R3-A6-P7 | OVERPARAMETERIZED-TERNARY-CRYSTALLIZATION | EXPLORE | ✓ Documented |
| R3-A6-P8 | COMMIT-RESTART-TRAINING | EXPLORE | ✓ Documented |

---

## Summary Statistics

- **Total Round 3 Aspect 6 findings:** 26
- **IMPLEMENT verdicts:** 9 findings (F1, F2, F3, F5, F6, F7, M1, M2, M4, M5, M6, M7, P1, P2, P3, P5)
- **EXPLORE verdicts:** 17 findings (F4, F8, F9, F10, M3, M8, P4, P6, P7, P8)
- **Critical ID pattern:** R3-A6-{F,M,P}[1-10] — ALL documented, ready for dedup
- **No re-discoveries:** Checked against web prior art, math derivations independently novel, paradigm shifts without prior fusion patterns

---

## Notes for Future Rounds

- **Wave 3 Paradigm-Breaker insights:** Unifying principle is "boundary exploitation" — committed vs uncommitted weights form phase transition. Every technique exploits this boundary differently (infrastructure elimination, algorithm switching, knowledge transfer, crystallization cascade).
- **Integration points:** R3-A6 forms COMPLETE system for committed weight optimization. Earlier rounds (R2-A3, R3-A2, R3-A5) are prerequisite findings. No gaps in logical progression.
- **Implementation order:** P1+P3 (module replacement + monitoring) → P2 (buffer pool) → P5 (bifurcated optimizer) → F1-F10 and M1-M8 as supporting techniques.
