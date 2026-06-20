# R11-A4: Ternary-Aware Optimizer — Synthesized Findings

**Round:** R11 | **Aspect:** A4 | **Waves:** W1-A, W1-B, W2-A, W2-B | **Date:** 2026-03-06

All 20 findings pass Gate v3 (G1 actionability + G2 applicability + G3 no-LEGO).

---

## Summary Table

| ID | Technique | File | Gain | Source |
|----|-----------|------|------|--------|
| R11-A4-W2-B-F4 | SHAPLEY-BOTTLENECK-DIAGNOSIS | train.py:4700 | 10-20% | W2-B |
| R11-A4-W2-B-F3 | MECHANISM-DESIGN-COMMITMENT-BONUS | ternary.py:2115 | 8-16% | W2-B |
| R11-A4-W2-A-F4 | TWO-TIMESCALE-SA-SCALE-COUPLING | train.py:2448 | 8-16% | W2-A |
| R11-A4-W1-B-F4 | LION-THRESHOLD-CROSSING-MOMENTUM-BOOST | lion.py:71 | 10-18% | W1-B |
| R11-A4-W1-A-F4 | GROUP-BOUNDARY-DISTANCE-ADAPTIVE-LR | lion.py:59 | 10-20% | W1-A |
| R11-A4-W1-A-F1 | TERNARY-FLIP-PROBABILITY-LR-SCALING | lion.py:59 | 8-18% | W1-A |
| R11-A4-W1-A-F5 | GROUP-FLIP-RATE-GATED-OPTIMIZER-PASS | ternary_sign.py:142 | 15-30% | W1-A |
| R11-A4-W2-B-F1 | STACKELBERG-SCALE-LEADER-LR | lion.py:55 | 5-12% | W2-B |
| R11-A4-W1-B-F1 | FRONTIER-WD-SUPPRESSION | lion.py:57 | 8-15% | W1-B |
| R11-A4-W2-A-F2 | MIXING-TIME-DERIVED-BETA1 | lion.py:50 | 6-14% | W2-A |
| R11-A4-W2-B-F5 | ESS-COMMITMENT-STABILITY-FILTER | commitment.py:128 | 6-14% | W2-B |
| R11-A4-W1-B-F3 | PER-GROUP-SIGMA-G-TURBULENCE-LR-DAMPING | train.py:1590 | 6-12% | W1-B |
| R11-A4-W2-A-F1 | ROBBINS-MONRO-SCHEDULE-COMMITMENT | ternary.py:2120 | 5-12% | W2-A |
| R11-A4-W1-A-F3 | TERNARY-UPDATE-DIRECTION-UTILITY-FILTER | lion.py:69 | 6-14% | W1-A |
| R11-A4-W1-B-F2 | COMMITTED-WD-ZERO-SHADOW | lion.py:57 | 5-10% | W1-B |
| R11-A4-W2-A-F5 | GEOMETRIC-ERGODICITY-RATE-CLIP | train.py:1590 | 5-11% | W2-A |
| R11-A4-W1-A-F2 | SIGN-CONSENSUS-LION-SKIP | lion.py:60 | 5-12% | W1-A |
| R11-A4-W2-A-F3 | MARTINGALE-STOPPED-OPTIMIZER | lion.py:55 | 4-9% | W2-A |
| R11-A4-W2-B-F2 | NASH-EQUILIBRIUM-COMMITMENT-REGULARIZER | commitment.py:130 | 4-9% | W2-B |
| R11-A4-W1-B-F5 | LAYER-FIRST-COMMIT-LR-RESTART | train.py:4717 | 4-9% | W1-B |

---

## Critical Conflict

**W1-B-F3 (PER-GROUP-SIGMA-G-TURBULENCE-LR-DAMPING) vs W2-A-F5 (GEOMETRIC-ERGODICITY-RATE-CLIP)**

- **W1-B-F3:** High sigma_g → reduce LR (damping prevents oscillation)
- **W2-A-F5:** High sigma_g (noise-dominated) → increase LR (maintain spectral gap)

**Verdict:** Implement both. W1-B-F3 first (lower risk: direct variance damping). W2-A-F5 second with empirical validation: measure mixing time per group; if W1-B-F3 under-damps, W2-A-F5's spectral-gap correction will activate.

---

## Implementation Priority (Recommended Order)

1. **W1-A-F5** (15-30% gain): Group flip-rate optimizer pass — reuses stability[]
2. **W1-A-F4** (10-20% gain): Boundary-distance adaptive LR — per-group modulation
3. **W1-B-F4** (10-18% gain): Threshold-crossing momentum boost — directional gating
4. **W2-B-F4** (10-20% gain): Shapley bottleneck diagnosis — logging + adaptive boost
5. **W1-A-F1** (8-18% gain): Flip-probability LR scaling — probabilistic frontier damping
6. **W2-B-F3** (8-16% gain): Mechanism-design commitment bonus — gradient incentive
7. **W2-A-F4** (8-16% gain): Two-timescale SA scale coupling — separate param groups
8. **W1-B-F1** (8-15% gain): Frontier WD suppression — boolean mask on WD
9. **W1-A-F3** (6-14% gain): Update-direction utility filter — bias filter
10. **W2-A-F2** (6-14% gain): Mixing-time-derived beta1 — Markov chain theory
11. W1-B-F3 (6-12% gain): Turbulence-LR damping — **implement first in conflict pair**
12. W2-A-F5 (5-11% gain): Ergodicity-rate clip — **empirical validation gate**
13. W2-B-F5 (6-14% gain): ESS commitment filter — geometric stability
14. W1-B-F2 (5-10% gain): Committed WD zero — simple zero on committed
15. W1-A-F2 (5-12% gain): Sign-consensus skip — amortized O(num_groups)
16. W2-A-F1 (5-12% gain): Robbins-Monro per-weight schedule — SA theory
17. W2-B-F1 (5-12% gain): Stackelberg best-response LR — game-theoretic coupling
18. W2-A-F3 (4-9% gain): Martingale-stopped optimizer — momentum zeroing on commit
19. W2-B-F2 (4-9% gain): Nash-equilibrium commitment regularizer — threshold adaptation
20. W1-B-F5 (4-9% gain): Layer-first-commit LR restart — per-layer mini-warmup

**Composability:** All findings are orthogonal; implement in gain-order with smoke tests after each batch (3-5 findings per batch).
