# v8/v9 Implementation Log

**Last updated:** 2026-03-09
**V8 status:** Research complete (R18-R23). 5 experiments, all negative. Root cause: random initialization.
**V9 status:** ACTIVE. Pipeline experiment confirmed learning without GD.

## V8 Implementations (in src/model_v7/ as vehicle)

See full log in docs/model-v7-architecture/implementation/implementation-log.md (IMPL-1 through IMPL-11).

## Summary Table

| # | Technique | Source | Result |
|---|-----------|--------|--------|
| IMPL-1 | Hurwitz A∈{-1,0} | R19-B | Bug fix ✓ |
| IMPL-2 | Causal deconfounding | R19-C | Analytical fix ✓ |
| IMPL-3 | HiPPO-LegS (Mamba-1) | R19-C | Already optimal ✓ |
| IMPL-4 | RLS Readout (fixed) | R21-B | -0.002 neutral |
| IMPL-5 | MESU Categorical | R21-B | +0.047 slightly worse |
| IMPL-6 | CUSUM Update Gate | R19-B | **-0.019 positive** |
| IMPL-7 | Kronecker Rank-1 State | R19-D | **-0.006 positive** |
| IMPL-8 | Local Loss/PredSim | R23-B-WEB-4 | **-1.18 positive (13.3%)** |
| IMPL-9 | N4SID Probe Init | R19-A | NEGATIVE (ill-conditioned) |
| IMPL-10 | Matrix Monoid (input-dep A) | R19-D | Neutral (needs longer run) |
| EXP-1 | SSC W_out | R22 | NEGATIVE (random init) |
| EXP-2 | DFA (synthetic) | R22 | NEGATIVE (random labels) |
| EXP-3 | DFA (wikitext) | R23 | NEGATIVE (64 steps insufficient) |
| EXP-4 | Bootstrap+OLS | R23 | NEGATIVE (single trajectory) |
| EXP-5 | EqProp | R23 | NEGATIVE (random init) |
| IMPL-11 | **V9 Pipeline** | R19/R23 | **+2% without GD ← PARADIGM SHIFT** |

## V9 Priorities (next)

1. Initialize x_proj analytically (embedding→SSM projections)
2. Local loss as default (IMPL-8 confirmed 13.3%)
3. Reversible SSM scan (28× VRAM gap)
4. DFA long-run test (1000+ steps)
