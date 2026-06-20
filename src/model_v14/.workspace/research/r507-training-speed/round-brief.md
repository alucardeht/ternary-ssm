# Research Round R507: Massive Scale Training Speed

## Objective
The project's north star is to train a modern-quality 7B-31B-equivalent model on a single consumer GPU, fast. This round (R507) is dedicated exclusively to finding exactly 100 VERY STRONG, paper-worthy discoveries that dramatically accelerate training speed.

## Current Architecture and State
- **Macro**: Ternary weights, State Space Model (SSM), Matmul-Free inference.
- **State**: The GPU pipeline has achieved 100% residency on the RTX 5070 with SISSF and GASSC compilers active. Current peak speed is ~20,142 tok/s for a 6L/d1024 configuration, which projects to 4.3 years to train 1T tokens on a 32B-equivalent scale.
- **Mathematical Map**: Forward pass is dominated by CSR Ternary Matrix-Vector multiplications and recurrences. Backward pass uses custom F1/F3 pathways to evade dense gradient materialization.

## Research Guidelines
- **Constraint**: The user demands 100 VERY STRONG findings.
- **Bar**: A candidate is only valid if it changes training dynamics, ternary geometry, or credit assignment. Local software micro-optimizations (like "unroll a loop") are explicitly rejected unless they expose a new mathematical paradigm.
- **Validation**: Every finding must pass strict validation: a rigorous proof object, cheap falsifier, and no hidden trade-offs.
- **Gain Floor**: 2x speedup is the headline threshold. Do not reject a finding >10% if it opens a new avenue, but push for fundamental architectural restructuring.

## Documentation Expectations
Agents must log their findings in sharded ledgers (e.g., `wave-01-part-01.md`). Do not create giant files.
