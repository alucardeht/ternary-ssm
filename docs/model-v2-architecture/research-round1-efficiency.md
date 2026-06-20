# Research Discoveries — Ternary SSM Architecture

**Date:** 2026-02-25 | 28 agents across 5 research rounds | Consolidated single source of truth

**This file is an index. The content has been split into two parts for agent readability (max ~400 lines each).**

---

## Index

### Part 1: Performance Baselines & Architecture (research-round1-efficiency-part1.md)

- **Part 1:** Current Measured Performance (inference, training, quality, compression)
- **Part 2:** Paper Projections (already published, skip in future research)
- **Part 3:** New Discoveries — Kernel & GPU Speed (masked reduction, fused kernel, CUDA graphs, FP4, integer pipeline)
- **Part 4:** CPU & Edge Deployment (compiled binary, SIMD, WebGPU, NPU, mesh inference)
- **Part 5:** Memory & Training Optimization (codebook training, SSD checkpointing, XOR-delta, layer streaming, adaptive precision)
- **Part 6:** Quality Improvements (multi-codebook, Tequila re-test, LOTION, evolutionary polish, ensemble, test-time compute, self-distillation)
- **Part 7:** Inference Architecture & System-Level (hierarchical softmax, speculative decoding, context-independent serving, state database, state algebra, split-brain)
- **Part 8:** Ternary-Unique Advantages (structural advantages impossible with FP16)

### Part 2: Roadmap & Strategy (research-round1-efficiency-part2.md)

- **Part 9:** Technique Synergies & Conflicts (top 10 synergies, hard conflicts, soft conflicts, redundancies)
- **Part 10:** Implementation Roadmap (dependency graph, critical path, phase-by-phase breakdown A-E)
- **Part 11:** Zero-Cost Validation Experiments (11 experiments to validate discoveries before implementation)
- **Part 12:** Unexplored Directions (next research round gaps and recommendations)
- **Part 13:** Strategic Competitive Advantages (the moat — 5 permanent advantages, killer applications, scaling story)
- **Part 14:** Projected End-State (all phases complete — metrics comparison)

---

## Quick Links

| Topic | File | Section |
|-------|------|---------|
| Measured performance | Part 1 | Part 1 |
| GPU speed improvements | Part 1 | Part 3 |
| Training optimization | Part 1 | Part 5 |
| Quality gap closure | Part 1 | Part 6 |
| What to do next (roadmap) | Part 2 | Part 10 |
| Validate before building | Part 2 | Part 11 |
| Strategic advantages | Part 2 | Part 13 |

---

## How to Use These Files

1. **For performance analysis:** Read Part 1
2. **For implementation planning:** Read Part 2 (specifically Part 10 roadmap)
3. **For validation experiments:** Read Part 2 (Part 11)
4. **For strategic context:** Read Part 2 (Part 13)
5. **For research gaps:** Read Part 2 (Part 12)

Each part is self-contained with clear section headers. Cross-references between parts use file names and part numbers.
