# Research Discoveries — Index

**This file has been split into 4 parts for agent readability (max ~350-400 lines each).**

All discoveries from 28 agents across 5 research rounds and Phase 28 implementation are now organized as:

## Parts Index

### [Part 1: Performance Baselines & GPU Speed](research-discoveries-part1.md)
- **Coverage:** Current measured performance (50M, 6M scales), paper projections, GPU kernel acceleration techniques (masked reduction, fused kernels, CUDA graphs, FP4, integer pipeline)
- **Size:** ~350 lines
- **Key metrics:** Baseline 75.6 tok/s GPU, 14.3 tok/s CPU, 7.4% quality gap vs FP16
- **Use this for:** Understanding current performance and identifying immediate kernel-level speedups

### [Part 2: CPU/Edge, Memory, Quality, and Ternary Advantages](research-discoveries-part2.md)
- **Coverage:** CPU & edge deployment (compiled binary, bitnet.cpp, WebGPU, mobile NPU), memory optimization (codebook training, SSD checkpointing, XOR-delta, layer streaming), quality improvements (K=2, Tequila, LOTION, polish, ensemble), and ternary-unique advantages
- **Size:** ~350 lines
- **Key techniques:** Codebook reduces training VRAM 100x, ensemble beats FP16 by 17.7%, XOR-delta enables 10 variants in 4.2 GB
- **Use this for:** Planning memory optimization and quality improvement work

### [Part 3: Synergies, Roadmap, Round 2 Findings](research-discoveries-part3.md)
- **Coverage:** Technique synergies (multiplicative gains), hard conflicts (FP4 vs integer pipeline), implementation roadmap (Tier 0-5 dependencies), zero-cost validation experiments, unexplored directions, competitive advantages (5 permanent structural advantages), end-state projections
- **Size:** ~350 lines
- **Key roadmap:** Phase A (1.5-2x speed, 4-8% quality), Phase B (15-40x speed), Phase C (75-200x speed), Phase D-E (end-state: 200-400 tok/s CPU, 80K concurrent sessions)
- **Use this for:** Implementation planning and validation strategy

### [Part 4: Memory Paradigms, Quality Improvements, Phase 28 Implementation](research-discoveries-part4.md)
- **Coverage:** Radical memory paradigms (self-generating weights, amortized decay, train-serve interleaving, evolutionary training), system-level paradigms (state-centric hierarchy, OS primitives, mmap DAG, circuit compilation), Round 3 quality improvements (20 techniques), paper citations, Phase 28 implementation results (findings 98-109)
- **Size:** ~340 lines
- **Key results:** Phase 28-A mega-kernel 1.17x speedup, Phase 28-B STEWS/angular reg/GaLore/d_state=64 implemented, Phase 28-D torch.compile selective + data loading optimized, POPCOUNT design completed
- **Use this for:** Advanced research directions and implementation status

---

## How to Use This Index

**For new agents starting research:**
1. Read **Part 1** first — understand the measured baselines and why they matter
2. Read **Part 3** (roadmap section) — see dependency graph and what's safe to work on
3. Pick your specific area:
   - GPU speed: Parts 1 + 3 roadmap Phase A-B
   - Quality: Part 2 + Part 4
   - Memory: Part 2 + Part 4 paradigms
   - CPU/edge: Part 2
   - Competitive strategy: Part 3

**For continued work:**
- Check Part 3 roadmap for what depends on what
- Review Part 4 Phase 28 results to avoid duplicate implementation
- Use zero-cost validation experiments (Part 3) before committing to implementation

**For reference:**
- All 142+ discoveries are indexed across 4 parts
- Paper citations (Part 4 appendix) point to external sources
- Findings from Phases 20-28 complete and documented
- No further research rounds planned until Phase 28 batch completes

---

## Quick Statistics

| Metric | Value |
|--------|-------|
| Total discoveries | 142+ |
| Research rounds | 3 (Round 1: 50 findings, Round 2: 57 findings, Round 3: 35+ findings) |
| Implementation phase results | 12 findings completed (Phase 28-A/B/D) |
| Validation experiments | 22 zero-cost experiments identified |
| Synergy combinations | 10 major (22-187x compound gains) |
| Competitive moats | 5 permanent structural advantages |
| Quality gap closure techniques | 20+ (from -7.4% to +17.7% ensemble) |
| Projected speedup pathways | 6 (ranging 1.5x → 200x depending on implementation) |

---

**Last updated:** 2026-02-26

**Status:** All parts split, indexed, and ready for agent access. Original file consolidates 7 research documents and 5 round reports.

See individual parts for detailed findings, roadmaps, and implementation status.
