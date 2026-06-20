# Ternary SSM Language Model

> **Note — June 2026:** I'm stepping away from this research for personal reasons. This repository is the complete, unfiltered output of months of independent work — the code, the math, the papers, the dead ends, and the breakthroughs. I'm releasing everything in the hope that it may be useful to someone, somewhere. If any of this contributes to your work in any way, a citation or mention would mean a lot. — *Julio*

A research project exploring a novel language model architecture built from first principles: **ternary weights {-1, 0, +1}**, **State-Space Models (SSM)**, and a **custom C/Vulkan GPU training pipeline** — no PyTorch, no CUDA, no ML frameworks.

The goal: train a 7B–31B parameter-equivalent model on a **single consumer GPU** (RTX 5070, 12GB VRAM).

> *"The user is looking for the airplane, not a faster car."*
> — Project Manifesto

---

## What Makes This Different

This is not an optimization of existing LLM recipes. It's a ground-up rethinking of how language models can be trained and deployed:

| Conventional LLMs | This Project |
|-------------------|-------------|
| FP16/BF16 weights (16 bits) | Ternary {-1, 0, +1} (2 bits) |
| Matrix multiplications | Add/subtract/skip only |
| PyTorch + CUDA | Handwritten C + Vulkan compute shaders |
| Transformer attention O(n²) | SSM recurrence O(n) |
| Cluster of A100/H100 GPUs | Single consumer RTX GPU |
| 14+ GB just for 7B weights | ~893 MB for 3.38B via Forward Programming |

### Core Innovations

1. **Matmul-Free Computation** — By constraining weights to `{-1, 0, +1}`, every "multiplication" becomes at most a negation or zero. No floating-point multiply-accumulate needed in the forward pass.

2. **Forward Programming (FP)** — Layer-sequential training where only one layer is active at a time, with all others frozen and packed to INT2. This enables training a 3.38B model in **893 MB VRAM** (vs ~56 GB for standard joint gradient descent).

3. **Custom Vulkan GPU Backend** — The entire training and inference pipeline is written from scratch in C with Vulkan compute shaders, giving fine-grained control over GPU memory, compute dispatch, and pipeline optimization.

4. **SSM Architecture** — Uses State-Space Model dynamics instead of attention mechanisms, providing fixed-size recurrent state with O(1) memory per token regardless of context length.

---

## The Three Barriers

The project identified three fundamental barriers to training large models on consumer hardware:

| Barrier | Status | Detail |
|---------|--------|--------|
| **VRAM** | ✅ Solved | FP + INT2 = 893 MB/stage for 3.38B (R130) |
| **Training Time** | 🔬 Open | 60B tokens ≈ 495 days on RTX 5070. Needs paradigm shift. |
| **Quality** | 🔬 Open | FP structural gap is 65-70%. Best closure: 89% with FP+1000 GD steps (R135) |

---

## Repository Structure

```
├── src/                          # All model implementations
│   ├── model_v2/                 # Early Python prototype
│   ├── model_v5/ – v7/           # Python iterations
│   ├── model_v10/                # First C prototype
│   ├── model_v11/                # Python production with validated pipeline
│   ├── model_v12/                # Pure C/Vulkan SSM (frozen baseline)
│   ├── model_v13/                # Ternary VSA Fractional Engine variant
│   ├── model_v14/                # Native Multimodal Clifford SSM (latest)
│   ├── engine/                   # CMake-based C engine with Vulkan shaders
│   ├── constructive_imagination/ # World-model substrate research
│   └── transformer_residency/    # Transformer comparison track (standby)
│
├── papers/                       # Academic paper drafts (9 papers)
│   ├── paper1-ternary-ssm/       # Core architecture paper (NeurIPS format)
│   ├── paper2-gradient-estimators/
│   ├── paper3-efficiency/
│   ├── paper4-dataset-quality/
│   ├── paper5-tssi-import/       # Ternary Sufficient-Statistic Import
│   ├── paper6-update-objects/
│   ├── paper7-factored-gradient-accumulation/
│   ├── paper8-forward-only-bptt/
│   └── paper9-jl-gradient-quotient/
│
├── memory/                       # Research documentation
│   ├── PROJECT-MANIFESTO.md      # Project vision and research standard
│   ├── model-mental-map.md       # Architecture overview with diagrams
│   ├── mathematical-research-map.md # Mathematical framework
│   ├── pipeline-reference.md     # Validated training pipeline
│   ├── baselines.md              # Performance baselines index
│   ├── strategic-direction.md    # Current research direction
│   ├── project-state.md          # Development state
│   ├── r500-summary.md           # R500 extraordinary round: 69 findings
│   ├── hardware-and-batch-policy.md # Hardware constraints
│   └── historical-results.md    # Experiment history
│
├── docs/                         # Technical documentation
│   ├── architecture-alternatives.md
│   ├── training-math-formalization.md
│   ├── hardware-analysis.md
│   ├── efficiency-techniques.md
│   ├── model-v*-architecture/    # Per-version architecture docs
│   └── training-happy-path.md
│
├── tools/                        # Utility scripts
├── scripts/                      # Build and maintenance scripts
└── tests/                        # Test suite
```

### Model Versions

The codebase shows the full evolution from Python prototypes to native C/Vulkan:

| Version | Language | Architecture | Status |
|---------|----------|-------------|--------|
| v2-v7 | Python | Early ternary experiments | Historical |
| v10 | C | First native prototype | Frozen |
| v11 | Python | Production pipeline with FP+BCD+SAC | Frozen (tag `v11-final`) |
| v12 | C/Vulkan | Pure native SSM training | Frozen baseline |
| v13 | C/Vulkan | Ternary VSA Fractional Engine | In development |
| v14 | C/Vulkan | Clifford Algebra SSM | Latest (active) |

### Building

The native models (v12-v14) use GCC with Vulkan SDK:

```bash
# Requirements: GCC, Vulkan SDK, Windows (tested on RTX 5070)
cd src/model_v12
make          # Produces train.exe
```

Compilation flags: `-std=c99 -O2 -mavx2 -mfma`

### Training Data

The project uses TinyShakespeare for development/testing. The dataset is not included in the repository — download it from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) and place it in `data/tinyshakespeare/input.txt`.

---

## Validated Training Pipeline

The current best pipeline (documented in `memory/pipeline-reference.md`):

### Stage 1: Forward Programming (FP)
```
For each layer k = 0 to (n_layers - 1):
  1. Unpack layer k from INT2 → FP32 on GPU
  2. Train 100-150 steps (CE loss, backward through layer k only)
  3. D4 gradient filter (sign consistency ≥ 5 steps)
  4. ABA early stopping (patience=20, min_steps=30)
  5. Pack layer k → INT2, move to CPU
```
**Measured (3.38B model):** 893 MB peak VRAM, ~29 min total

### Stage 2: mini-BCD (Block Coordinate Descent)
Post-FP refinement on subset of layers. ~28-32% gap closure vs joint GD.

### Validated Speed Gains
- Cache H_k incremental: **33× FP speedup**
- ABA early stopping: **47% fewer FP steps**

---

## Research Findings

Over 500 research rounds were conducted, producing dozens of validated findings. The most significant from R500 (69 mechanisms, 8 paradigm-breaking):

| Finding | Impact | Mechanism |
|---------|--------|-----------|
| **RBLSS** (Rao-Blackwell Layer-State Signal) | 40-60% FP gap closure | Proves E[g_RB] = E[g_joint] for CE softmax |
| **RSSM** (Reversible SSM State Reconstruction) | 99.95% trajectory VRAM reduction | Exact inversion for diagonal ternary SSM |
| **RCFA** (Recurrent Cotangent Forward Accumulator) | 4× context expansion | Proves λ_t^RCFA = λ_t^BPTT for constant diagonal A |
| **TSSI** (Ternary Sufficient-Statistic Import) | 3.5× speedup on byte-level pilot | Corpus-moment import with residual CE training |
| **SAOBM** (Sign-Accumulant Optimizer) | 87% optimizer state reduction | INT8 sign accumulator replaces FP32 moments |
| **TCIR** (Ternary Cotangent INT4) | 87.5% cotangent buffer reduction | Proves ternary SSM ρ(A) ≤ 1 enables INT4 |

**Projected combined impact:** 2-3× improvement (495 days → ~165-250 days)

See `memory/r500-summary.md` for the complete list and `papers/` for formal write-ups.

---

## Mathematical Framework

The project defines formal objectives and speed inequalities:

**Base objective:**
$$T_0(\varepsilon) = \inf \{ t : L(\theta_t) \leq L_{\text{target}} + \varepsilon \}$$

**Speed target:**
$$T_M(\varepsilon) \leq 0.50 \times T_0(\varepsilon) \quad \text{(aspirational 2×)}$$

All costs must be priced:
$$T_M = T_{\text{build}} + T_{\text{score}} + T_{\text{train}} + T_{\text{failed}} + T_{\text{repair}} + T_{\text{verify}} + T_{\text{realize}}$$

See `memory/mathematical-research-map.md` for the complete mathematical architecture.

---

## Key Baselines

| Metric | Value | Source |
|--------|-------|--------|
| 7B/60B-token training projection | ~495 days on RTX 5070 | R150 |
| FP 3.38B VRAM | 893 MB peak | R130 |
| 800-step dev run (400K params) | Loss 12.75 → 1.41 (88.9% reduction) | R336 |
| Vulkan fixture | Loss 11.7143, 728 MB VRAM | Smoke test |
| 6M GD inference | 166 tok/s | R296 |

---

## Hardware Requirements

- **GPU:** Vulkan-capable GPU (developed on RTX 5070, 12GB VRAM)
- **OS:** Windows (primary), Linux (should work with minor adjustments)
- **Compiler:** GCC with AVX2/FMA support
- **Vulkan SDK:** Required for GPU training path
- **RAM:** 16+ GB recommended

---

## Current State & What's Left Open

This project is being released as-is — **I will not be continuing this research.** The VRAM barrier is solved, but training time and quality gaps remain open problems. If you're interested in picking any of this up, here's where things stand:

**What works:**
- A complete C/Vulkan training pipeline (forward, backward, Adam optimizer, data loading, tokenizer)
- Forward Programming fits 3.38B parameters in 893 MB VRAM
- INT2 packing with negligible quality loss (97-99% sparsity, 83% compression)
- Vulkan GPU training path is functional (728 MB VRAM smoke tests pass)
- Loss convergence verified: 12.75 → 1.41 (88.9% reduction) on dev config
- Text generation works (`generate.exe` in v12)

**What's unsolved:**
- Training time: 60B tokens ≈ 495 days on RTX 5070. Needs a fundamental speedup.
- Quality: Forward Programming structural gap is 65-70%. Best closure was 89% with FP+1000 GD steps.
- The 69 mechanisms from R500 project a 2-3× improvement but were never fully implemented.

**Where the most promising leads are:**
- RBLSS (Rao-Blackwell Layer-State Signal) — mathematically proven to close the FP gap
- RSSM (Reversible SSM) — 99.95% trajectory VRAM reduction via exact inversion
- RCFA (Recurrent Cotangent Forward Accumulator) — 4× context expansion
- The v14 Clifford Algebra SSM — the hidden state as a Geometric Multivector in Cl_{3,1}(ℝ) is unexplored territory

See `memory/r500-summary.md` for the full list of validated mechanisms.

---

## Papers

Seven paper drafts exist in `papers/`, in varying stages of completion:

| Paper | Topic | Status |
|-------|-------|--------|
| **paper1-ternary-ssm** | Core architecture — ternary SSM + SWA + MoD | Complete (NeurIPS format) |
| **paper2-gradient-estimators** | Systematic ablation of 8 STE variants | Complete |
| **paper3-efficiency** | Multiply-free efficiency on consumer hardware | Complete |
| **paper4-dataset-quality** | Zero-parameter data quality score from compression theory | Complete |
| **paper5-tssi-import** | Ternary Sufficient-Statistic Import | Complete (no figures yet) |
| **paper6-update-objects** | Replaying training progress instead of tokens | Complete (no figures yet) |
| **paper7-factored-gradient-accumulation** | Avoiding dense LM-head gradient materialization | Draft (preliminary results) |

Papers 1-4 are submission-ready. Papers 5-7 need additional work. Feel free to build on any of them.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use any part of this work — code, ideas, mathematical formulations, or findings — please cite it:

```bibtex
@misc{lena2026ternary,
  title={Ternary State-Space Models: Hardware-Efficient Language Modeling
         with Custom C/Vulkan Training},
  author={Julio Jose Lena},
  year={2026},
  url={https://github.com/alucardeht/ternary-ssm}
}
```

Or simply mention this repository. It would mean a lot.

---

## A Final Note

This project was built entirely by one person on one GPU, driven by the belief that there has to be a fundamentally different way to train language models — not just a faster version of what already exists, but something structurally new.

I don't know if the ternary SSM path is the right one. But I know the questions it asks are worth asking: Can we train without multiplications? Can we fit billions of parameters in under a gigabyte? Can a single consumer GPU do what clusters do today?

I hope someone finds something useful here. Maybe it's a line of math that clicks, or a Vulkan shader trick, or just the conviction that the problem is worth solving differently.

Good luck.

— Julio
