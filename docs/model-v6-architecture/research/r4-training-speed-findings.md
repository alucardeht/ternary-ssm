# Research Round 4 - Novel Paradigms for Training Speed

**Context:** Research training speed for ternary SSM models
**Focus:** Novel paradigms - challenge assumptions, radical redesigns
**Date:** 2026-03-05

---

## Aspect 1: Optimizer - Learned Optimizers, Meta-Learning

### R4-A1-01: META-MIRROR-DESCENT
- **Src:** https://arxiv.org/abs/2509.02418 [Sep 2025] | **Gain:** O(ε^-2) convergence, reduces adaptation steps | **V:** PROMISING
- **Principle:** Learns a versatile distance-generating function that induces a nonlinear mirror map to capture complex loss geometries beyond quadratic approximation
- **Application:** Use learned preconditioning for faster per-task convergence in meta-learning scenarios

### R4-A1-02: META-GRADIENT-GENERATOR
- **Src:** https://arxiv.org/abs/2412.16901 [Dec 2024] | **Gain:** 4.2x faster adaptation, 7.4% accuracy improvement | **V:** PROMISING
- **Principle:** Learns to generate gradients using historical gradient information via gradient memory layer with self-supervised reconstruction
- **Application:** Pre-train optimizer on small unlabeled dataset, then deploy for faster online adaptation

### R4-A1-03: CUBIC-ML
- **Src:** https://arxiv.org/abs/2409.04585 [Sep 2024] | **Gain:** Optimizes training speed for 73B-405B parameter models at Meta | **V:** PROMISING
- **Principle:** Uses ML proxy model to predict training performance and automatically optimize co-design hyperparameters for distributed ML systems
- **Application:** Apply co-design optimization to find optimal training configuration for large-scale SSM training

### R4-A1-04: FEDMETANAS
- **Src:** https://arxiv.org/abs/2504.06457 [Apr 2025] | **Gain:** 50%+ speedup in architecture search | **V:** PROMISING
- **Principle:** Integrates meta-learning with neural architecture search to prune search space and eliminate retraining stage
- **Application:** Use meta-learning to quickly find optimal architecture configurations for ternary SSMs

---

## Aspect 2: Backward - Non-Backprop Alternatives

### R4-A2-01: RANDOM-NOISE-PRETRAINING
- **Src:** https://arxiv.org/abs/2405.16731 [May 2024] | **Gain:** Convergence speed comparable to backpropagation | **V:** SPECULATIVE
- **Principle:** Pretraining with random noise modifies forward weights to match backward synaptic feedback, enabling feedback alignment without weight transport
- **Application:** Use random noise pretraining to enable faster convergence with non-backprop methods

---

## Aspect 3: Data - Synthetic Data Generation

### R4-A3-01: METARATER
- **Src:** https://arxiv.org/abs/2504.14194 [Apr 2025] | **Gain:** 2x convergence speed, 3.23% downstream task improvement | **V:** PROMISING
- **Principle:** Multi-dimensional data selection using learned optimal weightings across professionalism, readability, reasoning, cleanliness dimensions
- **Application:** Apply meta-rater for data selection to double convergence speed in pre-training

### R4-A3-02: AUTOSYNTH
- **Src:** https://arxiv.org/abs/2309.11170 [Sep 2023] | **Gain:** 4056x speedup via surrogate network | **V:** SPECULATIVE
- **Principle:** Meta-learning strategy to automatically generate and search for optimal 3D training data using surrogate network
- **Application:** Use synthetic data generation with meta-learning for data-efficient training

---

## Aspect 4: Freezing - RL-Based Freezing Decisions

### R4-A4-01: SEER-FREEZING
- **Src:** https://arxiv.org/abs/2103.02886 [Mar 2021] | **Gain:** Significant computation and memory savings | **V:** PROMISING
- **Principle:** RL-based early freezing of lower layers of CNN encoders due to early convergence, with latent vector storage instead of images
- **Application:** Apply RL-based layer freezing after early convergence detection to save computation

---

## Aspect 5: Sign - Hardware-Aware Sign Operations

### R4-A5-01: HCiM-TERNARY
- **Src:** https://arxiv.org/abs/2403.13577 [Mar 2024] | **Gain:** 28% energy reduction vs ADC-based, 12% vs 4-bit ADC | **V:** PROMISING
- **Principle:** Algorithm-hardware co-design using ternary quantization with analog CiM crossbars and digital processing of scale factors
- **Application:** Use ternary quantization with analog compute-in-memory for hardware-efficient sign operations

---

## Aspect 6: Precision - Learned Quantization

### R4-A6-01: JAQ-FRAMEWORK
- **Src:** https://arxiv.org/abs/2501.05339 [Jan 2025] | **Gain:** 7% higher Top-1 accuracy, 0.15s hardware search time | **V:** PROMISING
- **Principle:** Joint architecture design and low-bit quantization with hardware-software co-exploration using channel-wise sparse quantization
- **Application:** Apply joint optimization of architecture and quantization for optimal accuracy-efficiency trade-off

### R4-A6-02: QUASAR-VIT
- **Src:** https://arxiv.org/abs/2407.18175 [Jul 2024] | **Gain:** 101.5-251.6 FPS on FPGA | **V:** PROMISING
- **Principle:** Hardware-oriented quantization-aware architecture search with row-wise flexible mixed-precision quantization
- **Application:** Use hardware-aware NAS with learned quantization for efficient ViT deployment

### R4-A6-03: SUB-BIT-NN
- **Src:** https://arxiv.org/abs/2110.09195 [Oct 2021] | **Gain:** 3.13-3.33x runtime speedup, 1.8x compression | **V:** PROMISING
- **Principle:** Kernel-aware optimization in fine-grained kernel space rather than per-weight binarization
- **Application:** Apply sub-bit neural network techniques to compress and accelerate binary/ternary networks

---

## Key Insights (max 3 bullets)

1. **Meta-learning optimizers** show 2-4x speedup potential through learned gradient generation and loss geometry capture
2. **Hardware-software co-design** is critical - joint optimization of architecture, quantization, and accelerators yields 7%+ accuracy gains
3. **Multi-dimensional data selection** (Meta-rater) provides 2x convergence speed by optimizing data quality across multiple dimensions

## Seed Questions for Wave 2 (max 2 questions)

1. Can learned optimizers be combined with ternary quantization without gradient degradation?
2. How to apply hardware-aware sign operations to SSM state-space computations efficiently?

---

*End of Round 4 Findings*
