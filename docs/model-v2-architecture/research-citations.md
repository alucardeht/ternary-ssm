## Appendix: Paper Citations

All techniques cited with publication dates:

| Citation | Details | Year |
|----------|---------|------|
| BitNet: Scaling Quantization to 1-bit Weights and Activations | Ternary quantization foundations | ICML 2024 |
| BitNet b1.58: More Learnable Bits per Weight | 1.58-bit quantization research | 2024 |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | SSM foundation (Mamba-2 SSD) | ICML 2024 |
| Mamba-2: Vision Mambas for efficient image and video understanding | Selective SSM, SSD variant | 2024 |
| SPIDER: Sparse Tensor Core Acceleration | 2:4 structured sparsity on FP16 | PPoPP 2026 |
| Tequila: Learning Ternary Quantization Levels | Deadzone reactivation | 2024 |
| LOTION: Loss Smoothing for Quantization | Smooth loss for discrete quantization | Harvard 2024 |
| COAT: Consistent Low-Precision Training | FP8 training validation | ICLR 2025 |
| GemLite: Distributed Gemini-Lite Inference | BitNet reference kernel | 2024 |
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computation | Custom kernel framework | OpenAI |
| CUTLASS: C++ Tensor Core Kernels | FP4 tensor core templates | NVIDIA |
| FLA (Flash-Linear-Attention) | Fused linear attention scan | Triton |
| Liger Kernels | Fused cross-entropy kernels | 2025 |
| Muon | SGD-like optimizer for ternary networks | 2024 |
| bitnet.cpp | C++ ternary SIMD inference | Microsoft 2025 |
| WebGPU | Browser GPU inference standard | W3C |
| SageAttention2++ | INT8 attention optimization | 2025 |
| PagedAttention | Efficient KV cache management | vLLM |
| ES-SSM: Elastic SSM with Spectral Truncation | Runtime compute-quality tradeoff | 2026 |
| CALM: Early Exit Framework | Confidence-based layer skipping | 2024 |
| Quamba2 | SSM quantization validated at scale | ICML 2025 |
| MambaQuant | Mamba-specific quantization | ICLR 2025 |
| GLA / RetNet | Linear attention alternatives | 2024 |
| BlackMamba / MoE-Mamba | MoE applied to SSM | 2024 |
| Jamba | AI21's 8:1 Mamba:Attention ratio | 2024 |
| Zamba | Strategic attention placement in SSM | 2024 |
| Multi-Token Prediction | Auxiliary heads for verification | Meta 2024 |

---

**Last updated:** 2026-02-25

**Purpose:** Single source of truth consolidating 7 research files (master + 4 rounds + baseline context). All 28 agent findings integrated. Supercedes individual round files.

**How to use:** Future research agents must read Part 1 (measured) + Part 2 (already projected) to avoid duplicate work. Parts 3-8 are novel discoveries. Parts 9-14 provide roadmap and validation strategy.

**Validation baseline:** All findings compared against OUR measured baseline (75.6 tok/s GPU, 14.3 tok/s CPU, 34.33 PPL), not generic benchmarks.
