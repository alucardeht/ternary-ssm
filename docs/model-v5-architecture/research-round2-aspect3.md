# Research Round 2, Aspect 3: Convergence Speed Optimization

## Wave 1 — Web Research (2 researchers, Feb 2026)

### Theme A: LR Schedules & Optimizer Configurations

#### R2-A3-F1: FOGZO — First-Order-Guided Zeroth-Order Gradient for STE
- **Source:** Yang et al., NeurIPS 2025, arXiv:2510.23926
- **Summary:** Combines STE's biased-but-fast gradients with zeroth-order unbiased-but-expensive gradients. Uses first-order (STE) guidance to direct ZO sampling for better gradient estimates. 796x cheaper than n-SPSA for equivalent quality.
- **Convergence gain:** 1-8% accuracy improvement at same iteration count vs pure STE. 1-22 perplexity improvement for LLaMA models up to 300M.
- **Applicability:** Directly targets STE bias problem. Hybrid approach keeps STE speed while reducing bias.
- **Verdict:** PROMISING — test as STE replacement or hybrid

#### R2-A3-F2: Shampoo Optimizer — Second-Order, Quantization-Robust
- **Source:** Vlassis et al., arXiv:2509.23500v2, 2025
- **Summary:** Comprehensive study of 6 optimizers (AdamW, PSGD, Shampoo, Muon, SOAP, Scion) under quantization. Shampoo consistently shows LOWEST accuracy degradation under QAT and PTQ. Kronecker-factored curvature approximation naturally smooths loss landscape.
- **Convergence gain:** Highest parameter efficiency under 4-bit QAT. Retains MORE effective capacity than Adam-trained models after quantization.
- **Applicability:** Available in PyTorch/JAX. Second-order curvature smooths the ternary loss landscape.
- **Verdict:** PROMISING — test Shampoo vs Adam for ternary SSM training

#### R2-A3-F3: Muon Optimizer — Matrix-Based, Weaker Error Requirements
- **Source:** Tang et al., arXiv:2510.21314v1, Oct 2025
- **Summary:** SVD-based sign operator avoids error amplification from inverse square root of variance. Requires WEAKER relative error conditions than Adam (1/T^(1/2) vs 1/T^2 for weights/second-moments).
- **Convergence gain:** O(T^(-1/4)) convergence rate with less strict precision. More robust to low-precision training.
- **Applicability:** Designed for matrix-shaped parameters. Needs evaluation on SSM components.
- **Verdict:** PROMISING — particularly interesting for severe ternary quantization errors

#### R2-A3-F4: Adam Asymmetric Precision — v Needs More Precision Than m
- **Source:** Tang et al., arXiv:2510.21314v1, Oct 2025 (same paper as F3)
- **Summary:** Adam is HIGHLY SENSITIVE to quantization of second moments (v_t) due to β₂ → 1. Second-moment precision must be q_V = O(1/T²) while first-moment can be q_M = O(1/T).
- **Convergence gain:** Proper precision allocation achieves O(T^(-1/4)) convergence matching full-precision.
- **Applicability:** Directly actionable — allocate more precision to v than m in our optimizer.
- **Verdict:** PROMISING — explains potential instability and provides exact precision requirements

#### R2-A3-F5: HeRo-Q — Hessian Conditioning for Stable Quantization
- **Source:** arXiv:2601.21626v1, ICML 2026
- **Summary:** Learnable rotation-compression matrix applied to weight space BEFORE quantization. Reduces largest Hessian eigenvalue by 3-4 orders of magnitude (10^7.6 → 10^4.0). W3A16 achieves 70.15% GSM8K vs 46.20% AWQ.
- **Applicability:** Designed for PTQ, but Hessian conditioning principle applicable to QAT.
- **Verdict:** NEEDS-MORE-WORK — requires adaptation to from-scratch QAT scenario

### Theme B: Initialization, Curriculum & Training Dynamics

#### R2-A3-F6: AdaGC — Adaptive Per-Tensor Gradient Clipping
- **Source:** Baidu/ERNIE team, arXiv:2502.11034v2, Feb 2025
- **Summary:** Per-tensor adaptive gradient clipping using EMA of gradient norms: clip_threshold = λ_rel × γ_{t-1,i}. Eliminates loss spikes on Llama-2 7B, Mixtral 8×1B, ERNIE 10B. Works with AdamW, Lion, Muon. Default: λ_rel=1.04, β=0.99.
- **Convergence gain:** Eliminates training instability, +1-2.5% downstream accuracy, 4.48% faster wall-clock.
- **Applicability:** DIRECTLY applicable. Ternary STE gradients are prone to spikes at threshold discontinuities. Per-tensor EMA perfectly addresses this.
- **Verdict:** PROMISING — state-of-the-art stability technique, proven on LLMs

#### R2-A3-F7: Annealing-Inspired Adaptive Ternary Training
- **Source:** Nature Communications, Jan 2025
- **Summary:** Adaptive simulated annealing for ternary weights: n_mirrors = floor(α × NMSE) + 1. Adjusts number of weight flips per epoch proportionally to error. 10x faster convergence than fixed-schedule SA.
- **Convergence gain:** 10x vs traditional SA, 7% accuracy improvement over Boolean weights.
- **Applicability:** Ternary-specific. Adaptive flip strategy could dynamically adjust quantization pressure.
- **Verdict:** PROMISING — proven convergence accelerator specifically for ternary

#### R2-A3-F8: Deep Progressive Training (Depth Expansion)
- **Source:** OpenReview ICLR submission 2025
- **Summary:** Start with 0-1 layers, progressively expand to target depth. For GPT-2 7B (60 layers), saves ~80% compute (5x speedup). Orthogonal to quantization-progressive training.
- **Convergence gain:** 5x acceleration (80% compute savings) at GPT-2 7B scale.
- **Applicability:** Compatible with ternary. Start 6-layer → expand to 12 → 24. Early layers converge fast, expansion reuses learned features.
- **Verdict:** PROMISING — depth-progressive is orthogonal to our quantization approach

#### R2-A3-F9: Cosine Annealing with Warmup + Warm Restarts
- **Source:** Multiple 2025-2026 sources, validated best practice
- **Summary:** Linear warmup prevents early instability from discrete weights, cosine decay helps escape saddle points. CosineAnnealingWarmRestarts enables exploration of new basins. Consensus: 10-20% faster convergence than step decay for quantized models.
- **Convergence gain:** 10-20% faster convergence vs step decay. Warmup prevents early divergence.
- **Applicability:** Trivial to implement. We have basic schedule; add warmup if missing.
- **Verdict:** PROMISING — low-hanging fruit, validated best practice

#### R2-A3-F10: Gradient Accumulation for Batch Scaling
- **Source:** General ML practice, widely validated 2025-2026
- **Summary:** Larger effective batch via gradient accumulation = smoother gradient estimates = less STE noise amplification. Enables 1.5-2x higher LR while maintaining stability. Diminishing returns beyond batch=2048 tokens.
- **Applicability:** Trivial. We already have grad_accum in config. Test larger effective batch sizes.
- **Verdict:** PROMISING — easy win, validated for quantized training

#### R2-A3-F11: Stochastic Depth for Deep Networks
- **Source:** Shadecoder 2025 guide, widely validated
- **Summary:** Randomly skip layers during training (drop probability linearly increasing by depth, 0.1-0.5). Implicit ensemble, reduces overfitting, smooths optimization.
- **Applicability:** Compatible but risky for SSM — skipping SSM layers might break temporal dependencies. Needs careful validation.
- **Verdict:** NEEDS-MORE-WORK — validate SSM layer skipping impact first

#### R2-A3-F12: OneCycleLR Policy
- **Source:** PyTorch/Fastai, 2025 validation
- **Summary:** Single cycle: LR ramps low → max (30%) → low (70%). Momentum inversely changes. Enables much higher peak LR without divergence. 2-3x faster convergence reported.
- **Applicability:** Aggressive, might conflict with ternary instability. High-risk, high-reward.
- **Verdict:** NEEDS-MORE-WORK — test AFTER baseline is solid

## Key Insights — Aspect 3 Wave 1

1. **Second-order optimizers (Shampoo, Muon) are MORE robust to quantization** — curvature-aware updates smooth the ternary loss landscape
2. **Adam's second moment v needs MORE precision than first moment m** — asymmetric precision allocation is actionable
3. **Per-tensor gradient clipping (AdaGC) eliminates spikes** — addresses ternary STE's discontinuity at thresholds
4. **STE bias is addressable** — FOGZO hybrid first-order/zeroth-order reduces bias without full ZO cost
5. **Depth-progressive training is orthogonal** — can combine with ternary training for 5x compute savings
6. **Ternary-specific optimization exists** — annealing-inspired adaptive flip scheduling is proven for ternary

## For Math-Paradigm-Breakers (Wave 2)

1. Can we derive the optimal learning rate schedule for ternary from first principles (accounting for discrete loss landscape)?
2. What is the theoretical convergence rate for ternary coordinate descent vs continuous SGD?
3. Is there a principled way to combine Shampoo's curvature with ternary-specific structure?
4. Can the STE bias be characterized mathematically as a function of distance to threshold?
5. What is the minimum number of gradient observations needed to make a reliable ternary flip decision?
