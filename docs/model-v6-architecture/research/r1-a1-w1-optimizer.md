# Wave 1 - Aspect 1: Optimizer (Energy-based, Sign, Lion)

## Research Context
- **Architecture:** Ternary SSM (State Space Model) with weights in {-1, 0, +1}
- **Focus:** Training speed optimization through optimizer choices
- **Date filter:** 2024-2026 papers only

## Findings

### ID: ZEUS-ENERGY-OPT
- **Src:** https://pytorch.org/blog/zeus/ [2024-11] | **Gain:** Energy measurement + optimization | **V:** PROMISING
- **Principle:** Zeus provides programmatic GPU energy measurement and optimization tools that find best ML/GPU configurations
- **Application:** Integrate ZeusMonitor to measure energy consumption and use GlobalPowerLimitOptimizer to auto-tune GPU power limit for training

### ID: LION-EVOLVED-SIGN
- **Src:** https://optimi.benjaminwarner.dev/optimizers/lion/ [2024] | **Gain:** 10x longer momentum history, reduced memory | **V:** PROMISING
- **Principle:** Lion uses two EMA factors - one for tracking momentum (beta1=0.9) and another for update (beta2=0.99), with sign operation for uniform magnitude updates
- **Application:** Use Lion with lr 3-10x smaller than AdamW, weight decay 3-10x higher. Supports Triton for speedup.

### ID: LION-TRITON-FAST
- **Src:** https://optimi.benjaminwarner.dev/optimizers/lion/ [2024] | **Gain:** Triton implementation significantly faster | **V:** PROMISING
- **Principle:** Lion optimizer with triton=True enables GPU-optimized kernel for faster parameter updates
- **Application:** Enable `triton=True` in Lion optimizer for production training

### ID: SIGN-MOMENTUM-REDUCED-MEMORY
- **Src:** https://optimi.benjaminwarner.dev/optimizers/lion/ [2024] | **Gain:** Reduced memory vs AdamW | **V:** PROMISING
- **Principle:** Lion only keeps gradient moving average (single momentum buffer) vs AdamW's two (momentum + variance)
- **Application:** For ternary SSM with limited VRAM, Lion reduces optimizer state memory by ~33%

### ID: LION-BATCH-SIZE
- **Src:** https://optimi.benjaminwarner.dev/optimizers/lion/ [2024] | **Gain:** Optimal batch 4096, works well at 64 | **V:** SPECULATIVE
- **Principle:** Lion shows better scaling with larger batch sizes, optimal at 4096 vs AdamW's 256
- **Application:** When training ternary SSM with large batch, Lion may outperform AdamW significantly

### ID: WEIGHT-DECAY-LION
- **Src:** https://optimi.benjaminwarner.dev/optimizers/lion/ [2024] | **Gain:** Proper weight decay coupling | **V:** PROMISING
- **Principle:** Since effective weight decay is multiplied by learning rate in Lion, increase weight decay proportionally when reducing lr
- **Application:** When using Lion with lr=1e-4 (3-10x smaller), set weight_decay=0.1-0.3 instead of 0.01

### ID: GRADIENT-RELEASE-LION
- **Src:** https://optimi.benjaminwarner.dev/optimizers/lion/ [2024] | **Gain:** Fuses step+zero_grad, reduces memory | **V:** PROMISING
- **Principle:** Lion supports gradient_release=True to fuse optimizer step with zero_grad as part of backward pass
- **Application:** Enable for ternary SSM training to reduce memory fragmentation and Python overhead

## Key Insights
- Lion optimizer offers 10x longer momentum tracking with single buffer, ideal for ternary weight training
- Triton implementation provides significant speedup for production use
- Energy-based optimization (Zeus) can reduce power consumption with minimal throughput loss
- Sign-based approaches align well with ternary weight nature

## Seed Questions for Wave 2
1. Can energy-based optimization be combined with sign-only training for synergistic gains?
2. How does Lion's sign-based update interact with ternary weight STE gradients?
