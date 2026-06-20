# Aspect Brief: Training Speed Optimization

## 1. State Snapshot
The GPU pipeline is 100% resident and verified. Peak token throughput on an RTX 5070 for `d_model=1024, d_state=128, L=6` is ~20,142 tok/s. Scaled to a 31B-equivalent class, this projects to ~4.3 years to train 1T tokens. 

## 2. Prior Attempts and Outcomes
- SISSF and GASSC compiler implementations successfully bypassed PyTorch and CUDA overheads.
- CPU fallbacks caused latency and UAF crashes (now fixed).
- TSMSQ (Ternary Shadow-Moment Shared Quantization) successfully reduced VRAM bounds.

## 3. Open Question
How can we fundamentally reshape the mathematical learning path, ternary geometry, or credit assignment to achieve at least 10x-100x theoretical gains in training time for a 31B-equivalent model?

## 4. Acceptance Criteria
- Must be a theoretical or mathematical paradigm shift.
- Must provide a clear proof object, invariant, or convergence guarantee.
- Must include a cheap falsifier.
- Must project a >10% gain (targeting >2x).

## 5. Rejection Criteria
- Code micro-optimizations (e.g., SIMD vectorization, Vulkan pipeline barriers).
- Combinations of existing ideas without a novel mechanism.
- Ideas that violate the Ternary SSM constraints or require full dense matmuls.

## 6. Out of Scope
- Local software engineering tuning.
- CPU/GPU memory shifts (unless they validate a deeper theoretical finding).
- External teacher dependencies (unless the workflow remains fully self-contained on a single GPU).
