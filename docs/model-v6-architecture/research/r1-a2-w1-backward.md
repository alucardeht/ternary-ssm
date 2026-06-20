# Wave 1 - Aspect 2: Backward Pass (Checkpointing, AMP, Fusion)

## Research Context
- **Architecture:** Ternary SSM (State Space Model) with weights in {-1, 0, +1}
- **Focus:** Training speed optimization through backward pass techniques
- **Date filter:** 2024-2026 papers only

## Findings

### ID: GRADIENT-CHECKPOINTING-TRADEOFF
- **Src:** https://www.gilesthomas.com/2024/09/fine-tuning-9 [2024-09] | **Gain:** Enables 2-16x larger batch sizes, 20-50% slowdown per batch | **V:** PROMISING
- **Principle:** Gradient checkpointing trades compute for memory by storing only subset of activations, recomputing rest during backward pass
- **Application:** For 8B model, enables batch size 96→112 with expandable segments. Use for OOM prevention.

### ID: DOUBLE-CHECKPOINTING-SSM
- **Src:** https://arxiv.org/abs/2412.11810 [2024-12] | **Gain:** 10x longer sequences or 4x larger networks | **V:** PROMISING
- **Principle:** Double Checkpointing optimizes local memory resources while minimizing recomputation overhead for sparse RNNs and SNNs
- **Application:** Apply to SSM layers for training on sequences 10x longer than otherwise feasible

### ID: AMP-TENSOR-CORE
- **Src:** https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html [2024] | **Gain:** 2-3X speedup on Tensor Core GPUs | **V:** PROMISING
- **Principle:** PyTorch AMP uses autocast to run ops in FP16/BF16 where faster, GradScaler to prevent gradient underflow
- **Application:** Wrap forward pass with `torch.autocast()` and use `GradScaler` for training

### ID: AMP-MEMORY-SAVINGS
- **Src:** https://www.codegenes.net/blog/pytorch-amp-example/ [2024] | **Gain:** 50% memory reduction | **V:** PROMISING
- **Principle:** Mixed precision reduces activation memory by half since FP16 uses 2 bytes vs FP32's 4
- **Application:** Enable AMP to fit larger batches in same VRAM, especially effective on memory-bound models

### ID: TORCH-COMPILE-FUSION
- **Src:** https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html [2024] | **Gain:** Graph optimization + kernel fusion | **V:** PROMISING
- **Principle:** torch.compile traces forward/backward into optimized graph, fusing operations to reduce kernel launch overhead
- **Application:** Use `torch.compile(mode="reduce-overhead")` for training with large batch sizes

### ID: FUSED-BACKWARD-UPDATE
- **Src:** https://arxiv.org/abs/2412.11810 [2024-12] | **Gain:** Reduced recomputation overhead | **V:** SPECULATIVE
- **Principle:** Combining backward pass with parameter update in single kernel reduces memory bandwidth
- **Application:** For ternary SSM, implement fused gradient accumulation + update kernel

### ID: SSM-CHECKPOINTING-STRATEGY
- **Src:** https://arxiv.org/abs/2412.11810 [2024-12] | **Gain:** Optimal for SSM architecture | **V:** PROMISING
- **Principle:** SSMs have different activation patterns than attention - checkpoint placement should account for selective SSM computation
- **Application:** Place checkpoints before SSM scan operations to maximize memory savings

### ID: SELECTIVE-ACTIVATION-CHECKPOINT
- **Src:** https://arxiv.org/abs/2412.11810 [2024-12] | **Gain:** Adaptive memory reduction | **V:** SPECULATIVE
- **Principle:** Selective checkpointing only recomputes activations that would exceed memory budget
- **Application:** For ternary SSM, checkpoint SSM state computations while keeping embedding gradients

## Key Insights
- Gradient checkpointing enables larger batches but adds 20-50% compute overhead per batch
- AMP provides 2-3x speedup on Tensor Core GPUs with 50% memory savings
- SSM architectures have unique checkpointing opportunities due to linear recurrence
- Double checkpointing extends to 10x longer sequences for RNN/SSM architectures

## Seed Questions for Wave 2
1. How does AMP interact with ternary weight gradients during backward pass?
2. Can checkpointing placement be optimized for SSM's selective scan operations?
